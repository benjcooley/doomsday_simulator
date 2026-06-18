// ARM SPIKE — does merely ARMING a rested planet inject heat, with no impactor at all?
// Reproduces the global cliff: the planet comes to rest (heatGate=0, settleDrag=1/60), then we
// flip BOTH flags the way first-contact does (heatGate=1, settleDrag=0). Any temperature rise is
// the planet releasing its OWN held settling energy — Bug 2 — since nothing hits it.
//   Phase S (settle): heatGate 0, settleDrag 1/60     — "come to rest without heating"
//   Phase A (arm):    heatGate 1, settleDrag 0         — the instant an impact flips the switches
// Variants:
//   relax=true  → a COOL relax first (settleDrag 0, heatGate 0) to reach TRUE equilibrium before
//                 settling+arming. If the spike vanishes, "freeze the settled state" is the fix.
//   dampZ x2    → doubles every material's damping. If the spike is unchanged, damping is being
//                 clamped/ignored (the dt-stability cMax dominates) — confirms that hypothesis.
import { gpuDevice, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline } from './util.mjs';
import { makeSortChain, sortChainResources } from './chain.mjs';
import { gravParams, makeGravPipes, gravResources } from './grav.mjs';
import { makeSimParams } from './simrun.mjs';
import { buildBlob, PHYS } from '../js/blob.js';
import { MAT_TYPES } from '../js/bodies.js';
import { fastForceWGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

const G = 6.674e-5;
const VISR = { IRON: 1.35, ROCK: 1.45, CRUST: 1.45, WATER: 1.75, ICE: 1.55, GAS: 1.8, LAVA: 1.45, ARMED: 1.0 };
const SETTLE_STEPS = +(process.env.SETTLE || 400);
const ARM_STEPS = +(process.env.ARM || 300);
const RELAX_STEPS = +(process.env.RELAX || 400);
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, ...a);

const device = await gpuDevice();
const chain = await makeSortChain(device);
const gravPipes = await makeGravPipes(device);

function wireBlob(blob) {
  const N = blob.count, M = blob.M, R = blob.R, rp = blob.rp;
  const mAvg = M / N;
  const vEsc = Math.sqrt(2 * G * M / R);
  const gSurf = G * M / (R * R);
  const shells = Math.max(2, R / (2 * rp));
  const k = Math.max(mAvg * (vEsc / rp) ** 2, shells * mAvg * gSurf / (0.08 * rp));
  const damp = 2 * 0.45 * Math.sqrt(k * mAvg);
  let minMass = Infinity;
  for (let i = 0; i < N; i++) minMass = Math.min(minMass, blob.mass[i]);
  const dtStable = 0.10 * Math.sqrt(minMass / k);

  const mats = new Float32Array(16 * 16);
  const slotCp = [];
  blob.matsUsed.forEach((mu, s) => {
    const mt = MAT_TYPES[mu.type];
    const coolK = 5.67e-8 * 4 * Math.PI * (rp * 1e6) ** 2 / (mAvg * 1e24 * mt.cp);
    const heatK = 1e12 / mt.cp;
    mats.set([
      rp, k, k * mt.cohF, damp * mt.dampZ / 0.45,
      mu.isSurface ? 1 : 0, heatK, mt.Tsol, mt.Tliq,
      mt.Tvap, mt.emis, VISR[mu.type] || 1.6, mu.type === 'GAS' ? 1 : 0,
      mt.base[0], mt.base[1], mt.base[2], coolK,
    ], s * 16);
    slotCp[s] = mt.cp;
  });

  const pm = new Float32Array(N * 4), vt = new Float32Array(N * 4), meta = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    pm[i * 4] = blob.pos[i * 3]; pm[i * 4 + 1] = blob.pos[i * 3 + 1]; pm[i * 4 + 2] = blob.pos[i * 3 + 2];
    pm[i * 4 + 3] = blob.mass[i];
    vt[i * 4 + 3] = blob.temp[i];
    meta[i] = (s_(blob, i) & 15) | (0 << 4);
  }
  return { N, R, rp, k, dtStable, mats, pm, vt, meta, slotCp, mass: blob.mass };
}
function s_(blob, i) { return blob.matLocal[i]; }  // local slot == matsUsed index in single-body wire

function measure(pos, vel, meta, mass, slotCp, n) {
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += pos[i * 4]; cy += pos[i * 4 + 1]; cz += pos[i * 4 + 2]; }
  cx /= n; cy /= n; cz /= n;
  const r = new Float64Array(n), T = new Float64Array(n);
  let rms = 0, thermalJ = 0;
  for (let i = 0; i < n; i++) {
    r[i] = Math.hypot(pos[i * 4] - cx, pos[i * 4 + 1] - cy, pos[i * 4 + 2] - cz);
    const t = vel[i * 4 + 3]; T[i] = t;
    rms += vel[i * 4] ** 2 + vel[i * 4 + 1] ** 2 + vel[i * 4 + 2] ** 2;
    if (t > 320) thermalJ += mass[i] * 1e24 * (slotCp[meta[i] & 15] || 1000) * (t - 320);
  }
  r.sort((a, b) => a - b); T.sort((a, b) => a - b);
  const q = (arr, p) => arr[Math.min(arr.length - 1, (arr.length * p) | 0)];
  return { r50: q(r, 0.5), tP50: q(T, 0.5), tP99: q(T, 0.99), tMax: T[n - 1], rms: Math.sqrt(rms / n) * 1000, thermalJ };
}

function buildSides(pipe, paramsBuf, matsBuf, pos, vel, metaBuf, bodyDyn, fgBuf, n, gp, CELL) {
  return [0, 1].map((p) => {
    const res = sortChainResources(device, chain, { posBuf: pos[p], n }, CELL);
    const grav = gravResources(device, gravPipes, pos[p], n, gp);
    const bg0 = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } }, { binding: 1, resource: { buffer: matsBuf } },
        { binding: 2, resource: { buffer: pos[p] } }, { binding: 3, resource: { buffer: vel[p] } },
        { binding: 4, resource: { buffer: pos[1 - p] } }, { binding: 5, resource: { buffer: vel[1 - p] } },
        { binding: 6, resource: { buffer: metaBuf } }, { binding: 7, resource: { buffer: bodyDyn } },
      ],
    });
    const bg1 = device.createBindGroup({
      layout: pipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: fgBuf } }, { binding: 1, resource: { buffer: res.offsetsBuf } },
        { binding: 2, resource: { buffer: res.sortedBuf } }, { binding: 3, resource: { buffer: grav.gravBuf } },
      ],
    });
    return { res, grav, bg0, bg1 };
  });
}

async function run(N, { relax = false, dampMul = 1, packComp = null, settle = SETTLE_STEPS } = {}) {
  // optional damping override (restore after)
  const saved = {};
  if (dampMul !== 1) for (const k of Object.keys(MAT_TYPES)) { saved[k] = MAT_TYPES[k].dampZ; MAT_TYPES[k].dampZ *= dampMul; }
  const pcSaved = PHYS.packComp;
  if (packComp !== null) PHYS.packComp = packComp;
  const blob = buildBlob('earth', 6.371, 5.97, N, {});
  PHYS.packComp = pcSaved;
  if (dampMul !== 1) for (const k of Object.keys(saved)) MAT_TYPES[k].dampZ = saved[k];

  const w = wireBlob(blob);
  const n = w.N;
  const CELL = 4 * w.rp;
  const dt = Math.min(Math.max(w.dtStable * 5, 0.3), 8);
  const gp = gravParams(w.pm, n);

  const paramsBuf = uniformBuf(device, new Uint32Array(makeSimParams({ n, dt, gConst: G, heatGate: 0, settleDrag: 1 / 60 })), 'P');
  const matsBuf = uniformBuf(device, w.mats, 'mats');
  const pos = [storageBuf(device, w.pm, 'p0'), emptyBuf(device, n * 16, 'p1')];
  const vel = [storageBuf(device, w.vt, 'v0'), emptyBuf(device, n * 16, 'v1')];
  const metaBuf = storageBuf(device, w.meta, 'meta');
  const bodyDyn = uniformBuf(device, new Float32Array(16 * 12), 'bd');

  const fg = new ArrayBuffer(48);
  const ff = new Float32Array(fg), fu = new Uint32Array(fg);
  ff[0] = 1 / CELL; fu[1] = HASH_SIZE; ff[2] = gp.massScale; ff[3] = gp.soft2;
  ff[4] = gp.origin[0]; ff[5] = gp.origin[1]; ff[6] = gp.origin[2]; ff[7] = gp.extentInv; ff[8] = gp.extent;
  const fgBuf = uniformBuf(device, new Uint32Array(fg), 'fg');

  const pipe = await makePipeline(device, fastForceWGSL({ nearGravity: false, farGravity: true }), 'simFast', 'fast');
  const sides = buildSides(pipe, paramsBuf, matsBuf, pos, vel, metaBuf, bodyDyn, fgBuf, n, gp, CELL);

  let ping = 0;
  const stepN = async (count) => {
    for (let s = 0; s < count; s += 50) {
      const enc = device.createCommandEncoder();
      for (let k = 0; k < Math.min(50, count - s); k++) {
        const S = sides[ping];
        S.res.encode(enc); S.grav.encode(enc);
        const cp = enc.beginComputePass();
        cp.setPipeline(pipe); cp.setBindGroup(0, S.bg0); cp.setBindGroup(1, S.bg1);
        cp.dispatchWorkgroups(Math.ceil(n / 256)); cp.end();
        ping = 1 - ping;
      }
      device.queue.submit([enc.finish()]);
    }
  };
  const snap = async () => measure(
    await readBack(device, pos[ping], n * 16, Float32Array),
    await readBack(device, vel[ping], n * 16, Float32Array), w.meta, w.mass, w.slotCp, n);

  const setP = (heatGate, settleDrag) =>
    device.queue.writeBuffer(paramsBuf, 0, new Uint32Array(makeSimParams({ n, dt, gConst: G, heatGate, settleDrag })));

  log(`  build+wire done, N=${n}, dt=${dt.toFixed(3)} (dtStable=${w.dtStable.toFixed(4)})`);
  if (relax) { setP(0, 0); await stepN(RELAX_STEPS); log('  relax done'); }   // cool relax to TRUE equilibrium
  setP(0, 1 / 60); await stepN(settle);                   // Phase S: come to rest, no heat
  const rested = await snap(); log(`  settled: rms=${rested.rms.toFixed(3)} Tp99=${rested.tP99.toFixed(0)}`);
  setP(1, 0); await stepN(ARM_STEPS);                     // Phase A: ARM — flip both flags
  const armed = await snap(); log(`  armed:   rms=${armed.rms.toFixed(3)} Tp99=${armed.tP99.toFixed(0)}`);

  return { N: n, dt, dtStable: w.dtStable, rested, armed };
}

const chicxJ = 4.2e23;
console.log('ARM SPIKE — heat injected by ARMING a rested planet (no impactor). 63k = above the dt-floor.\n');
console.log('  case                | rested rms | armed rms | rested Tp99 | armed Tp99 | ΔthermalJ (×Chicx)');
const ALL = [
  ['63k pc0 settle1000',  63000, { packComp: 0, settle: 1000 }],
  ['63k pc0 settle1400',  63000, { packComp: 0, settle: 1400 }],
  ['63k pc0 settle1800',  63000, { packComp: 0, settle: 1800 }],
];
// CASES env picks a subset by index, e.g. CASES=0 for just the baseline (fast sanity run)
const cases = process.env.CASES ? process.env.CASES.split(',').map((i) => ALL[+i]) : ALL;
for (const [name, N, opt] of cases) {
  const r = await run(N, opt);
  const dJ = r.armed.thermalJ - r.rested.thermalJ;
  console.log(`  ${name.padEnd(20)}| ${r.rested.rms.toFixed(3).padStart(9)} | ${r.armed.rms.toFixed(3).padStart(8)} | ${r.rested.tP99.toFixed(0).padStart(10)} | ${r.armed.tP99.toFixed(0).padStart(9)} | ${(dJ / chicxJ).toExponential(2).padStart(10)}`);
}
