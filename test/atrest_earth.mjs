// AT-REST EARTH — does an undisturbed planet stay cool and still as particle count rises?
// Builds the REAL layered Earth (buildBlob) at several N, wires the shipping FAST engine with
// the exact material constants particles.js derives, runs it at the sim's idle params
// (self-gravity ON, heatGate OFF, settle-drag ON toward the rigid reference), and tracks:
//   - temperature percentiles  (p50 / p99 / max)  → is energy being injected?
//   - RMS internal speed (km/s)                    → mechanical churn (should be ~0)
//   - r50 / r90 radius                             → is the lattice breathing / collapsing?
// The question: are these INVARIANT to N, or does a 50k+ Earth heat / churn at rest?
import { gpuDevice, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, done } from './util.mjs';
import { makeSortChain, sortChainResources } from './chain.mjs';
import { gravParams, makeGravPipes, gravResources } from './grav.mjs';
import { makeSimParams } from './simrun.mjs';
import { buildBlob } from '../js/blob.js';
import { MAT_TYPES } from '../js/bodies.js';
import { fastForceWGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

const G = 6.674e-5;
const COUNTS = [50000, 200000, 700000];
const PRES = [1.0, 0.90];
const STEPS = 600;
const CHECKS = [100, 300, 600];
const VISR = { IRON: 1.35, ROCK: 1.45, CRUST: 1.45, WATER: 1.75, ICE: 1.55, GAS: 1.8, LAVA: 1.45, ARMED: 1.0 };

const device = await gpuDevice();
const chain = await makeSortChain(device);
const gravPipes = await makeGravPipes(device);

// Build mats[16] + meta from a blob, mirroring particles.js addBlob exactly.
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
  const slotOf = [];
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
    slotOf.push(s);
  });

  const pm = new Float32Array(N * 4), vt = new Float32Array(N * 4), meta = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    pm[i * 4] = blob.pos[i * 3]; pm[i * 4 + 1] = blob.pos[i * 3 + 1]; pm[i * 4 + 2] = blob.pos[i * 3 + 2];
    pm[i * 4 + 3] = blob.mass[i];
    vt[i * 4 + 3] = blob.temp[i];
    meta[i] = (slotOf[blob.matLocal[i]] & 15) | (0 << 4);
  }
  return { N, R, rp, k, dtStable, mats, pm, vt, meta };
}

function profile(pos, vel, n) {
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += pos[i * 4]; cy += pos[i * 4 + 1]; cz += pos[i * 4 + 2]; }
  cx /= n; cy /= n; cz /= n;
  const r = new Float64Array(n), T = new Float64Array(n);
  let rms = 0;
  for (let i = 0; i < n; i++) {
    r[i] = Math.hypot(pos[i * 4] - cx, pos[i * 4 + 1] - cy, pos[i * 4 + 2] - cz);
    T[i] = vel[i * 4 + 3];
    rms += vel[i * 4] ** 2 + vel[i * 4 + 1] ** 2 + vel[i * 4 + 2] ** 2;
  }
  r.sort((a, b) => a - b); T.sort((a, b) => a - b);
  const q = (arr, p) => arr[Math.min(arr.length - 1, (arr.length * p) | 0)];
  return {
    r50: q(r, 0.5), r90: q(r, 0.9),
    tP50: q(T, 0.5), tP99: q(T, 0.99), tMax: T[n - 1],
    rms: Math.sqrt(rms / n) * 1000,
  };
}

async function runAtRest(N, pre) {
  const blob = buildBlob('earth', 6.371, 5.97, N, {});
  if (pre !== 1) for (let i = 0; i < blob.count * 3; i++) blob.pos[i] *= pre;
  const w = wireBlob(blob);
  const n = w.N;
  const CELL = 4 * w.rp;
  const dt = Math.min(Math.max(w.dtStable * 5, 0.3), 8);
  const gp = gravParams(w.pm, n);

  // idle params: self-gravity ON (farGravity), heatGate OFF (no event), settle-drag ON.
  const params = makeSimParams({ n, dt, gConst: G, heatGate: 0, settleDrag: 1 / 60, settleBoost: 1 });
  const paramsBuf = uniformBuf(device, new Uint32Array(params), 'P');
  const matsBuf = uniformBuf(device, w.mats, 'mats');
  const pos = [storageBuf(device, w.pm, 'p0'), emptyBuf(device, n * 16, 'p1')];
  const vel = [storageBuf(device, w.vt, 'v0'), emptyBuf(device, n * 16, 'v1')];
  const metaBuf = storageBuf(device, w.meta, 'meta');
  // rigid reference for settle drag: body 0 at rest at origin, no spin
  const bodyDyn = uniformBuf(device, new Float32Array(16 * 12), 'bd');

  const fg = new ArrayBuffer(48);
  const ff = new Float32Array(fg), fu = new Uint32Array(fg);
  ff[0] = 1 / CELL; fu[1] = HASH_SIZE; ff[2] = gp.massScale; ff[3] = gp.soft2;
  ff[4] = gp.origin[0]; ff[5] = gp.origin[1]; ff[6] = gp.origin[2]; ff[7] = gp.extentInv;
  ff[8] = gp.extent;
  const fgBuf = uniformBuf(device, new Uint32Array(fg), 'fg');

  const pipe = await makePipeline(device, fastForceWGSL({ nearGravity: false, farGravity: true }), 'simFast', 'fast');
  const sides = [0, 1].map((p) => {
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

  const spawn = profile(w.pm, w.vt, n);
  const out = [{ step: 0, ...spawn }];
  let ping = 0;
  for (let s = 0; s < STEPS; s += 50) {
    const enc = device.createCommandEncoder();
    for (let k = 0; k < 50; k++) {
      const S = sides[ping];
      S.res.encode(enc); S.grav.encode(enc);
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe); cp.setBindGroup(0, S.bg0); cp.setBindGroup(1, S.bg1);
      cp.dispatchWorkgroups(Math.ceil(n / 256));
      cp.end();
      ping = 1 - ping;
    }
    device.queue.submit([enc.finish()]);
    if (CHECKS.includes(s + 50)) {
      out.push({
        step: s + 50, ...profile(
          await readBack(device, pos[ping], n * 16, Float32Array),
          await readBack(device, vel[ping], n * 16, Float32Array), n),
      });
    }
  }
  return { N: n, R: w.R, rp: w.rp, dt, dtStable: w.dtStable, rows: out };
}

console.log('AT-REST EARTH — collapse & popping vs particle count, pre-compress 1.00 vs 0.90\n');
console.log('   N      pre | spawn-r50→final | collapse | peak-rms(pop) | final-rms | ΔTp99');
for (const C of COUNTS) {
  for (const pre of PRES) {
    const r = await runAtRest(C, pre);
    const f = r.rows[0], l = r.rows[r.rows.length - 1];
    let peak = 0; for (const o of r.rows) peak = Math.max(peak, o.rms);
    const coll = (l.r50 - f.r50) / f.r50 * 100;
    console.log(`  ${String(r.N).padStart(6)} ${pre.toFixed(2)} | ${f.r50.toFixed(3)}→${l.r50.toFixed(3)}   | ${coll.toFixed(1).padStart(5)}%  |    ${peak.toFixed(3)}     |   ${l.rms.toFixed(3)}   | ${(l.tP99-f.tP99>=0?'+':'')}${(l.tP99-f.tP99).toFixed(0)}K`);
  }
}
