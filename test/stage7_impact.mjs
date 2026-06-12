// STAGE 7: full IMPACT parity — two blocks collide at 10 km/s with heating enabled,
// 1,000 steps in both engines.
//
// ACCEPTED MODEL ERROR (user-approved 2026-06-12): monopole gravity carries ~5-7% mid-field
// force error (stage 6b INFO). In free-expanding debris (no contact), it integrates into
// ~2%R median-radius drift per 1000 steps — envelope (r90), temperature, momentum all <1%.
// Visually indistinguishable; revisit with CIC deposit if runtime A/B ever shows it.
// Gates: r50 ≤ 2.5%R per 1000 steps, r90 ≤ 1%R, meanT ≤ 5%, chaos floor reported.
import { gpuDevice, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, done } from './util.mjs';
import { makeSortChain, sortChainResources } from './chain.mjs';
import { makeMats, makeSimParams } from './simrun.mjs';
import { gravParams, makeGravPipes, gravResources } from './grav.mjs';
import { fixtureCollision } from './simrun.mjs';
import { SIM_WGSL } from '../js/shaders_sim.js';
import { fastForceWGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
const chain = await makeSortChain(device);
const gravPipes = await makeGravPipes(device);
const mats = makeMats();
const G = 6.674e-5;
const DT = 2;
const STEPS = 1000;
const CHECKS = [250, 500, 750, 1000];
const CELL = 0.84;
let allPass = true;

const fix = fixtureCollision(3000, 0.63, 0.01, 5);   // two blocks, 10 km/s impact
fix.R = 8;
console.log(`impact fixture: n=${fix.n} closing 10 km/s, heat ON`);
const params = makeSimParams({ n: fix.n, dt: DT, gConst: G, heatGate: 1 });
const gp = gravParams(fix.pm, fix.n);

function profile(pos, vel, n) {
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += pos[i * 4]; cy += pos[i * 4 + 1]; cz += pos[i * 4 + 2]; }
  cx /= n; cy /= n; cz /= n;
  const r = new Float64Array(n);
  let rms = 0;
  for (let i = 0; i < n; i++) {
    r[i] = Math.hypot(pos[i * 4] - cx, pos[i * 4 + 1] - cy, pos[i * 4 + 2] - cz);
    rms += vel[i * 4] ** 2 + vel[i * 4 + 1] ** 2 + vel[i * 4 + 2] ** 2;
  }
  r.sort();
  let meanT = 0;
  for (let i = 0; i < n; i++) meanT += vel[i * 4 + 3];
  return { r50: r[(n / 2) | 0], r90: r[(n * 0.9) | 0], rms: Math.sqrt(rms / n) * 1000, meanT: meanT / n };
}

// ---------- N² multi-step ----------
async function runN2() {
  const paramsBuf = uniformBuf(device, new Uint32Array(params), 'P');
  const matsBuf = storageBuf(device, mats, 'mats');
  const pos = [storageBuf(device, fix.pm, 'p0'), emptyBuf(device, fix.n * 16, 'p1')];
  const vel = [storageBuf(device, fix.vt, 'v0'), emptyBuf(device, fix.n * 16, 'v1')];
  const metaBuf = storageBuf(device, fix.meta, 'meta');
  const bodyDyn = storageBuf(device, new Float32Array(16 * 12), 'bd');
  const dbg = emptyBuf(device, 128, 'dbg');
  const pipe = await makePipeline(device, SIM_WGSL, 'sim', 'n2');
  const bgs = [0, 1].map((p) => device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: matsBuf } },
      { binding: 2, resource: { buffer: pos[p] } },
      { binding: 3, resource: { buffer: vel[p] } },
      { binding: 4, resource: { buffer: pos[1 - p] } },
      { binding: 5, resource: { buffer: vel[1 - p] } },
      { binding: 6, resource: { buffer: metaBuf } },
      { binding: 7, resource: { buffer: bodyDyn } },
      { binding: 8, resource: { buffer: dbg } },
    ],
  }));
  const out = [];
  let ping = 0;
  for (let s = 0; s < STEPS; s += 50) {
    const enc = device.createCommandEncoder();
    for (let k = 0; k < 50; k++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe); cp.setBindGroup(0, bgs[ping]); cp.dispatchWorkgroups(Math.ceil(fix.n / 256));
      cp.end();
      ping = 1 - ping;
    }
    device.queue.submit([enc.finish()]);
    if (CHECKS.includes(s + 50)) {
      out.push(profile(
        await readBack(device, pos[ping], fix.n * 16, Float32Array),
        await readBack(device, vel[ping], fix.n * 16, Float32Array), fix.n));
    }
  }
  return out;
}

// ---------- fast multi-step (grid + grav rebuilt every step from current positions) ----------
async function runFast() {
  const paramsBuf = uniformBuf(device, new Uint32Array(params), 'P');
  const matsBuf = uniformBuf(device, mats, 'mats');
  const pos = [storageBuf(device, fix.pm, 'p0'), emptyBuf(device, fix.n * 16, 'p1')];
  const vel = [storageBuf(device, fix.vt, 'v0'), emptyBuf(device, fix.n * 16, 'v1')];
  const metaBuf = storageBuf(device, fix.meta, 'meta');
  const bodyDyn = uniformBuf(device, new Float32Array(16 * 12), 'bd');
  const fg = new ArrayBuffer(48);
  const ff = new Float32Array(fg), fu = new Uint32Array(fg);
  ff[0] = 1 / CELL; fu[1] = HASH_SIZE; ff[2] = gp.massScale; ff[3] = gp.soft2;
  ff[4] = gp.origin[0]; ff[5] = gp.origin[1]; ff[6] = gp.origin[2]; ff[7] = gp.extentInv;
  ff[8] = gp.extent;
  const fgBuf = uniformBuf(device, new Uint32Array(fg), 'fg');
  const pipe = await makePipeline(device, fastForceWGSL({ nearGravity: false, farGravity: true }), 'simFast', 'fast');

  const sides = [0, 1].map((p) => {
    const res = sortChainResources(device, chain, { posBuf: pos[p], n: fix.n }, CELL);
    const grav = gravResources(device, gravPipes, pos[p], fix.n, gp);
    const bg0 = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: matsBuf } },
        { binding: 2, resource: { buffer: pos[p] } },
        { binding: 3, resource: { buffer: vel[p] } },
        { binding: 4, resource: { buffer: pos[1 - p] } },
        { binding: 5, resource: { buffer: vel[1 - p] } },
        { binding: 6, resource: { buffer: metaBuf } },
        { binding: 7, resource: { buffer: bodyDyn } },
      ],
    });
    const bg1 = device.createBindGroup({
      layout: pipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: fgBuf } },
        { binding: 1, resource: { buffer: res.offsetsBuf } },
        { binding: 2, resource: { buffer: res.sortedBuf } },
        { binding: 3, resource: { buffer: grav.gravBuf } },
      ],
    });
    return { res, grav, bg0, bg1 };
  });

  const out = [];
  let ping = 0;
  for (let s = 0; s < STEPS; s += 50) {
    const enc = device.createCommandEncoder();
    for (let k = 0; k < 50; k++) {
      const S = sides[ping];
      S.res.encode(enc);
      S.grav.encode(enc);
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe); cp.setBindGroup(0, S.bg0); cp.setBindGroup(1, S.bg1);
      cp.dispatchWorkgroups(Math.ceil(fix.n / 256));
      cp.end();
      ping = 1 - ping;
    }
    device.queue.submit([enc.finish()]);
    if (CHECKS.includes(s + 50)) {
      out.push(profile(
        await readBack(device, pos[ping], fix.n * 16, Float32Array),
        await readBack(device, vel[ping], fix.n * 16, Float32Array), fix.n));
    }
  }
  return out;
}

// chaos floor: the same N² engine vs itself with ONE particle perturbed by 1e-9 km/s —
// in a 22,000K chaotic debris cloud this measures how fast ANY difference amplifies.
async function runN2Perturbed() {
  const save = fix.vt[0];
  fix.vt[0] = Math.fround(fix.vt[0] + 1e-9);
  const out = await runN2();
  fix.vt[0] = save;
  return out;
}

const t0 = performance.now();
const n2 = await runN2();
const t1 = performance.now();
const fast = await runFast();
const t2 = performance.now();
const n2p = await runN2Perturbed();
console.log(`runtime: n2=${((t1 - t0) / 1000).toFixed(1)}s fast=${((t2 - t1) / 1000).toFixed(1)}s for ${STEPS} steps`);

for (let c = 0; c < CHECKS.length; c++) {
  const dr50 = Math.abs(n2[c].r50 - fast[c].r50) / fix.R;
  const dr90 = Math.abs(n2[c].r90 - fast[c].r90) / fix.R;
  const dT = Math.abs(n2[c].meanT - fast[c].meanT) / Math.max(n2[c].meanT, 1);
  // chaos floor: divergence of the SAME engine under a 1e-9 perturbation
  const f50 = Math.abs(n2[c].r50 - n2p[c].r50) / fix.R;
  const f90 = Math.abs(n2[c].r90 - n2p[c].r90) / fix.R;
  const ok = (dr50 <= Math.max(0.025 * (CHECKS[c] / 1000), 1.5 * f50)) && (dr90 <= Math.max(0.01, 1.5 * f90)) && dT <= 0.05;
  allPass = allPass && ok;
  console.log(`${ok ? 'PASS' : 'FAIL'} step ${CHECKS[c]}: ` +
    `r50 Δ${(dr50 * 100).toFixed(2)}%R (chaos floor ${(f50 * 100).toFixed(2)}%) ` +
    `r90 Δ${(dr90 * 100).toFixed(2)}%R (floor ${(f90 * 100).toFixed(2)}%) | ` +
    `meanT Δ${(dT * 100).toFixed(2)}% | rms n2=${n2[c].rms.toFixed(2)} fast=${fast[c].rms.toFixed(2)}`);
}

done(allPass);
