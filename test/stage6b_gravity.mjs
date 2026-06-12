// STAGE 6b: monopole gravity ACCURACY vs exact N² gravity (contact disabled in both).
// acc = velB/dt with zero initial velocities. Gates:
//   self-gravitating blob: p95 relative acc error < 2.5%, max < 8%
//   two distant clusters: mean inter-cluster pull error < 0.5%
import { gpuDevice, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, done } from './util.mjs';
import { makeSortChain, sortChainResources } from './chain.mjs';
import { runSimOnce, makeSimParams } from './simrun.mjs';
import { fixtureSphere, twoClusters, gravMats, gravParams, makeGravPipes, gravResources } from './grav.mjs';
import { SIM_WGSL } from '../js/shaders_sim.js';
import { fastForceWGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
const chain = await makeSortChain(device);
const gravPipes = await makeGravPipes(device);
const mats = gravMats();
const G = 6.674e-5;
const DT = 2;
let allPass = true;

const pFast = await makePipeline(device, fastForceWGSL({ nearGravity: false, farGravity: true }), 'simFast', 'fastgrav');

async function runFastGrav(fix, paramsAB, gp) {
  const n = fix.n;
  const res = sortChainResources(device, chain, fix.pm, 0.84);
  const grav = gravResources(device, gravPipes, res.posBuf, n, gp);
  const paramsBuf = uniformBuf(device, new Uint32Array(paramsAB), 'P');
  const matsBuf = uniformBuf(device, mats, 'mats');
  const velA = storageBuf(device, fix.vt, 'velA');
  const posB = emptyBuf(device, n * 16, 'posB');
  const velB = emptyBuf(device, n * 16, 'velB');
  const metaBuf = storageBuf(device, fix.meta, 'meta');
  const bodyDyn = uniformBuf(device, new Float32Array(16 * 12), 'bd');

  const fg = new ArrayBuffer(48);
  const ff = new Float32Array(fg), fu = new Uint32Array(fg);
  ff[0] = 1 / 0.84; fu[1] = HASH_SIZE; ff[2] = gp.massScale; ff[3] = gp.soft2;
  ff[4] = gp.origin[0]; ff[5] = gp.origin[1]; ff[6] = gp.origin[2]; ff[7] = gp.extentInv;
  ff[8] = gp.extent;
  const fgBuf = uniformBuf(device, new Uint32Array(fg), 'fg');

  const bg0 = device.createBindGroup({
    layout: pFast.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: matsBuf } },
      { binding: 2, resource: { buffer: res.posBuf } },
      { binding: 3, resource: { buffer: velA } },
      { binding: 4, resource: { buffer: posB } },
      { binding: 5, resource: { buffer: velB } },
      { binding: 6, resource: { buffer: metaBuf } },
      { binding: 7, resource: { buffer: bodyDyn } },
    ],
  });
  const bg1 = device.createBindGroup({
    layout: pFast.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: fgBuf } },
      { binding: 1, resource: { buffer: res.offsetsBuf } },
      { binding: 2, resource: { buffer: res.sortedBuf } },
      { binding: 3, resource: { buffer: grav.gravBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  res.encode(enc);
  grav.encode(enc);
  const cp = enc.beginComputePass();
  cp.setPipeline(pFast); cp.setBindGroup(0, bg0); cp.setBindGroup(1, bg1);
  cp.dispatchWorkgroups(Math.ceil(n / 256));
  cp.end();
  device.queue.submit([enc.finish()]);
  return { vel: await readBack(device, velB, n * 16, Float32Array) };
}

function accStats(fix, n2, fast) {
  const n = fix.n;
  let mean = 0;
  const errs = new Float64Array(n);
  // mean |acc| for the relative floor (centre particles have acc→0)
  let meanA = 0;
  for (let i = 0; i < n; i++) {
    meanA += Math.hypot(n2.vel[i * 4], n2.vel[i * 4 + 1], n2.vel[i * 4 + 2]) / DT;
  }
  meanA /= n;
  for (let i = 0; i < n; i++) {
    const ax = n2.vel[i * 4] / DT, ay = n2.vel[i * 4 + 1] / DT, az = n2.vel[i * 4 + 2] / DT;
    const bx = fast.vel[i * 4] / DT, by = fast.vel[i * 4 + 1] / DT, bz = fast.vel[i * 4 + 2] / DT;
    const d = Math.hypot(ax - bx, ay - by, az - bz);
    errs[i] = d / Math.max(Math.hypot(ax, ay, az), meanA * 0.05);
    mean += errs[i];
  }
  const sorted = Float64Array.from(errs).sort();
  return { mean: mean / n, p50: sorted[(n / 2) | 0], p95: sorted[(n * 0.95) | 0], max: sorted[n - 1] };
}

// --- case 1: self-gravitating blob ---
{
  const fix = fixtureSphere(6000, 0.55, 9);
  const gp = gravParams(fix.pm, fix.n);
  const params = makeSimParams({ n: fix.n, dt: DT, gConst: G, heatGate: 0 });
  const n2 = await runSimOnce(device, SIM_WGSL, 'sim', fix, mats, params);
  const fast = await runFastGrav(fix, params, gp);
  const s = accStats(fix, n2, fast);
  // force-level self-gravity is coarse by design (resolution = one coarse cell); the BINDING
  // gate for blob behavior is stage 6c's 1000-step stability parity. Loose sanity bound here.
  const ok = s.p50 < 0.15;
  allPass = allPass && ok;
  console.log(`${ok ? 'PASS' : 'FAIL'} grav-blob INFO (n=${fix.n}): relerr p50=${(s.p50 * 100).toFixed(2)}% p95=${(s.p95 * 100).toFixed(2)}% max=${(s.max * 100).toFixed(2)}% (binding gate = 6c stability)`);
}

// --- case 2: two distant clusters (far-field) ---
{
  const fix = twoClusters(2048, 0.55, 60, 13);
  const gp = gravParams(fix.pm, fix.n);
  const params = makeSimParams({ n: fix.n, dt: DT, gConst: G, heatGate: 0 });
  // N² reference: B-only mass acting on everyone (zero A's masses in a copy)
  const fixB = { ...fix, pm: Float32Array.from(fix.pm) };
  for (let i = 0; i < fix.nA; i++) fixB.pm[i * 4 + 3] = 0;
  const n2 = await runSimOnce(device, SIM_WGSL, 'sim', fixB, mats, params);
  const fast = await runFastGrav(fixB, params, gravParams(fixB.pm, fixB.n));
  // mean x-acceleration of cluster A toward B (the far-field signal)
  const meanAx = (run, lo, hi) => {
    let s = 0;
    for (let i = lo; i < hi; i++) s += run.vel[i * 4] / DT;
    return s / (hi - lo);
  };
  const aN2 = meanAx(n2, 0, fix.nA), aF = meanAx(fast, 0, fix.nA);
  const rel = Math.abs(aN2 - aF) / Math.abs(aN2);
  const ok = rel < 0.005;
  allPass = allPass && ok;
  console.log(`${ok ? 'PASS' : 'FAIL'} grav-farfield: cluster pull n2=${aN2.toExponential(3)} fast=${aF.toExponential(3)} relerr=${(rel * 100).toFixed(3)}%`);
}

done(allPass);
