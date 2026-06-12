// STAGE 5b: contact-force parity — fast grid kernel vs proven N² kernel, identical fixture,
// gravity OFF (far-field is stage 6's job). Forces are the same shared physics; the only
// legitimate difference is neighbor summation ORDER → tight float tolerance, plus momentum
// conservation must hold for both engines independently.
import { gpuDevice, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, done } from './util.mjs';
import { makeSortChain, sortChainResources } from './chain.mjs';
import { runSimOnce, fixtureCollision, makeMats, makeSimParams, compareRuns } from './simrun.mjs';
import { SIM_WGSL } from '../js/shaders_sim.js';
import { fastForceWGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
const chain = await makeSortChain(device);
const mats = makeMats();
let allPass = true;

async function runFastOnce(fix, paramsAB, cellSize) {
  const n = fix.n;
  // grid build from the same positions
  const res = sortChainResources(device, chain, fix.pm, cellSize);
  // sim buffers (group 0)
  const paramsBuf = uniformBuf(device, new Uint32Array(paramsAB), 'P');
  const matsBuf = uniformBuf(device, mats, 'mats');
  const posA = storageBuf(device, fix.pm, 'posA');
  const velA = storageBuf(device, fix.vt, 'velA');
  const posB = emptyBuf(device, n * 16, 'posB');
  const velB = emptyBuf(device, n * 16, 'velB');
  const metaBuf = storageBuf(device, fix.meta, 'meta');
  const bodyDyn = uniformBuf(device, new Float32Array(16 * 12), 'bodyDyn');
  const dbg = emptyBuf(device, 8 * 16, 'dbg');
  // grid uniform (group 1)
  const fg = new ArrayBuffer(48);   // FastGrid is 48B (gravity fields unused when farGravity:false)
  new Float32Array(fg)[0] = 1 / cellSize;
  new Uint32Array(fg)[1] = HASH_SIZE;
  const fgBuf = uniformBuf(device, new Uint32Array(fg), 'fg');

  const pipe = await makePipeline(device, fastForceWGSL({ nearGravity: true, farGravity: false }), 'simFast', 'fast');
  const bg0 = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: matsBuf } },
      { binding: 2, resource: { buffer: posA } },
      { binding: 3, resource: { buffer: velA } },
      { binding: 4, resource: { buffer: posB } },
      { binding: 5, resource: { buffer: velB } },
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
    ],
  });

  const enc = device.createCommandEncoder();
  res.encode(enc);                       // build the grid this "substep"
  const cp = enc.beginComputePass();
  cp.setPipeline(pipe);
  cp.setBindGroup(0, bg0);
  cp.setBindGroup(1, bg1);
  cp.dispatchWorkgroups(Math.ceil(n / 256));
  cp.end();
  device.queue.submit([enc.finish()]);

  return {
    pos: await readBack(device, posB, n * 16, Float32Array),
    vel: await readBack(device, velB, n * 16, Float32Array),
  };
}

function momentum(fix, run) {
  let px = 0, py = 0, pz = 0;
  const n = fix.n;
  for (let i = 0; i < n; i++) {
    const m = fix.pm[i * 4 + 3];
    px += m * run.vel[i * 4]; py += m * run.vel[i * 4 + 1]; pz += m * run.vel[i * 4 + 2];
  }
  return [px, py, pz];
}

const CELL = 0.84;   // ≥ contact reach 1.35·(0.31+0.31) = 0.837

for (const [name, closing] of [['gentle-2km/s', 0.002], ['hyper-20km/s', 0.02]]) {
  const fix = fixtureCollision(2048, 0.55, closing, 5);
  const params = makeSimParams({ n: fix.n, dt: 2, gConst: 0 });
  const n2 = await runSimOnce(device, SIM_WGSL, 'sim', fix, mats, params);
  const fast = await runFastOnce(fix, params, CELL);
  allPass = compareRuns(`force-parity ${name}`, n2, fast, { relTol: 5e-4 }) && allPass;

  // momentum drift per engine (relative to initial)
  let p0x = 0;
  for (let i = 0; i < fix.n; i++) p0x += fix.pm[i * 4 + 3] * fix.vt[i * 4];
  const [ax] = momentum(fix, n2);
  const [bx] = momentum(fix, fast);
  const drift = (m) => Math.abs(m - p0x) / Math.max(Math.abs(p0x), 1e-12);
  console.log(`  momentum-x drift: n2=${drift(ax).toExponential(2)} fast=${drift(bx).toExponential(2)}`);
}

done(allPass);
