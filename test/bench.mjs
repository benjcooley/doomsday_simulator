// Benchmark: N² kernel vs fast grid kernel, dense lattice (contact-heavy worst case).
import { gpuDevice, storageBuf, emptyBuf, uniformBuf, makePipeline, fixtureLattice, done } from './util.mjs';
import { makeSortChain, sortChainResources } from './chain.mjs';
import { fixtureCollision, makeMats, makeSimParams } from './simrun.mjs';
import { SIM_WGSL } from '../js/shaders_sim.js';
import { fastForceWGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
const chain = await makeSortChain(device);
const mats = makeMats();
const CELL = 0.84;
const REPS = 30;

async function bench(n) {
  const fix = fixtureCollision(n / 2, 0.55, 0.002, 5);
  const params = makeSimParams({ n: fix.n, dt: 2, gConst: 6.674e-5 });

  // --- N² ---
  const mk = (data) => storageBuf(device, data, 'b');
  const paramsBuf = uniformBuf(device, new Uint32Array(params), 'P');
  const common = {
    posA: mk(fix.pm), velA: mk(fix.vt), posB: emptyBuf(device, fix.n * 16), velB: emptyBuf(device, fix.n * 16),
    meta: mk(fix.meta), dbg: emptyBuf(device, 128),
  };
  const pN2 = await makePipeline(device, SIM_WGSL, 'sim', 'n2');
  const bgN2 = device.createBindGroup({
    layout: pN2.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: mk(mats) } },
      { binding: 2, resource: { buffer: common.posA } },
      { binding: 3, resource: { buffer: common.velA } },
      { binding: 4, resource: { buffer: common.posB } },
      { binding: 5, resource: { buffer: common.velB } },
      { binding: 6, resource: { buffer: common.meta } },
      { binding: 7, resource: { buffer: mk(new Float32Array(16 * 12)) } },
      { binding: 8, resource: { buffer: common.dbg } },
    ],
  });
  const runN2 = () => {
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(pN2); cp.setBindGroup(0, bgN2); cp.dispatchWorkgroups(Math.ceil(fix.n / 256)); cp.end();
    device.queue.submit([enc.finish()]);
  };
  runN2(); await device.queue.onSubmittedWorkDone();   // warm
  let t0 = performance.now();
  for (let r = 0; r < REPS; r++) runN2();
  await device.queue.onSubmittedWorkDone();
  const msN2 = (performance.now() - t0) / REPS;

  // --- fast (grid rebuild + force each rep, like production) ---
  const res = sortChainResources(device, chain, fix.pm, CELL);
  const fg = new ArrayBuffer(16);
  new Float32Array(fg)[0] = 1 / CELL;
  new Uint32Array(fg)[1] = HASH_SIZE;
  const pFast = await makePipeline(device, fastForceWGSL({ nearGravity: true }), 'simFast', 'fast');
  const bg0 = device.createBindGroup({
    layout: pFast.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: uniformBuf(device, mats, 'mats') } },
      { binding: 2, resource: { buffer: res.posBuf } },
      { binding: 3, resource: { buffer: common.velA } },
      { binding: 4, resource: { buffer: common.posB } },
      { binding: 5, resource: { buffer: common.velB } },
      { binding: 6, resource: { buffer: common.meta } },
      { binding: 7, resource: { buffer: uniformBuf(device, new Float32Array(16 * 12), 'bd') } },
      { binding: 8, resource: { buffer: common.dbg } },
    ],
  });
  const bg1 = device.createBindGroup({
    layout: pFast.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf(device, new Uint32Array(fg), 'fg') } },
      { binding: 1, resource: { buffer: res.offsetsBuf } },
      { binding: 2, resource: { buffer: res.sortedBuf } },
    ],
  });
  const runFast = () => {
    const enc = device.createCommandEncoder();
    res.encode(enc);                       // full grid rebuild, like production
    const cp = enc.beginComputePass();
    cp.setPipeline(pFast); cp.setBindGroup(0, bg0); cp.setBindGroup(1, bg1);
    cp.dispatchWorkgroups(Math.ceil(fix.n / 256)); cp.end();
    device.queue.submit([enc.finish()]);
  };
  runFast(); await device.queue.onSubmittedWorkDone();
  t0 = performance.now();
  for (let r = 0; r < REPS; r++) runFast();
  await device.queue.onSubmittedWorkDone();
  const msFast = (performance.now() - t0) / REPS;

  console.log(`n=${fix.n}: N²=${msN2.toFixed(2)}ms  fast=${msFast.toFixed(2)}ms  speedup=${(msN2 / msFast).toFixed(1)}×  (NOTE: fast lacks far gravity until stage 6)`);
}

await bench(4096);
await bench(16384);
await bench(32768);
done(true);
