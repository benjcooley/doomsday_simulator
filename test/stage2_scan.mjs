// STAGE 2: exclusive prefix scan — bit-exact vs JS, on synthetic arrays + a real histogram.
import { gpuDevice, fixtureCloud, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, expectEqual, rng, done } from './util.mjs';
import { SCAN_WGSL, HASH_SIZE, jsCellHash } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
let allPass = true;

const pBlocks = await makePipeline(device, SCAN_WGSL, 'scanBlocks', 'scan');
const pSums = await makePipeline(device, SCAN_WGSL, 'scanSums', 'scan');
const pAdd = await makePipeline(device, SCAN_WGSL, 'addBack', 'scan');

async function runScan(name, src) {
  const n = src.length;
  const sp = new Uint32Array([n, 0, 0, 0]);
  const spBuf = uniformBuf(device, sp, 'sp');
  const srcBuf = storageBuf(device, src, 'src');
  const dstBuf = emptyBuf(device, n * 4, 'dst');
  const sumsBuf = emptyBuf(device, Math.ceil(n / 256) * 4, 'sums');

  const bgBlocks = device.createBindGroup({
    layout: pBlocks.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: spBuf } },
      { binding: 1, resource: { buffer: srcBuf } },
      { binding: 2, resource: { buffer: dstBuf } },
      { binding: 3, resource: { buffer: sumsBuf } },
    ],
  });
  const bgSums = device.createBindGroup({
    layout: pSums.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: spBuf } },
      { binding: 3, resource: { buffer: sumsBuf } },
    ],
  });
  const bgAdd = device.createBindGroup({
    layout: pAdd.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: spBuf } },
      { binding: 2, resource: { buffer: dstBuf } },
      { binding: 3, resource: { buffer: sumsBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  let cp = enc.beginComputePass();
  cp.setPipeline(pBlocks); cp.setBindGroup(0, bgBlocks); cp.dispatchWorkgroups(Math.ceil(n / 256)); cp.end();
  cp = enc.beginComputePass();
  cp.setPipeline(pSums); cp.setBindGroup(0, bgSums); cp.dispatchWorkgroups(1); cp.end();
  cp = enc.beginComputePass();
  cp.setPipeline(pAdd); cp.setBindGroup(0, bgAdd); cp.dispatchWorkgroups(Math.ceil(n / 256)); cp.end();
  device.queue.submit([enc.finish()]);

  const gpuOut = await readBack(device, dstBuf, n * 4, Uint32Array);

  const ref = new Uint32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) { ref[i] = acc; acc += src[i]; }

  allPass = expectEqual(name, gpuOut, ref) && allPass;
}

// synthetic: random counts incl zeros, sizes that are NOT multiples of 256
{
  const r = rng(7);
  const a = new Uint32Array(1000);
  for (let i = 0; i < a.length; i++) a[i] = (r() * 5) | 0;
  await runScan('scan-random-1000', a);
}
{
  const r = rng(8);
  const a = new Uint32Array(HASH_SIZE);
  for (let i = 0; i < a.length; i++) a[i] = r() < 0.05 ? ((r() * 9) | 0) + 1 : 0;
  await runScan('scan-sparse-131072', a);
}
// real histogram from a fixture
{
  const pm = fixtureCloud(8192, 50, 11);
  const counts = new Uint32Array(HASH_SIZE);
  for (let i = 0; i < 8192; i++) counts[jsCellHash(pm[i * 4], pm[i * 4 + 1], pm[i * 4 + 2], 1 / 1.3)]++;
  await runScan('scan-real-histogram', counts);
}

done(allPass);
