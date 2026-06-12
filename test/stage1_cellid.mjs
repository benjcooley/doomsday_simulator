// STAGE 1: cell-ID assignment + histogram — bit-exact vs JS reference, on two fixture shapes.
import { gpuDevice, fixtureCloud, fixtureLattice, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, expectEqual } from './util.mjs';
import { done } from './util.mjs';
import { GRID_WGSL, HASH_SIZE, jsCellHash } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
let allPass = true;

async function runCase(name, pm, cellSize) {
  const n = pm.length / 4;
  const invCell = 1 / cellSize;

  // GPU
  const gp = new ArrayBuffer(16);
  new Float32Array(gp)[0] = invCell;
  new Uint32Array(gp)[1] = n;
  new Uint32Array(gp)[2] = HASH_SIZE;
  const gpBuf = uniformBuf(device, new Uint8Array(gp), 'gp');
  const posBuf = storageBuf(device, pm, 'pos');
  const cellOfBuf = emptyBuf(device, n * 4, 'cellOf');
  const countsBuf = emptyBuf(device, HASH_SIZE * 4, 'counts');

  const pClear = await makePipeline(device, GRID_WGSL, 'clearCounts', 'grid');
  const pCount = await makePipeline(device, GRID_WGSL, 'countCells', 'grid');
  // layout:'auto' includes ONLY statically-used bindings — entries must match per pipeline
  const bgClear = device.createBindGroup({
    layout: pClear.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpBuf } },
      { binding: 3, resource: { buffer: countsBuf } },
    ],
  });
  const bgCount = device.createBindGroup({
    layout: pCount.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpBuf } },
      { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: cellOfBuf } },
      { binding: 3, resource: { buffer: countsBuf } },
    ],
  });
  console.log(`  [${name}] pipelines + bind groups ready`);
  const enc = device.createCommandEncoder();
  let cp = enc.beginComputePass();
  cp.setPipeline(pClear); cp.setBindGroup(0, bgClear); cp.dispatchWorkgroups(Math.ceil(HASH_SIZE / 256));
  cp.end();
  cp = enc.beginComputePass();
  cp.setPipeline(pCount); cp.setBindGroup(0, bgCount); cp.dispatchWorkgroups(Math.ceil(n / 256));
  cp.end();
  device.queue.submit([enc.finish()]);
  console.log(`  [${name}] dispatched, reading back…`);

  const gpuCellOf = await readBack(device, cellOfBuf, n * 4, Uint32Array);
  const gpuCounts = await readBack(device, countsBuf, HASH_SIZE * 4, Uint32Array);

  // JS reference
  const refCellOf = new Uint32Array(n);
  const refCounts = new Uint32Array(HASH_SIZE);
  for (let i = 0; i < n; i++) {
    const h = jsCellHash(pm[i * 4], pm[i * 4 + 1], pm[i * 4 + 2], invCell);
    refCellOf[i] = h;
    refCounts[h]++;
  }

  allPass = expectEqual(`${name}: cellOf`, gpuCellOf, refCellOf) && allPass;
  allPass = expectEqual(`${name}: histogram`, gpuCounts, refCounts) && allPass;
  // sanity: total count equals N
  const total = refCounts.reduce((a, b) => a + b, 0);
  console.log(`  (${name}: ${n} particles in ${refCounts.filter(Boolean).length} occupied cells, total=${total})`);
}

await runCase('cloud-8k', fixtureCloud(8192, 50, 11), 1.3);
await runCase('lattice-4k', fixtureLattice(4096, 0.6, 22), 1.3);
await runCase('cloud-boundary', fixtureCloud(4096, 0.65, 33), 1.3);   // many cell-edge positions


done(allPass);
