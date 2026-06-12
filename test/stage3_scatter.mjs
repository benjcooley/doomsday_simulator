// STAGE 3: full counting-sort chain (clear → count → scan → scatter).
// Oracle: sortedIdx is a permutation, and each cell's segment equals the JS reference
// membership set (order within a cell is nondeterministic — compared as sorted sets).
import { gpuDevice, fixtureCloud, fixtureLattice, storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, done } from './util.mjs';
import { GRID_WGSL, SCAN_WGSL, SCATTER_WGSL, HASH_SIZE, jsCellHash } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
let allPass = true;

const pClear = await makePipeline(device, GRID_WGSL, 'clearCounts', 'grid');
const pCount = await makePipeline(device, GRID_WGSL, 'countCells', 'grid');
const pBlocks = await makePipeline(device, SCAN_WGSL, 'scanBlocks', 'scan');
const pSums = await makePipeline(device, SCAN_WGSL, 'scanSums', 'scan');
const pAdd = await makePipeline(device, SCAN_WGSL, 'addBack', 'scan');
const pScatter = await makePipeline(device, SCATTER_WGSL, 'scatter', 'scatter');

async function runCase(name, pm, cellSize) {
  const n = pm.length / 4;
  const invCell = 1 / cellSize;

  const gp = new ArrayBuffer(16);
  new Float32Array(gp)[0] = invCell;
  new Uint32Array(gp)[1] = n;
  new Uint32Array(gp)[2] = HASH_SIZE;
  const gpBuf = uniformBuf(device, new Uint32Array(gp), 'gp');
  const spBuf = uniformBuf(device, new Uint32Array([HASH_SIZE, 0, 0, 0]), 'sp');

  const posBuf = storageBuf(device, pm, 'pos');
  const cellOfBuf = emptyBuf(device, n * 4, 'cellOf');
  const countsBuf = emptyBuf(device, HASH_SIZE * 4, 'counts');
  const offsetsBuf = emptyBuf(device, HASH_SIZE * 4, 'offsets');
  const sumsBuf = emptyBuf(device, (HASH_SIZE / 256) * 4, 'sums');
  const cursorBuf = emptyBuf(device, HASH_SIZE * 4, 'cursor');
  const sortedBuf = emptyBuf(device, n * 4, 'sorted');

  const bg = (pipe, list) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: list });
  const bgClear = bg(pClear, [
    { binding: 0, resource: { buffer: gpBuf } }, { binding: 3, resource: { buffer: countsBuf } }]);
  const bgCount = bg(pCount, [
    { binding: 0, resource: { buffer: gpBuf } }, { binding: 1, resource: { buffer: posBuf } },
    { binding: 2, resource: { buffer: cellOfBuf } }, { binding: 3, resource: { buffer: countsBuf } }]);
  const bgBlocks = bg(pBlocks, [
    { binding: 0, resource: { buffer: spBuf } }, { binding: 1, resource: { buffer: countsBuf } },
    { binding: 2, resource: { buffer: offsetsBuf } }, { binding: 3, resource: { buffer: sumsBuf } }]);
  const bgSums = bg(pSums, [
    { binding: 0, resource: { buffer: spBuf } }, { binding: 3, resource: { buffer: sumsBuf } }]);
  const bgAdd = bg(pAdd, [
    { binding: 0, resource: { buffer: spBuf } }, { binding: 2, resource: { buffer: offsetsBuf } },
    { binding: 3, resource: { buffer: sumsBuf } }]);
  const bgScatter = bg(pScatter, [
    { binding: 0, resource: { buffer: gpBuf } }, { binding: 1, resource: { buffer: cellOfBuf } },
    { binding: 2, resource: { buffer: cursorBuf } }, { binding: 3, resource: { buffer: sortedBuf } }]);

  const enc = device.createCommandEncoder();
  const run = (pipe, group, wgs) => {
    const cp = enc.beginComputePass();
    cp.setPipeline(pipe); cp.setBindGroup(0, group); cp.dispatchWorkgroups(wgs); cp.end();
  };
  run(pClear, bgClear, Math.ceil(HASH_SIZE / 256));
  run(pCount, bgCount, Math.ceil(n / 256));
  run(pBlocks, bgBlocks, Math.ceil(HASH_SIZE / 256));
  run(pSums, bgSums, 1);
  run(pAdd, bgAdd, Math.ceil(HASH_SIZE / 256));
  enc.copyBufferToBuffer(offsetsBuf, 0, cursorBuf, 0, HASH_SIZE * 4);   // cursor = offsets
  run(pScatter, bgScatter, Math.ceil(n / 256));
  device.queue.submit([enc.finish()]);

  const sorted = await readBack(device, sortedBuf, n * 4, Uint32Array);
  const offsets = await readBack(device, offsetsBuf, HASH_SIZE * 4, Uint32Array);

  // JS reference membership
  const refCells = new Map();
  const refCounts = new Uint32Array(HASH_SIZE);
  for (let i = 0; i < n; i++) {
    const h = jsCellHash(pm[i * 4], pm[i * 4 + 1], pm[i * 4 + 2], invCell);
    refCounts[h]++;
    if (!refCells.has(h)) refCells.set(h, []);
    refCells.get(h).push(i);
  }

  // permutation check
  const seen = new Uint8Array(n);
  let perm = true;
  for (let k = 0; k < n; k++) {
    const v = sorted[k];
    if (v >= n || seen[v]) { perm = false; break; }
    seen[v] = 1;
  }
  console.log(perm ? `PASS ${name}: permutation` : `FAIL ${name}: not a permutation`);
  allPass = allPass && perm;

  // per-cell set equality
  let cellsOK = true, checked = 0;
  for (const [h, members] of refCells) {
    const start = offsets[h];
    const seg = Array.from(sorted.slice(start, start + refCounts[h])).sort((a, b) => a - b);
    const want = members.slice().sort((a, b) => a - b);
    if (seg.length !== want.length || seg.some((v, i) => v !== want[i])) {
      console.error(`FAIL ${name}: cell ${h} segment mismatch`);
      cellsOK = false;
      break;
    }
    checked++;
  }
  if (cellsOK) console.log(`PASS ${name}: per-cell membership (${checked} cells)`);
  allPass = allPass && cellsOK;
}

await runCase('sort-cloud-8k', fixtureCloud(8192, 50, 11), 1.3);
await runCase('sort-lattice-4k', fixtureLattice(4096, 0.6, 22), 1.3);
await runCase('sort-collide-16k', (() => {
  // hash-collision stressor: huge extent → cell coords large → collisions guaranteed
  const a = fixtureCloud(16384, 40000, 44);
  return a;
})(), 1.3);

done(allPass);
