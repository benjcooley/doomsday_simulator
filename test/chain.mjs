// test/chain.mjs — reusable GPU counting-sort chain (clear → count → scan → scatter)
// used by stages 4+ so every test exercises the same production pass sequence.
import { storageBuf, emptyBuf, uniformBuf, makePipeline } from './util.mjs';
import { GRID_WGSL, SCAN_WGSL, SCATTER_WGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

export async function makeSortChain(device) {
  return {
    pClear: await makePipeline(device, GRID_WGSL, 'clearCounts', 'grid'),
    pCount: await makePipeline(device, GRID_WGSL, 'countCells', 'grid'),
    pBlocks: await makePipeline(device, SCAN_WGSL, 'scanBlocks', 'scan'),
    pSums: await makePipeline(device, SCAN_WGSL, 'scanSums', 'scan'),
    pAdd: await makePipeline(device, SCAN_WGSL, 'addBack', 'scan'),
    pScatter: await makePipeline(device, SCATTER_WGSL, 'scatter', 'scatter'),
  };
}

// builds all buffers, encodes the full chain, returns { buffers..., encode(enc) }
// `pm` may be a Float32Array (uploads a fresh pos buffer) or { posBuf, n } to sort an
// EXISTING evolving buffer (multi-step runs rebuild the grid from current positions).
export function sortChainResources(device, chain, pm, cellSize) {
  const ext = !(pm instanceof Float32Array);
  const n = ext ? pm.n : pm.length / 4;
  const gp = new ArrayBuffer(16);
  new Float32Array(gp)[0] = 1 / cellSize;
  new Uint32Array(gp)[1] = n;
  new Uint32Array(gp)[2] = HASH_SIZE;
  const gpBuf = uniformBuf(device, new Uint32Array(gp), 'gp');
  const spBuf = uniformBuf(device, new Uint32Array([HASH_SIZE, 0, 0, 0]), 'sp');
  const posBuf = ext ? pm.posBuf : storageBuf(device, pm, 'pos');
  const cellOfBuf = emptyBuf(device, n * 4, 'cellOf');
  const countsBuf = emptyBuf(device, HASH_SIZE * 4, 'counts');
  const offsetsBuf = emptyBuf(device, HASH_SIZE * 4, 'offsets');
  const sumsBuf = emptyBuf(device, (HASH_SIZE / 256) * 4, 'sums');
  const cursorBuf = emptyBuf(device, HASH_SIZE * 4, 'cursor');
  const sortedBuf = emptyBuf(device, n * 4, 'sorted');

  const bg = (pipe, list) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: list });
  const groups = {
    clear: bg(chain.pClear, [
      { binding: 0, resource: { buffer: gpBuf } }, { binding: 3, resource: { buffer: countsBuf } }]),
    count: bg(chain.pCount, [
      { binding: 0, resource: { buffer: gpBuf } }, { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: cellOfBuf } }, { binding: 3, resource: { buffer: countsBuf } }]),
    blocks: bg(chain.pBlocks, [
      { binding: 0, resource: { buffer: spBuf } }, { binding: 1, resource: { buffer: countsBuf } },
      { binding: 2, resource: { buffer: offsetsBuf } }, { binding: 3, resource: { buffer: sumsBuf } }]),
    sums: bg(chain.pSums, [
      { binding: 0, resource: { buffer: spBuf } }, { binding: 3, resource: { buffer: sumsBuf } }]),
    add: bg(chain.pAdd, [
      { binding: 0, resource: { buffer: spBuf } }, { binding: 2, resource: { buffer: offsetsBuf } },
      { binding: 3, resource: { buffer: sumsBuf } }]),
    scatter: bg(chain.pScatter, [
      { binding: 0, resource: { buffer: gpBuf } }, { binding: 1, resource: { buffer: cellOfBuf } },
      { binding: 2, resource: { buffer: cursorBuf } }, { binding: 3, resource: { buffer: sortedBuf } }]),
  };

  const encode = (enc) => {
    const run = (pipe, group, wgs) => {
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe); cp.setBindGroup(0, group); cp.dispatchWorkgroups(wgs); cp.end();
    };
    run(chain.pClear, groups.clear, Math.ceil(HASH_SIZE / 256));
    run(chain.pCount, groups.count, Math.ceil(n / 256));
    run(chain.pBlocks, groups.blocks, Math.ceil(HASH_SIZE / 256));
    run(chain.pSums, groups.sums, 1);
    run(chain.pAdd, groups.add, Math.ceil(HASH_SIZE / 256));
    enc.copyBufferToBuffer(offsetsBuf, 0, cursorBuf, 0, HASH_SIZE * 4);
    run(chain.pScatter, groups.scatter, Math.ceil(n / 256));
  };

  return { n, gpBuf, spBuf, posBuf, cellOfBuf, countsBuf, offsetsBuf, cursorBuf, sortedBuf, encode };
}
