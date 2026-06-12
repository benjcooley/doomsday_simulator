// STAGE 4: neighbor completeness — the 27-cell walk must find EVERY pair within reach.
// Oracle: brute-force JS with f32-tolerance bracketing. For each particle:
//   lower = pairs within reach·(1-1e-5)   (must-find set)
//   upper = pairs within reach·(1+1e-5)   (may-find set)
//   assert lower ≤ gpuCount ≤ upper; where lower==upper (no borderline pairs) the
//   neighbor index-sum must match EXACTLY. A missed cell or double-counted collision
//   slot breaks these immediately; f32 rounding on borderline pairs cannot.
import { gpuDevice, fixtureCloud, fixtureLattice, emptyBuf, uniformBuf, readBack, makePipeline, done } from './util.mjs';
import { makeSortChain, sortChainResources } from './chain.mjs';
import { NEIGHBOR_WGSL, HASH_SIZE } from '../js/shaders_sim_fast.js';

const device = await gpuDevice();
const chain = await makeSortChain(device);
const pWalk = await makePipeline(device, NEIGHBOR_WGSL, 'walkNeighbors', 'walk');
let allPass = true;

async function runCase(name, pm, cellSize, reach) {
  const res = sortChainResources(device, chain, pm, cellSize);
  const n = res.n;
  const wp = new ArrayBuffer(16);
  new Float32Array(wp)[0] = 1 / cellSize;
  new Uint32Array(wp)[1] = n;
  new Uint32Array(wp)[2] = HASH_SIZE;
  new Float32Array(wp)[3] = Math.fround(reach) * Math.fround(reach);
  const wpBuf = uniformBuf(device, new Uint32Array(wp), 'wp');
  const outCount = emptyBuf(device, n * 4, 'outCount');
  const outISum = emptyBuf(device, n * 4, 'outISum');

  const bgWalk = device.createBindGroup({
    layout: pWalk.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wpBuf } },
      { binding: 1, resource: { buffer: res.posBuf } },
      { binding: 2, resource: { buffer: res.offsetsBuf } },
      { binding: 3, resource: { buffer: res.countsBuf } },
      { binding: 4, resource: { buffer: res.sortedBuf } },
      { binding: 5, resource: { buffer: outCount } },
      { binding: 6, resource: { buffer: outISum } },
    ],
  });

  const enc = device.createCommandEncoder();
  res.encode(enc);
  const cp = enc.beginComputePass();
  cp.setPipeline(pWalk); cp.setBindGroup(0, bgWalk); cp.dispatchWorkgroups(Math.ceil(n / 256));
  cp.end();
  device.queue.submit([enc.finish()]);

  const gCount = await readBack(device, outCount, n * 4, Uint32Array);
  const gISum = await readBack(device, outISum, n * 4, Uint32Array);

  // brute-force reference with tolerance brackets
  const r2lo = (reach * (1 - 1e-5)) ** 2;
  const r2hi = (reach * (1 + 1e-5)) ** 2;
  let bad = 0, exactChecked = 0, totalPairs = 0;
  for (let i = 0; i < n && bad < 5; i++) {
    let lo = 0, hi = 0, isum = 0;
    const xi = pm[i * 4], yi = pm[i * 4 + 1], zi = pm[i * 4 + 2];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = pm[j * 4] - xi, dy = pm[j * 4 + 1] - yi, dz = pm[j * 4 + 2] - zi;
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < r2lo) { lo++; isum += j; }
      if (r2 < r2hi) hi++;
    }
    totalPairs += lo;
    if (gCount[i] < lo || gCount[i] > hi) {
      console.error(`FAIL ${name}: particle ${i}: gpu=${gCount[i]} expected [${lo},${hi}]`);
      bad++; continue;
    }
    if (lo === hi) {
      exactChecked++;
      if (gISum[i] !== (isum >>> 0)) {
        console.error(`FAIL ${name}: particle ${i}: index-sum ${gISum[i]} != ${isum >>> 0}`);
        bad++;
      }
    }
  }
  const ok = bad === 0;
  allPass = allPass && ok;
  if (ok) console.log(`PASS ${name} (${n} particles, ~${totalPairs} pairs, ${exactChecked} exact index-sum checks)`);
}

// dense lattice: spacing 0.6, cell 1.3, reach 1.3 → ~30+ neighbors each, heavy boundary crossing
await runCase('walk-lattice-4k', fixtureLattice(4096, 0.6, 22), 1.3, 1.3);
// cloud at moderate density, reach < cell
await runCase('walk-cloud-4k', fixtureCloud(4096, 12, 11), 1.3, 0.9);
// hash-collision regime: huge extent (same as stage-3 stressor), sparse pairs
await runCase('walk-collide-8k', fixtureCloud(8192, 40000, 44), 1.3, 1.3);
// tight clump: everyone within a few cells — max segment lengths
await runCase('walk-clump-2k', fixtureCloud(2048, 1.4, 77), 1.3, 1.3);

done(allPass);
