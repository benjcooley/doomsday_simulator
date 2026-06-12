// STAGE 6a: fixed-point monopole deposit — bit-exact vs the JS mirror.
import { gpuDevice, storageBuf, readBack, expectEqual, done } from './util.mjs';
import { fixtureSphere, twoClusters, gravParams, jsDeposit, makeGravPipes, gravResources } from './grav.mjs';
import { fixtureCollision } from './simrun.mjs';

const device = await gpuDevice();
const pipes = await makeGravPipes(device);
let allPass = true;

async function runCase(name, fix) {
  const gp = gravParams(fix.pm, fix.n);
  const posBuf = storageBuf(device, fix.pm, 'pos');
  const res = gravResources(device, pipes, posBuf, fix.n, gp);
  const enc = device.createCommandEncoder();
  res.encode(enc);
  device.queue.submit([enc.finish()]);
  const gpu = await readBack(device, res.gravBuf, 4096 * 16, Uint32Array);
  const ref = jsDeposit(fix.pm, fix.n, gp);
  allPass = expectEqual(name, gpu, ref) && allPass;
  const occupied = Array.from({ length: 4096 }, (_, c) => ref[c * 4 + 3]).filter(Boolean).length;
  console.log(`  (${fix.n} particles → ${occupied} occupied gravity cells)`);
}

await runCase('deposit-sphere-4k', fixtureSphere(4096, 0.55, 9));
await runCase('deposit-clusters', twoClusters(2048, 0.55, 60, 13));
await runCase('deposit-collision', fixtureCollision(2048, 0.55, 0.002, 5));

done(allPass);
