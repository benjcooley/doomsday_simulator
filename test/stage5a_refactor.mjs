// STAGE 5a: the chunk refactor changed NOTHING — pre-split snapshot kernel vs reassembled
// kernel, identical contact-rich fixture, one substep, outputs must be BIT-EQUAL.
import { readFileSync } from 'node:fs';
import { gpuDevice, done } from './util.mjs';
import { runSimOnce, fixtureCollision, makeMats, makeSimParams, compareRuns } from './simrun.mjs';
import { SIM_WGSL } from '../js/shaders_sim.js';

const device = await gpuDevice();
const OLD_WGSL = readFileSync('/tmp/sim_wgsl_before.txt', 'utf8');
console.log(`old: ${OLD_WGSL.length} chars, new: ${SIM_WGSL.length} chars`);

const mats = makeMats();
let allPass = true;

for (const [name, opts] of [
  ['contact-only (g=0)', { gConst: 0 }],
  ['contact+gravity', { gConst: 6.674e-5 }],
  ['with-sun (gm=132.7)', { gConst: 6.674e-5, gmSun: 132.71244 }],
]) {
  const fix = fixtureCollision(2048, 0.55, 0.002, 5);
  const params = makeSimParams({ n: fix.n, dt: 2, ...opts });
  const oldRun = await runSimOnce(device, OLD_WGSL, 'sim', fix, mats, params);
  const newRun = await runSimOnce(device, SIM_WGSL, 'sim', fix, mats, params);
  allPass = compareRuns(name, oldRun, newRun, { bitExact: true }) && allPass;
}

done(allPass);
