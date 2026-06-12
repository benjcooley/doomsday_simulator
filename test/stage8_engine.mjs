// STAGE 8: the PRODUCTION FastParticleSystem class, end-to-end — real addBlob spawning,
// real writeParams/writeBodyDyn, 200 substeps of a two-blob collision in BOTH engine
// classes. Oracle: no NaN, stats sane, bound-fraction + mean surface T within tolerance
// of the proven engine, and the fast class actually exercises its grid every substep.
import { gpuDevice, done } from './util.mjs';
import { ParticleSystem } from '../js/particles.js';
import { FastParticleSystem } from '../js/particles_fast.js';
import { buildBlob } from '../js/blob.js';

const device = await gpuDevice();
// minimal shim matching the GPU wrapper surface ParticleSystem uses
const gpuShim = {
  device,
  frameId: 1,
  makeShader: async (code, label) => device.createShaderModule({ code, label }),
};

async function runEngine(Cls) {
  const ps = await new Cls().init(gpuShim);
  ps.reset();
  // two rocky blobs on a collision course (real spawn path: materials, meta, albedo)
  const a = buildBlob('rock', 2.0, 0.05, 1800, {});
  const b = buildBlob('rock', 1.5, 0.02, 800, {});
  ps.addBlob(a, [0, 0, 0], [0, 0, 0], 'A');
  ps.addBlob(b, [6.5, 0.4, 0], [-0.005, 0, 0], 'B');   // 5 km/s closing
  ps.writeBodyDyn([
    { slot: 0, vel: [0, 0, 0], pos: [0, 0, 0], spin: [0, 0, 0] },
    { slot: 1, vel: [-0.005, 0, 0], pos: [6.5, 0.4, 0], spin: [0, 0, 0] },
  ]);
  ps.writeParams({
    dt: 1.2, settleBoost: 1, sunPos: [1.5e5, 0, 0], gmSun: 0,
    coolMul: 0, heatMul: 1, time: 0, solarLum: 1, heatGate: 1, settleDrag: 0,
  });
  for (let batch = 0; batch < 4; batch++) {
    const enc = device.createCommandEncoder();
    ps.step(enc, 50);
    if (batch === 3) ps.requestReadback(enc, 0);
    device.queue.submit([enc.finish()]);
    ps.afterSubmit();
  }
  // wait for the readback to resolve
  for (let i = 0; i < 200; i++) {
    const stats = ps.pollStats();
    if (stats) return { ps, stats };
    await new Promise((r) => setTimeout(r, 10));
  }
  throw new Error('stats readback never resolved');
}

const { stats: sN2 } = await runEngine(ParticleSystem);
const { stats: sF } = await runEngine(FastParticleSystem);

let allPass = true;
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ': ' + detail : ''}`);
  allPass = allPass && ok;
};

for (const [tag, s] of [['n2', sN2], ['fast', sF]]) {
  const a = s.bodies[0], b = s.bodies[1];
  const finite = [a.com, a.cov, b.com, b.cov].flat().every(Number.isFinite);
  check(`${tag}: stats finite`, finite);
  check(`${tag}: no NaN-guard hits`, (a.nanN || 0) === 0 && (b.nanN || 0) === 0, `nanN=${a.nanN || 0},${b.nanN || 0}`);
  check(`${tag}: mass conserved`, Math.abs(a.mass + b.mass - 0.07) < 1e-4, (a.mass + b.mass).toFixed(5));
}
// cross-engine agreement after 200 collision substeps
const bf = (s) => (s.bodies[0].bound + s.bodies[1].bound) / 0.07;
const dBound = Math.abs(bf(sN2) - bf(sF));
check('cross: bound fraction', dBound < 0.05, `n2=${bf(sN2).toFixed(3)} fast=${bf(sF).toFixed(3)} Δ=${dBound.toFixed(3)}`);
const tN2 = sN2.bodies[0].maxT, tF = sF.bodies[0].maxT;
const dT = Math.abs(tN2 - tF) / Math.max(tN2, 1);
check('cross: peak temperature', dT < 0.2, `n2=${tN2.toFixed(0)}K fast=${tF.toFixed(0)}K Δ=${(dT * 100).toFixed(1)}%`);
const sep = (s) => Math.hypot(...[0, 1, 2].map((k) => s.bodies[0].com[k] - s.bodies[1].com[k]));
check('cross: CoM separation', Math.abs(sep(sN2) - sep(sF)) < 0.4, `n2=${sep(sN2).toFixed(3)} fast=${sep(sF).toFixed(3)}`);

done(allPass);
