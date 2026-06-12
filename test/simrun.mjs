// test/simrun.mjs — run one substep of a full sim kernel (N² or fast) on a fixture
import { storageBuf, emptyBuf, uniformBuf, readBack, makePipeline, rng } from './util.mjs';

// rock-like material in slot 0 (values mirror the live earth-crust calibration scale)
export function makeMats() {
  const m = new Float32Array(16 * 16);
  m.set([
    0.31, 1e-6, 3e-7, 2e-5,        // a: radius, k, cohK, damp
    1, 1e9, 1400, 1700,            // b: surfaceFlag, heatK, Tsol, Tliq
    3500, 0, 1.6, 0,               // c: Tvap, emis, visR, isGas
    0.45, 0.36, 0.28, 1e-19,       // d: base rgb, coolK
  ], 0);
  return m;
}

// SimParams ArrayBuffer (80B) — layout must match SIM_STRUCTS_WGSL exactly
export function makeSimParams({ dt = 2, n, gConst = 0, eps2 = 0.024, sunPos = [1.5e5, 0, 0], gmSun = 0,
  settleBoost = 1, vClamp = 1, coolMul = 0, heatMul = 1, solarLum = 1, heatGate = 1, settleDrag = 0 }) {
  const ab = new ArrayBuffer(80);
  const f = new Float32Array(ab), u = new Uint32Array(ab);
  f[0] = dt; u[1] = n; u[2] = Math.ceil(n / 256); f[3] = settleBoost;
  f[4] = sunPos[0]; f[5] = sunPos[1]; f[6] = sunPos[2]; f[7] = gmSun;
  f[8] = gConst; f[9] = eps2; f[10] = vClamp; f[11] = coolMul;
  f[12] = heatMul; f[13] = 0; f[14] = solarLum; f[15] = heatGate;
  f[16] = settleDrag; u[17] = 0xffffffff; u[18] = 0; f[19] = 0;
  return ab;
}

// contact-rich fixture: two lattice blocks, B closing on A at `closing` Mm/s
export function fixtureCollision(nPerBlock = 2048, spacing = 0.55, closing = 0.002, seed = 5) {
  const r = rng(seed);
  const n = nPerBlock * 2;
  const pm = new Float32Array(n * 4);
  const vt = new Float32Array(n * 4);
  const meta = new Uint32Array(n);
  const side = Math.ceil(Math.cbrt(nPerBlock));
  const width = side * spacing;
  for (let b = 0; b < 2; b++) {
    for (let k = 0; k < nPerBlock; k++) {
      const i = b * nPerBlock + k;
      const x = k % side, y = ((k / side) | 0) % side, z = (k / (side * side)) | 0;
      pm[i * 4] = (x - side / 2) * spacing + (r() - 0.5) * 0.04 + (b === 1 ? width * 0.72 : 0);
      pm[i * 4 + 1] = (y - side / 2) * spacing + (r() - 0.5) * 0.04;
      pm[i * 4 + 2] = (z - side / 2) * spacing + (r() - 0.5) * 0.04;
      pm[i * 4 + 3] = 1e-4;
      vt[i * 4] = b === 1 ? -closing : 0;
      vt[i * 4 + 3] = 300;
      meta[i] = 0 | (b << 4);     // matId 0, bodySlot b
    }
  }
  return { pm, vt, meta, n };
}

// run one substep of `entry` in `code`; returns {pos, vel} after the step
export async function runSimOnce(device, code, entry, fix, mats, paramsAB, extra = null) {
  const n = fix.n;
  const paramsBuf = uniformBuf(device, new Uint32Array(paramsAB), 'P');
  const matsBuf = storageBuf(device, mats, 'mats');
  const posA = storageBuf(device, fix.pm, 'posA');
  const velA = storageBuf(device, fix.vt, 'velA');
  const posB = emptyBuf(device, n * 16, 'posB');
  const velB = emptyBuf(device, n * 16, 'velB');
  const metaBuf = storageBuf(device, fix.meta, 'meta');
  const bodyDyn = storageBuf(device, new Float32Array(16 * 12), 'bodyDyn');
  const dbg = emptyBuf(device, 8 * 16, 'dbg');

  const pipe = await makePipeline(device, code, entry, 'simrun-' + entry);
  const entries = [
    { binding: 0, resource: { buffer: paramsBuf } },
    { binding: 1, resource: { buffer: matsBuf } },
    { binding: 2, resource: { buffer: posA } },
    { binding: 3, resource: { buffer: velA } },
    { binding: 4, resource: { buffer: posB } },
    { binding: 5, resource: { buffer: velB } },
    { binding: 6, resource: { buffer: metaBuf } },
    { binding: 7, resource: { buffer: bodyDyn } },
    { binding: 8, resource: { buffer: dbg } },
  ];
  if (extra) extra.augment(entries);   // fast engine adds group(1) resources via callback

  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
  const enc = device.createCommandEncoder();
  if (extra) extra.encodePre(enc);     // grid build passes
  const cp = enc.beginComputePass();
  cp.setPipeline(pipe);
  cp.setBindGroup(0, bg);
  if (extra) extra.setGroups(cp);
  cp.dispatchWorkgroups(Math.ceil(n / 256));
  cp.end();
  device.queue.submit([enc.finish()]);

  return {
    pos: await readBack(device, posB, n * 16, Float32Array),
    vel: await readBack(device, velB, n * 16, Float32Array),
  };
}

export function compareRuns(name, a, b, { bitExact = false, relTol = 2e-5, absTol = 1e-9 } = {}) {
  let maxRel = 0, maxAbs = 0, bitDiff = 0, worst = -1;
  for (let i = 0; i < a.pos.length; i++) {
    for (const [x, y] of [[a.pos[i], b.pos[i]], [a.vel[i], b.vel[i]]]) {
      if (x !== y) bitDiff++;
      const abs = Math.abs(x - y);
      const rel = abs / Math.max(Math.abs(x), Math.abs(y), 1e-20);
      if (abs > absTol && rel > maxRel) { maxRel = rel; worst = i >> 2; }
      maxAbs = Math.max(maxAbs, abs);
    }
  }
  const ok = bitExact ? bitDiff === 0 : (maxRel <= relTol);
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: bitDiffs=${bitDiff} maxAbs=${maxAbs.toExponential(2)} maxRel=${maxRel.toExponential(2)}${worst >= 0 ? ' worst@' + worst : ''}`);
  return ok;
}
