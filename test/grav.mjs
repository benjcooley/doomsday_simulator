// test/grav.mjs — shared gravity-stage helpers: sphere fixtures, contact-free materials,
// coarse-grid parameterization, and the bit-exact JS mirror of the fixed-point deposit.
import { rng, storageBuf, emptyBuf, uniformBuf, makePipeline } from './util.mjs';
import { GRAV_DEPOSIT_WGSL } from '../js/shaders_sim_fast.js';

const fr = Math.fround;

export function fixtureSphere(nMax, spacing = 0.55, seed = 9, center = [0, 0, 0]) {
  const r = rng(seed);
  const side = Math.ceil(Math.cbrt((nMax * 6) / Math.PI)) + 2;
  const R = (side / 2 - 1) * spacing;
  const pts = [];
  for (let z = 0; z < side && pts.length < nMax; z++) {
    for (let y = 0; y < side && pts.length < nMax; y++) {
      for (let x = 0; x < side && pts.length < nMax; x++) {
        const px = (x - side / 2) * spacing + (r() - 0.5) * 0.03;
        const py = (y - side / 2) * spacing + (r() - 0.5) * 0.03;
        const pz = (z - side / 2) * spacing + (r() - 0.5) * 0.03;
        if (px * px + py * py + pz * pz <= R * R) pts.push([px, py, pz]);
      }
    }
  }
  const n = pts.length;
  const pm = new Float32Array(n * 4), vt = new Float32Array(n * 4), meta = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    pm[i * 4] = pts[i][0] + center[0];
    pm[i * 4 + 1] = pts[i][1] + center[1];
    pm[i * 4 + 2] = pts[i][2] + center[2];
    pm[i * 4 + 3] = 1e-4;
    vt[i * 4 + 3] = 300;
  }
  return { pm, vt, meta, n, R };
}

export function twoClusters(nPer, spacing, sep, seed = 13) {
  const a = fixtureSphere(nPer, spacing, seed, [-sep / 2, 0, 0]);
  const b = fixtureSphere(nPer, spacing, seed + 1, [sep / 2, 0, 0]);
  const n = a.n + b.n;
  const pm = new Float32Array(n * 4), vt = new Float32Array(n * 4), meta = new Uint32Array(n);
  pm.set(a.pm, 0); pm.set(b.pm, a.n * 4);
  vt.set(a.vt, 0); vt.set(b.vt, a.n * 4);
  for (let i = 0; i < b.n; i++) meta[a.n + i] = 1 << 4;
  return { pm, vt, meta, n, nA: a.n, nB: b.n, R: a.R };
}

// contact-free materials: gravity isolation (radius→0 ⇒ no reach, k/coh/damp zero)
export function gravMats() {
  const m = new Float32Array(16 * 16);
  m.set([
    1e-6, 0, 0, 0,
    1, 1e9, 1400, 1700,
    3500, 0, 1.6, 0,
    0.45, 0.36, 0.28, 0,
  ], 0);
  m.set(m.slice(0, 16), 16);   // slot 1 identical (two-cluster bodySlot)
  return m;
}

// coarse-grid parameterization from positions (cubic AABB) — f32-rounded once, used
// identically by the GPU uniform and the JS mirror
export function gravParams(pm, n) {
  let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity], totalMass = 0;
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) {
      const v = pm[i * 4 + a];
      if (v < mn[a]) mn[a] = v;
      if (v > mx[a]) mx[a] = v;
    }
    totalMass += pm[i * 4 + 3];
  }
  const extent = fr(Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) * 1.002 + 1e-6);
  const origin = [fr(mn[0]), fr(mn[1]), fr(mn[2])];
  const massScale = fr(3.8e9 / totalMass);
  const cell = extent / 16;
  return { origin, extent: fr(extent), extentInv: fr(1 / extent), massScale, soft2: fr((0.65 * cell) ** 2) };
}

// bit-exact JS mirror of depositGrav (fround chain replicates f32 ops, trunc replicates u32())
export function jsDeposit(pm, n, gp) {
  const cells = new Uint32Array(4096 * 4);
  for (let i = 0; i < n; i++) {
    const m = pm[i * 4 + 3];
    if (m <= 0) continue;
    const pn = [0, 1, 2].map((a) => {
      const v = fr(fr(pm[i * 4 + a] - gp.origin[a]) * gp.extentInv);
      return Math.min(Math.max(v, 0), fr(0.999999));
    });
    const ci = pn.map((v) => Math.trunc(fr(v * 16)));
    const flat = (ci[0] + ci[1] * 16 + ci[2] * 256) * 4;
    const mfp = Math.trunc(fr(m * gp.massScale)) >>> 0;
    const f32mfp = fr(mfp);
    cells[flat + 0] = (cells[flat + 0] + Math.trunc(fr(f32mfp * pn[0]))) >>> 0;
    cells[flat + 1] = (cells[flat + 1] + Math.trunc(fr(f32mfp * pn[1]))) >>> 0;
    cells[flat + 2] = (cells[flat + 2] + Math.trunc(fr(f32mfp * pn[2]))) >>> 0;
    cells[flat + 3] = (cells[flat + 3] + mfp) >>> 0;
  }
  return cells;
}

export async function makeGravPipes(device) {
  return {
    pClearG: await makePipeline(device, GRAV_DEPOSIT_WGSL, 'clearGrav', 'grav'),
    pDeposit: await makePipeline(device, GRAV_DEPOSIT_WGSL, 'depositGrav', 'grav'),
  };
}

export function gravResources(device, pipes, posBuf, n, gp) {
  const dp = new ArrayBuffer(32);
  const f = new Float32Array(dp), u = new Uint32Array(dp);
  f[0] = gp.origin[0]; f[1] = gp.origin[1]; f[2] = gp.origin[2]; f[3] = gp.extentInv;
  f[4] = gp.massScale; u[5] = n;
  const dpBuf = uniformBuf(device, new Uint32Array(dp), 'dp');
  const gravBuf = emptyBuf(device, 4096 * 16, 'grav');
  // clearGrav statically uses ONLY the grav buffer — auto-layout drops the uniform binding
  const bgClear = device.createBindGroup({
    layout: pipes.pClearG.getBindGroupLayout(0),
    entries: [{ binding: 2, resource: { buffer: gravBuf } }],
  });
  const bgDep = device.createBindGroup({
    layout: pipes.pDeposit.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dpBuf } },
      { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: gravBuf } },
    ],
  });
  const encode = (enc) => {
    let cp = enc.beginComputePass();
    cp.setPipeline(pipes.pClearG); cp.setBindGroup(0, bgClear); cp.dispatchWorkgroups(64); cp.end();
    cp = enc.beginComputePass();
    cp.setPipeline(pipes.pDeposit); cp.setBindGroup(0, bgDep); cp.dispatchWorkgroups(Math.ceil(n / 256)); cp.end();
  };
  return { gravBuf, dpBuf, encode };
}
