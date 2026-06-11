// test_kernel.mjs — standalone, deterministic transcription of the GPU pair-physics kernel.
// Run: node test_kernel.mjs
// Fires moon particles at earth particles and prints every force, every step.
import { MAT_TYPES } from './js/bodies.js';

// ---- material calibration, copied exactly from particles.js addBlob ----
function calibrate(M, R, count, matType) {
  const G_SIM = 6.674e-5;
  const mAvg = M / count;
  const vEsc = Math.sqrt(2 * G_SIM * M / R);
  const gSurf = G_SIM * M / (R * R);
  const rp = (R / Math.cbrt(count)) * 0.9;           // close to packSphere's result
  const shells = Math.max(2, R / (2 * rp));
  const k1 = mAvg * Math.pow(vEsc / rp, 2);
  const k2 = shells * mAvg * gSurf / (0.08 * rp);
  const k = Math.max(k1, k2);
  const damp = 2 * 0.45 * Math.sqrt(k * mAvg);
  const mt = MAT_TYPES[matType];
  return {
    rad: rp, k, cohK: k * mt.cohF, damp: damp * mt.dampZ / 0.45,
    heatK: 1e12 / mt.cp, Tsol: mt.Tsol, Tliq: mt.Tliq, Tvap: mt.Tvap,
    mass: mAvg * mt.densMul,
  };
}

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const cohScale = (m, T) => Math.max(0, 1 - 0.85 * smooth(m.Tsol, m.Tliq, T) - smooth(m.Tvap * 0.8, m.Tvap, T));
const dampScale = (m, T) => {
  const melt = smooth(m.Tsol, m.Tliq, T), vap = smooth(m.Tvap * 0.8, m.Tvap * 1.4, T);
  return (1 + (0.3 - 1) * melt) * (1 - vap) + 0.05 * vap;
};
const vapOf = (m, T) => smooth(m.Tvap * 0.8, m.Tvap * 1.4, T);

// ---- the kernel inner loop, transcribed line-for-line from shaders_sim.js ----
// returns {acc:[ax,ay,az], heat} acting on particle a from particle b
function pairForce(a, b, dt, settleBoost = 1, heatGate = 1, heatMul = 1, log = null) {
  const G = 6.674e-5;
  const d = [b.p[0] - a.p[0], b.p[1] - a.p[1], b.p[2] - a.p[2]];
  const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const out = { acc: [0, 0, 0], heat: 0, terms: {} };
  if (!(b.m > 0) || r2 <= 1e-12) return out;
  const eps2 = Math.pow(0.5 * Math.min(a.mat.rad, b.mat.rad), 2);
  const invR = 1 / Math.sqrt(r2 + eps2);
  for (let q = 0; q < 3; q++) out.acc[q] += d[q] * (G * b.m * invR * invR * invR);

  const h = a.mat.rad + b.mat.rad;
  const reach = h * 1.35;
  if (r2 < reach * reach) {
    const r = Math.sqrt(r2);
    const n = d.map((x) => x / Math.max(r, 1e-6));
    const kMax = 0.20 * a.m / (dt * dt);
    const cMax = 0.08 * a.m / dt;
    const cohI = a.mat.cohK * cohScale(a.mat, a.T);
    const cohJ = b.mat.cohK * cohScale(b.mat, b.T);
    const dampI = a.mat.damp * dampScale(a.mat, a.T);
    const dampJ = b.mat.damp * dampScale(b.mat, b.T);
    const dv = [b.v[0] - a.v[0], b.v[1] - a.v[1], b.v[2] - a.v[2]];
    const vn = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
    const pen = h - r;
    // SHOCK-SOFTENING (candidate fix): pairs in violent relative motion lose solid damping —
    // the shock mobilizes material; without this the lattice is memory foam that eats kicks
    const rel2early = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
    const shockPair = smooth(6.4e-7, 6.25e-6, rel2early);   // 0.8 km/s -> 2.5 km/s
    const kp = Math.min(Math.min(a.mat.k, b.mat.k), kMax);
    const cohp = Math.min(Math.min(cohI, cohJ), kMax * 0.5);
    const cp = Math.min(Math.min(dampI, dampJ) * settleBoost, cMax) * (1 - 0.92 * shockPair);
    let f = 0;
    const vapI = vapOf(a.mat, a.T), vapJ = vapOf(b.mat, b.T);
    if (pen > 0) {
      const vapP = Math.max(vapI, vapJ);
      f = -kp * Math.min(pen, 0.6 * a.mat.rad * (1 + 4 * vapP));
    } else {
      f = cohp * (r - h);
    }
    const w = 1 - smooth(h * 0.9, reach, r);
    const fdamp = cp * vn * w;
    const vt = [dv[0] - n[0] * vn, dv[1] - n[1] * vn, dv[2] - n[2] * vn];
    const ftan = cp * 0.35 * w;
    for (let q = 0; q < 3; q++) out.acc[q] += (n[q] * (f + fdamp) + vt[q] * ftan) / a.m;
    out.terms.spring = f; out.terms.fdamp = fdamp; out.terms.vn = vn; out.terms.pen = pen;

    // hypervelocity exchange
    let exch = 0;
    if (vn < 0 && pen > -0.1 * h) {
      const hyp = smooth(1.2e-7, 4.0e-6, vn * vn);
      if (hyp > 0) {
        const muJ = b.m / Math.max(a.m + b.m, 1e-12);
        const alpha = hyp * 0.5;
        exch = vn * muJ * alpha;
        for (let q = 0; q < 3; q++) out.acc[q] += n[q] * (exch / Math.max(dt, 1e-6));
        out.heat += 0.5 * muJ * vn * vn * alpha * a.mat.heatK * heatMul * heatGate;
      }
    }
    out.terms.exch = exch;

    const relSpeed2 = vn * vn + vt[0] * vt[0] + vt[1] * vt[1] + vt[2] * vt[2];
    const impactGate = smooth(2.25e-6, 1.6e-5, relSpeed2);
    const diss = (cp * vn * vn + cp * 0.35 * (relSpeed2 - vn * vn)) * w;
    const dTraw = 0.5 * diss * dt * a.mat.heatK / Math.max(a.m, 1e-12) * heatMul;
    const dTcap = 0.25 * relSpeed2 * a.mat.heatK;
    out.heat += Math.min(dTraw, dTcap) * heatGate * impactGate;
  }
  return out;
}

// ---- TEST A: one moon rock particle at 10 km/s vs one earth crust particle at rest ----
console.log('=== TEST A: 1 moon particle (10 km/s) -> 1 earth crust particle (at rest) ===');
const earthMat = calibrate(5.972, 6.371, 9830, 'CRUST');
const moonMat = calibrate(0.0735, 1.737, 420, 'ROCK');
console.log('earth particle:', JSON.stringify({ m: earthMat.mass.toExponential(2), rad: earthMat.rad.toFixed(3), k: earthMat.k.toExponential(2), damp: earthMat.damp.toExponential(2) }));
console.log('moon  particle:', JSON.stringify({ m: moonMat.mass.toExponential(2), rad: moonMat.rad.toFixed(3), k: moonMat.k.toExponential(2), damp: moonMat.damp.toExponential(2) }));

const dt = 2.0;
let E = { p: [0, 0, 0], v: [0, 0, 0], T: 290, m: earthMat.mass, mat: earthMat };
let Mo = { p: [(earthMat.rad + moonMat.rad) * 1.2, 0, 0], v: [-0.010, 0, 0], T: 300, m: moonMat.mass, mat: moonMat };

console.log('step |   r(Mm) |   vn(km/s) | spring | exch dv(km/s) | vE(km/s) | vM(km/s) |   T_E(K)');
for (let s = 0; s < 30; s++) {
  const fE = pairForce(E, Mo, dt);
  const fM = pairForce(Mo, E, dt);
  for (let q = 0; q < 3; q++) {
    E.v[q] += fE.acc[q] * dt;
    Mo.v[q] += fM.acc[q] * dt;
  }
  E.T += fE.heat; Mo.T += fM.heat;
  for (let q = 0; q < 3; q++) { E.p[q] += E.v[q] * dt; Mo.p[q] += Mo.v[q] * dt; }
  const r = Math.hypot(Mo.p[0] - E.p[0], Mo.p[1] - E.p[1], Mo.p[2] - E.p[2]);
  if (s < 12 || s % 5 === 0) {
    console.log(
      String(s).padStart(4), '|',
      r.toFixed(4).padStart(7), '|',
      ((fE.terms.vn ?? 0) * 1000).toFixed(2).padStart(10), '|',
      (fE.terms.spring ?? 0).toExponential(1).padStart(8), '|',
      ((fE.terms.exch ?? 0) * 1000).toFixed(3).padStart(9), '|',
      (E.v[0] * 1000).toFixed(3).padStart(8), '|',
      (Mo.v[0] * 1000).toFixed(3).padStart(8), '|',
      E.T.toFixed(0).padStart(8),
    );
  }
}
console.log('\nVERDICT A: earth particle final speed', (Math.hypot(...E.v) * 1000).toFixed(2), 'km/s',
  '(should be ~2+ km/s if momentum transfer works; ~0 reproduces the bug)');

// ---- TEST B: moon particle vs earth particle WITH a lattice neighbor behind it ----
console.log('\n=== TEST B: same impact, but earth particle has 6 lattice neighbors (springs+damping to them) ===');
E = { p: [0, 0, 0], v: [0, 0, 0], T: 290, m: earthMat.mass, mat: earthMat };
Mo = { p: [(earthMat.rad + moonMat.rad) * 1.2, 0, 0], v: [-0.010, 0, 0], T: 300, m: moonMat.mass, mat: moonMat };
const s0 = earthMat.rad * 2;
const neighbors = [
  { p: [-s0, 0, 0] }, { p: [0, s0, 0] }, { p: [0, -s0, 0] },
  { p: [0, 0, s0] }, { p: [0, 0, -s0] }, { p: [-s0 * 0.5, s0 * 0.87, 0] },
].map((n) => ({ ...n, v: [0, 0, 0], T: 290, m: earthMat.mass, mat: earthMat, fixed: false }));

for (let s = 0; s < 60; s++) {
  const bodies = [E, Mo, ...neighbors];
  const accs = bodies.map(() => [0, 0, 0]);
  const heats = bodies.map(() => 0);
  for (let i = 0; i < bodies.length; i++) {
    for (let j = 0; j < bodies.length; j++) {
      if (i === j) continue;
      const f = pairForce(bodies[i], bodies[j], dt);
      for (let q = 0; q < 3; q++) accs[i][q] += f.acc[q];
      heats[i] += f.heat;
    }
  }
  bodies.forEach((b, i) => {
    for (let q = 0; q < 3; q++) { b.v[q] += accs[i][q] * dt; }
    b.T += heats[i];
  });
  bodies.forEach((b) => { for (let q = 0; q < 3; q++) b.p[q] += b.v[q] * dt; });
  if (s % 10 === 0 || s === 59) {
    console.log(`step ${String(s).padStart(2)}: vE=${(Math.hypot(...E.v) * 1000).toFixed(3)} km/s  xE=${E.p[0].toFixed(4)} Mm  T_E=${E.T.toFixed(0)}K  vMoon=${(Math.hypot(...Mo.v) * 1000).toFixed(2)} km/s  vNbr0=${(Math.hypot(...neighbors[0].v) * 1000).toFixed(3)} km/s`);
  }
}
console.log('\nVERDICT B: struck earth particle displaced', E.p[0].toFixed(4), 'Mm =', (E.p[0] / earthMat.rad).toFixed(2), 'particle radii',
  '(visible motion needs |displacement| >> 0.1 radii)');
