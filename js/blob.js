// blob.js — builds particle-blob bodies: layered shell packing, texture-painted albedo, spin, temps
import { MAT_TYPES, RECIPES } from './bodies.js';
import { sampleEquirect } from './gpu.js';
import { vcross, vnorm, vsub, vscale, vdot } from './mathx.js';

let _seed = 12345;
function rnd() { _seed = (_seed * 1664525 + 1013904223) >>> 0; return _seed / 4294967296; }

// Earth-fixed basis from pole + sidereal angle, in ecliptic frame
export function bodyBasis(pole, theta) {
  const ze = vnorm(pole);
  let xe = vsub([1, 0, 0], vscale(ze, ze[0]));
  xe = vnorm(xe);
  const ye = vcross(ze, xe);
  const c = Math.cos(theta), s = Math.sin(theta);
  const xr = [xe[0] * c + ye[0] * s, xe[1] * c + ye[1] * s, xe[2] * c + ye[2] * s];
  const yr = vcross(ze, xr);
  return [xr, yr, ze];
}

// concentric-shell fibonacci packing; returns positions (unit-scaled to R) and spacing
function packSphere(R, targetCount) {
  // binary search spacing s so the shell construction yields ~targetCount
  const countFor = (s) => {
    let total = 1; // center particle
    for (let r = s; r < R - s * 0.3; r += s) {
      total += Math.max(4, Math.round(4 * Math.PI * r * r / (s * s) * 0.95));
    }
    return total;
  };
  let lo = R / Math.cbrt(targetCount) * 0.3, hi = R / Math.cbrt(targetCount) * 4;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (countFor(mid) > targetCount) lo = mid; else hi = mid;
  }
  const s = (lo + hi) / 2;
  const pts = [[0, 0, 0]];
  const GA = Math.PI * (3 - Math.sqrt(5));
  for (let r = s; r < R - s * 0.3; r += s) {
    const n = Math.max(4, Math.round(4 * Math.PI * r * r / (s * s) * 0.95));
    const rot = rnd() * Math.PI * 2, tilt = rnd() * 0.3;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    for (let k = 0; k < n; k++) {
      const zu = 1 - 2 * (k + 0.5) / n;
      const ru = Math.sqrt(Math.max(0, 1 - zu * zu));
      const th = GA * k + rot;
      let x = ru * Math.cos(th), y = ru * Math.sin(th), z = zu;
      const y2 = y * ct - z * st, z2 = y * st + z * ct;  // small random tilt to decorrelate shells
      pts.push([x * r, y2 * r, z2 * r]);
    }
  }
  return { pts, spacing: s };
}

function packAlbedo(r, g, b) {
  const c = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
  return (c(r) | (c(g) << 8) | (c(b) << 16) | (255 << 24)) >>> 0;
}

/**
 * Build a blob.
 * @param recipeName key of RECIPES
 * @param R radius (Mm), M mass (1e24 kg), targetCount particles
 * @param opts { textures: {earthDay, moon, mars, jupiter}, pole: [x,y,z], theta: sidereal rad,
 *               spinRadPerSec, T0override }
 * @returns { pos:Float64Array(3N), mass:Float64Array(N), vel:Float64Array(3N), temp:Float64Array(N),
 *            albedo:Uint32Array(N), matLocal:Uint8Array(N), matsUsed:[{type, name, isSurface}], rp, count }
 */
export function buildBlob(recipeName, R, M, targetCount, opts = {}) {
  _seed = 1234567 + Math.round(R * 1000);
  const recipe = RECIPES[recipeName];
  const { pts, spacing } = packSphere(R, targetCount);
  const N = pts.length;
  const rp = spacing / 2;
  const pole = opts.pole || [0, 0, 1];
  const theta = opts.theta || 0;
  const [xe, ye, ze] = bodyBasis(pole, theta);
  const tex = opts.textures || {};

  const pos = new Float64Array(N * 3);
  const mass = new Float64Array(N);
  const vel = new Float64Array(N * 3);
  const temp = new Float64Array(N);
  const albedo = new Uint32Array(N);
  const matLocal = new Uint8Array(N);

  // collect distinct materials used by this recipe (locally indexed)
  const matsUsed = [];
  const matIndex = {};
  const useMat = (type, isSurface) => {
    if (matIndex[type] === undefined) {
      matIndex[type] = matsUsed.length;
      matsUsed.push({ type, isSurface: !!isSurface });
    } else if (isSurface) matsUsed[matIndex[type]].isSurface = true;
    return matIndex[type];
  };

  // layer temperature profiles (inner→outer K), defaults if not given
  // interior temps deliberately capped BELOW the 2200K conduction threshold: still molten
  // (iron Tliq 1850), still glows orange when excavated, but pristine interiors can never
  // start a conductive thermal creep toward the surface — only impact heat conducts
  const layerTemps = {
    IRON: [2050, 1750], ROCK: [1950, 1000], CRUST: [320, 288], WATER: [288, 288],
    ICE: [255, 255], GAS: [2100, 165], LAVA: [1900, 1900],
  };

  // normalize layer bands to the OUTERMOST PACKED SHELL, not the theoretical radius —
  // otherwise the surface-classification band (0.985-1.0) sits in empty space above the
  // lattice and no particle ever becomes ocean/ice
  let rMax = 0;
  for (const p of pts) {
    const rr = Math.hypot(p[0], p[1], p[2]);
    if (rr > rMax) rMax = rr;
  }
  rMax = Math.max(rMax, R * 0.5);

  for (let i = 0; i < N; i++) {
    const p = pts[i];
    pos[i * 3] = p[0]; pos[i * 3 + 1] = p[1]; pos[i * 3 + 2] = p[2];
    const r = Math.hypot(p[0], p[1], p[2]);
    const rf = Math.min(1, r / rMax);

    // find layer
    let li = 0, from = 0;
    for (; li < recipe.layers.length; li++) {
      if (rf <= recipe.layers[li].to + 1e-9) break;
      from = recipe.layers[li].to;
    }
    li = Math.min(li, recipe.layers.length - 1);
    const layer = recipe.layers[li];
    const layerT = (layer.to - from) > 1e-9 ? (rf - from) / (layer.to - from) : 1;

    // lat/lon in body frame for texture painting
    const d = r > 1e-9 ? [p[0] / r, p[1] / r, p[2] / r] : [0, 0, 1];
    const lat = Math.asin(Math.max(-1, Math.min(1, vdot(d, ze))));
    const lon = Math.atan2(vdot(d, ye), vdot(d, xe));

    let matType = layer.mat, col = [0.5, 0.5, 0.5];
    if (matType === 'SURFACE_SPECIAL') {
      // Earth surface: classify from day map
      const c = sampleEquirect(tex.earthDay, lon, lat);
      const lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
      const isIce = Math.abs(lat) > 58 * Math.PI / 180 && lum > 0.55;
      const isOcean = !isIce && (c[2] > c[0] * 1.12 && c[2] > 0.10 && c[2] >= c[1] * 0.85);
      if (isIce) { matType = 'ICE'; col = [0.82, 0.88, 0.96]; }
      else if (isOcean) { matType = 'WATER'; col = [0.04 + c[2] * 0.12, 0.10 + c[2] * 0.25, 0.25 + c[2] * 0.55]; }
      else { matType = 'CRUST'; col = c; }
    } else if (layer.paint && tex[layer.paint]) {
      col = sampleEquirect(tex[layer.paint], lon, lat);
    } else if (layer.tint) {
      col = layer.tint.slice();
    } else {
      col = MAT_TYPES[matType].base.slice();
    }
    if (layer.speckle) {
      const f = 1 - layer.speckle / 2 + layer.speckle * rnd();
      col = [col[0] * f, col[1] * f, col[2] * f];
    } else {
      const f = 0.93 + 0.14 * rnd();
      col = [col[0] * f, col[1] * f, col[2] * f];
    }

    const mt = MAT_TYPES[matType];
    const isSurf = matType === 'WATER' || matType === 'ICE' ||
      li === recipe.layers.length - 1 ||
      (recipe.layers.length >= 4 && li === recipe.layers.length - 2);
    matLocal[i] = useMat(matType, isSurf);
    mass[i] = mt.densMul;
    const tprof = (recipe.temps && recipe.temps[matType]) || layerTemps[matType] || [300, 300];
    temp[i] = opts.T0override || (tprof[0] + (tprof[1] - tprof[0]) * layerT);
    albedo[i] = packAlbedo(col[0], col[1], col[2]);

    // spin velocity ω × r about pole
    if (opts.spinRadPerSec) {
      const w = vscale(ze, opts.spinRadPerSec);
      const v = vcross(w, p);
      vel[i * 3] = v[0]; vel[i * 3 + 1] = v[1]; vel[i * 3 + 2] = v[2];
    }
  }

  // renormalize mass to exactly M
  let sum = 0;
  for (let i = 0; i < N; i++) sum += mass[i];
  const sc = M / sum;
  for (let i = 0; i < N; i++) mass[i] *= sc;

  return { pos, mass, vel, temp, albedo, matLocal, matsUsed, rp, count: N, R, M, recipeName };
}
