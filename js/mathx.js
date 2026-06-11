// mathx.js — minimal column-major mat4 + vec3 helpers (doubles in JS, Float32 on upload)

export const DEG = Math.PI / 180;

export function v3(x = 0, y = 0, z = 0) { return [x, y, z]; }
export function vadd(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function vsub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function vscale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
export function vdot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function vcross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function vlen(a) { return Math.hypot(a[0], a[1], a[2]); }
export function vnorm(a) { const l = vlen(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
export function vlerp(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

// Column-major 4x4, matching WGSL mat4x4f layout. m[col*4+row].
export function midentity() {
  return new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

export function mmul(a, b) { // a*b
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

export function mlookAt(eye, target, up) {
  const f = vnorm(vsub(target, eye));        // forward
  const s = vnorm(vcross(f, up));            // right
  const u = vcross(s, f);                    // true up
  // view matrix (world -> view), looking down -Z
  const o = midentity();
  o[0] = s[0]; o[4] = s[1]; o[8] = s[2];
  o[1] = u[0]; o[5] = u[1]; o[9] = u[2];
  o[2] = -f[0]; o[6] = -f[1]; o[10] = -f[2];
  o[12] = -vdot(s, eye); o[13] = -vdot(u, eye); o[14] = vdot(f, eye);
  return o;
}

// Reversed-Z infinite perspective: depth 1 at near, 0 at infinity. Clip z in [0,1].
export function mperspectiveRevZ(fovY, aspect, near) {
  const f = 1 / Math.tan(fovY / 2);
  const o = new Float64Array(16);
  o[0] = f / aspect;
  o[5] = f;
  o[10] = 0; o[11] = -1;
  o[14] = near;
  return o;
}

export function mrotAxis(axis, ang) {
  const [x, y, z] = vnorm(axis), c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  const o = midentity();
  o[0] = t * x * x + c; o[4] = t * x * y - s * z; o[8] = t * x * z + s * y;
  o[1] = t * x * y + s * z; o[5] = t * y * y + c; o[9] = t * y * z - s * x;
  o[2] = t * x * z - s * y; o[6] = t * y * z + s * x; o[10] = t * z * z + c;
  return o;
}

export function mscaleUni(s) { const o = midentity(); o[0] = s; o[5] = s; o[10] = s; return o; }
export function mtranslate(p) { const o = midentity(); o[12] = p[0]; o[13] = p[1]; o[14] = p[2]; return o; }

export function mtransformPoint(m, p) {
  const x = p[0], y = p[1], z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
    w,
  ];
}

export function clamp(x, a, b) { return x < a ? a : (x > b ? b : x); }
export function lerp(a, b, t) { return a + (b - a) * t; }
export function smoothstep(a, b, x) { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

// piecewise-linear curve eval: pts = [[x0,y0],[x1,y1],...] sorted by x
export function curve(pts, x) {
  if (x <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    if (x <= pts[i][0]) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
  }
  return pts[pts.length - 1][1];
}

export function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }
export function fmtSI(n, unit = '') {
  if (!isFinite(n)) return '—';
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(2) + ' T' + unit;
  if (a >= 1e9) return (n / 1e9).toFixed(2) + ' B' + unit;
  if (a >= 1e6) return (n / 1e6).toFixed(2) + ' M' + unit;
  if (a >= 1e3) return (n / 1e3).toFixed(1) + ' k' + unit;
  return n.toFixed(1) + ' ' + unit;
}
