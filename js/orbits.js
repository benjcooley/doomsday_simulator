// orbits.js — solar system ephemeris (JPL approximate Keplerian elements, doubles, heliocentric ecliptic J2000)
// Units: positions in Mm (1 Mm = 1000 km). 1 AU = 149597.8707 Mm.

export const AU = 149597.8707;          // Mm
export const G_SIM = 6.674e-5;          // Mm^3 / (1e24 kg) / s^2
export const GM_SUN = 132712.44;        // Mm^3/s^2  (1.32712440018e11 km^3/s^2 * 1e-9)
export const GM_EARTH = 0.3986004;      // Mm^3/s^2
export const OBLIQUITY = 23.43928 * Math.PI / 180;

export function dateToJD(date) {
  return date.getTime() / 86400000 + 2440587.5;
}
export function jdToDate(jd) {
  return new Date((jd - 2440587.5) * 86400000);
}
// Greenwich mean sidereal angle (radians) — drives Earth texture rotation
export function gmst(jd) {
  const d = jd - 2451545.0;
  const deg = 280.46061837 + 360.98564736629 * d;
  return ((deg % 360) + 360) % 360 * Math.PI / 180;
}

// JPL "Approximate Positions of the Planets" Table 1 (valid 1800–2050 AD)
// [a(AU), e, I(deg), L(deg), peri(deg), node(deg)] + rates per Julian century
const ELEMS = {
  mercury: [[0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
            [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]],
  venus:   [[0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
            [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]],
  earth:   [[1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
            [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]],
  mars:    [[1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
            [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]],
  jupiter: [[5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
            [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]],
  saturn:  [[9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
            [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]],
  uranus:  [[19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
            [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589]],
  neptune: [[30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
            [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664]],
};

const D2R = Math.PI / 180;

function solveKepler(M, e) {
  // M in radians; Newton iteration
  let E = e < 0.8 ? M : Math.PI;
  for (let i = 0; i < 10; i++) {
    const dE = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E -= dE;
    if (Math.abs(dE) < 1e-12) break;
  }
  return E;
}

// position from elements at jd; returns [x,y,z] heliocentric ecliptic, in Mm
function elemsToPos(el, rates, jd) {
  const T = (jd - 2451545.0) / 36525.0;
  const a = (el[0] + rates[0] * T) * AU;
  const e = el[1] + rates[1] * T;
  const I = (el[2] + rates[2] * T) * D2R;
  const L = (el[3] + rates[3] * T) * D2R;
  const peri = (el[4] + rates[4] * T) * D2R;
  const node = (el[5] + rates[5] * T) * D2R;
  const w = peri - node;                    // argument of perihelion
  let M = L - peri;
  M = ((M % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const E = solveKepler(M, e);
  const xp = a * (Math.cos(E) - e);
  const yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
  const cw = Math.cos(w), sw = Math.sin(w), cn = Math.cos(node), sn = Math.sin(node), ci = Math.cos(I), si = Math.sin(I);
  return [
    (cw * cn - sw * sn * ci) * xp + (-sw * cn - cw * sn * ci) * yp,
    (cw * sn + sw * cn * ci) * xp + (-sw * sn + cw * cn * ci) * yp,
    (sw * si) * xp + (cw * si) * yp,
  ];
}

export function planetPos(name, jd) {
  const e = ELEMS[name];
  return elemsToPos(e[0], e[1], jd);
}

export function planetVel(name, jd) {
  const h = 60 / 86400; // 60 s
  const p0 = planetPos(name, jd - h), p1 = planetPos(name, jd + h);
  return [(p1[0] - p0[0]) / 120, (p1[1] - p0[1]) / 120, (p1[2] - p0[2]) / 120]; // Mm/s
}

// Simplified lunar ephemeris: precessing Keplerian orbit (good to ~ a degree — fine for entertainment)
export function moonGeo(jd) {
  const d = jd - 2451545.0;
  const a = 384.400 * 1000 / 1000;          // 384.400 Mm... keep explicit
  const A = 384.748;                         // semi-major axis Mm
  const e = 0.0549;
  const i = 5.145 * D2R;
  const node = (125.08 - 0.0529538083 * d) * D2R;       // regressing node, 18.6 y
  const peri = (318.15 + 0.1643573223 * d) * D2R;       // argument of perigee, 8.85 y advance
  const M = ((115.3654 + 13.0649929509 * d) % 360) * D2R;
  const E = solveKepler(((M % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI), e);
  const xp = A * (Math.cos(E) - e);
  const yp = A * Math.sqrt(1 - e * e) * Math.sin(E);
  const cw = Math.cos(peri), sw = Math.sin(peri), cn = Math.cos(node), sn = Math.sin(node), ci = Math.cos(i), si = Math.sin(i);
  return [
    (cw * cn - sw * sn * ci) * xp + (-sw * cn - cw * sn * ci) * yp,
    (cw * sn + sw * cn * ci) * xp + (-sw * sn + cw * cn * ci) * yp,
    (sw * si) * xp + (cw * si) * yp,
  ];
}

export function moonGeoVel(jd) {
  const h = 30 / 86400;
  const p0 = moonGeo(jd - h), p1 = moonGeo(jd + h);
  return [(p1[0] - p0[0]) / 60, (p1[1] - p0[1]) / 60, (p1[2] - p0[2]) / 60];
}

// Orbit polyline: n points around the full ellipse at fixed jd elements (heliocentric, Mm)
export function orbitPolyline(name, jd, n = 256) {
  const [el, rates] = ELEMS[name];
  const T = (jd - 2451545.0) / 36525.0;
  const a = (el[0] + rates[0] * T) * AU;
  const e = el[1] + rates[1] * T;
  const I = (el[2] + rates[2] * T) * D2R;
  const peri = (el[4] + rates[4] * T) * D2R;
  const node = (el[5] + rates[5] * T) * D2R;
  const w = peri - node;
  const cw = Math.cos(w), sw = Math.sin(w), cn = Math.cos(node), sn = Math.sin(node), ci = Math.cos(I), si = Math.sin(I);
  const pts = new Float64Array((n + 1) * 3);
  for (let k = 0; k <= n; k++) {
    const E = (k / n) * 2 * Math.PI;
    const xp = a * (Math.cos(E) - e);
    const yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
    pts[k * 3] = (cw * cn - sw * sn * ci) * xp + (-sw * cn - cw * sn * ci) * yp;
    pts[k * 3 + 1] = (cw * sn + sw * cn * ci) * xp + (-sw * sn + cw * cn * ci) * yp;
    pts[k * 3 + 2] = (sw * si) * xp + (cw * si) * yp;
  }
  return pts;
}

// Earth's north pole direction in ecliptic frame (ignoring precession)
export function earthPoleEcliptic() {
  return [0, -Math.sin(OBLIQUITY), Math.cos(OBLIQUITY)];
}

export const PLANET_NAMES = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'];
