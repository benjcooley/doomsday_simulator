// bodies.js — body catalog, particle material recipes, impactor templates
// Units: R in Mm, M in 1e24 kg.

export const CATALOG = {
  sun:     { name: 'Sun',     R: 695.7,  M: 1.989e6, tex: '2k_sun.jpg' },
  mercury: { name: 'Mercury', R: 2.4397, M: 0.33011, tex: '2k_mercury.jpg', rotH: 1407.6 },
  venus:   { name: 'Venus',   R: 6.0518, M: 4.8675,  tex: '2k_venus_atmosphere.jpg', rotH: -5832.5 },
  earth:   { name: 'Earth',   R: 6.371,  M: 5.97237, tex: '2k_earth_daymap.jpg', rotH: 23.934 },
  moon:    { name: 'Moon',    R: 1.7374, M: 0.07342, tex: '2k_moon.jpg', rotH: 655.7 },
  mars:    { name: 'Mars',    R: 3.3895, M: 0.64171, tex: '2k_mars.jpg', rotH: 24.62 },
  jupiter: { name: 'Jupiter', R: 69.911, M: 1898.2,  tex: '2k_jupiter.jpg', rotH: 9.925 },
  saturn:  { name: 'Saturn',  R: 58.232, M: 568.34,  tex: '2k_saturn.jpg', rotH: 10.66, ring: { tex: '2k_saturn_ring_alpha.png', inner: 74.5, outer: 137.0 } },
  uranus:  { name: 'Uranus',  R: 25.362, M: 86.813,  tex: '2k_uranus.jpg', rotH: -17.24 },
  neptune: { name: 'Neptune', R: 24.622, M: 102.413, tex: '2k_neptune.jpg', rotH: 16.11 },
};

// Material archetypes — per-instance numbers (radius, stiffness) get filled in by the blob
// builder for each spawned body. These are the physical/visual personalities.
//   cohF: cohesion as a fraction of contact stiffness (how "solid")
//   Tsol/Tliq: melting band (K); Tvap: cohesion fully gone + vapor glow
//   cp: specific heat J/(kg K)  → heatK = 1e12/cp (K per (Mm/s)^2 of specific energy)
//   dampZ: dashpot damping ratio
export const MAT_TYPES = {
  IRON:  { cohF: 0.45, dampZ: 0.55, Tsol: 1700, Tliq: 1850, Tvap: 3300, cp: 450,  base: [0.42, 0.40, 0.43], emis: 0.0, densMul: 1.55 },
  ROCK:  { cohF: 0.32, dampZ: 0.50, Tsol: 1400, Tliq: 1700, Tvap: 3500, cp: 1000, base: [0.45, 0.36, 0.28], emis: 0.0, densMul: 0.92 },
  CRUST: { cohF: 0.30, dampZ: 0.50, Tsol: 1350, Tliq: 1650, Tvap: 3500, cp: 1000, base: [0.40, 0.34, 0.26], emis: 0.0, densMul: 0.88 },
  // WATER/ICE = blue/white rock. At particle resolution an "ocean" particle is a 70-270 km
  // column that is >98% rock with a thin water film, so it behaves like crust: same cohesion,
  // damping, density, melt point. This is the ONLY change on top of the known-good physics —
  // it stops the low-cohesion "jelly" squeezing over the land. cp/Tvap stay water-like so the
  // ocean still boils to steam (not flash-vaporized rock), to avoid over-cooking impacts.
  WATER: { cohF: 0.30, dampZ: 0.50, Tsol: 1350, Tliq: 1650, Tvap: 600,  cp: 4184, base: [0.06, 0.18, 0.42], emis: 0.0, densMul: 0.88 },
  ICE:   { cohF: 0.30, dampZ: 0.50, Tsol: 1350, Tliq: 1650, Tvap: 550,  cp: 2100, base: [0.75, 0.85, 0.95], emis: 0.0, densMul: 0.88 },
  GAS:   { cohF: 0.0,  dampZ: 0.90, Tsol: 1e6,  Tliq: 2e6,  Tvap: 3e6,  cp: 12000, base: [0.85, 0.72, 0.55], emis: 0.02, densMul: 1.0 },
  LAVA:  { cohF: 0.10, dampZ: 0.70, Tsol: 1400, Tliq: 1700, Tvap: 3500, cp: 1000, base: [0.35, 0.28, 0.22], emis: 0.0, densMul: 0.92 },
};

// recipes: layered composition by radius fraction; 'paint' = sample planet texture for albedo
export const RECIPES = {
  earth: {
    layers: [
      { to: 0.545, mat: 'IRON', tint: [0.50, 0.46, 0.44] },
      { to: 0.94,  mat: 'ROCK', tint: [0.55, 0.30, 0.12] },   // mantle — hot rock look on excavation
      { to: 0.985, mat: 'CRUST', paint: 'earthDay' },
      { to: 1.0,   mat: 'SURFACE_SPECIAL' },                  // ocean/land/ice decided per-particle from texture
    ],
    spin: 'earth', T0: 288,
  },
  moon:    { layers: [{ to: 0.4, mat: 'IRON', tint: [0.45, 0.43, 0.42] }, { to: 1.0, mat: 'ROCK', paint: 'moon' }],
             temps: { IRON: [1650, 1400], ROCK: [1250, 250] }, T0: 250 },
  mars:    { layers: [{ to: 0.5, mat: 'IRON', tint: [0.5, 0.45, 0.42] }, { to: 1.0, mat: 'ROCK', paint: 'mars' }],
             temps: { IRON: [2050, 1750], ROCK: [1500, 210] }, T0: 210 },
  jupiter: { layers: [{ to: 0.20, mat: 'IRON', tint: [0.6, 0.55, 0.5] }, { to: 1.0, mat: 'GAS', paint: 'jupiter' }], T0: 165 },
  venus:   { layers: [{ to: 0.52, mat: 'IRON', tint: [0.52, 0.48, 0.44] }, { to: 1.0, mat: 'ROCK', paint: 'venus' }],
             temps: { IRON: [2050, 1750], ROCK: [1700, 740] }, T0: 737 },
  rock:    { layers: [{ to: 1.0, mat: 'ROCK', speckle: 0.25 }], T0: 200 },
  pluto:   { layers: [{ to: 0.55, mat: 'ROCK', tint: [0.3, 0.26, 0.22] }, { to: 1.0, mat: 'ICE', paint: 'pluto' }],
             temps: { ROCK: [600, 300], ICE: [44, 44] }, T0: 44 },
  iron:    { layers: [{ to: 1.0, mat: 'IRON', speckle: 0.15 }], T0: 200 },
  comet:   { layers: [{ to: 0.6, mat: 'ROCK', tint: [0.25, 0.22, 0.2] }, { to: 1.0, mat: 'ICE', speckle: 0.35 }], T0: 120 },
  earth2:  { layers: [
      { to: 0.545, mat: 'IRON', tint: [0.50, 0.46, 0.44] },
      { to: 0.94,  mat: 'ROCK', tint: [0.55, 0.30, 0.12] },
      { to: 0.985, mat: 'CRUST', paint: 'earthDay' },
      { to: 1.0,   mat: 'SURFACE_SPECIAL' },
    ], T0: 288 },
};

// Densities for custom impactor mass-from-diameter (kg/m^3)
export const DENSITY = { rock: 3000, iron: 7800, comet: 900, earth2: 5514, mars: 3933, jupiter: 1326, moon: 3344 };

// ☄ Impactor templates for the sidebar lab. d = diameter km. Sub-resolution ones run "energy-true".
export const IMPACTORS = [
  { id: 'bennu',   name: 'Bennu',            d: 0.49,   recipe: 'rock',  vDef: 12.7, blurb: 'OSIRIS-REx’s friend. City-killer class.', skulls: 1 },
  { id: 'apophis', name: 'Apophis',          d: 0.375,  recipe: 'rock',  vDef: 12.6, blurb: 'God of chaos, scheduled 2029 flyby.', skulls: 1 },
  { id: 'yr4',     name: '2024 YR4',         d: 0.055,  recipe: 'rock',  vDef: 17.0, blurb: 'The one that made the news.', skulls: 1 },
  { id: 'tunguska',name: 'Tunguska stone',   d: 0.06,   recipe: 'rock',  vDef: 27.0, blurb: 'Flattens forests, ruins picnics.', skulls: 1 },
  { id: 'chicx',   name: 'Chicxulub killer', d: 10,     recipe: 'rock',  vDef: 20.0, blurb: 'Ask the dinosaurs how it went.', skulls: 3 },
  { id: 'chicxXL', name: 'Chicxulub XL',     d: 120,    recipe: 'rock',  vDef: 24.0, blurb: 'The dino-killer’s big brother. Visible crater guaranteed.', skulls: 4 },
  { id: 'halley',  name: 'Halley-class comet', d: 11,   recipe: 'comet', vDef: 55.0, blurb: 'Dirty snowball at highway-to-hell velocity.', skulls: 3 },
  { id: 'ironslug',name: 'Iron slug (500 km)', d: 500,  recipe: 'iron',  vDef: 30.0, blurb: 'A cannonball the size of Texas… the movie lied, this is worse.', skulls: 5 },
  { id: 'ceres',   name: 'Ceres',            d: 940,    recipe: 'rock',  vDef: 18.0, blurb: 'Largest asteroid. No longer on our side.', skulls: 5 },
  { id: 'pluto',   name: 'Pluto',            d: 2376,   recipe: 'comet', vDef: 22.0, blurb: 'It’s still mad about 2006. Revenge is a dish served at 22 km/s.', skulls: 5 },
  { id: 'theia',   name: 'Theia (Mars-size)', d: 6779,  recipe: 'mars',  vDef: 9.5,  blurb: 'The original Moon-maker. Round two.', skulls: 5 },
  { id: 'earth2',  name: 'Earth 2.0',        d: 12742,  recipe: 'earth2', vDef: 11.2, blurb: 'There can be only one.', skulls: 5 },
  { id: 'jupiter', name: 'Jupiter',          d: 139822, recipe: 'jupiter', vDef: 13.0, blurb: 'You don’t hit Jupiter. Jupiter hits you.', skulls: 5 },
];

export function impactorMass(d_km, recipe) {  // → 1e24 kg
  const rho = DENSITY[recipe] || 3000;
  const r_m = d_km * 500;                      // m
  return (4 / 3) * Math.PI * r_m ** 3 * rho / 1e24;
}

export const POP_2026 = 8.23e9;
