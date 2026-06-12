// shaders_sim.js — the physics kernel. One fused O(N²) tiled pass:
// pairwise gravity + DEM contact (spring/dashpot/cohesion) + collision heating +
// melt-dependent cohesion + sub-resolution "armed payload" energy deposit +
// solar gravity/heating + radiative cooling + semi-implicit Euler integration.
//
// Units: length Mm, mass 1e24 kg, time s, temperature K. G = 6.674e-5.
//
// SHARED-CHUNK STRUCTURE: the physics lives in exported chunks consumed by BOTH engines —
// this proven N² kernel below, and the fast grid engine (shaders_sim_fast.js). A physics
// change lands once, both engines inherit it. Validated by test/stage5: the reassembled
// N² kernel must produce identical results to the pre-split snapshot.

export const SIM_STRUCTS_WGSL = /* wgsl */`
struct SimParams {
  dt: f32,
  nActive: u32,
  numTiles: u32,
  settleBoost: f32,
  sunPos: vec3f,
  gmSun: f32,
  gConst: f32,
  eps2: f32,
  vClamp: f32,
  coolMul: f32,
  heatMul: f32,
  time: f32,
  solarLum: f32,
  heatGate: f32,
  settleDrag: f32,
  debugIdxBits: f32,   // bitcast u32 particle index to dump diagnostics for (0xffffffff = off)
  consumedBits: f32,   // bitcast u32: bitmask of body slots the Sun has eaten (kill all their particles)
  pad9: f32,
}

// per-body rigid reference motion (settle drag target): velocity, center, spin
struct BodyDyn { vc: vec4f, cp: vec4f, om: vec4f }

// material instance table (shared with renderer)
// a = (radius, k, cohK, damp)
// b = (reachMul, heatK, Tsol, Tliq)
// c = (Tvap, emis, visR, isGas)
// d = (baseR, baseG, baseB, coolK)
struct MatI { a: vec4f, b: vec4f, c: vec4f, d: vec4f }
`;

// group(0) binding layout — IDENTICAL in both engines so the integrate tail is shared verbatim
export const SIM_BINDINGS_WGSL = /* wgsl */`
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> mats: array<MatI, 16>;
@group(0) @binding(2) var<storage, read> posA: array<vec4f>;
@group(0) @binding(3) var<storage, read> velA: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> posB: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> velB: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> pmeta: array<u32>;
@group(0) @binding(7) var<storage, read> bodyDyn: array<BodyDyn, 16>;
@group(0) @binding(8) var<storage, read_write> dbg: array<vec4f, 8>;
`;

export const SIM_HELPERS_WGSL = /* wgsl */`
fn meltOf(b: vec4f, c: vec4f, T: f32) -> f32 {
  return smoothstep(b.z, b.w, T);
}
fn cohScale(b: vec4f, c: vec4f, T: f32) -> f32 {
  let melt = smoothstep(b.z, b.w, T);
  let vap = smoothstep(c.x * 0.8, c.x, T);
  return max(0.0, 1.0 - 0.85 * melt - vap);
}
// molten material flows (low viscosity), vapor expands freely — solid-state damping must
// collapse with temperature or impact sites absorb every kick like memory foam
fn dampScale(b: vec4f, c: vec4f, T: f32) -> f32 {
  let melt = smoothstep(b.z, b.w, T);
  let vap = smoothstep(c.x * 0.8, c.x * 1.4, T);
  return mix(mix(1.0, 0.3, melt), 0.05, vap);
}
// neighbor's interaction properties, packed: (radius [neg if armed], k [neg if interior], cohEff, dampEff)
fn neighborAux(mtj: u32, vtj: vec4f) -> vec4f {
  let mj_ = mats[mtj & 15u];
  var rj = mj_.a.x;
  if ((mtj & 512u) != 0u) { rj = -rj; }
  var kj = mj_.a.y;
  if (mj_.b.x < 0.5) { kj = -kj; }   // sign of k encodes "j is interior material"
  return vec4f(rj, kj, mj_.a.z * cohScale(mj_.b, mj_.c, vtj.w),
    mj_.a.w * dampScale(mj_.b, mj_.c, vtj.w));
}
`;

// self-state material derivations (after pm/vt/meta of particle i are loaded)
export const SELF_MATS_WGSL = /* wgsl */`
    let m = mats[mymeta & 15u];
    radI = m.a.x; kI = m.a.y; heatKI = m.b.y;
    matB = m.b; matC = m.c; coolK = m.d.w;
    armedI = (mymeta & 512u) != 0u;
    cohI = m.a.z * cohScale(matB, matC, Ti);
    vapI = smoothstep(matC.x * 0.8, matC.x * 1.4, Ti);   // vaporized fraction (drives plume pressure)
    dampI = m.a.w * dampScale(matB, matC, Ti);           // hot material loses solid damping
`;

// ---- THE pair interaction: the hard-won physics, single source for both engines ----
// Wrapper contract: caller provides, in scope:
//   pj   : vec4f — neighbor position + mass
//   pvel : vec4f — neighbor velocity + temperature (or armed payload budget)
//   aux  : vec4f — neighborAux() of the neighbor
// plus particle-self state (pi, mi, vi, Ti, radI, kI, cohI, dampI, heatKI, matB, matC, vapI,
// armedI) and accumulators (acc, dvContact, heat, nPayloadHits, dContacts, dTouch, dMinVn,
// dMaxHyp, dMaxPen) — names exactly as declared in both kernels' preambles.
export function pairPhysicsWGSL({ nearGravity = true } = {}) {
  return /* wgsl */`
        let mj = pj.w;
        let d = pj.xyz - pi;
        let r2 = dot(d, d);
        if (mj > 0.0 && r2 > 1e-12) {
${nearGravity ? `          let invR = inverseSqrt(r2 + P.eps2);
          acc = acc + d * (P.gConst * mj * invR * invR * invR);
` : ''}
          let rj = abs(aux.x);
          let armedJ = aux.x < 0.0;
          let h = radI + rj;
          let reach = h * 1.35;

          // sub-resolution impactor blast: deposit the payload over a ~4-radius zone with
          // falloff — a real Chicxulub makes a 1000 km glowing scar and a debris spray,
          // not a faint blush on the one particle it touched
          if (armedI && !armedJ) {
            // donor budget drains over the same widened blast radius the receivers use
            let blastD = 4.0 * rj;
            if (r2 < blastD * blastD) { nPayloadHits = nPayloadHits + 1.0; }
          }
          if (armedJ && !armedI) {
            let blast = 4.0 * radI;
            if (r2 < blast * blast) {
              let rr = sqrt(r2);
              let fall = 1.0 - rr / blast;
              let Ej = pvel.w;
              let frac = clamp(0.6 * P.dt, 0.0, 0.2);
              let chunk = Ej * frac * fall;
              heat = heat + chunk * heatKI / max(mi, 1e-12) * 0.5 * P.heatMul;
              let nb = d * (1.0 / max(rr, 1e-6));
              let kick = min(0.004, sqrt(max(0.0, 2.0 * chunk / max(mi, 1e-9))) * 0.35);
              dvContact = dvContact - nb * kick;
            }
          }
          if (r2 < reach * reach) {
            let r = sqrt(r2);
            let n = d * (1.0 / max(r, 1e-6));
            // dt-aware clamps: unconditionally stable at any timestep (springs go soft, never explosive).
            // cMax budgets TOTAL damping across ~8 simultaneous neighbors, not per pair.
            let kMax = 0.20 * mi / (P.dt * P.dt);
            let cMax = 0.08 * mi / P.dt;
            let kp = min(min(kI, abs(aux.y)), kMax);
            let cohp = min(min(cohI, aux.z), kMax * 0.5);
            let dv = pvel.xyz - vi;
            let vn = dot(dv, n);
            let pen = h - r;
            // SHOCK-SOFTENING (telemetry-derived): impact kicks measured 1-7 km/s but decayed to
            // zero within ~1500s — damping smothered the excavation 1.5 radii in. Fast-moving
            // pairs lose solid damping so ejecta carries; idle churn measures ~10 m/s, far below
            // the 150 m/s threshold, so the boiling-Earth bug cannot return through this gate.
            // band sits ABOVE settle churn (~0.25 km/s measured) so ordinary internal motion
            // stays damped; only genuine impact-speed material (>0.4 km/s) gets mobilized
            let shockSoft = 1.0 - 0.93 * smoothstep(1.6e-7, 2.25e-6, dot(dv, dv)); // 0.4->1.5 km/s
            let cp = min(min(dampI, aux.w) * P.settleBoost, cMax) * shockSoft;
            dContacts = dContacts + 1.0;
            if (r < h * 1.08) { dTouch = dTouch + 1.0; }   // true touching neighbors (occlusion)
            dMinVn = min(dMinVn, vn);
            dMaxPen = max(dMaxPen, pen);
            var f = 0.0;
            if (pen > 0.0) {
              // force-capped repulsion (plastic yield): deep interpenetration can't store
              // explosive spring energy across timestep changes. Vaporized material pushes
              // MUCH harder (impact vapor plume — this is what excavates craters and throws
              // the splash). The boost uses the PAIR's hotter side so hot vapor shoves cold
              // rock just as hard as it shoves itself (Newton's 3rd law — without this, the
              // impactor blows itself to orbit while the target never feels the blast).
              let vapJ2 = smoothstep(matC.x * 0.8, matC.x * 1.4, pvel.w);
              let vapP = max(vapI, vapJ2);
              f = -kp * min(pen, 0.6 * radI * (1.0 + 8.0 * vapP));
            } else {
              f = cohp * (r - h);                  // cohesion (pulls i toward j)
            }
            let w = 1.0 - smoothstep(h * 0.9, reach, r);
            // compression (shock) heating: P·dV work while material is actively rammed —
            // the shock passing THROUGH an impactor processes its interior, so hit-and-run
            // fragments come out scorched and half-molten, not showroom-fresh. Double-gated
            // (fast closing AND deep penetration) so idle churn can never re-cook the planet.
            if (vn < -7e-4 && pen > 0.1 * radI) {
              let dqc = 0.3 * abs(f) * (-vn) * P.dt * heatKI / max(mi, 1e-12) * P.heatMul * P.heatGate;
              heat = heat + min(dqc, 0.25 * vn * vn * heatKI);
            }
            let fdamp = cp * vn * w;
            let vt = dv - n * vn;
            // tangential dead-zone below ~80 m/s: rigid planetary rotation shears neighbors at
            // ~33 m/s and must NOT be damped (it was spinning Earth down ~50% per sim-day)
            let ftan = cp * 0.35 * w * smoothstep(9e-10, 6.4e-9, dot(vt, vt));
            if (!armedI) {
              dvContact = dvContact + ((n * (f + fdamp) + vt * ftan) / max(mi, 1e-12)) * P.dt;
              // hypervelocity regime: impact pressure (~rho v^2) dwarfs material strength, but
              // the spring force is capped at static strength — so fast closings exchange
              // momentum DIRECTLY (inelastic step toward the pair's COM velocity). This is what
              // lets an impactor plow, excavate a crater, and throw ejecta instead of oozing in.
              if (vn < 0.0 && pen > -0.1 * h) {
                let hyp = smoothstep(1.2e-7, 4.0e-6, vn * vn);   // 0.35 km/s -> 2 km/s closing
                if (hyp > 0.0) {
                  let muJ = mj / max(mi + mj, 1e-12);
                  let alpha = hyp * 0.5;
                  dvContact = dvContact + n * (vn * muJ * alpha);
                  // exchange MOMENTUM starts at 0.35 km/s (shock propagation needs it), but its
                  // HEAT obeys the global ≥1.5 km/s rule — settle-churn tails must not cook
                  heat = heat + 0.5 * muJ * vn * vn * alpha * heatKI * P.heatMul * P.heatGate
                    * smoothstep(2.25e-6, 1.6e-5, vn * vn);
                  dMaxHyp = max(dMaxHyp, hyp);
                }
              }
            }
            // heat from dissipation, energy-bounded: a pair can never deposit more than ~half
            // the pair's relative specific KE in one substep (kills numerical over-heating).
            // Sub-300 m/s granular churn (molten-interior convection, settling creaks) does
            // not count as impact heating — real impacts arrive at km/s.
            // nothing real can hit a planet below escape velocity (~11 km/s for Earth), so
            // sub-1.5 km/s contact speeds are numerical rattle by definition — never heat
            let relSpeed2 = vn * vn + dot(vt, vt);
            let impactGate = smoothstep(2.25e-6, 1.6e-5, relSpeed2);  // 1.5 km/s → 4 km/s
            let diss = (cp * vn * vn + cp * 0.35 * dot(vt, vt)) * w;
            let dTraw = 0.5 * diss * P.dt * heatKI / max(mi, 1e-12) * P.heatMul;
            let dTcap = 0.25 * relSpeed2 * heatKI;
            heat = heat + min(dTraw, dTcap) * P.heatGate * impactGate;

            // thermal conduction from IMPACT-GRADE melt (>2200K): magma pools smooth into
            // gradients, and molten ejecta scorches whatever ground it lands on — global
            // heating after a big impact emerges from the worldwide fallback blanket.
            // Solid-on-solid does not conduct (real crust insulates for megayears), so the
            // pristine interior gradient can never cook the surface from below.
            let Tj = pvel.w;
            let condS = smoothstep(2200.0, 3800.0, max(Ti, Tj));
            if (condS > 0.0 && aux.x > 0.0) {
              // exponential transfer fraction: exactly framerate- AND density-independent and
              // unconditionally stable (never overshoots past equilibrium), unlike a min() clamp
              var dq = (Tj - Ti) * (1.0 - exp(-P.dt * 4e-4 * condS)) * w;
              // receivers SATURATE in the orange band (1400-2100K): scorch halos hold a
              // visible white→orange→red gradient instead of equilibrating straight to
              // white. Only direct impact heating exceeds this ceiling.
              if (dq > 0.0) { dq = dq * (1.0 - smoothstep(1400.0, 2100.0, Ti)); }
              heat = heat + dq;
            }
            // warm-tier conduction strictly SURFACE↔SURFACE (both sides land/ocean/ice): the
            // burn front crawls along the surface, while the crust-mantle boundary is sealed
            // by material type — no temperature margin to leak through.
            if (matB.x > 0.5 && aux.y > 0.0) {
              let hotSide = max(Ti, Tj);
              if (hotSide > 1150.0 && hotSide < 2600.0) {
                var dq2 = (Tj - Ti) * (1.0 - exp(-P.dt * 1.5e-4)) * w;
                if (dq2 > 0.0) { dq2 = dq2 * (1.0 - smoothstep(1100.0, 1700.0, Ti)); }
                heat = heat + dq2;
              }
            }

          }
        }
`;
}

// ---- post-loop: sun gravity, contact-dv cap, settle drag, integrate, heat/cool, guards,
// debug taps, occlusion bits, sun-eater, write-out. Shared by both engines; the fast engine
// omits the debug tap (its dbg storage slot is spent on the gravity monopole grid). ----
export function integrateTailWGSL({ debugTap = true } = {}) {
  return /* wgsl */`
  // sun gravity — DIFFERENTIAL (tidal): particle pull minus the frame anchor's pull, so the
  // Earth blob stays pinned at the origin instead of all particles sliding toward the Sun.
  // Must match the CPU mirror integration in sim.js.
  let dsun = P.sunPos - pi;
  let r2s = dot(dsun, dsun);
  let invRs = inverseSqrt(max(r2s, 1.0));
  let invRa = inverseSqrt(max(dot(P.sunPos, P.sunPos), 1.0));
  let aAnchor = P.sunPos * (P.gmSun * invRa * invRa * invRa);
  acc = acc + dsun * (P.gmSun * invRs * invRs * invRs) - aAnchor;

  // cap TOTAL contact velocity change per substep (~6 km/s): many simultaneous contacts must
  // not sum into 100s of km/s shrapnel — this keeps impact energy grinding at the interface
  let dvLen = length(dvContact);
  if (dvLen > 6e-3) { dvContact = dvContact * (6e-3 / dvLen); }

  var v = vi + acc * P.dt + dvContact;
  // settle drag: pull toward the body's rigid reference motion (bulk + spin) so gravitational
  // compaction proceeds quasi-statically — discards Kelvin-Helmholtz churn without killing spin
  if (P.settleDrag > 0.0) {
    let bd = bodyDyn[(mymeta >> 4u) & 15u];
    let vRef = bd.vc.xyz + cross(bd.om.xyz, pi - bd.cp.xyz);
    v = v + (vRef - v) * min(0.5, P.settleDrag * P.dt);
  }
  let sp = length(v);
  if (sp > P.vClamp) { v = v * (P.vClamp / sp); }
  var p = pi + v * P.dt;

  var T = Ti;
  var newMeta = mymeta;
  if (armedI) {
    let frac = clamp(0.6 * P.dt, 0.0, 0.2);
    T = max(0.0, Ti * (1.0 - frac * min(nPayloadHits, 10.0)));
    if (T < Ti * 0.02 && nPayloadHits > 0.0) {
      newMeta = (newMeta & ~512u) | 256u;   // disarm → becomes a molten ember
      T = 3400.0;
    }
  } else {
    T = T + heat;
    // solar radiative equilibrium pull-up (slow) + Stefan-Boltzmann cooling
    // radiative exchange with space scales with EXPOSURE: free-flying ejecta radiates fully
    // (glowing rain dims as it falls), crater surfaces cool at partial rate, deeply buried
    // material (12+ contacts) is insulated and only loses heat via conduction to the surface
    let exposure = clamp(1.0 - dContacts / 12.0, 0.03, 1.0);
    let dSun = 1.0 / invRs;
    let teq = 278.6 * sqrt(149597.87 / max(dSun, 700.0)) * pow(max(P.solarLum, 0.01), 0.25);
    if (teq > T) {
      T = T + (teq - T) * (1.0 - exp(-P.dt / 40000.0)) * exposure;
    }
    // atmospheric bake: hot air (post-impact ejecta re-entry) cooks exposed surface
    // material everywhere on the planet, not just near the crater
    let atmT = bodyDyn[(mymeta >> 4u) & 15u].vc.w;
    if (matB.x > 0.5 && atmT > T) {
      T = T + (atmT - T) * (1.0 - exp(-P.dt / 9000.0)) * max(exposure, 0.15);
    }
    let t2 = T * T;
    T = T - coolK * t2 * t2 * P.dt * P.coolMul * exposure;
    T = clamp(T, 2.7, 26000.0);
    if (T > matB.w) { newMeta = newMeta | 256u; }   // ever-molten flag
  }

  // NaN guard — freeze instead of exploding (bit 11 records that it fired, for diagnostics)
  if (p.x != p.x || p.y != p.y || p.z != p.z || v.x != v.x || v.y != v.y || v.z != v.z || T != T) {
    p = pi; v = vec3f(0.0); T = max(Ti, 300.0);
    newMeta = newMeta | 2048u;
  }

${debugTap ? `  if (i == bitcast<u32>(P.debugIdxBits)) {
    dbg[0] = vec4f(dContacts, dMinVn, dMaxPen, dMaxHyp);
    dbg[1] = vec4f(dvContact, length(dvContact));
    dbg[2] = vec4f(acc * P.dt, heat);
    dbg[3] = vec4f(v, T);
    dbg[4] = vec4f(P.dt, mi, radI, kI);
    dbg[5] = vec4f(vi, Ti);
  }
` : ''}
  // adjacency count → meta bits 12-15: the renderer culls deeply-buried particles
  newMeta = (newMeta & ~(15u << 12u)) | (u32(min(dTouch, 15.0)) << 12u);

  // THE SUN IS A UNIVERSAL DESTROYER — the CPU flags a whole body's slot once its path crosses
  // the Sun's 5× kill radius, and every particle of that body is annihilated at once: all motion
  // halted, mass→0 (gone from physics + render), parked dead-still at the Sun's center.
  let consumed = bitcast<u32>(P.consumedBits);
  if ((consumed & (1u << ((mymeta >> 4u) & 15u))) != 0u) {
    p = P.sunPos; v = vec3f(0.0); mi = 0.0;
    newMeta = newMeta | 4096u;
  }

  posB[i] = vec4f(p, mi);
  velB[i] = vec4f(v, T);
  pmeta[i] = newMeta;
`;
}

// back-compat: the tail with debug taps (used by the N² kernel below)
export const INTEGRATE_TAIL_WGSL = integrateTailWGSL({ debugTap: true });

// accumulator + self-state declarations shared by both kernels' preambles
export const SELF_DECLS_WGSL = /* wgsl */`
  var pi = vec3f(0.0); var mi = 0.0;
  var vi = vec3f(0.0); var Ti = 0.0;
  var mymeta = 0u;
  var radI = 0.0; var kI = 0.0; var cohI = 0.0; var dampI = 0.0; var heatKI = 0.0;
  var matB = vec4f(0.0); var matC = vec4f(0.0); var coolK = 0.0;
  var armedI = false;
  var vapI = 0.0;

  var acc = vec3f(0.0);
  var dvContact = vec3f(0.0);   // contact-force velocity change, capped after the loop:
                                // a particle buried among 50 neighbors must not sum 50 kicks
  var heat = 0.0;
  var nPayloadHits = 0.0;
  // debug tap accumulators (only written out for the P.debugIdxBits particle)
  var dContacts = 0.0;
  var dTouch = 0.0;
  var dMinVn = 0.0;
  var dMaxHyp = 0.0;
  var dMaxPen = 0.0;
`;

// ================= the proven N² kernel, assembled from the shared chunks =================
export const SIM_WGSL = /* wgsl */`
${SIM_STRUCTS_WGSL}
${SIM_BINDINGS_WGSL}
var<workgroup> tPos: array<vec4f, 256>;
var<workgroup> tVel: array<vec4f, 256>;
var<workgroup> tAux: array<vec4f, 256>;   // (radius [neg if armed], k, cohEff, damp)
${SIM_HELPERS_WGSL}
@compute @workgroup_size(256)
fn sim(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lidv: vec3u) {
  let i = gid.x;
  let lid = lidv.x;
  let isOn = i < P.nActive;
${SELF_DECLS_WGSL}
  if (isOn) {
    let pm = posA[i]; pi = pm.xyz; mi = pm.w;
    let vt = velA[i]; vi = vt.xyz; Ti = vt.w;
    mymeta = pmeta[i];
${SELF_MATS_WGSL}
  }

  for (var t = 0u; t < P.numTiles; t = t + 1u) {
    let j = t * 256u + lid;
    if (j < P.nActive) {
      let pmj = posA[j];
      let vtj = velA[j];
      tPos[lid] = pmj;
      tVel[lid] = vtj;
      tAux[lid] = neighborAux(pmeta[j], vtj);
    } else {
      tPos[lid] = vec4f(3e7, 3e7, 3e7, 0.0);
      tVel[lid] = vec4f(0.0);
      tAux[lid] = vec4f(0.001, 0.0, 0.0, 0.0);
    }
    workgroupBarrier();

    if (isOn) {
      for (var jj = 0u; jj < 256u; jj = jj + 1u) {
        let pj = tPos[jj];
        let pvel = tVel[jj];
        let aux = tAux[jj];
${pairPhysicsWGSL({ nearGravity: true })}
      }
    }
    workgroupBarrier();
  }

  if (!isOn) { return; }
${INTEGRATE_TAIL_WGSL}
}

// --- thermal-only step: O(N), unconditionally stable at ANY dt ---
// Runs while mechanics are frozen (settled aftermath): radiative cooling uses the exact
// closed-form solution of dT/dt = -k T^4, so days can pass per frame while the glow fades.
@group(0) @binding(0) var<uniform> PT: SimParams;
@group(0) @binding(1) var<storage, read> matsT: array<MatI, 16>;
@group(0) @binding(2) var<storage, read_write> velTh: array<vec4f>;
@group(0) @binding(3) var<storage, read> pmetaT: array<u32>;
@group(0) @binding(4) var<storage, read> bodyDynT: array<BodyDyn, 16>;

@compute @workgroup_size(256)
fn thermal(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= PT.nActive) { return; }
  let mt = pmetaT[i];
  if ((mt & 512u) != 0u) { return; }              // armed payloads keep their energy budget
  let m = matsT[mt & 15u];
  var T = velTh[i].w;
  let touch = f32((mt >> 12u) & 15u);
  let exposure = clamp(1.0 - touch / 12.0, 0.03, 1.0);
  // atmosphere bake (exposed surface materials)
  let atmT = bodyDynT[(mt >> 4u) & 15u].vc.w;
  if (m.b.x > 0.5 && atmT > T) {
    T = T + (atmT - T) * (1.0 - exp(-PT.dt / 9000.0)) * max(exposure, 0.15);
  }
  // exact radiative cooling: T(t) = (T0^-3 + 3 k t)^(-1/3)
  let kEff = m.d.w * PT.coolMul * exposure;
  if (kEff > 0.0 && T > 3.0) {
    T = pow(1.0 / (T * T * T) + 3.0 * kEff * PT.dt, -1.0 / 3.0);
  }
  T = clamp(T, 2.7, 26000.0);
  velTh[i] = vec4f(velTh[i].xyz, T);
}

// --- frame rebase / rigid body shift (with rigid ROTATION for frozen ride-along) ---
// slot == 0xffffffff: shift ALL particles by -dp/-dv (frame rebase, no rotation).
// otherwise: particles of that body slot rotate by rotAA (axis, angle) about com, then
// translate by +dp/+dv. Rotation keeps planets spinning while mechanics are frozen —
// positions orbit the spin axis and the spin-component of velocity rotates with them.
struct Shift {
  dp: vec3f, n: u32,
  dv: vec3f, slot: u32,
  rotAA: vec4f,    // xyz = unit axis, w = angle (radians); w == 0 → no rotation
  com: vec4f,      // rotation centre (body CoM)
  vcom: vec4f,     // body bulk velocity — only the spin part of velocity rotates
}
@group(0) @binding(0) var<uniform> S: Shift;
@group(0) @binding(1) var<storage, read_write> posR: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> velR: array<vec4f>;
@group(0) @binding(3) var<storage, read> metaR: array<u32>;

fn rodrigues(v: vec3f, k: vec3f, c: f32, s: f32) -> vec3f {
  return v * c + cross(k, v) * s + k * (dot(k, v) * (1.0 - c));
}

@compute @workgroup_size(256)
fn rebase(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= S.n) { return; }
  if (S.slot == 0xffffffffu) {
    let pm = posR[i];
    posR[i] = vec4f(pm.xyz - S.dp, pm.w);
    let vt = velR[i];
    velR[i] = vec4f(vt.xyz - S.dv, vt.w);
  } else {
    if (((metaR[i] >> 4u) & 15u) == S.slot) {
      let pm = posR[i];
      let vt = velR[i];
      var p = pm.xyz;
      var v = vt.xyz;
      if (S.rotAA.w != 0.0) {
        let c = cos(S.rotAA.w);
        let sn = sin(S.rotAA.w);
        p = S.com.xyz + rodrigues(p - S.com.xyz, S.rotAA.xyz, c, sn);
        v = S.vcom.xyz + rodrigues(v - S.vcom.xyz, S.rotAA.xyz, c, sn);
      }
      posR[i] = vec4f(p + S.dp, pm.w);
      velR[i] = vec4f(v + S.dv, vt.w);
    }
  }
}
`;
