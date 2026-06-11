// shaders_render.js — all scene WGSL. Render frame = world relative to focus body (fp32-safe).

const FRAME = /* wgsl */`
struct Frame {
  viewProj: mat4x4f,
  camPos: vec3f, tanHalf: f32,
  camRight: vec3f, aspect: f32,
  camUp: vec3f, time: f32,
  camFwd: vec3f, exposure: f32,
  sunPosRel: vec3f, pixScale: f32,
  partOffset: vec3f, starBoost: f32,
}
@group(0) @binding(0) var<uniform> F: Frame;
`;

const NOISE = /* wgsl */`
fn hash31(p3in: vec3f) -> f32 {
  var p3 = fract(p3in * 0.1031);
  p3 = p3 + dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
fn vnoise(x: vec3f) -> f32 {
  let p = floor(x);
  let f = fract(x);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash31(p + vec3f(0.0, 0.0, 0.0)), hash31(p + vec3f(1.0, 0.0, 0.0)), u.x),
        mix(hash31(p + vec3f(0.0, 1.0, 0.0)), hash31(p + vec3f(1.0, 1.0, 0.0)), u.x), u.y),
    mix(mix(hash31(p + vec3f(0.0, 0.0, 1.0)), hash31(p + vec3f(1.0, 0.0, 1.0)), u.x),
        mix(hash31(p + vec3f(0.0, 1.0, 1.0)), hash31(p + vec3f(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}
`;

// ---------- background stars (equirect texture by view ray) ----------
export const STARS_WGSL = FRAME + /* wgsl */`
@group(1) @binding(0) var starTex: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;

struct VOut { @builtin(position) pos: vec4f, @location(0) ndc: vec2f }

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  o.pos = vec4f(xy, 0.0, 1.0);
  o.ndc = xy;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let dir = normalize(F.camFwd + F.camRight * (in.ndc.x * F.tanHalf * F.aspect) + F.camUp * (in.ndc.y * F.tanHalf));
  let u = atan2(dir.y, dir.x) / 6.2831853 + 0.5;
  let v = acos(clamp(dir.z, -1.0, 1.0)) / 3.14159265;
  var c = textureSampleLevel(starTex, smp, vec2f(u, v), 0.0).rgb;
  // prettify: gentle saturation lift so the galactic dust glows warm/blue instead of grey,
  // a soft gamma to deepen the gaps, and the brightest band pushed a touch for the Milky Way.
  let lum = dot(c, vec3f(0.299, 0.587, 0.114));
  c = mix(vec3f(lum), c, 1.35);                       // +saturation
  c = pow(max(c, vec3f(0.0)), vec3f(1.18));            // deepen the void between stars
  c = c + c * smoothstep(0.25, 0.8, lum) * 0.6;        // bloomable lift on the galactic plane
  // deep-space ambient floor — never pure black, faint cold tint with a hint of warmth
  let ambient = vec3f(0.013, 0.016, 0.028);
  return vec4f(ambient + c * F.starBoost, 1.0);
}
`;

// ---------- shared sphere draw (planets + sun + body shells) ----------
export const SPHERE_WGSL = FRAME + NOISE + /* wgsl */`
struct DrawU {
  model: mat4x4f,
  params: vec4f,   // x: emissive mult (sun), y: ring inner, z: ring outer, w: fade (shells)
}
@group(1) @binding(0) var<uniform> D: DrawU;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var smp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) wpos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
}
@vertex fn vs(@location(0) p: vec3f, @location(1) uv: vec2f) -> VOut {
  var o: VOut;
  let wp = (D.model * vec4f(p, 1.0)).xyz;
  o.wpos = wp;
  o.nrm = normalize((D.model * vec4f(p, 0.0)).xyz);
  o.uv = uv;
  o.pos = F.viewProj * vec4f(wp, 1.0);
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let c = textureSample(tex, smp, in.uv).rgb;
  let N = normalize(in.nrm);
  // body shells burn away like Earth's globe (noise cut + glowing edge), not a plain fade
  let fade = select(D.params.w, 1.0, D.params.w <= 0.0);
  if (fade < 0.999) {
    let dis = (1.0 - fade) * 1.12;
    let nz = vnoise(N * 3.4) * 0.65 + vnoise(N * 9.1) * 0.35;
    if (nz < dis - 0.06) { discard; }
    if (nz < dis + 0.05) {
      return vec4f(vec3f(6.0, 1.6, 0.25) * (1.0 - (nz - dis) / 0.05), 1.0);
    }
  }
  let L = normalize(F.sunPosRel - in.wpos);
  let V = normalize(F.camPos - in.wpos);
  let diff = max(dot(N, L), 0.0);
  var col = c * (diff * 1.25 + 0.012);
  if (D.params.x > 0.0) {
    let limb = pow(max(dot(N, V), 0.0), 0.45);
    col = c * (0.45 + 0.75 * limb) * D.params.x;
  }
  return vec4f(col, 1.0);
}
`;

// ---------- sun corona billboard (additive) ----------
export const CORONA_WGSL = FRAME + /* wgsl */`
struct DrawU { model: mat4x4f, params: vec4f }  // params.x: intensity, params.y: world radius of quad
@group(1) @binding(0) var<uniform> D: DrawU;

struct VOut { @builtin(position) pos: vec4f, @location(0) q: vec2f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let q = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  let center = D.model[3].xyz;
  let wp = center + (F.camRight * q.x + F.camUp * q.y) * D.params.y;
  o.pos = F.viewProj * vec4f(wp, 1.0);
  o.q = q;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let r = length(in.q);
  if (r > 1.0) { discard; }
  let glow = exp(-r * 4.5) * 3.0 + pow(max(1.0 - r, 0.0), 3.0) * 0.7;
  let col = vec3f(1.0, 0.58, 0.24) * glow * D.params.x;
  return vec4f(col, 1.0);
}
`;

// ---------- saturn ring ----------
export const RING_WGSL = FRAME + /* wgsl */`
struct DrawU { model: mat4x4f, params: vec4f }  // y: inner, z: outer (Mm)
@group(1) @binding(0) var<uniform> D: DrawU;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var smp: sampler;

struct VOut { @builtin(position) pos: vec4f, @location(0) wpos: vec3f, @location(1) rad: f32, @location(2) nrm: vec3f }
@vertex fn vs(@location(0) p: vec3f, @location(1) uv: vec2f) -> VOut {
  var o: VOut;
  let wp = (D.model * vec4f(p, 1.0)).xyz;
  o.wpos = wp;
  o.rad = length(p.xy);
  o.nrm = normalize((D.model * vec4f(0.0, 0.0, 1.0, 0.0)).xyz);
  o.pos = F.viewProj * vec4f(wp, 1.0);
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let u = clamp((in.rad - D.params.y) / (D.params.z - D.params.y), 0.0, 1.0);
  let c = textureSample(tex, smp, vec2f(u, 0.5));
  let L = normalize(F.sunPosRel - in.wpos);
  let lit = abs(dot(in.nrm, L)) * 0.85 + 0.10;
  return vec4f(c.rgb * lit, c.a * 0.96);
}
`;

// ---------- hero globe (pristine Earth) ----------
export const GLOBE_WGSL = FRAME + NOISE + /* wgsl */`
struct GlobeU {
  model: mat4x4f,
  dissolve: f32, cloudRot: f32, cityDim: f32, cloudOp: f32,
  atmoDensity: f32, radius: f32, pad0: f32, pad1: f32,
}
@group(1) @binding(0) var<uniform> GU: GlobeU;
@group(1) @binding(1) var dayTex: texture_2d<f32>;
@group(1) @binding(2) var nightTex: texture_2d<f32>;
@group(1) @binding(3) var cloudTex: texture_2d<f32>;
@group(1) @binding(4) var smp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) wpos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
  @location(3) opos: vec3f,
}
@vertex fn vs(@location(0) p: vec3f, @location(1) uv: vec2f) -> VOut {
  var o: VOut;
  let wp = (GU.model * vec4f(p, 1.0)).xyz;
  o.wpos = wp;
  o.nrm = normalize((GU.model * vec4f(p, 0.0)).xyz);
  o.uv = uv;
  o.opos = p;
  o.pos = F.viewProj * vec4f(wp, 1.0);
  return o;
}

@fragment fn fsSurface(in: VOut) -> @location(0) vec4f {
  // sample first — textureSample requires uniform control flow
  let day = textureSample(dayTex, smp, in.uv).rgb;
  let night = textureSample(nightTex, smp, in.uv).rgb;
  // dissolve cut
  if (GU.dissolve > 0.001) {
    let n = vnoise(in.opos * 3.4) * 0.65 + vnoise(in.opos * 9.1) * 0.35;
    let cut = GU.dissolve * 1.12;
    if (n < cut - 0.06) { discard; }
    if (n < cut + 0.05) {
      return vec4f(vec3f(6.0, 1.6, 0.25) * (1.0 - (n - cut) / 0.05), 1.0);
    }
  }
  let N = normalize(in.nrm);
  let L = normalize(F.sunPosRel - in.wpos);
  let V = normalize(F.camPos - in.wpos);
  let d = dot(N, L);
  let dl = max(d, 0.0);
  let ocean = clamp((day.b - max(day.r, day.g)) * 5.0, 0.0, 1.0);
  let R = reflect(-L, N);
  let spec = pow(max(dot(R, V), 0.0), 110.0) * ocean * 1.4 * dl;
  let lights = night * smoothstep(0.05, -0.12, d) * GU.cityDim * 2.0;
  var col = day * (dl * 1.30 + 0.008) + vec3f(spec) + lights;
  return vec4f(col, 1.0);
}

@fragment fn fsClouds(in: VOut) -> @location(0) vec4f {
  let N = normalize(in.nrm);
  let L = normalize(F.sunPosRel - in.wpos);
  let c = textureSample(cloudTex, smp, vec2f(in.uv.x + GU.cloudRot, in.uv.y)).rgb;
  let lum = dot(c, vec3f(0.333));
  let dl = max(dot(N, L), 0.0);
  var a = lum * GU.cloudOp * (1.0 - GU.dissolve * 1.6);
  a = clamp(a, 0.0, 1.0);
  let col = vec3f(1.0, 0.99, 0.97) * (dl * 1.25 + 0.02);
  return vec4f(col * a, a);
}

@fragment fn fsAtmo(in: VOut) -> @location(0) vec4f {
  let N = normalize(in.nrm);
  let L = normalize(F.sunPosRel - in.wpos);
  let V = normalize(F.camPos - in.wpos);
  let rim = pow(1.0 - abs(dot(V, N)), 2.8);
  let d = dot(N, L);
  let day = clamp(d + 0.25, 0.0, 1.0);
  var col = mix(vec3f(0.02, 0.05, 0.22), vec3f(0.22, 0.52, 1.0), day) * rim * 1.6;
  col = col + vec3f(1.0, 0.32, 0.06) * pow(max(1.0 - abs(d), 0.0), 7.0) * rim * 0.9;
  let a = GU.atmoDensity * (1.0 - GU.dissolve);
  return vec4f(col * a, 0.0);
}
`;

// ---------- particle sphere impostors ----------
export const PARTICLES_WGSL = FRAME + /* wgsl */`
struct MatI { a: vec4f, b: vec4f, c: vec4f, d: vec4f }
@group(1) @binding(0) var<storage, read> posM: array<vec4f>;
@group(1) @binding(1) var<storage, read> velT: array<vec4f>;
@group(1) @binding(2) var<storage, read> pmeta: array<u32>;
@group(1) @binding(3) var<storage, read> albedo: array<u32>;
@group(1) @binding(4) var<storage, read> mats: array<MatI, 16>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) center: vec3f,       // render-frame world
  @location(1) wpos: vec3f,         // corner world pos (ray target)
  @location(2) @interpolate(flat) rv: f32,
  @location(3) @interpolate(flat) T: f32,
  @location(4) @interpolate(flat) col: vec3f,
  @location(5) @interpolate(flat) flags: u32,
  @location(6) @interpolate(flat) emisP: vec2f,  // material self-emis, isGas
}

// Planckian locus chroma (sRGB fit): dull red → cherry → orange → yellow-white → white → blue-white
fn blackbody(T: f32) -> vec3f {
  let t = clamp(T, 700.0, 12000.0);
  var c: vec3f;
  if (t < 1000.0) { c = mix(vec3f(1.0, 0.12, 0.0), vec3f(1.0, 0.28, 0.01), (t - 700.0) / 300.0); }
  else if (t < 1500.0) { c = mix(vec3f(1.0, 0.28, 0.01), vec3f(1.0, 0.52, 0.10), (t - 1000.0) / 500.0); }
  else if (t < 2200.0) { c = mix(vec3f(1.0, 0.52, 0.10), vec3f(1.0, 0.70, 0.32), (t - 1500.0) / 700.0); }
  else if (t < 3300.0) { c = mix(vec3f(1.0, 0.70, 0.32), vec3f(1.0, 0.84, 0.60), (t - 2200.0) / 1100.0); }
  else if (t < 5000.0) { c = mix(vec3f(1.0, 0.84, 0.60), vec3f(1.0, 0.95, 0.88), (t - 3300.0) / 1700.0); }
  else if (t < 6600.0) { c = mix(vec3f(1.0, 0.95, 0.88), vec3f(1.0, 1.0, 1.0), (t - 5000.0) / 1600.0); }
  else { c = mix(vec3f(1.0, 1.0, 1.0), vec3f(0.75, 0.84, 1.0), clamp((t - 6600.0) / 5400.0, 0.0, 1.0)); }
  return c;
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let pm = posM[ii];
  if (pm.w <= 0.0) {
    o.pos = vec4f(2e9, 2e9, 2.0, 1.0);
    return o;
  }
  let mt = pmeta[ii];
  // occlusion cull: deeply buried particles (11+ touching neighbors) can never be seen —
  // when an impact tears the surface open, exposed interior pops in automatically
  if (((mt >> 12u) & 15u) >= 11u) {
    o.pos = vec4f(2e9, 2e9, 2.0, 1.0);
    return o;
  }
  let m = mats[mt & 15u];
  let vt = velT[ii];
  let center = pm.xyz + F.partOffset;
  var rv = m.a.x * m.c.z;
  // keep distant debris visible: enforce ~1.1 px minimum
  let dist = length(center - F.camPos);
  let minR = 1.1 * dist / max(F.pixScale, 1.0);
  rv = max(rv, minR);
  let q = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  let wp = center + (F.camRight * q.x + F.camUp * q.y) * (rv * 1.42);
  o.pos = F.viewProj * vec4f(wp, 1.0);
  o.center = center;
  o.wpos = wp;
  o.rv = rv;
  var T = vt.w;
  if ((mt & 512u) != 0u) { T = 2600.0; }   // armed impactor: render as hot tracer
  o.T = T;
  let alb = unpack4x8unorm(albedo[ii]).rgb;
  var lin = pow(alb, vec3f(2.2));
  if ((mt & 256u) != 0u) {                 // ever-molten: cooled lava turns basalt-dark
    lin = mix(lin, vec3f(0.045, 0.04, 0.038), 0.7);
  }
  // ice melts: frigid white albedo becomes seawater above freezing, THEN the scorch/steam
  // chain applies — ice caps must not stay snow-white through an apocalypse
  let isIceV = lin.b > 0.5 && lin.r > 0.35 && lin.b >= lin.r;
  if (isIceV) {
    lin = mix(lin, vec3f(0.02, 0.07, 0.18), smoothstep(274.0, 295.0, T));
  }
  // scorch ramp COMPLETES to near-black BEFORE the glow begins (~700K vs glow onset 700-980):
  // otherwise sunlit half-burnt land renders as light brown between the char and the glow.
  // Burn scars are black; black is what makes the red rim read.
  let scorch = smoothstep(340.0, 700.0, T);
  if (scorch > 0.0) {
    let isWaterV = lin.b > lin.r * 1.6 && lin.b > 0.15;
    let charT = select(vec3f(0.018, 0.016, 0.014), vec3f(0.38, 0.40, 0.43), isWaterV);
    lin = mix(lin, charT, scorch * 0.97);
  }
  o.col = lin;
  o.flags = mt;
  o.emisP = vec2f(m.c.y, m.c.w);
  return o;
}

struct FOut { @location(0) col: vec4f, @builtin(frag_depth) depth: f32 }

@fragment fn fs(in: VOut) -> FOut {
  let ro = F.camPos;
  let rd = normalize(in.wpos - ro);
  let oc = ro - in.center;
  let b = dot(rd, oc);
  let disc = b * b - (dot(oc, oc) - in.rv * in.rv);
  if (disc < 0.0) { discard; }
  let t = -b - sqrt(disc);
  if (t < 0.0) { discard; }
  let hit = ro + rd * t;
  let N = (hit - in.center) / in.rv;
  let L = normalize(F.sunPosRel - hit);
  var diff = max(dot(N, L), 0.0);
  if (in.emisP.y > 0.5) { diff = max(dot(N, L) * 0.6 + 0.4, 0.0); }  // gas: wrap light
  let V = -rd;
  let R = reflect(-L, N);
  let isWater = in.col.b > in.col.r * 1.6 && in.col.b > 0.15;
  var spec = 0.0;
  if (isWater) { spec = pow(max(dot(R, V), 0.0), 60.0) * 0.9 * diff; }
  var col = in.col * (diff * 1.30 + 0.015) + vec3f(spec);
  // thermal emission
  let T = in.T;
  // DESIGNED incandescence gradient (entertainment-first, per user): saturated stops chosen
  // to survive ACES, REPLACING the surface color as it heats — additive glow over sunlit
  // albedo can only ever produce cream/tan, never movie-grade orange.
  // Stops are NUMERICALLY INVERTED through ACES+gamma so each band displays the intended
  // screen color exactly: ember red -> deep red -> SATURATED ORANGE -> golden yellow ->
  // yellow-white -> white. (Eyeballed stops come out brown/tan: ACES lifts midtones hard.)
  let glowP = smoothstep(760.0, 1060.0, T);
  if (glowP > 0.0) {
    let t = clamp(T, 800.0, 6000.0);
    var g: vec3f;
    if (t < 1100.0) { g = mix(vec3f(0.10, 0.004, 0.0003), vec3f(0.30, 0.012, 0.001), (t - 800.0) / 300.0); }
    else if (t < 1500.0) { g = mix(vec3f(0.30, 0.012, 0.001), vec3f(1.70, 0.18, 0.014), (t - 1100.0) / 400.0); }
    else if (t < 2200.0) { g = mix(vec3f(1.70, 0.18, 0.014), vec3f(2.60, 0.56, 0.026), (t - 1500.0) / 700.0); }
    else if (t < 3200.0) { g = mix(vec3f(2.60, 0.56, 0.026), vec3f(4.50, 1.40, 0.18), (t - 2200.0) / 1000.0); }
    else { g = mix(vec3f(4.50, 1.40, 0.18), vec3f(6.0, 5.5, 4.5), clamp((t - 3200.0) / 1300.0, 0.0, 1.0)); }
    col = mix(col, g, glowP);
  }
  col = col + in.col * in.emisP.x;
  var f: FOut;
  let clip = F.viewProj * vec4f(hit, 1.0);
  f.depth = clip.z / clip.w;
  f.col = vec4f(col, 1.0);
  return f;
}
`;

// ---------- polylines (orbits / trails / aim) ----------
export const LINES_WGSL = FRAME + /* wgsl */`
struct LineU { offset: vec3f, alpha: f32 }
@group(1) @binding(0) var<uniform> LU: LineU;

struct VOut { @builtin(position) pos: vec4f, @location(0) col: vec4f }
@vertex fn vs(@location(0) p: vec3f, @location(1) c: vec4f) -> VOut {
  var o: VOut;
  o.pos = F.viewProj * vec4f(p + LU.offset, 1.0);
  o.col = c;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  return vec4f(in.col.rgb * in.col.a * LU.alpha, in.col.a * LU.alpha);
}
`;

// ---------- decorative asteroid belt (Kepler in vertex shader) ----------
export const BELT_WGSL = FRAME + /* wgsl */`
struct BeltU { d2000: f32, alpha: f32, count: u32, focusOffX: f32, focusOff: vec3f, pad: f32 }
@group(1) @binding(0) var<uniform> BU: BeltU;
@group(1) @binding(1) var<storage, read> elems: array<vec4f>;  // pairs: [a,e,inc,node],[w,M0,size,pad]

struct VOut { @builtin(position) pos: vec4f, @location(0) q: vec2f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let ai = vi / 6u;
  let corner = vi % 6u;
  if (ai >= BU.count) { o.pos = vec4f(2e9, 2e9, 2.0, 1.0); return o; }
  let e1 = elems[ai * 2u];
  let e2 = elems[ai * 2u + 1u];
  let a = e1.x; let ecc = e1.y; let inc = e1.z; let node = e1.w;
  let w = e2.x;
  let n = sqrt(132.71244 / (a * a * a));      // rad/s  (GM_SUN, Mm^3/s^2)
  var M = e2.y + n * BU.d2000 * 86400.0;
  M = M - floor(M / 6.2831853) * 6.2831853;
  var E = M + ecc * sin(M);
  E = M + ecc * sin(E);
  let xp = a * (cos(E) - ecc);
  let yp = a * sqrt(1.0 - ecc * ecc) * sin(E);
  let cw = cos(w); let sw = sin(w); let cn = cos(node); let sn = sin(node); let ci = cos(inc); let si = sin(inc);
  let pos = vec3f(
    (cw * cn - sw * sn * ci) * xp + (-sw * cn - cw * sn * ci) * yp,
    (cw * sn + sw * cn * ci) * xp + (-sw * sn + cw * cn * ci) * yp,
    (sw * si) * xp + (cw * si) * yp,
  );
  let wp = pos + BU.focusOff;
  var clip = F.viewProj * vec4f(wp, 1.0);
  var qq = vec2f(0.0);
  if (corner == 1u || corner == 4u || corner == 5u) { qq.x = 1.0; } else { qq.x = -1.0; }
  if (corner == 2u || corner == 3u || corner == 5u) { qq.y = 1.0; } else { qq.y = -1.0; }
  let hPx = F.pixScale * 2.0 * F.tanHalf;
  let pxNdc = 2.0 / max(hPx, 64.0);
  clip = vec4f(clip.xy + qq * e2.z * pxNdc * clip.w * vec2f(1.0 / F.aspect, 1.0), clip.zw);
  o.pos = clip;
  o.q = qq;
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let r = length(in.q);
  if (r > 1.0) { discard; }
  return vec4f(vec3f(0.42, 0.38, 0.33) * BU.alpha * (1.0 - r * 0.7), 0.0);
}
`;
