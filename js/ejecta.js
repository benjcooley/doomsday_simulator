// ejecta.js — impact ejecta: a separate, lightweight pool of TINY particles thrown out on a hit.
// One-way coupled: Earth is a single point mass (Earth-centred frame, Earth at origin); ejecta
// feels its gravity and flies ballistically, but never perturbs Earth back (excavated mass is a
// tiny fraction of the planet). Cratering physics: a ~45° curtain with a power-law speed spread —
// slow ejecta arcs back as rays, fast ejecta goes suborbital, escape-velocity ejecta forms a ring.
import { GM_EARTH } from './orbits.js';

export const EMAX = 49152;

const COMPUTE_WGSL = /* wgsl */`
struct EParams {
  dt: f32, count: u32, earthGM: f32, earthR: f32,
  coolMul: f32, pad0: f32, pad1: f32, pad2: f32,
}
@group(0) @binding(0) var<uniform> P: EParams;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4f>;   // xyz (Earth-centred Mm), w = life left (s)
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;   // xyz (Mm/s), w = temperature (K)

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) { return; }
  var pm = pos[i];
  if (pm.w <= 0.0) { return; }                       // dead slot
  var vt = vel[i];
  let r = length(pm.xyz);
  if (r > 1e-5) {
    let a = pm.xyz * (-P.earthGM / (r * r * r));      // Earth as a single gravity point
    vt = vec4f(vt.xyz + a * P.dt, vt.w);
  }
  pm = vec4f(pm.xyz + vt.xyz * P.dt, pm.w - P.dt);
  // radiative cool toward a warm floor
  let T = max(300.0, vt.w - vt.w * 0.0006 * P.coolMul * P.dt);
  vt.w = T;
  // landed (fell back below the surface) or expired → kill
  if (length(pm.xyz) < P.earthR * 0.992 || pm.w <= 0.0) { pm.w = 0.0; }
  pos[i] = pm;
  vel[i] = vt;
}
`;

// rendered by renderer.js (it owns the HDR pass); this module just provides the pipeline + buffers.
export const EJECTA_RENDER_WGSL = /* wgsl */`
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
struct ERender { earthRender: vec3f, sizeMm: f32 }
@group(1) @binding(0) var<uniform> U: ERender;
@group(1) @binding(1) var<storage, read> pos: array<vec4f>;
@group(1) @binding(2) var<storage, read> vel: array<vec4f>;
@group(1) @binding(3) var<storage, read> col: array<u32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) q: vec2f,
  @location(1) @interpolate(flat) rgb: vec3f,
}

fn blackbody(T: f32) -> vec3f {
  let t = clamp(T, 600.0, 12000.0);
  if (t < 1300.0) { return mix(vec3f(0.55, 0.05, 0.0), vec3f(1.0, 0.2, 0.02), (t - 600.0) / 700.0); }
  if (t < 2400.0) { return mix(vec3f(1.0, 0.2, 0.02), vec3f(1.0, 0.55, 0.15), (t - 1300.0) / 1100.0); }
  if (t < 5000.0) { return mix(vec3f(1.0, 0.55, 0.15), vec3f(1.0, 0.9, 0.7), (t - 2400.0) / 2600.0); }
  return mix(vec3f(1.0, 0.9, 0.7), vec3f(0.8, 0.85, 1.0), clamp((t - 5000.0) / 7000.0, 0.0, 1.0));
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let pm = pos[ii];
  if (pm.w <= 0.0) { o.pos = vec4f(2e9, 2e9, 2.0, 1.0); return o; }
  let center = U.earthRender + pm.xyz;
  let q = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  // tiny world size, but keep a ~1px minimum so distant dust stays visible
  let dist = length(center - F.camPos);
  let rad = max(U.sizeMm, 0.9 * dist / max(F.pixScale, 1.0));
  let wp = center + (F.camRight * q.x + F.camUp * q.y) * rad;
  o.pos = F.viewProj * vec4f(wp, 1.0);
  o.q = q;
  let T = vel[ii].w;
  let alb = unpack4x8unorm(col[ii]).rgb;
  let glow = smoothstep(500.0, 1100.0, T);
  let life = clamp(pm.w / 600.0, 0.0, 1.0);           // fade out over the last ~600 s of flight
  o.rgb = (alb * 0.5 + blackbody(T) * (0.6 + glow * 2.2)) * (0.25 + 0.75 * life);
  return o;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let r = dot(in.q, in.q);
  if (r > 1.0) { discard; }
  let a = (1.0 - r);
  return vec4f(in.rgb * a, a);                         // additive (premultiplied)
}
`;

let _seed = 99173;
function rnd() { _seed = (_seed * 1664525 + 1013904223) >>> 0; return _seed / 4294967296; }

export class EjectaSystem {
  async init(gpu) {
    this.gpu = gpu;
    const d = gpu.device, B = GPUBufferUsage;
    this.pos = d.createBuffer({ size: EMAX * 16, usage: B.STORAGE | B.COPY_DST, label: 'ejPos' });
    this.vel = d.createBuffer({ size: EMAX * 16, usage: B.STORAGE | B.COPY_DST, label: 'ejVel' });
    this.col = d.createBuffer({ size: EMAX * 4, usage: B.STORAGE | B.COPY_DST, label: 'ejCol' });
    this.params = d.createBuffer({ size: 32, usage: B.UNIFORM | B.COPY_DST, label: 'ejParams' });
    this.renderU = d.createBuffer({ size: 16, usage: B.UNIFORM | B.COPY_DST, label: 'ejRenderU' });
    // zero the life field so nothing renders until spawned
    d.queue.writeBuffer(this.pos, 0, new Float32Array(EMAX * 4));
    const mod = await gpu.makeShader(COMPUTE_WGSL, 'ejecta-compute');
    this.pipe = d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    this.bg = d.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.pos } },
        { binding: 2, resource: { buffer: this.vel } },
      ],
    });
    this.head = 0;          // ring-buffer write head
    this.active = 0;        // high-water mark of slots ever used (dispatch bound)
    this.earthR = 6.371;
    this.earthGM = GM_EARTH;
    return this;
  }

  // Spawn an ejecta curtain. opts:
  //   point: impact point in EARTH-CENTRED Mm (on/near the surface)
  //   normal: outward surface normal (unit)
  //   vImpact: impact speed (Mm/s), energyScale: 0..1-ish drama knob, color: [r,g,b], count
  spawn({ point, normal, vImpact, count, color, hot = 2600 }) {
    const n = Math.min(count | 0, EMAX);
    if (n <= 0) return;
    // build a tangent basis around the surface normal
    let t1 = [normal[1], -normal[2], normal[0]];
    const d0 = t1[0] * normal[0] + t1[1] * normal[1] + t1[2] * normal[2];
    t1 = [t1[0] - normal[0] * d0, t1[1] - normal[1] * d0, t1[2] - normal[2] * d0];
    const l1 = Math.hypot(...t1) || 1; t1 = [t1[0] / l1, t1[1] / l1, t1[2] / l1];
    const t2 = [
      normal[1] * t1[2] - normal[2] * t1[1],
      normal[2] * t1[0] - normal[0] * t1[2],
      normal[0] * t1[1] - normal[1] * t1[0],
    ];
    const vEsc = Math.sqrt(2 * this.earthGM / this.earthR);   // ~0.0112 Mm/s
    const vMax = Math.min(Math.max(vImpact * 0.35, vEsc * 1.3), 0.06);
    const vMin = vEsc * 0.18;
    const c = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
    const packed = (c(color[0]) | (c(color[1]) << 8) | (c(color[2]) << 16) | (255 << 24)) >>> 0;

    const pm = new Float32Array(n * 4), vt = new Float32Array(n * 4), cl = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      // launch site jittered slightly across the crater floor
      const jr = 0.18 * this.earthR * Math.sqrt(rnd());
      const ja = rnd() * Math.PI * 2;
      const off = [
        t1[0] * Math.cos(ja) * jr + t2[0] * Math.sin(ja) * jr,
        t1[1] * Math.cos(ja) * jr + t2[1] * Math.sin(ja) * jr,
        t1[2] * Math.cos(ja) * jr + t2[2] * Math.sin(ja) * jr,
      ];
      const px = point[0] + off[0], py = point[1] + off[1], pz = point[2] + off[2];
      // ejecta curtain: ~45° cone around the normal, power-law speed (most slow, few fast)
      const elev = (38 + rnd() * 22) * Math.PI / 180;     // 38–60° from surface
      const az = rnd() * Math.PI * 2;
      const horiz = Math.cos(elev), up = Math.sin(elev);
      const dir = [
        normal[0] * up + (t1[0] * Math.cos(az) + t2[0] * Math.sin(az)) * horiz,
        normal[1] * up + (t1[1] * Math.cos(az) + t2[1] * Math.sin(az)) * horiz,
        normal[2] * up + (t1[2] * Math.cos(az) + t2[2] * Math.sin(az)) * horiz,
      ];
      const speed = Math.min(vMax, vMin * Math.pow(Math.max(rnd(), 1e-3), -0.5));
      const life = 800 + rnd() * 2500;   // sim-seconds — long enough for the ballistic arc
      const slot = (this.head + i) % EMAX;
      pm[i * 4] = px; pm[i * 4 + 1] = py; pm[i * 4 + 2] = pz; pm[i * 4 + 3] = life;
      vt[i * 4] = dir[0] * speed; vt[i * 4 + 1] = dir[1] * speed; vt[i * 4 + 2] = dir[2] * speed;
      vt[i * 4 + 3] = hot * (0.7 + 0.6 * rnd());
      cl[i] = packed;
    }
    // wrap-aware upload (ring buffer)
    const q = this.gpu.device.queue;
    const first = Math.min(n, EMAX - this.head);
    q.writeBuffer(this.pos, this.head * 16, pm, 0, first * 4);
    q.writeBuffer(this.vel, this.head * 16, vt, 0, first * 4);
    q.writeBuffer(this.col, this.head * 4, cl, 0, first);
    if (first < n) {
      q.writeBuffer(this.pos, 0, pm, first * 4, (n - first) * 4);
      q.writeBuffer(this.vel, 0, vt, first * 4, (n - first) * 4);
      q.writeBuffer(this.col, 0, cl, first, n - first);
    }
    this.head = (this.head + n) % EMAX;
    this.active = Math.min(EMAX, Math.max(this.active, this.head, first < n ? EMAX : this.head));
  }

  step(encoder, dt, coolMul) {
    if (this.active === 0) return;
    const a = new ArrayBuffer(32), f = new Float32Array(a), u = new Uint32Array(a);
    f[0] = dt; u[1] = this.active; f[2] = this.earthGM; f[3] = this.earthR; f[4] = coolMul ?? 1;
    this.gpu.device.queue.writeBuffer(this.params, 0, a);
    const cp = encoder.beginComputePass();
    cp.setPipeline(this.pipe); cp.setBindGroup(0, this.bg);
    cp.dispatchWorkgroups(Math.ceil(this.active / 128));
    cp.end();
  }

  setRenderU(earthRender, sizeMm) {
    this.gpu.device.queue.writeBuffer(this.renderU, 0, new Float32Array([earthRender[0], earthRender[1], earthRender[2], sizeMm]));
  }

  reset() {
    this.head = 0; this.active = 0;
    this.gpu.device.queue.writeBuffer(this.pos, 0, new Float32Array(EMAX * 4));
  }
}
