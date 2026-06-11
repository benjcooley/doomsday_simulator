// shaders_post.js — HDR post: threshold → downsample chain → tent upsample (additive) → ACES composite

export const POST_WGSL = /* wgsl */`
struct PostU {
  texel: vec2f,        // 1/srcSize
  knee: f32,           // bloom soft knee
  thresh: f32,
  strength: f32,       // bloom strength (composite)
  exposure: f32,
  vignette: f32,
  pad0: f32,
}
@group(0) @binding(0) var<uniform> U: PostU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  o.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(xy.x, 1.0 - xy.y);
  return o;
}

fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + max(c.r, max(c.g, c.b))); }

@fragment fn fsThreshold(in: VOut) -> @location(0) vec4f {
  // 4-tap karis-weighted box (anti-firefly) + soft knee threshold
  let o = U.texel * 0.5;
  let c0 = textureSample(src, smp, in.uv + vec2f(-o.x, -o.y)).rgb;
  let c1 = textureSample(src, smp, in.uv + vec2f(o.x, -o.y)).rgb;
  let c2 = textureSample(src, smp, in.uv + vec2f(-o.x, o.y)).rgb;
  let c3 = textureSample(src, smp, in.uv + vec2f(o.x, o.y)).rgb;
  let w0 = karis(c0); let w1 = karis(c1); let w2 = karis(c2); let w3 = karis(c3);
  var c = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / max(w0 + w1 + w2 + w3, 1e-4);
  let br = max(c.r, max(c.g, c.b));
  let soft = clamp(br - U.thresh + U.knee, 0.0, 2.0 * U.knee);
  let w = max(soft * soft / (4.0 * U.knee + 1e-4), br - U.thresh) / max(br, 1e-4);
  return vec4f(c * max(w, 0.0), 1.0);
}

@fragment fn fsDown(in: VOut) -> @location(0) vec4f {
  let o = U.texel;
  var c = textureSample(src, smp, in.uv).rgb * 0.25;
  c = c + textureSample(src, smp, in.uv + vec2f(-o.x, -o.y)).rgb * 0.1875;
  c = c + textureSample(src, smp, in.uv + vec2f(o.x, -o.y)).rgb * 0.1875;
  c = c + textureSample(src, smp, in.uv + vec2f(-o.x, o.y)).rgb * 0.1875;
  c = c + textureSample(src, smp, in.uv + vec2f(o.x, o.y)).rgb * 0.1875;
  return vec4f(c, 1.0);
}

@fragment fn fsUp(in: VOut) -> @location(0) vec4f {
  let o = U.texel;
  var c = textureSample(src, smp, in.uv).rgb * 0.25;
  c = c + textureSample(src, smp, in.uv + vec2f(o.x, 0.0)).rgb * 0.125;
  c = c + textureSample(src, smp, in.uv + vec2f(-o.x, 0.0)).rgb * 0.125;
  c = c + textureSample(src, smp, in.uv + vec2f(0.0, o.y)).rgb * 0.125;
  c = c + textureSample(src, smp, in.uv + vec2f(0.0, -o.y)).rgb * 0.125;
  c = c + textureSample(src, smp, in.uv + vec2f(o.x, o.y)).rgb * 0.0625;
  c = c + textureSample(src, smp, in.uv + vec2f(-o.x, o.y)).rgb * 0.0625;
  c = c + textureSample(src, smp, in.uv + vec2f(o.x, -o.y)).rgb * 0.0625;
  c = c + textureSample(src, smp, in.uv + vec2f(-o.x, -o.y)).rgb * 0.0625;
  return vec4f(c, 1.0);
}

fn aces(x: vec3f) -> vec3f {
  return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

@fragment fn fsComposite(in: VOut) -> @location(0) vec4f {
  var c = textureSample(src, smp, in.uv).rgb;
  c = c + textureSample(bloomTex, smp, in.uv).rgb * U.strength;
  c = c * U.exposure;
  c = aces(c);
  let d = in.uv - vec2f(0.5);
  c = c * (1.0 - U.vignette * dot(d, d) * 2.2);
  c = pow(c, vec3f(1.0 / 2.2));
  c = c + (hash12(in.pos.xy) - 0.5) * (1.5 / 255.0);
  return vec4f(c, 1.0);
}
`;
