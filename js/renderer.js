// renderer.js — render graph: scene (MSAA HDR) → bloom chain → ACES composite
import { STARS_WGSL, SPHERE_WGSL, CORONA_WGSL, RING_WGSL, GLOBE_WGSL, PARTICLES_WGSL, LINES_WGSL, BELT_WGSL } from './shaders_render.js';
import { EJECTA_RENDER_WGSL } from './ejecta.js';
import { POST_WGSL } from './shaders_post.js';
import { mmul, mlookAt, mperspectiveRevZ, vsub, vadd, vnorm, vcross, mtransformPoint } from './mathx.js';
import { CATALOG } from './bodies.js';
import { AU } from './orbits.js';

const HDR = 'rgba16float';
const BLOOM_LEVELS = 6;

function sphereMesh(stacks = 48, slices = 96) {
  const verts = [], idx = [];
  for (let i = 0; i <= stacks; i++) {
    const th = (i / stacks) * Math.PI;          // 0 at +Z pole
    const z = Math.cos(th), r = Math.sin(th);
    for (let j = 0; j <= slices; j++) {
      const ph = (j / slices) * 2 * Math.PI - Math.PI;
      verts.push(Math.cos(ph) * r, Math.sin(ph) * r, z, j / slices, i / stacks);
    }
  }
  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i * (slices + 1) + j, b = a + slices + 1;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { v: new Float32Array(verts), i: new Uint32Array(idx) };
}

function ringMesh(inner, outer, segs = 160) {
  const verts = [], idx = [];
  for (let j = 0; j <= segs; j++) {
    const ph = (j / segs) * 2 * Math.PI;
    const c = Math.cos(ph), s = Math.sin(ph);
    verts.push(c * inner, s * inner, 0, 0, 0, c * outer, s * outer, 0, 1, 0);
  }
  for (let j = 0; j < segs; j++) {
    const a = j * 2;
    idx.push(a, a + 1, a + 2, a + 2, a + 1, a + 3);
  }
  return { v: new Float32Array(verts), i: new Uint32Array(idx) };
}

export class Renderer {
  async init(gpu, particleSystem, opts = {}) {
    this.gpu = gpu;
    this.ps = particleSystem;
    const d = gpu.device;

    // textures
    this.tex = {};
    const want = ['2k_sun.jpg', '2k_mercury.jpg', '2k_venus_atmosphere.jpg',
      '2k_earth_daymap.jpg', '2k_earth_nightmap.jpg', '2k_earth_clouds.jpg', '2k_moon.jpg', '2k_mars.jpg',
      '2k_jupiter.jpg', '2k_saturn.jpg', '2k_saturn_ring_alpha.png', '2k_uranus.jpg', '2k_neptune.jpg',
      '2k_pluto.jpg'];
    await Promise.all(want.map(async (f) => {
      this.tex[f] = await gpu.loadTexture('assets/tex/' + f, { fallback: { color: '#556', noise: true } });
    }));
    // Skybox: adaptive 8K/4K by GPU class, loaded WITHOUT mips (the shader samples level 0 only,
    // and a black-hole lens will re-sample it by bent ray direction — full-res is what it warps).
    this.skyTex = (opts.sky8k ? '8k' : '4k') + '_stars_milky_way.jpg';
    this.tex[this.skyTex] = await gpu.loadTexture('assets/tex/' + this.skyTex, { mips: false, fallback: { color: '#223', noise: true } });

    // meshes
    const sm = sphereMesh();
    this.sphereVB = gpu.buf(sm.v, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    this.sphereIB = gpu.buf(sm.i, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);
    this.sphereCount = sm.i.length;
    const rm = ringMesh(74.5, 137.0);
    this.ringVB = gpu.buf(rm.v, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    this.ringIB = gpu.buf(rm.i, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);
    this.ringCount = rm.i.length;

    // uniforms
    this.frameBuf = gpu.uniform(176, 'frame');
    this.frameArr = new Float32Array(44);
    this.drawStride = 256;
    this.drawBuf = d.createBuffer({ size: this.drawStride * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'draws' });
    this.drawArr = new Float32Array(this.drawStride * 16 / 4);
    this.globeBuf = gpu.uniform(96, 'globe');
    this.lineUBufs = { orbits: gpu.uniform(16, 'lu0'), trails: gpu.uniform(16, 'lu1'), aim: gpu.uniform(16, 'lu2'), ghost: gpu.uniform(16, 'lu3') };
    this.beltBuf = gpu.uniform(48, 'belt');

    // line vertex buffers
    this.orbitVB = d.createBuffer({ size: 600000, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: 'orbits' });
    this.orbitRanges = [];
    this.trailVB = d.createBuffer({ size: 16 * 560 * 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: 'trails' });
    this.trailRanges = [];
    this.ghostVB = d.createBuffer({ size: 16 * 256 * 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: 'ghosts' });
    this.ghostRanges = [];
    this.aimVB = d.createBuffer({ size: 16 * 1024, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: 'aim' });
    this.aimCount = 0;

    // asteroid belt
    const NB = 2600;
    const belt = new Float32Array(NB * 8);
    let seed = 777;
    const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    for (let i = 0; i < NB; i++) {
      belt[i * 8] = (2.1 + rnd() * 1.25) * AU;
      belt[i * 8 + 1] = rnd() * 0.16;
      belt[i * 8 + 2] = (rnd() - 0.5) * 0.30;
      belt[i * 8 + 3] = rnd() * Math.PI * 2;
      belt[i * 8 + 4] = rnd() * Math.PI * 2;
      belt[i * 8 + 5] = rnd() * Math.PI * 2;
      belt[i * 8 + 6] = 0.7 + rnd() * 1.3;
      belt[i * 8 + 7] = 0;
    }
    this.beltCount = NB;
    this.beltElems = gpu.buf(belt, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    await this._makePipelines();
    this.resize();
    return this;
  }

  async _makePipelines() {
    const gpu = this.gpu, d = gpu.device;
    const dsOpaque = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' };
    const dsBlend = { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater' };
    const dsAlways = { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' };
    const ms = { count: 4 };
    const premul = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    const additive = { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } };
    const sphereVerts = [{ arrayStride: 20, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x2' }] }];
    const lineVerts = [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'unorm8x4' }] }];

    const mk = async (label, code, vsEntry, fsEntry, opts) => {
      const mod = await gpu.makeShader(code, label);
      return d.createRenderPipeline({
        label, layout: 'auto',
        vertex: { module: mod, entryPoint: vsEntry, buffers: opts.verts || [] },
        fragment: { module: mod, entryPoint: fsEntry, targets: [{ format: opts.fmt || HDR, blend: opts.blend }] },
        primitive: { topology: opts.topo || 'triangle-list', cullMode: opts.cull || 'none' },
        depthStencil: opts.ds,
        multisample: opts.noMs ? undefined : ms,
      });
    };

    this.pStars = await mk('stars', STARS_WGSL, 'vs', 'fs', { ds: dsAlways });
    this.pSphere = await mk('sphere', SPHERE_WGSL, 'vs', 'fs', { ds: dsOpaque, verts: sphereVerts, cull: 'back' });
    this.pCorona = await mk('corona', CORONA_WGSL, 'vs', 'fs', { ds: dsBlend, blend: additive, topo: 'triangle-strip' });
    this.pRing = await mk('ring', RING_WGSL, 'vs', 'fs', { ds: dsBlend, blend: premul, verts: sphereVerts });
    this.pGlobe = await mk('globe', GLOBE_WGSL, 'vs', 'fsSurface', { ds: dsOpaque, verts: sphereVerts, cull: 'back' });
    this.pClouds = await mk('clouds', GLOBE_WGSL, 'vs', 'fsClouds', { ds: dsBlend, blend: premul, verts: sphereVerts, cull: 'back' });
    this.pAtmo = await mk('atmo', GLOBE_WGSL, 'vs', 'fsAtmo', { ds: dsBlend, blend: additive, verts: sphereVerts, cull: 'front' });
    this.pParts = await mk('particles', PARTICLES_WGSL, 'vs', 'fs', { ds: dsOpaque, topo: 'triangle-strip' });
    this.pEjecta = await mk('ejecta', EJECTA_RENDER_WGSL, 'vs', 'fs', { ds: dsBlend, blend: additive, topo: 'triangle-strip' });
    this.pLines = await mk('lines', LINES_WGSL, 'vs', 'fs', { ds: dsBlend, blend: premul, topo: 'line-strip', verts: lineVerts });
    this.pBelt = await mk('belt', BELT_WGSL, 'vs', 'fs', { ds: dsBlend, blend: additive });

    const postMod = await gpu.makeShader(POST_WGSL, 'post');
    const mkPost = (entry, fmt, blend) => d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: postMod, entryPoint: 'vs' },
      fragment: { module: postMod, entryPoint: entry, targets: [{ format: fmt, blend }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pThresh = mkPost('fsThreshold', HDR);
    this.pDown = mkPost('fsDown', HDR);
    this.pUp = mkPost('fsUp', HDR, { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } });
    this.pComposite = mkPost('fsComposite', gpu.format);
  }

  resize() {
    const gpu = this.gpu, d = gpu.device;
    gpu.configure();
    const w = gpu.width, h = gpu.height;
    this.msaaTex?.destroy?.(); this.depthTex?.destroy?.(); this.hdrTex?.destroy?.();
    this.msaaTex = d.createTexture({ size: [w, h], format: HDR, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthTex = d.createTexture({ size: [w, h], format: 'depth32float', sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.hdrTex = d.createTexture({ size: [w, h], format: HDR, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.bloom = [];
    let bw = Math.max(2, w >> 1), bh = Math.max(2, h >> 1);
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      this.bloom.push({
        tex: d.createTexture({ size: [bw, bh], format: HDR, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
        w: bw, h: bh,
        ubuf: this.gpu.uniform(32, 'post' + i),
        ubuf2: this.gpu.uniform(32, 'postup' + i),
      });
      bw = Math.max(2, bw >> 1); bh = Math.max(2, bh >> 1);
    }
    this.compBuf = this.compBuf || this.gpu.uniform(32, 'comp');
    this._bindCache = {};
  }

  _bg(key, pipe, entries) {
    if (!this._bindCache[key]) {
      this._bindCache[key] = this.gpu.device.createBindGroup({ layout: pipe.getBindGroupLayout(entries.group ?? 0), entries: entries.list });
    }
    return this._bindCache[key];
  }

  writeOrbits(segments) {
    // segments: [{pts: Float64Array(helio Mm xyz...), color: [r,g,b,a]}]
    let total = 0;
    for (const s of segments) total += s.pts.length / 3;
    const arr = new ArrayBuffer(total * 16);
    const f = new Float32Array(arr), u8 = new Uint8Array(arr);
    this.orbitRanges = [];
    let off = 0;
    for (const s of segments) {
      const n = s.pts.length / 3;
      const c = s.color.map((x) => Math.round(x * 255));
      for (let i = 0; i < n; i++) {
        f[(off + i) * 4] = s.pts[i * 3]; f[(off + i) * 4 + 1] = s.pts[i * 3 + 1]; f[(off + i) * 4 + 2] = s.pts[i * 3 + 2];
        const b = (off + i) * 16 + 12;
        u8[b] = c[0]; u8[b + 1] = c[1]; u8[b + 2] = c[2]; u8[b + 3] = c[3];
      }
      this.orbitRanges.push({ start: off, count: n });
      off += n;
    }
    this.gpu.device.queue.writeBuffer(this.orbitVB, 0, arr);
  }

  writeTrail(slotIdx, f32rgba, count) {
    this.gpu.device.queue.writeBuffer(this.trailVB, slotIdx * 560 * 16, f32rgba, 0, count * 4);
    this.trailRanges[slotIdx] = { start: slotIdx * 560, count };
  }
  clearTrails() { this.trailRanges = []; }

  writeGhost(slotIdx, f32rgba, count) {
    this.gpu.device.queue.writeBuffer(this.ghostVB, slotIdx * 256 * 16, f32rgba, 0, count * 4);
    this.ghostRanges[slotIdx] = { start: slotIdx * 256, count };
  }
  clearGhosts() { this.ghostRanges = []; }

  writeAim(arrBytes, count) {
    // size is in Float32Array ELEMENTS (4 per vertex), not bytes
    if (count > 0) this.gpu.device.queue.writeBuffer(this.aimVB, 0, arrBytes, 0, count * 4);
    this.aimCount = count;
  }

  projectToScreen(posRel) {
    if (!this._lastVP) return null;
    const c = mtransformPoint(this._lastVP, posRel);
    if (c[3] <= 0) return null;
    return [(c[0] * 0.5 + 0.5) * this.gpu.canvas.clientWidth, (0.5 - c[1] * 0.5) * this.gpu.canvas.clientHeight];
  }

  render(scene) {
    const gpu = this.gpu, d = gpu.device, q = d.queue;
    const cam = scene.camera;
    const aspect = gpu.width / gpu.height;
    const eyeRel = cam.eyeRel();
    const near = Math.max(0.002, cam.dist * 1e-3);
    const view = mlookAt(eyeRel, [0, 0, 0], [0, 0, 1]);
    const proj = mperspectiveRevZ(cam.fov, aspect, near);
    const vp = mmul(proj, view);
    this._lastVP = vp;
    const fwd = vnorm(vsub([0, 0, 0], eyeRel));
    let upRef = [0, 0, 1];
    if (Math.abs(fwd[2]) > 0.999) upRef = [1, 0, 0];
    const right = vnorm(vcross(fwd, upRef));
    const up = vcross(right, fwd);
    const tanHalf = Math.tan(cam.fov / 2);

    const F = this.frameArr;
    for (let i = 0; i < 16; i++) F[i] = vp[i];
    F.set(eyeRel, 16); F[19] = tanHalf;
    F.set(right, 20); F[23] = aspect;
    F.set(up, 24); F[27] = scene.timeSec || 0;
    F.set(fwd, 28); F[31] = scene.exposure;
    F.set(scene.sunPosRel, 32); F[35] = gpu.height / (2 * tanHalf);
    F.set(scene.partOffset, 36); F[39] = scene.starBoost;
    F[40] = scene.hideMask || 0;   // body slots whose particles hide under an opaque shell/globe
    q.writeBuffer(this.frameBuf, 0, F);

    // per-draw uniforms (sun + planets + corona + ring)
    const DA = this.drawArr;
    DA.fill(0);
    let di = 0;
    const drawIdx = [];
    for (const p of scene.planets) {
      const base = di * 64;
      DA.set(p.model, base);
      DA[base + 16] = p.emis || 0;
      DA[base + 17] = p.ringInner || 0;
      DA[base + 18] = p.ringOuter || 0;
      DA[base + 19] = p.fade || 0;     // 0 → shader treats as opaque (legacy entries)
      drawIdx.push({ p, di });
      di++;
      if (di >= 15) break;
    }
    // corona slot
    const coronaDi = di;
    if (scene.sunRel) {
      const base = di * 64;
      DA[base + 12] = scene.sunRel[0]; DA[base + 13] = scene.sunRel[1]; DA[base + 14] = scene.sunRel[2]; DA[base + 15] = 1;
      DA[base + 16] = 1.0;                       // corona intensity
      DA[base + 17] = scene.coronaScale || 4000;  // quad radius Mm
      di++;
    }
    q.writeBuffer(this.drawBuf, 0, DA);

    if (scene.globe) {
      const g = scene.globe;
      const GA = new Float32Array(24);
      GA.set(g.model, 0);
      GA[16] = g.dissolve; GA[17] = g.cloudRot; GA[18] = g.cityDim; GA[19] = g.cloudOp;
      GA[20] = g.atmoDensity; GA[21] = g.R;
      q.writeBuffer(this.globeBuf, 0, GA);
    }

    if (scene.belt) {
      const BA = new Float32Array(12);
      BA[0] = scene.belt.d2000; BA[1] = scene.belt.alpha;
      new Uint32Array(BA.buffer)[2] = this.beltCount;
      BA.set(scene.belt.focusOff, 4);
      q.writeBuffer(this.beltBuf, 0, BA);
    }

    for (const [k, lu] of Object.entries(this.lineUBufs)) {
      const off = scene.lineOffsets[k] || [0, 0, 0];
      const la = new Float32Array([off[0], off[1], off[2], scene.lineAlphas?.[k] ?? 1]);
      q.writeBuffer(lu, 0, la);
    }

    const enc = d.createCommandEncoder();

    // particle sim runs first in same encoder (caller already encoded compute)
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.msaaTex.createView(), resolveTarget: this.hdrTex.createView(),
        loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTex.createView(), depthLoadOp: 'clear', depthClearValue: 0, depthStoreOp: 'discard',
      },
    });

    // stars
    pass.setPipeline(this.pStars);
    pass.setBindGroup(0, this._bg('f-stars', this.pStars, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
    pass.setBindGroup(1, this._bg('stars1', this.pStars, { group: 1, list: [{ binding: 0, resource: this.tex[this.skyTex].createView() }, { binding: 1, resource: gpu.sampler }] }));
    pass.draw(3);

    // spheres (sun + planets)
    pass.setPipeline(this.pSphere);
    pass.setBindGroup(0, this._bg('f-sph', this.pSphere, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
    pass.setVertexBuffer(0, this.sphereVB);
    pass.setIndexBuffer(this.sphereIB, 'uint32');
    for (const { p, di: idx } of drawIdx) {
      if (p.ringInner) continue;   // ring pseudo-entry — drawn in the blended phase
      pass.setBindGroup(1, this._bg('sph-' + p.texKey + '-' + idx, this.pSphere, {
        group: 1, list: [
          { binding: 0, resource: { buffer: this.drawBuf, offset: idx * this.drawStride, size: 80 } },
          { binding: 1, resource: this.tex[p.texKey].createView() },
          { binding: 2, resource: gpu.sampler },
        ],
      }));
      pass.drawIndexed(this.sphereCount);
    }

    // globe (pristine Earth)
    if (scene.globe && scene.globe.dissolve < 0.999) {
      pass.setPipeline(this.pGlobe);
      pass.setBindGroup(0, this._bg('f-globe', this.pGlobe, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
      pass.setBindGroup(1, this._bg('globe1', this.pGlobe, {
        group: 1, list: [
          { binding: 0, resource: { buffer: this.globeBuf } },
          { binding: 1, resource: this.tex['2k_earth_daymap.jpg'].createView() },
          { binding: 2, resource: this.tex['2k_earth_nightmap.jpg'].createView() },
          { binding: 4, resource: gpu.sampler },
        ],
      }));
      pass.setVertexBuffer(0, this.sphereVB);
      pass.setIndexBuffer(this.sphereIB, 'uint32');
      pass.drawIndexed(this.sphereCount);
    }

    // particles
    if (scene.particles && this.ps.activeN > 0) {
      const cur = this.ps.current();
      pass.setPipeline(this.pParts);
      pass.setBindGroup(0, this._bg('f-parts', this.pParts, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
      const key = 'parts-' + this.ps.ping;
      pass.setBindGroup(1, this._bg(key, this.pParts, {
        group: 1, list: [
          { binding: 0, resource: { buffer: cur.pos } },
          { binding: 1, resource: { buffer: cur.vel } },
          { binding: 2, resource: { buffer: this.ps.metaBuf } },
          { binding: 3, resource: { buffer: this.ps.albedoBuf } },
          { binding: 4, resource: { buffer: this.ps.matsBuf } },
        ],
      }));
      pass.draw(4, this.ps.activeN);
    }

    // impact ejecta (tiny additive motes) — drawn after the opaque particles, before the globe blend
    if (this.ejecta && this.ejecta.active > 0 && scene.showEjecta) {
      pass.setPipeline(this.pEjecta);
      pass.setBindGroup(0, this._bg('f-ejecta', this.pEjecta, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
      pass.setBindGroup(1, this._bg('ejecta1', this.pEjecta, {
        group: 1, list: [
          { binding: 0, resource: { buffer: this.ejecta.renderU } },
          { binding: 1, resource: { buffer: this.ejecta.pos } },
          { binding: 2, resource: { buffer: this.ejecta.vel } },
          { binding: 3, resource: { buffer: this.ejecta.col } },
        ],
      }));
      pass.draw(4, this.ejecta.active);
    }

    // ---- blended ----
    const saturn = drawIdx.find((x) => x.p.ringInner);
    if (saturn) {
      pass.setPipeline(this.pRing);
      pass.setBindGroup(0, this._bg('f-ring', this.pRing, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
      pass.setBindGroup(1, this._bg('ring1-' + saturn.di, this.pRing, {
        group: 1, list: [
          { binding: 0, resource: { buffer: this.drawBuf, offset: saturn.di * this.drawStride, size: 80 } },
          { binding: 1, resource: this.tex['2k_saturn_ring_alpha.png'].createView() },
          { binding: 2, resource: gpu.sampler },
        ],
      }));
      pass.setVertexBuffer(0, this.ringVB);
      pass.setIndexBuffer(this.ringIB, 'uint32');
      pass.drawIndexed(this.ringCount);
    }

    if (scene.globe && scene.globe.dissolve < 0.999) {
      pass.setVertexBuffer(0, this.sphereVB);
      pass.setIndexBuffer(this.sphereIB, 'uint32');
      if (scene.globe.cloudOp > 0.01) {
        pass.setPipeline(this.pClouds);
        pass.setBindGroup(0, this._bg('f-clouds', this.pClouds, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
        pass.setBindGroup(1, this._bg('clouds1', this.pClouds, {
          group: 1, list: [
            { binding: 0, resource: { buffer: this.globeBuf } },
            { binding: 3, resource: this.tex['2k_earth_clouds.jpg'].createView() },
            { binding: 4, resource: gpu.sampler },
          ],
        }));
        pass.drawIndexed(this.sphereCount);
      }
      if (scene.globe.atmoDensity > 0.01) {
        pass.setPipeline(this.pAtmo);
        pass.setBindGroup(0, this._bg('f-atmo', this.pAtmo, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
        pass.setBindGroup(1, this._bg('atmo1', this.pAtmo, {
          group: 1, list: [
            { binding: 0, resource: { buffer: this.globeBuf } },
          ],
        }));
        pass.drawIndexed(this.sphereCount);
      }
    }

    // corona
    if (scene.sunRel) {
      pass.setPipeline(this.pCorona);
      pass.setBindGroup(0, this._bg('f-cor', this.pCorona, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
      pass.setBindGroup(1, this._bg('cor1-' + coronaDi, this.pCorona, {
        group: 1, list: [{ binding: 0, resource: { buffer: this.drawBuf, offset: coronaDi * this.drawStride, size: 80 } }],
      }));
      pass.draw(4);
    }

    // belt
    if (scene.belt && scene.belt.alpha > 0.01) {
      pass.setPipeline(this.pBelt);
      pass.setBindGroup(0, this._bg('f-belt', this.pBelt, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
      pass.setBindGroup(1, this._bg('belt1', this.pBelt, {
        group: 1, list: [
          { binding: 0, resource: { buffer: this.beltBuf } },
          { binding: 1, resource: { buffer: this.beltElems } },
        ],
      }));
      pass.draw(this.beltCount * 6);
    }

    // lines: orbits, trails, aim
    pass.setPipeline(this.pLines);
    pass.setBindGroup(0, this._bg('f-lines', this.pLines, { list: [{ binding: 0, resource: { buffer: this.frameBuf } }] }));
    if (scene.showOrbits && this.orbitRanges.length) {
      pass.setBindGroup(1, this._bg('lu-orbits', this.pLines, { group: 1, list: [{ binding: 0, resource: { buffer: this.lineUBufs.orbits } }] }));
      pass.setVertexBuffer(0, this.orbitVB);
      for (const r of this.orbitRanges) pass.draw(r.count, 1, r.start);
    }
    if (scene.showTrails) {
      pass.setBindGroup(1, this._bg('lu-trails', this.pLines, { group: 1, list: [{ binding: 0, resource: { buffer: this.lineUBufs.trails } }] }));
      pass.setVertexBuffer(0, this.trailVB);
      for (const r of this.trailRanges) { if (r && r.count > 1) pass.draw(r.count, 1, r.start); }
    }
    if (scene.showGhosts && this.ghostRanges.length) {
      pass.setBindGroup(1, this._bg('lu-ghost', this.pLines, { group: 1, list: [{ binding: 0, resource: { buffer: this.lineUBufs.ghost } }] }));
      pass.setVertexBuffer(0, this.ghostVB);
      for (const r of this.ghostRanges) { if (r && r.count > 1) pass.draw(r.count, 1, r.start); }
    }
    if (this.aimCount > 1) {
      pass.setBindGroup(1, this._bg('lu-aim', this.pLines, { group: 1, list: [{ binding: 0, resource: { buffer: this.lineUBufs.aim } }] }));
      pass.setVertexBuffer(0, this.aimVB);
      pass.draw(this.aimCount);
    }

    pass.end();

    // ---- bloom ----
    const writeU = (buf, w, h, extra = {}) => {
      const a = new Float32Array([1 / w, 1 / h, 0.5, scene.bloomThresh ?? 1.0, scene.bloomStrength ?? 0.65, scene.exposure, 0.20, 0]);
      q.writeBuffer(buf, 0, a);
    };
    const fsq = (pipe, target, srcView, ubuf, loadOp = 'clear', cacheKey) => {
      const bg = this._bg(cacheKey, pipe, {
        list: [
          { binding: 0, resource: { buffer: ubuf } },
          { binding: 1, resource: srcView },
          { binding: 2, resource: gpu.samplerClamp },
        ],
      });
      const rp = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp, clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }] });
      rp.setPipeline(pipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
    };

    const hdrView = this.hdrTex.createView();
    writeU(this.bloom[0].ubuf, gpu.width, gpu.height);
    fsq(this.pThresh, this.bloom[0].tex.createView(), hdrView, this.bloom[0].ubuf, 'clear', 'pp-thresh');
    for (let i = 1; i < BLOOM_LEVELS; i++) {
      writeU(this.bloom[i].ubuf, this.bloom[i - 1].w, this.bloom[i - 1].h);
      fsq(this.pDown, this.bloom[i].tex.createView(), this.bloom[i - 1].tex.createView(), this.bloom[i].ubuf, 'clear', 'pp-down' + i);
    }
    for (let i = BLOOM_LEVELS - 2; i >= 0; i--) {
      writeU(this.bloom[i].ubuf2, this.bloom[i + 1].w, this.bloom[i + 1].h);
      fsq(this.pUp, this.bloom[i].tex.createView(), this.bloom[i + 1].tex.createView(), this.bloom[i].ubuf2, 'load', 'pp-up' + i);
    }

    // composite
    const ca = new Float32Array([1 / gpu.width, 1 / gpu.height, 0.5, 1.0, scene.bloomStrength ?? 0.65, scene.exposure, 0.20, 0]);
    q.writeBuffer(this.compBuf, 0, ca);
    const canvasView = gpu.ctx.getCurrentTexture().createView();
    const bgc = this._bg('pp-comp', this.pComposite, {
      list: [
        { binding: 0, resource: { buffer: this.compBuf } },
        { binding: 1, resource: hdrView },
        { binding: 2, resource: gpu.samplerClamp },
        { binding: 3, resource: this.bloom[0].tex.createView() },
      ],
    });
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: canvasView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }] });
    rp.setPipeline(this.pComposite); rp.setBindGroup(0, bgc); rp.draw(3); rp.end();

    return enc;
  }
}
