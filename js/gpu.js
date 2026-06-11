// gpu.js — WebGPU device init + resource helpers

export class GPU {
  async init(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser. Use Chrome/Edge/Safari 26+.');
    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!this.adapter) throw new Error('No WebGPU adapter found.');
    this.device = await this.adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(268435456, this.adapter.limits.maxStorageBufferBindingSize),
      },
    });
    this.device.addEventListener('uncapturederror', (e) => {
      console.error('[WebGPU uncaptured]', e.error?.message || e.error);
    });
    this.device.lost.then((info) => {
      if (info.reason !== 'destroyed') {
        console.error('[WebGPU device lost]', info.message);
        const el = document.getElementById('error-overlay');
        if (el) { el.style.display = 'flex'; el.querySelector('.err-msg').textContent = 'GPU device lost — reload the page.'; }
      }
    });
    this.canvas = canvas;
    this.ctx = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.configure();
    this.sampler = this.device.createSampler({
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      addressModeU: 'repeat', addressModeV: 'clamp-to-edge', maxAnisotropy: 4,
    });
    this.samplerClamp = this.device.createSampler({
      magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    this._mipPipe = null;
    return this;
  }

  configure() {
    const w = Math.max(2, Math.floor(this.canvas.clientWidth * this.dpr));
    const h = Math.max(2, Math.floor(this.canvas.clientHeight * this.dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
    }
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.width = w; this.height = h;
  }

  async makeShader(code, label) {
    const mod = this.device.createShaderModule({ code, label });
    const info = await mod.getCompilationInfo();
    let fatal = false;
    for (const m of info.messages) {
      const tag = `[WGSL ${label}] L${m.lineNum}:${m.linePos} ${m.message}`;
      if (m.type === 'error') { console.error(tag); fatal = true; }
      else console.warn(tag);
    }
    if (fatal) throw new Error(`Shader '${label}' failed to compile (see console).`);
    return mod;
  }

  buf(sizeOrData, usage, label) {
    const isData = sizeOrData instanceof Float32Array || sizeOrData instanceof Uint32Array || sizeOrData instanceof Int32Array || sizeOrData instanceof Uint16Array;
    const size = isData ? Math.ceil(sizeOrData.byteLength / 4) * 4 : Math.ceil(sizeOrData / 4) * 4;
    const b = this.device.createBuffer({ size, usage, label });
    if (isData) this.device.queue.writeBuffer(b, 0, sizeOrData.buffer, sizeOrData.byteOffset, sizeOrData.byteLength);
    return b;
  }

  uniform(byteSize, label) {
    return this.device.createBuffer({ size: Math.ceil(byteSize / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
  }

  tex2D(w, h, format, usage, mips = 1, label = '') {
    return this.device.createTexture({ size: { width: w, height: h }, format, usage, mipLevelCount: mips, label });
  }

  async loadTexture(url, { srgb = true, mips = true, fallback = null } = {}) {
    let bitmap = null;
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
    } catch (e) {
      console.warn(`texture ${url} failed (${e.message}) — using fallback`);
      bitmap = await this._fallbackBitmap(fallback);
    }
    const mipCount = mips ? Math.floor(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1 : 1;
    const fmt = srgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
    const tex = this.tex2D(bitmap.width, bitmap.height, fmt,
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT, mipCount, url);
    this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex }, { width: bitmap.width, height: bitmap.height });
    if (mipCount > 1) await this.genMips(tex, bitmap.width, bitmap.height, mipCount, fmt);
    return tex;
  }

  async _fallbackBitmap(spec) {
    // procedural fallback: flat or two-tone noise canvas
    const c = new OffscreenCanvas(256, 128);
    const g = c.getContext('2d');
    const col = (spec && spec.color) || '#777788';
    g.fillStyle = col; g.fillRect(0, 0, 256, 128);
    if (spec && spec.noise) {
      for (let i = 0; i < 900; i++) {
        g.fillStyle = `rgba(255,255,255,${Math.random() * 0.12})`;
        g.fillRect(Math.random() * 256, Math.random() * 128, 2, 2);
      }
    }
    return await createImageBitmap(c);
  }

  async genMips(tex, w, h, mipCount, fmt) {
    if (!this._mipPipe) this._mipPipe = {};
    if (!this._mipPipe[fmt]) {
      const mod = await this.makeShader(`
        @group(0) @binding(0) var src: texture_2d<f32>;
        @group(0) @binding(1) var smp: sampler;
        struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
          var o: VOut;
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
          o.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
          o.uv = vec2f(xy.x, 1.0 - xy.y);
          return o;
        }
        @fragment fn fs(in: VOut) -> @location(0) vec4f { return textureSample(src, smp, in.uv); }
      `, 'mipgen');
      this._mipPipe[fmt] = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: 'fs', targets: [{ format: fmt }] },
        primitive: { topology: 'triangle-list' },
      });
    }
    const pipe = this._mipPipe[fmt];
    const enc = this.device.createCommandEncoder();
    for (let i = 1; i < mipCount; i++) {
      const bg = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: tex.createView({ baseMipLevel: i - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: this.samplerClamp },
        ],
      });
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: tex.createView({ baseMipLevel: i, mipLevelCount: 1 }), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
      });
      rp.setPipeline(pipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
    }
    this.device.queue.submit([enc.finish()]);
  }

  // Decode an image into ImageData via the GPU (robust where canvas-2D readback is broken):
  // upload → blit to small rgba8 target → copyTextureToBuffer → map.
  async imageDataViaGPU(url, w = 512, h = 256) {
    try {
      const tex = await this.loadTexture(url, { srgb: false, mips: false });
      if (!this._mipPipe) this._mipPipe = {};
      if (!this._mipPipe['rgba8unorm']) await this.genMips(this.tex2D(2, 2, 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT, 2), 2, 2, 2, 'rgba8unorm');
      const pipe = this._mipPipe['rgba8unorm'];
      const target = this.tex2D(w, h, 'rgba8unorm', GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
      const bytesPerRow = w * 4;            // 512/1024-wide → already 256-aligned
      const stage = this.device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const bg = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: tex.createView() },
          { binding: 1, resource: this.samplerClamp },
        ],
      });
      const enc = this.device.createCommandEncoder();
      const rp = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'clear', clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1 }, storeOp: 'store' }] });
      rp.setPipeline(pipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
      enc.copyTextureToBuffer({ texture: target }, { buffer: stage, bytesPerRow }, { width: w, height: h });
      this.device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const data = new Uint8ClampedArray(stage.getMappedRange().slice(0));
      stage.unmap(); stage.destroy(); target.destroy(); tex.destroy();
      return { width: w, height: h, data };
    } catch (e) {
      console.warn(`imageDataViaGPU ${url} failed: ${e.message} — falling back to canvas decode`);
      return this.loadImageData(url, w, h);
    }
  }

  // Decode an image into ImageData for CPU sampling (albedo painting)
  async loadImageData(url, w = 512, h = 256) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const bm = await createImageBitmap(await resp.blob());
      let g;
      try {
        g = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
      } catch (_) {
        g = null;
      }
      if (!g) {
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        g = c.getContext('2d', { willReadFrequently: true });
      }
      g.drawImage(bm, 0, 0, w, h);
      const data = g.getImageData(0, 0, w, h);
      // sanity: detect an all-uniform decode (some headless paths silently no-op)
      const d = data.data;
      let varies = false;
      for (let i = 4; i < 4000 && !varies; i += 16) {
        if (Math.abs(d[i] - d[0]) > 6 || Math.abs(d[i + 1] - d[1]) > 6) varies = true;
      }
      if (!varies) console.warn(`imageData ${url}: decode looks uniform — painting may be flat`);
      return data;
    } catch (e) {
      console.warn(`imageData ${url} failed: ${e.message}`);
      return null;
    }
  }
}

// sample ImageData at lon/lat (equirect), returns [r,g,b] 0..1 sRGB
export function sampleEquirect(img, lon, lat) {
  if (!img) return [0.5, 0.5, 0.5];
  let u = (lon / (2 * Math.PI) + 0.5) % 1; if (u < 0) u += 1;
  const v = Math.min(0.999, Math.max(0, 0.5 - lat / Math.PI));
  const x = Math.min(img.width - 1, Math.floor(u * img.width));
  const y = Math.min(img.height - 1, Math.floor(v * img.height));
  const i = (y * img.width + x) * 4;
  return [img.data[i] / 255, img.data[i + 1] / 255, img.data[i + 2] / 255];
}
