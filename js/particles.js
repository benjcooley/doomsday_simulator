// particles.js — GPU particle state: buffers, sim dispatch, spawning, async readback stats
import { SIM_WGSL } from './shaders_sim.js';
import { MAT_TYPES } from './bodies.js';
import { G_SIM } from './orbits.js';

export const MAXN = 65536;
// drawn-sphere size vs physics radius: was 1.6-1.85, which visually swallowed ~2-radius
// impact deformation — slimmer spheres let craters and dents actually show
const VISR = { IRON: 1.35, ROCK: 1.45, CRUST: 1.45, WATER: 1.55, ICE: 1.45, GAS: 1.8, LAVA: 1.45, ARMED: 1.0 };

export class ParticleSystem {
  async init(gpu) {
    this.gpu = gpu;
    const d = gpu.device;
    const B = GPUBufferUsage;
    this.pos = [
      d.createBuffer({ size: MAXN * 16, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC, label: 'posA' }),
      d.createBuffer({ size: MAXN * 16, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC, label: 'posB' }),
    ];
    this.vel = [
      d.createBuffer({ size: MAXN * 16, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC, label: 'velA' }),
      d.createBuffer({ size: MAXN * 16, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC, label: 'velB' }),
    ];
    this.metaBuf = d.createBuffer({ size: MAXN * 4, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC, label: 'meta' });
    this.albedoBuf = d.createBuffer({ size: MAXN * 4, usage: B.STORAGE | B.COPY_DST, label: 'albedo' });
    this.matsBuf = d.createBuffer({ size: 16 * 64, usage: B.STORAGE | B.COPY_DST, label: 'mats' });
    this.paramsBuf = d.createBuffer({ size: 80, usage: B.UNIFORM | B.COPY_DST, label: 'simParams' });
    this.shiftBuf = d.createBuffer({ size: 32, usage: B.UNIFORM | B.COPY_DST, label: 'shift' });
    this.bodyDynBuf = d.createBuffer({ size: 16 * 48, usage: B.STORAGE | B.COPY_DST, label: 'bodyDyn' });
    this.bodyDynArr = new Float32Array(16 * 12);
    this.dbgBuf = d.createBuffer({ size: 128, usage: B.STORAGE | B.COPY_SRC | B.COPY_DST, label: 'dbg' });
    this.debugIdx = 0xffffffff;

    const mod = await gpu.makeShader(SIM_WGSL, 'sim');
    this.simPipe = d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'sim' } });
    this.rebasePipe = d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'rebase' } });
    this.thermalPipe = d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'thermal' } });

    this.simBG = [0, 1].map((ping) => d.createBindGroup({
      layout: this.simPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.matsBuf } },
        { binding: 2, resource: { buffer: this.pos[ping] } },
        { binding: 3, resource: { buffer: this.vel[ping] } },
        { binding: 4, resource: { buffer: this.pos[1 - ping] } },
        { binding: 5, resource: { buffer: this.vel[1 - ping] } },
        { binding: 6, resource: { buffer: this.metaBuf } },
        { binding: 7, resource: { buffer: this.bodyDynBuf } },
        { binding: 8, resource: { buffer: this.dbgBuf } },
      ],
    }));
    this.rebaseBG = [0, 1].map((ping) => d.createBindGroup({
      layout: this.rebasePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.shiftBuf } },
        { binding: 1, resource: { buffer: this.pos[ping] } },
        { binding: 2, resource: { buffer: this.vel[ping] } },
        { binding: 3, resource: { buffer: this.metaBuf } },
      ],
    }));
    this._shiftQueue = [];

    this.thermalBG = [0, 1].map((ping) => d.createBindGroup({
      layout: this.thermalPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.matsBuf } },
        { binding: 2, resource: { buffer: this.vel[ping] } },
        { binding: 3, resource: { buffer: this.metaBuf } },
        { binding: 4, resource: { buffer: this.bodyDynBuf } },
      ],
    }));

    this.staging = [0, 1].map(() => ({
      buf: d.createBuffer({ size: MAXN * 36 + 128, usage: B.COPY_DST | B.MAP_READ, label: 'staging' }),
      pending: false,
    }));

    this.paramsArr = new ArrayBuffer(80);
    this.pf = new Float32Array(this.paramsArr);
    this.pu = new Uint32Array(this.paramsArr);
    this.reset();
    return this;
  }

  reset() {
    this.activeN = 0;
    this.ping = 0;
    this.matSlots = [];      // CPU info per material slot: {type, cp, T0, isSurface, bodySlot, rad, k}
    this.bodies = [];        // {slot, name, offset, count, M0, R0, recipe, dtStable}
    this.matData = new Float32Array(16 * 16);
    this.minRp = 1.0;
    this.dtStable = 6.0;
    this._lastSurf = [];
    this.epoch = (this.epoch || 0) + 1;   // invalidates in-flight readbacks
  }

  _allocMatSlot(info, vals) {
    const s = this.matSlots.length;
    if (s >= 16) throw new Error('out of material slots');
    this.matSlots.push(info);
    this.matData.set(vals, s * 16);
    return s;
  }

  // Add a blob built by blob.js at local position/velocity (Mm, Mm/s)
  addBlob(blob, posL, velL, name) {
    const N = blob.count;
    if (this.activeN + N > MAXN) throw new Error('particle budget exceeded');
    const bodySlot = this.bodies.length;
    if (bodySlot >= 16) throw new Error('out of body slots');

    const mAvg = blob.M / N;
    const vEsc = Math.sqrt(2 * G_SIM * blob.M / blob.R);
    const gSurf = G_SIM * blob.M / (blob.R * blob.R);
    const shells = Math.max(2, blob.R / (2 * blob.rp));
    const k1 = mAvg * Math.pow(vEsc / blob.rp, 2);
    const k2 = shells * mAvg * gSurf / (0.08 * blob.rp);
    const k = Math.max(k1, k2);
    const damp = 2 * 0.45 * Math.sqrt(k * mAvg);

    // material slots for this blob
    const slotOf = [];
    for (const mu of blob.matsUsed) {
      const mt = MAT_TYPES[mu.type];
      const coolK = 5.67e-8 * 4 * Math.PI * Math.pow(blob.rp * 1e6, 2) / (mAvg * 1e24 * mt.cp);
      const heatK = 1e12 / mt.cp;
      slotOf.push(this._allocMatSlot(
        { type: mu.type, cp: mt.cp, T0: 300, isSurface: mu.isSurface, bodySlot, rad: blob.rp, k },
        [
          blob.rp, k, k * mt.cohF, damp * mt.dampZ / 0.45,
          mu.isSurface ? 1 : 0, heatK, mt.Tsol, mt.Tliq,
          mt.Tvap, mt.emis, VISR[mu.type] || 1.6, mu.type === 'GAS' ? 1 : 0,
          mt.base[0], mt.base[1], mt.base[2], coolK,
        ]));
    }

    const off = this.activeN;
    const pm = new Float32Array(N * 4);
    const vt = new Float32Array(N * 4);
    const me = new Uint32Array(N);
    for (let i = 0; i < N; i++) {
      pm[i * 4] = blob.pos[i * 3] + posL[0];
      pm[i * 4 + 1] = blob.pos[i * 3 + 1] + posL[1];
      pm[i * 4 + 2] = blob.pos[i * 3 + 2] + posL[2];
      pm[i * 4 + 3] = blob.mass[i];
      vt[i * 4] = blob.vel[i * 3] + velL[0];
      vt[i * 4 + 1] = blob.vel[i * 3 + 1] + velL[1];
      vt[i * 4 + 2] = blob.vel[i * 3 + 2] + velL[2];
      vt[i * 4 + 3] = blob.temp[i];
      me[i] = (slotOf[blob.matLocal[i]] & 15) | (bodySlot << 4);
    }
    const q = this.gpu.device.queue;
    for (const b of this.pos) q.writeBuffer(b, off * 16, pm);
    for (const b of this.vel) q.writeBuffer(b, off * 16, vt);
    q.writeBuffer(this.metaBuf, off * 4, me);
    q.writeBuffer(this.albedoBuf, off * 4, blob.albedo);
    q.writeBuffer(this.matsBuf, 0, this.matData);

    // stability is set by the LIGHTEST particle (ocean water, not the iron core) under the
    // combined stiffness of ~8 simultaneous contacts — scan the whole blob, not a core sample
    let minMass = Infinity;
    for (let i = 0; i < N; i++) if (blob.mass[i] < minMass) minMass = blob.mass[i];
    const dtStable = 0.10 * Math.sqrt(minMass / k);
    this.activeN += N;
    this.minRp = Math.min(this.minRp, blob.rp);
    this.dtStable = Math.min(this.dtStable, dtStable);
    const body = { slot: bodySlot, name, offset: off, count: N, M0: blob.M, R0: blob.R, recipe: blob.recipeName, dtStable };
    this.bodies.push(body);
    return body;
  }

  // Sub-resolution impactor: one particle carrying its real kinetic energy as a payload
  addArmedImpactor(massSim, energySim, posL, velL, visRad, name) {
    if (this.activeN + 1 > MAXN) throw new Error('particle budget exceeded');
    const bodySlot = this.bodies.length;
    const slot = this._allocMatSlot(
      { type: 'ARMED', cp: 1000, T0: 300, isSurface: false, bodySlot, rad: visRad, k: 1e-12 },
      [visRad, 1e-12, 0, 1e-12, 0, 1e9, 1e9, 2e9, 3e9, 0.4, 2.0, 0,
        1, 0.6, 0.3, 1e-12]);
    const off = this.activeN;
    const q = this.gpu.device.queue;
    const pm = new Float32Array([posL[0], posL[1], posL[2], massSim]);
    const vt = new Float32Array([velL[0], velL[1], velL[2], energySim]);
    const me = new Uint32Array([(slot & 15) | (bodySlot << 4) | 512]);
    for (const b of this.pos) q.writeBuffer(b, off * 16, pm);
    for (const b of this.vel) q.writeBuffer(b, off * 16, vt);
    q.writeBuffer(this.metaBuf, off * 4, me);
    q.writeBuffer(this.albedoBuf, off * 4, new Uint32Array([0xff5090ff]));
    q.writeBuffer(this.matsBuf, 0, this.matData);
    this.activeN += 1;
    const body = { slot: bodySlot, name, offset: off, count: 1, M0: massSim, R0: visRad, recipe: 'armed', dtStable: 99 };
    this.bodies.push(body);
    return body;
  }

  current() { return { pos: this.pos[this.ping], vel: this.vel[this.ping] }; }

  // per-body rigid reference motion for settle drag: [{slot, vel, pos, spin}]
  writeBodyDyn(list) {
    const a = this.bodyDynArr;
    a.fill(0);
    for (const b of list) {
      const o = b.slot * 12;
      a[o] = b.vel[0]; a[o + 1] = b.vel[1]; a[o + 2] = b.vel[2];
      a[o + 3] = b.atmT || 0;                                // atmosphere temp (vc.w)
      a[o + 4] = b.pos[0]; a[o + 5] = b.pos[1]; a[o + 6] = b.pos[2];
      const s = b.spin || [0, 0, 0];
      a[o + 8] = s[0]; a[o + 9] = s[1]; a[o + 10] = s[2];
    }
    this.gpu.device.queue.writeBuffer(this.bodyDynBuf, 0, a);
  }

  writeParams(p) {
    const f = this.pf, u = this.pu;
    f[0] = p.dt; u[1] = this.activeN; u[2] = Math.ceil(this.activeN / 256); f[3] = p.settleBoost;
    f[4] = p.sunPos[0]; f[5] = p.sunPos[1]; f[6] = p.sunPos[2]; f[7] = p.gmSun;
    f[8] = G_SIM; f[9] = Math.pow(0.5 * this.minRp, 2); f[10] = p.vClamp ?? 1.0; f[11] = p.coolMul;
    f[12] = p.heatMul; f[13] = p.time; f[14] = p.solarLum; f[15] = p.heatGate ?? 1;
    f[16] = p.settleDrag ?? 0; u[17] = this.debugIdx >>> 0;
    this.gpu.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsArr);
  }

  step(encoder, substeps) {
    if (this.activeN === 0) return;
    const n = Math.ceil(this.activeN / 256);
    for (let s = 0; s < substeps; s++) {
      const cp = encoder.beginComputePass();
      cp.setPipeline(this.simPipe);
      cp.setBindGroup(0, this.simBG[this.ping]);
      cp.dispatchWorkgroups(n);
      cp.end();
      this.ping = 1 - this.ping;
    }
  }

  // slot === undefined → frame rebase (all particles, subtract). slot number → rigid shift (add) for one body.
  // Multiple shifts per frame need distinct uniform buffers (writeBuffer is ordered before the whole submit).
  // thermal-only evolution while mechanics sleep: O(N), any timestep (params.dt pre-written)
  thermalStep(encoder) {
    if (this.activeN === 0) return;
    const cp = encoder.beginComputePass();
    cp.setPipeline(this.thermalPipe);
    cp.setBindGroup(0, this.thermalBG[this.ping]);
    cp.dispatchWorkgroups(Math.ceil(this.activeN / 256));
    cp.end();
  }

  rebase(encoder, dp, dv, slot) {
    if (this.activeN === 0) return;
    const a = new ArrayBuffer(32);
    const f = new Float32Array(a), u = new Uint32Array(a);
    f[0] = dp[0]; f[1] = dp[1]; f[2] = dp[2]; u[3] = this.activeN;
    f[4] = dv[0]; f[5] = dv[1]; f[6] = dv[2]; u[7] = slot === undefined ? 0xffffffff : slot;
    if (!this._shiftPool) this._shiftPool = [];
    if (this._shiftFrame !== this.gpu.frameId) { this._shiftFrame = this.gpu.frameId; this._shiftUsed = 0; }
    if (this._shiftUsed >= this._shiftPool.length) {
      const buf = this.gpu.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bgs = [0, 1].map((ping) => this.gpu.device.createBindGroup({
        layout: this.rebasePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buf } },
          { binding: 1, resource: { buffer: this.pos[ping] } },
          { binding: 2, resource: { buffer: this.vel[ping] } },
          { binding: 3, resource: { buffer: this.metaBuf } },
        ],
      }));
      this._shiftPool.push({ buf, bgs });
    }
    const sp = this._shiftPool[this._shiftUsed++];
    this.gpu.device.queue.writeBuffer(sp.buf, 0, a);
    const cp = encoder.beginComputePass();
    cp.setPipeline(this.rebasePipe);
    cp.setBindGroup(0, sp.bgs[this.ping]);
    cp.dispatchWorkgroups(Math.ceil(this.activeN / 256));
    cp.end();
  }

  // queue an async readback of pos/vel/meta; resolves stats later via pollStats()
  requestReadback(encoder, simTimeTag) {
    if (this.activeN === 0) return;
    const st = this.staging.find((s) => !s.pending);
    if (!st) return;
    const n = this.activeN;
    encoder.copyBufferToBuffer(this.pos[this.ping], 0, st.buf, 0, n * 16);
    encoder.copyBufferToBuffer(this.vel[this.ping], 0, st.buf, MAXN * 16, n * 16);
    encoder.copyBufferToBuffer(this.metaBuf, 0, st.buf, MAXN * 32, n * 4);
    encoder.copyBufferToBuffer(this.dbgBuf, 0, st.buf, MAXN * 36, 128);
    st.pending = true;
    st.epoch = this.epoch;
    st.n = n;
    st.simTimeTag = simTimeTag || 0;
    st.mapPromise = null;
    st.queued = true;
  }

  afterSubmit() {
    for (const st of this.staging) {
      if (st.queued) {
        st.queued = false;
        st.mapPromise = st.buf.mapAsync(GPUMapMode.READ).then(() => { st.mapped = true; }).catch(() => { st.pending = false; });
      }
    }
  }

  pollStats() {
    for (const st of this.staging) {
      if (st.pending && st.mapped) {
        st.mapped = false;
        let stats = null;
        if (st.epoch === this.epoch) {
          const ab = st.buf.getMappedRange();
          stats = this._computeStats(new Float32Array(ab, 0, st.n * 4), new Float32Array(ab, MAXN * 16, st.n * 4), new Uint32Array(ab, MAXN * 32, st.n));
          stats.simTimeTag = st.simTimeTag;
          stats.dbg = Array.from(new Float32Array(ab, MAXN * 36, 32));
          stats.dbgIdx = this.debugIdx;
        }
        st.buf.unmap();
        st.pending = false;
        if (stats) return stats;
      }
    }
    return null;
  }

  _computeStats(pm, vt, me) {
    const bodies = this.bodies.map((b) => ({
      name: b.name, slot: b.slot, M0: b.M0, R0: b.R0,
      mass: 0, com: [0, 0, 0], cov: [0, 0, 0], bound: 0, surfT: 0, surfN: 0,
      maxT: 0, molten: 0, count: 0,
    }));
    const n = me.length;
    for (let i = 0; i < n; i++) {
      const slot = (me[i] >> 4) & 15;
      const b = bodies[slot];
      if (!b) continue;
      const m = pm[i * 4 + 3];
      b.mass += m; b.count++;
      b.com[0] += m * pm[i * 4]; b.com[1] += m * pm[i * 4 + 1]; b.com[2] += m * pm[i * 4 + 2];
      b.cov[0] += m * vt[i * 4]; b.cov[1] += m * vt[i * 4 + 1]; b.cov[2] += m * vt[i * 4 + 2];
      const T = vt[i * 4 + 3];
      if (T > b.maxT && !(me[i] & 512)) b.maxT = T;
      if (me[i] & 256) b.molten++;
    }
    for (const b of bodies) {
      if (b.mass > 0) {
        b.com = b.com.map((v) => v / b.mass);
        b.cov = b.cov.map((v) => v / b.mass);
      }
    }
    // second pass: boundedness + surface temps + thermal energy
    let thermalJ = 0;
    let hotIdx = -1, hotT = 0;
    let crustLiv = 0, crustTot = 0;
    const tHist = [0, 0, 0, 0, 0];   // <800 | 800-1400 | 1400-2600 (ORANGE BAND) | 2600-4000 | >4000
    const matT = this.matSlots.map(() => ({ sum: 0, n: 0 }));
    for (let i = 0; i < n; i++) {
      const mtAcc = matT[me[i] & 15];
      if (mtAcc) { mtAcc.sum += vt[i * 4 + 3]; mtAcc.n++; }
      const slot = (me[i] >> 4) & 15;
      const b = bodies[slot];
      if (!b) continue;
      const ms = this.matSlots[me[i] & 15];
      const m = pm[i * 4 + 3];
      const dx = pm[i * 4] - b.com[0], dy = pm[i * 4 + 1] - b.com[1], dz = pm[i * 4 + 2] - b.com[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const boundR = Math.max(2.6 * b.R0, b.R0 + 12);
      const T = vt[i * 4 + 3];
      const wx = vt[i * 4] - b.cov[0], wy = vt[i * 4 + 1] - b.cov[1], wz = vt[i * 4 + 2] - b.cov[2];
      const w2 = wx * wx + wy * wy + wz * wz;
      b.rms2 = (b.rms2 || 0) + m * w2;
      if (w2 > (b.maxV2 || 0)) b.maxV2 = w2;
      if (w2 > 2.5e-7) b.fastN = (b.fastN || 0) + 1;     // > 0.5 km/s internal
      if (me[i] & 2048) b.nanN = (b.nanN || 0) + 1;      // NaN guard ever fired
      if (slot === 0 && !(me[i] & 512) && T > hotT) { hotT = T; hotIdx = i; }  // hottest earth particle
      if (!(me[i] & 512)) {
        tHist[T < 800 ? 0 : T < 1400 ? 1 : T < 2600 ? 2 : T < 4000 ? 3 : 4]++;
      }
      if (dist < boundR) {
        b.bound += m;
        if (ms && ms.isSurface) { b.surfT += T; b.surfN++; }
        // population basis: land (CRUST) particles still attached to Earth at livable temps.
        // Ejected land leaves boundR (dead); molten land fails the temp test (dead).
        if (slot === 0 && ms && ms.type === 'CRUST') {
          crustTot++;
          if (T < 322) crustLiv++;
        }
      }
      if (ms && !(me[i] & 512)) {
        const dT = T - 320;
        if (dT > 0) thermalJ += m * 1e24 * ms.cp * dT;
      }
    }
    if (!this._lastSurf) this._lastSurf = [];
    for (const b of bodies) {
      if (b.surfN > 8) {
        b.surfT = b.surfT / b.surfN;
        this._lastSurf[b.slot] = b.surfT;
      } else {
        b.surfT = this._lastSurf[b.slot] ?? 288;   // too few surface particles left — hold last good
      }
      b.boundFrac = b.M0 > 0 ? b.bound / b.M0 : 0;
      b.moltenFrac = b.count > 0 ? b.molten / b.count : 0;
      b.rmsV = b.mass > 0 ? Math.sqrt((b.rms2 || 0) / b.mass) * 1000 : 0;   // km/s internal motion
      b.maxV = Math.sqrt(b.maxV2 || 0) * 1000;
      b.fastN = b.fastN || 0;
      b.nanN = b.nanN || 0;
    }
    const matTemps = this.matSlots.map((ms, i) => ({
      type: ms.type, body: ms.bodySlot, surf: ms.isSurface,
      T: matT[i].n ? Math.round(matT[i].sum / matT[i].n) : 0, n: matT[i].n,
    }));
    this.debugIdx = hotIdx >= 0 ? hotIdx : 0xffffffff;   // kernel dumps this particle next frames
    return { bodies, thermalJ, matTemps, hotIdx, hotT, crustLiv, crustTot, tHist, t: performance.now() };
  }
}
