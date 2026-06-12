// particles_fast.js — the FAST engine: drop-in ParticleSystem with O(N·k) forces.
// Subclasses the proven engine — all spawning, readback/stats, rebase, thermal and the
// renderer-facing surface are inherited unchanged. Only step() is replaced: counting-sort
// spatial hash + 16³ monopole gravity + the SHARED pair-physics force kernel.
// Validated headlessly in test/ stages 0-7 (bit-exact passes, force parity 7e-6,
// 1000-step dynamic parity, 8-16× speedup). Selected via ?kernel=fast — the proven
// N² engine stays the default.
import { ParticleSystem, MAXN } from './particles.js';
import { GRID_WGSL, SCAN_WGSL, SCATTER_WGSL, GRAV_DEPOSIT_WGSL, fastForceWGSL, HASH_SIZE } from './shaders_sim_fast.js';

const fr = Math.fround;
const GRAV_CELLS = 4096;

export class FastParticleSystem extends ParticleSystem {
  async init(gpu) {
    await super.init(gpu);
    this.isFast = true;
    const d = gpu.device;
    const B = GPUBufferUsage;

    // uniform-typed copies of the small tables (the fast kernel binds them as var<uniform>
    // to stay inside WebGPU's 8-storage-buffers-per-stage baseline)
    this.matsBufU = d.createBuffer({ size: 16 * 64, usage: B.UNIFORM | B.COPY_DST, label: 'matsU' });
    this.bodyDynBufU = d.createBuffer({ size: 16 * 48, usage: B.UNIFORM | B.COPY_DST, label: 'bodyDynU' });

    // grid + gravity buffers
    this.cellOfBuf = d.createBuffer({ size: MAXN * 4, usage: B.STORAGE, label: 'cellOf' });
    this.countsBuf = d.createBuffer({ size: HASH_SIZE * 4, usage: B.STORAGE | B.COPY_SRC, label: 'counts' });
    this.offsetsBuf = d.createBuffer({ size: HASH_SIZE * 4, usage: B.STORAGE | B.COPY_SRC, label: 'offsets' });
    this.cursorBuf = d.createBuffer({ size: HASH_SIZE * 4, usage: B.STORAGE | B.COPY_DST, label: 'cursor' });
    this.blockSumsBuf = d.createBuffer({ size: (HASH_SIZE / 256) * 4, usage: B.STORAGE, label: 'blockSums' });
    this.sortedBuf = d.createBuffer({ size: MAXN * 4, usage: B.STORAGE, label: 'sorted' });
    this.gravBuf = d.createBuffer({ size: GRAV_CELLS * 16, usage: B.STORAGE, label: 'gravCells' });
    this.gpGridBuf = d.createBuffer({ size: 16, usage: B.UNIFORM | B.COPY_DST, label: 'gpGrid' });
    this.spScanBuf = d.createBuffer({ size: 16, usage: B.UNIFORM | B.COPY_DST, label: 'spScan' });
    d.queue.writeBuffer(this.spScanBuf, 0, new Uint32Array([HASH_SIZE, 0, 0, 0]));
    this.fgBuf = d.createBuffer({ size: 48, usage: B.UNIFORM | B.COPY_DST, label: 'fg' });
    this.dpBuf = d.createBuffer({ size: 32, usage: B.UNIFORM | B.COPY_DST, label: 'dp' });

    // pipelines
    this.pGClear = await this._pipe(gpu, GRID_WGSL, 'clearCounts');
    this.pGCount = await this._pipe(gpu, GRID_WGSL, 'countCells');
    this.pSBlocks = await this._pipe(gpu, SCAN_WGSL, 'scanBlocks');
    this.pSSums = await this._pipe(gpu, SCAN_WGSL, 'scanSums');
    this.pSAdd = await this._pipe(gpu, SCAN_WGSL, 'addBack');
    this.pScatter = await this._pipe(gpu, SCATTER_WGSL, 'scatter');
    this.pVClear = await this._pipe(gpu, GRAV_DEPOSIT_WGSL, 'clearGrav');
    this.pVDep = await this._pipe(gpu, GRAV_DEPOSIT_WGSL, 'depositGrav');
    this.pForce = await this._pipe(gpu, fastForceWGSL({ nearGravity: false, farGravity: true }), 'simFast');

    // bind groups (ping-dependent sets reference the ping-pong pos/vel buffers)
    const bg = (pipe, list) => d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: list });
    this.bgGClear = bg(this.pGClear, [
      { binding: 0, resource: { buffer: this.gpGridBuf } }, { binding: 3, resource: { buffer: this.countsBuf } }]);
    this.bgGCount = [0, 1].map((p) => bg(this.pGCount, [
      { binding: 0, resource: { buffer: this.gpGridBuf } }, { binding: 1, resource: { buffer: this.pos[p] } },
      { binding: 2, resource: { buffer: this.cellOfBuf } }, { binding: 3, resource: { buffer: this.countsBuf } }]));
    this.bgSBlocks = bg(this.pSBlocks, [
      { binding: 0, resource: { buffer: this.spScanBuf } }, { binding: 1, resource: { buffer: this.countsBuf } },
      { binding: 2, resource: { buffer: this.offsetsBuf } }, { binding: 3, resource: { buffer: this.blockSumsBuf } }]);
    this.bgSSums = bg(this.pSSums, [
      { binding: 0, resource: { buffer: this.spScanBuf } }, { binding: 3, resource: { buffer: this.blockSumsBuf } }]);
    this.bgSAdd = bg(this.pSAdd, [
      { binding: 0, resource: { buffer: this.spScanBuf } }, { binding: 2, resource: { buffer: this.offsetsBuf } },
      { binding: 3, resource: { buffer: this.blockSumsBuf } }]);
    this.bgScatter = bg(this.pScatter, [
      { binding: 0, resource: { buffer: this.gpGridBuf } }, { binding: 1, resource: { buffer: this.cellOfBuf } },
      { binding: 2, resource: { buffer: this.cursorBuf } }, { binding: 3, resource: { buffer: this.sortedBuf } }]);
    // clearGrav statically uses only the grav buffer (auto-layout drops the uniform)
    this.bgVClear = bg(this.pVClear, [{ binding: 2, resource: { buffer: this.gravBuf } }]);
    this.bgVDep = [0, 1].map((p) => bg(this.pVDep, [
      { binding: 0, resource: { buffer: this.dpBuf } }, { binding: 1, resource: { buffer: this.pos[p] } },
      { binding: 2, resource: { buffer: this.gravBuf } }]));
    this.bgForce0 = [0, 1].map((p) => bg(this.pForce, [
      { binding: 0, resource: { buffer: this.paramsBuf } },
      { binding: 1, resource: { buffer: this.matsBufU } },
      { binding: 2, resource: { buffer: this.pos[p] } },
      { binding: 3, resource: { buffer: this.vel[p] } },
      { binding: 4, resource: { buffer: this.pos[1 - p] } },
      { binding: 5, resource: { buffer: this.vel[1 - p] } },
      { binding: 6, resource: { buffer: this.metaBuf } },
      { binding: 7, resource: { buffer: this.bodyDynBufU } },
    ]));
    this.bgForce1 = d.createBindGroup({
      layout: this.pForce.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.fgBuf } },
        { binding: 1, resource: { buffer: this.offsetsBuf } },
        { binding: 2, resource: { buffer: this.sortedBuf } },
        { binding: 3, resource: { buffer: this.gravBuf } },
      ],
    });

    this._fastReset();
    return this;
  }

  async _pipe(gpu, code, entry) {
    const mod = await gpu.makeShader(code, 'fast-' + entry);
    return gpu.device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: entry } });
  }

  _fastReset() {
    this._spawnBounds = [];
    this._massTotal = 0;
    this._aabb = null;
    this._cellSize = 1;
  }

  reset() {
    super.reset();
    if (this.matsBufU) this._fastReset();   // (super.reset runs once before init finishes)
  }

  // ---- keep the uniform-typed copies and bounds in sync with spawns ----
  _afterSpawn(posL, R, M) {
    this.gpu.device.queue.writeBuffer(this.matsBufU, 0, this.matData);
    this._spawnBounds.push({ c: posL.slice(), r: Math.max(R * 2.2, 2) });
    this._massTotal += M;
    const maxRad = Math.max(...this.matSlots.map((s) => s.rad || 0.01));
    this._cellSize = Math.max(4 * maxRad, 0.05);
  }

  addBlob(blob, posL, velL, name) {
    const body = super.addBlob(blob, posL, velL, name);
    this._afterSpawn(posL, blob.R, blob.M);
    return body;
  }

  addArmedImpactor(massSim, energySim, posL, velL, visRad, name) {
    const body = super.addArmedImpactor(massSim, energySim, posL, velL, visRad, name);
    this._afterSpawn(posL, Math.max(visRad, 1), massSim);
    return body;
  }

  writeBodyDyn(list) {
    super.writeBodyDyn(list);
    this.gpu.device.queue.writeBuffer(this.bodyDynBufU, 0, this.bodyDynArr);
  }

  // piggyback an AABB on the existing stats readback (refreshes the gravity grid bounds)
  _computeStats(pm, vt, me) {
    const stats = super._computeStats(pm, vt, me);
    const n = me.length;
    let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
    for (let i = 0; i < n; i++) {
      if (pm[i * 4 + 3] <= 0) continue;
      const x = pm[i * 4], y = pm[i * 4 + 1], z = pm[i * 4 + 2];
      if (x < mnx) mnx = x; if (x > mxx) mxx = x;
      if (y < mny) mny = y; if (y > mxy) mxy = y;
      if (z < mnz) mnz = z; if (z > mxz) mxz = z;
    }
    if (mnx < Infinity) this._aabb = { mn: [mnx, mny, mnz], mx: [mxx, mxy, mxz] };
    return stats;
  }

  _bounds() {
    // freshest knowledge: stats AABB if available, else union of spawn bounds
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    if (this._aabb) {
      mn = this._aabb.mn.slice(); mx = this._aabb.mx.slice();
    } else {
      for (const b of this._spawnBounds) {
        for (let a = 0; a < 3; a++) {
          mn[a] = Math.min(mn[a], b.c[a] - b.r);
          mx[a] = Math.max(mx[a], b.c[a] + b.r);
        }
      }
    }
    if (mn[0] === Infinity) { mn = [-10, -10, -10]; mx = [10, 10, 10]; }
    return { mn, mx };
  }

  writeParams(p) {
    super.writeParams(p);
    const q = this.gpu.device.queue;
    const { mn, mx } = this._bounds();
    const extent = fr(Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) * 1.002 + 1e-6);
    const massScale = fr(3.8e9 / Math.max(this._massTotal, 1e-9));
    const soft2 = fr((0.65 * extent / 16) ** 2);
    const invCell = fr(1 / this._cellSize);

    const fg = new ArrayBuffer(48);
    const ff = new Float32Array(fg), fu = new Uint32Array(fg);
    ff[0] = invCell; fu[1] = HASH_SIZE; ff[2] = massScale; ff[3] = soft2;
    ff[4] = mn[0]; ff[5] = mn[1]; ff[6] = mn[2]; ff[7] = fr(1 / extent);
    ff[8] = extent;
    q.writeBuffer(this.fgBuf, 0, fg);

    const dp = new ArrayBuffer(32);
    const df = new Float32Array(dp), du = new Uint32Array(dp);
    df[0] = mn[0]; df[1] = mn[1]; df[2] = mn[2]; df[3] = fr(1 / extent);
    df[4] = massScale; du[5] = this.activeN;
    q.writeBuffer(this.dpBuf, 0, dp);

    const gg = new ArrayBuffer(16);
    new Float32Array(gg)[0] = invCell;
    new Uint32Array(gg)[1] = this.activeN;
    new Uint32Array(gg)[2] = HASH_SIZE;
    q.writeBuffer(this.gpGridBuf, 0, gg);
  }

  step(encoder, substeps) {
    if (this.activeN === 0) return;
    const n = Math.ceil(this.activeN / 256);
    const H = Math.ceil(HASH_SIZE / 256);
    const pass = (pipe, group, wgs, group1) => {
      const cp = encoder.beginComputePass();
      cp.setPipeline(pipe);
      cp.setBindGroup(0, group);
      if (group1) cp.setBindGroup(1, group1);
      cp.dispatchWorkgroups(wgs);
      cp.end();
    };
    for (let s = 0; s < substeps; s++) {
      const p = this.ping;
      // counting-sort spatial hash (validated stages 1-4)
      pass(this.pGClear, this.bgGClear, H);
      pass(this.pGCount, this.bgGCount[p], n);
      pass(this.pSBlocks, this.bgSBlocks, H);
      pass(this.pSSums, this.bgSSums, 1);
      pass(this.pSAdd, this.bgSAdd, H);
      encoder.copyBufferToBuffer(this.offsetsBuf, 0, this.cursorBuf, 0, HASH_SIZE * 4);
      pass(this.pScatter, this.bgScatter, n);
      // gravity monopoles (validated stages 6a-6c)
      pass(this.pVClear, this.bgVClear, Math.ceil((GRAV_CELLS * 4) / 256));
      pass(this.pVDep, this.bgVDep[p], n);
      // shared-physics force kernel (validated stages 5b, 7)
      pass(this.pForce, this.bgForce0[p], n, this.bgForce1);
      this.ping = 1 - p;
    }
  }
}
