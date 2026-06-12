// shaders_sim_fast.js — the FAST engine: counting-sort spatial hash + grid-walk force kernel.
// Built stage-by-stage under test/: each pass has a bit-exact JS reference in the headless
// harness before the next is added. The pair PHYSICS is shared with the proven engine
// (imported from shaders_sim.js) — this file only adds the neighbor-finding machinery.
import { SIM_STRUCTS_WGSL, SIM_HELPERS_WGSL, SELF_DECLS_WGSL, SELF_MATS_WGSL, pairPhysicsWGSL, integrateTailWGSL } from './shaders_sim.js';

export const HASH_SIZE = 1 << 20;   // 1M hash cells — load factor stays sane at 500k particles

// hash math — MUST stay identical to jsCellHash below (validated bit-exact in stage 1)
const HASH_FNS = /* wgsl */`
const HASH_MASK: u32 = ${HASH_SIZE - 1}u;

fn cellCoord(p: vec3f, invCell: f32) -> vec3i {
  return vec3i(floor(p * invCell));
}
fn cellHash(c: vec3i) -> u32 {
  let ux = bitcast<u32>(c.x) * 0x9E3779B1u;
  let uy = bitcast<u32>(c.y) * 0x85EBCA77u;
  let uz = bitcast<u32>(c.z) * 0xC2B2AE3Du;
  return (ux ^ uy ^ uz) & HASH_MASK;
}
`;

export const GRID_WGSL = /* wgsl */`
struct GridParams {
  invCell: f32,     // 1 / cellSize
  nActive: u32,
  hashSize: u32,
  pad0: f32,
}
${HASH_FNS}
@group(0) @binding(0) var<uniform> GP: GridParams;
@group(0) @binding(1) var<storage, read> posG: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cellOf: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;

// pass A: clear the histogram
@compute @workgroup_size(256)
fn clearCounts(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < GP.hashSize) { atomicStore(&counts[gid.x], 0u); }
}

// pass B: cell id per particle + histogram
@compute @workgroup_size(256)
fn countCells(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GP.nActive) { return; }
  let h = cellHash(cellCoord(posG[i].xyz, GP.invCell));
  cellOf[i] = h;
  atomicAdd(&counts[h], 1u);
}
`;

// ---- stage 2: exclusive prefix scan over the histogram (3 dispatches) ----
export const SCAN_WGSL = /* wgsl */`
struct ScanParams { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> SP: ScanParams;
@group(0) @binding(1) var<storage, read> srcS: array<u32>;
@group(0) @binding(2) var<storage, read_write> dstS: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;

var<workgroup> tmp: array<u32, 256>;

// pass 1: per-256-block exclusive scan (Hillis-Steele, double-barriered) + block totals
@compute @workgroup_size(256)
fn scanBlocks(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lidv: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = gid.x;
  let li = lidv.x;
  var v = 0u;
  if (i < SP.n) { v = srcS[i]; }
  tmp[li] = v;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var t = tmp[li];
    if (li >= off) { t = t + tmp[li - off]; }
    workgroupBarrier();
    tmp[li] = t;
    workgroupBarrier();
  }
  if (i < SP.n) { dstS[i] = tmp[li] - v; }            // inclusive → exclusive
  if (li == 255u) { blockSums[wid.x] = tmp[255u]; }
}

// pass 2: serial exclusive scan of the (≤512) block sums — one thread, bulletproof
@compute @workgroup_size(1)
fn scanSums() {
  let nb = (SP.n + 255u) / 256u;
  var acc = 0u;
  for (var b = 0u; b < nb; b = b + 1u) {
    let t = blockSums[b];
    blockSums[b] = acc;
    acc = acc + t;
  }
}

// pass 3: add scanned block offsets back
@compute @workgroup_size(256)
fn addBack(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  if (gid.x < SP.n) { dstS[gid.x] = dstS[gid.x] + blockSums[wid.x]; }
}
`;

// ---- stage 3: scatter into sorted order (cursor = copy of offsets, atomically advanced) ----
export const SCATTER_WGSL = /* wgsl */`
struct GridParams2 { invCell: f32, nActive: u32, hashSize: u32, pad0: f32 }
@group(0) @binding(0) var<uniform> GP2: GridParams2;
@group(0) @binding(1) var<storage, read> cellOfS: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIdx: array<u32>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= GP2.nActive) { return; }
  let slot = atomicAdd(&cursor[cellOfS[i]], 1u);
  sortedIdx[slot] = i;
}
`;

// ---- stage 4: 27-cell neighbor walk (with hash-collision dedupe) ----
// Used by the validation harness to prove completeness; the production force kernel
// uses this exact same walk structure around the shared pair-physics chunk.
export const NEIGHBOR_WGSL = /* wgsl */`
struct WalkParams {
  invCell: f32,
  nActive: u32,
  hashSize: u32,
  reach2: f32,     // interaction reach², f32
}
${HASH_FNS}
@group(0) @binding(0) var<uniform> WP: WalkParams;
@group(0) @binding(1) var<storage, read> posW: array<vec4f>;
@group(0) @binding(2) var<storage, read> offsetsW: array<u32>;
@group(0) @binding(3) var<storage, read> countsW: array<u32>;
@group(0) @binding(4) var<storage, read> sortedW: array<u32>;
@group(0) @binding(5) var<storage, read_write> outCount: array<u32>;
@group(0) @binding(6) var<storage, read_write> outISum: array<u32>;

@compute @workgroup_size(256)
fn walkNeighbors(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= WP.nActive) { return; }
  let pi = posW[i].xyz;
  let c = cellCoord(pi, WP.invCell);

  // gather the ≤27 neighbor-cell hashes, deduped: two adjacent cells may collide into the
  // same slot, and walking a slot twice would double-count every pair in it
  var hashes: array<u32, 27>;
  var nh = 0u;
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let h = cellHash(c + vec3i(dx, dy, dz));
        var dup = false;
        for (var k = 0u; k < nh; k = k + 1u) {
          if (hashes[k] == h) { dup = true; break; }
        }
        if (!dup) { hashes[nh] = h; nh = nh + 1u; }
      }
    }
  }

  var count = 0u;
  var isum = 0u;
  for (var k = 0u; k < nh; k = k + 1u) {
    let h = hashes[k];
    let start = offsetsW[h];
    let end = start + countsW[h];
    for (var s = start; s < end; s = s + 1u) {
      let j = sortedW[s];
      if (j == i) { continue; }
      let d = posW[j].xyz - pi;
      let r2 = dot(d, d);
      if (r2 < WP.reach2) {
        count = count + 1u;
        isum = isum + j;
      }
    }
  }
  outCount[i] = count;
  outISum[i] = isum;
}
`;

// ---- coarse gravity grid: 16³ mass monopoles in fixed point (deposit + clear) ----
// CoM uses scale-cancelling ratios, so only the mass needs massScale to decode.
export const GRAV_DEPOSIT_WGSL = /* wgsl */`
struct DepositParams {
  gOrigin: vec3f, gExtentInv: f32,
  massScale: f32, nActive: u32, pad0: f32, pad1: f32,
}
@group(0) @binding(0) var<uniform> DP: DepositParams;
@group(0) @binding(1) var<storage, read> posD: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> gravA: array<atomic<u32>>;   // 4096 cells × (mx,my,mz,m)

@compute @workgroup_size(256)
fn clearGrav(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < 16384u) { atomicStore(&gravA[gid.x], 0u); }
}

@compute @workgroup_size(256)
fn depositGrav(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= DP.nActive) { return; }
  let pm = posD[i];
  if (pm.w <= 0.0) { return; }
  let pn = clamp((pm.xyz - DP.gOrigin) * DP.gExtentInv, vec3f(0.0), vec3f(0.999999));
  let ci = vec3u(pn * 16.0);
  let flat = (ci.x + ci.y * 16u + ci.z * 256u) * 4u;
  let mfp = u32(pm.w * DP.massScale);
  atomicAdd(&gravA[flat + 0u], u32(f32(mfp) * pn.x));
  atomicAdd(&gravA[flat + 1u], u32(f32(mfp) * pn.y));
  atomicAdd(&gravA[flat + 2u], u32(f32(mfp) * pn.z));
  atomicAdd(&gravA[flat + 3u], mfp);
}
`;

// ---- stage 5/6: the fast FORCE kernel — shared pair physics around the validated walk,
// plus tiled 16³-monopole far gravity. group(0) keeps the N² slot numbers, but mats and
// bodyDyn are UNIFORM and there is no dbg slot — WebGPU's baseline guarantees only 8
// storage buffers per stage, and this lands exactly there:
// posA, velA, posB, velB, pmeta + offsetsF, sortedF, gravCells.
export function fastForceWGSL({ nearGravity = false, farGravity = true } = {}) {
  return /* wgsl */`
${SIM_STRUCTS_WGSL}
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<uniform> mats: array<MatI, 16>;
@group(0) @binding(2) var<storage, read> posA: array<vec4f>;
@group(0) @binding(3) var<storage, read> velA: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> posB: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> velB: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> pmeta: array<u32>;
@group(0) @binding(7) var<uniform> bodyDyn: array<BodyDyn, 16>;
struct FastGrid {
  invCell: f32, hashSize: u32, massScale: f32, soft2: f32,
  gOrigin: vec3f, gExtentInv: f32,
  gExtent: f32, pad5: f32, pad6: f32, pad7: f32,
}
${HASH_FNS}
@group(1) @binding(0) var<uniform> FG: FastGrid;
@group(1) @binding(1) var<storage, read> offsetsF: array<u32>;
@group(1) @binding(2) var<storage, read> sortedF: array<u32>;
@group(1) @binding(3) var<storage, read> gravCells: array<vec4<u32>, 4096>;
${SIM_HELPERS_WGSL}
var<workgroup> gTile: array<vec4f, 256>;   // decoded (CoM xyz, mass) cells

@compute @workgroup_size(256)
fn simFast(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lidv: vec3u) {
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

  if (isOn) {
    // 27-cell deduped walk (completeness + collision behavior validated in stage 4)
    let cc = cellCoord(pi, FG.invCell);
    var hashes: array<u32, 27>;
    var nh = 0u;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
      for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
          let hsh = cellHash(cc + vec3i(dx, dy, dz));
          var dup = false;
          for (var k2 = 0u; k2 < nh; k2 = k2 + 1u) {
            if (hashes[k2] == hsh) { dup = true; break; }
          }
          if (!dup) { hashes[nh] = hsh; nh = nh + 1u; }
        }
      }
    }
    for (var k = 0u; k < nh; k = k + 1u) {
      let hh = hashes[k];
      let segStart = offsetsF[hh];
      let segEnd = select(offsetsF[hh + 1u], P.nActive, hh + 1u >= FG.hashSize);
      for (var s = segStart; s < segEnd; s = s + 1u) {
        let j = sortedF[s];
        if (j == i) { continue; }
        let pj = posA[j];
        let pvel = velA[j];
        let aux = neighborAux(pmeta[j], pvel);
${pairPhysicsWGSL({ nearGravity })}
      }
    }
  }
${farGravity ? `
  // far gravity: all 4096 coarse monopoles, tiled through workgroup memory. Gravity comes
  // ENTIRELY from these cells (near-gravity in the pair chunk is off), softened at ~half a
  // coarse cell so own-cell/self attraction cannot spike. Barriers are workgroup-uniform.
  for (var t = 0u; t < 16u; t = t + 1u) {
    let c4 = gravCells[t * 256u + lid];
    var cm = vec4f(0.0);
    if (c4.w > 0u) {
      let inv = 1.0 / f32(c4.w);                       // CoM ratio — fixed-point scale cancels
      cm = vec4f(FG.gOrigin + vec3f(f32(c4.x), f32(c4.y), f32(c4.z)) * inv * FG.gExtent,
        f32(c4.w) / FG.massScale);
    }
    gTile[lid] = cm;
    workgroupBarrier();
    if (isOn) {
      for (var jj = 0u; jj < 256u; jj = jj + 1u) {
        let cell = gTile[jj];
        if (cell.w > 0.0) {
          let dg = cell.xyz - pi;
          let r2g = dot(dg, dg) + FG.soft2;
          let invRg = inverseSqrt(r2g);
          acc = acc + dg * (P.gConst * cell.w * invRg * invRg * invRg);
        }
      }
    }
    workgroupBarrier();
  }
` : ''}
  if (!isOn) { return; }
${integrateTailWGSL({ debugTap: false })}
}
`;
}

// JS mirror of the WGSL hash — single source of truth for tests.
// Math.fround replicates the GPU's f32 multiply exactly (IEEE: f32 op == fround(f64 op on f32 inputs)),
// so boundary positions land in the same cell on both sides — bit-exact comparisons hold.
export function jsCellHash(x, y, z, invCell) {
  const ic = Math.fround(invCell);
  const cx = Math.floor(Math.fround(Math.fround(x) * ic)) | 0;
  const cy = Math.floor(Math.fround(Math.fround(y) * ic)) | 0;
  const cz = Math.floor(Math.fround(Math.fround(z) * ic)) | 0;
  const ux = Math.imul(cx, 0x9E3779B1) >>> 0;
  const uy = Math.imul(cy, 0x85EBCA77) >>> 0;
  const uz = Math.imul(cz, 0xC2B2AE3D) >>> 0;
  return ((ux ^ uy ^ uz) >>> 0) & (HASH_SIZE - 1);
}
