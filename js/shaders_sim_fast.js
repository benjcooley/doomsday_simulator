// shaders_sim_fast.js — the FAST engine's grid passes (counting-sort spatial hash).
// Built stage-by-stage under test/: each pass has a bit-exact JS reference in the headless
// harness before the next is added. The pair PHYSICS is shared with the proven engine
// (see shaders_sim.js) — this file only contains the neighbor-finding machinery.

export const HASH_SIZE = 1 << 17;   // 131072 hash cells (open hash over unbounded space)

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
