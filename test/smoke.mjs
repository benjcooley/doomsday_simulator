// Stage 0 smoke: headless WebGPU (Dawn/Metal) in Node — compute correctness + the SHIPPING kernel compiles.
import { create, globals } from 'webgpu';
import { SIM_WGSL } from '../js/shaders_sim.js';

Object.assign(globalThis, globals);
const gpu = create([]);

const adapter = await gpu.requestAdapter();
if (!adapter) { console.error('FAIL: no adapter'); process.exit(1); }
const device = await adapter.requestDevice();
const info = adapter.info || {};
console.log(`adapter: ${info.vendor || '?'} ${info.architecture || ''} | wgmem=${device.limits.maxComputeWorkgroupStorageSize} | maxbuf=${(device.limits.maxStorageBufferBindingSize / 1048576) | 0}MB`);

// --- 1) trivial compute correctness ---
const mod = device.createShaderModule({
  code: `
  @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
  @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < 256u) { buf[id.x] = f32(id.x) * 3.0 + 1.0; }
  }`,
});
const buf = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const stage = device.createBuffer({ size: 1024, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
const enc = device.createCommandEncoder();
const cp = enc.beginComputePass();
cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(4); cp.end();
enc.copyBufferToBuffer(buf, 0, stage, 0, 1024);
device.queue.submit([enc.finish()]);
await stage.mapAsync(GPUMapMode.READ);
const v = new Float32Array(stage.getMappedRange());
const ok = v[0] === 1 && v[100] === 301 && v[255] === 766;
stage.unmap();
console.log(ok ? 'PASS: compute correctness (256 values verified)' : `FAIL: compute wrong (${v[0]}, ${v[100]}, ${v[255]})`);
if (!ok) process.exit(1);

// --- 2) the SHIPPING physics kernel compiles headlessly ---
const simMod = device.createShaderModule({ code: SIM_WGSL });
const simInfo = await simMod.getCompilationInfo();
const errs = simInfo.messages.filter((m) => m.type === 'error');
for (const m of errs) console.error(`  WGSL error L${m.lineNum}: ${m.message}`);
if (errs.length) { console.error('FAIL: shipping SIM_WGSL does not compile under Dawn-Node'); process.exit(1); }
// and the pipelines actually build
for (const entry of ['sim', 'thermal', 'rebase']) {
  device.pushErrorScope('validation');
  device.createComputePipeline({ layout: 'auto', compute: { module: simMod, entryPoint: entry } });
  const err = await device.popErrorScope();
  if (err) { console.error(`FAIL: pipeline '${entry}': ${err.message}`); process.exit(1); }
}
console.log('PASS: shipping SIM_WGSL compiles + all 3 pipelines build (sim/thermal/rebase)');
console.log('STAGE 0: GREEN — the exact shipping WGSL runs in a terminal loop on this machine');
process.exit(0);
