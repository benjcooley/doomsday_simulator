// test/util.mjs — shared harness: headless device, seeded fixtures, buffer helpers, comparators
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);

// CRITICAL: the Dawn instance + adapter must stay referenced for the process lifetime —
// if V8 garbage-collects them while the device is in use, dawn-node SEGFAULTS (exit 139)
// at whatever allocation happens to trigger GC. Module-level keep-alive prevents it.
const _keepAlive = [];

export async function gpuDevice() {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('no adapter');
  const device = await adapter.requestDevice();
  _keepAlive.push(gpu, adapter, device);
  device.addEventListener?.('uncapturederror', (e) => {
    console.error('[GPU]', e.error?.message);
    process.exitCode = 1;
  });
  return device;
}

// deterministic LCG — identical sequences in JS reference and fixture generation
export function rng(seed = 12345) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// fixtures: Float32Array posM (xyzm) layouts
export function fixtureCloud(n, extent = 50, seed = 1) {
  const r = rng(seed);
  const pm = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    pm[i * 4] = (r() * 2 - 1) * extent;
    pm[i * 4 + 1] = (r() * 2 - 1) * extent;
    pm[i * 4 + 2] = (r() * 2 - 1) * extent;
    pm[i * 4 + 3] = 1e-4 * (0.5 + r());
  }
  return pm;
}

export function fixtureLattice(n, spacing = 0.6, seed = 2) {
  // dense cubic-ish blob — the contact-heavy case
  const r = rng(seed);
  const side = Math.ceil(Math.cbrt(n));
  const pm = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const x = i % side, y = ((i / side) | 0) % side, z = (i / (side * side)) | 0;
    pm[i * 4] = (x - side / 2) * spacing + (r() - 0.5) * 0.05;
    pm[i * 4 + 1] = (y - side / 2) * spacing + (r() - 0.5) * 0.05;
    pm[i * 4 + 2] = (z - side / 2) * spacing + (r() - 0.5) * 0.05;
    pm[i * 4 + 3] = 1e-4;
  }
  return pm;
}

export function storageBuf(device, data, label) {
  const buf = device.createBuffer({
    size: Math.ceil(data.byteLength / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

export function emptyBuf(device, byteSize, label) {
  return device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label,
  });
}

export function uniformBuf(device, data, label) {
  const buf = device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

export async function readBack(device, buf, byteSize, Type = Float32Array) {
  const stage = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, stage, 0, byteSize);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const out = new Type(stage.getMappedRange().slice(0));
  stage.unmap();
  return out;
}

export async function makePipeline(device, code, entry, label) {
  // NOTE: getCompilationInfo() segfaults intermittently in dawn-node — do NOT call it here.
  // pushErrorScope catches both compile and pipeline-validation errors reliably.
  device.pushErrorScope('validation');
  const mod = device.createShaderModule({ code, label });
  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: entry } });
  const err = await device.popErrorScope();
  if (err) throw new Error(`pipeline '${label}:${entry}': ${err.message}`);
  return pipe;
}

export function expectEqual(name, got, want) {
  if (got.length !== want.length) {
    console.error(`FAIL ${name}: length ${got.length} != ${want.length}`);
    return false;
  }
  for (let i = 0; i < got.length; i++) {
    if (got[i] !== want[i]) {
      console.error(`FAIL ${name}: index ${i}: got ${got[i]} want ${want[i]}`);
      return false;
    }
  }
  console.log(`PASS ${name} (${got.length} values bit-equal)`);
  return true;
}

// dawn-node's live native handles prevent Node from draining at end-of-script —
// flush stdout for a tick, then exit explicitly.
export function done(ok) {
  console.log(ok ? '\nRESULT: GREEN' : '\nRESULT: RED');
  setTimeout(() => process.exit(ok ? 0 : 1), 80);
}
