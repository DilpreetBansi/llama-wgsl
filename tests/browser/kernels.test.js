// Each WGSL kernel against a float64 CPU reference on random inputs.
import * as K from '../../src/kernels.js';
import { createDevice, ropeTables } from '../../src/model.js';
import { halfToFloat } from '../../src/safetensors.js';
import { assert, maxAbsDiff, rng, run, test } from './harness.js';

const { device } = await createDevice();
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

function buffer(data, usage = S) {
  const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage });
  device.queue.writeBuffer(b, 0, data);
  return b;
}
const zeros = (n) => buffer(new Float32Array(n));
const uniform = (u32) => buffer(new Uint32Array(u32), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

async function read(buf, n, Type = Float32Array) {
  const dst = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, dst, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const out = new Type(dst.getMappedRange().slice(0));
  dst.destroy();
  return out;
}

async function dispatch(code, resources, gx, gy = 1) {
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(gx, gy);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}

// Random weights in each storage format, returned as {bytes to upload, exact float values}.
function randomWeights(n, dtype, rand) {
  const values = new Float64Array(n);
  if (dtype === 'f32') {
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = values[i] = Math.fround(0.3 * rand());
    return { bytes: f, values };
  }
  const u16 = new Uint16Array(n);
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  for (let i = 0; i < n; i++) {
    if (dtype === 'bf16') {
      f[0] = 0.3 * rand();
      u16[i] = u[0] >>> 16; // truncate to bf16
      u[0] = u16[i] << 16;
      values[i] = f[0];
    } else {
      // random finite f16 with magnitude in [2^-8, 2^1)
      const sign = rand() > 0 ? 0 : 0x8000;
      const exp = 7 + Math.floor(Math.abs(rand() * 1000)) % 9;
      const mant = Math.floor(Math.abs(rand() * 1e6)) & 0x3ff;
      u16[i] = sign | (exp << 10) | mant;
      values[i] = halfToFloat(u16[i]);
    }
  }
  return { bytes: u16, values };
}

function randomVec(n, rand, scale = 1) {
  const f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = scale * rand();
  return f;
}

// |error| bound for an f32 dot product of length k, relative to sum |w x|.
const dotTol = (absSum) => 2e-6 * absSum + 1e-6;

for (const dtype of ['bf16', 'f16', 'f32']) {
  for (const [rows, cols] of [[100, 64], [70, 576], [37, 1536], [9, 2048], [8, 8]]) {
    test(`matvec ${dtype} ${rows}x${cols}`, async () => {
      const rand = rng(rows * cols);
      const w = randomWeights(rows * cols, dtype, rand);
      const x = randomVec(cols, rand);
      const y = zeros(rows);
      await dispatch(K.matvec({ rows, K: cols, dtype }), [buffer(w.bytes), buffer(x), y], ...K.matvecGroups(rows, cols));
      const got = await read(y, rows);
      let worst = 0;
      for (let r = 0; r < rows; r++) {
        let ref = 0;
        let abs = 0;
        for (let k = 0; k < cols; k++) {
          ref += w.values[r * cols + k] * x[k];
          abs += Math.abs(w.values[r * cols + k] * x[k]);
        }
        const err = Math.abs(got[r] - ref);
        assert(err <= dotTol(abs), `row ${r}: got ${got[r]}, want ${ref}`);
        worst = Math.max(worst, err);
      }
      return `max err ${worst.toExponential(1)}`;
    });
  }
}

test('matvec residual + bias + output offset', async () => {
  const rows = 50;
  const cols = 72;
  const rand = rng(3);
  const w = randomWeights(rows * cols, 'bf16', rand);
  const x = randomVec(cols, rand);
  const b = randomVec(rows, rand);
  const y0 = randomVec(rows + 10, rand);
  const y = buffer(y0);
  const code = K.matvec({ rows, K: cols, dtype: 'bf16', residual: true, bias: true, outOffset: 10 });
  await dispatch(code, [buffer(w.bytes), buffer(x), y, buffer(b)], ...K.matvecGroups(rows, cols));
  const got = await read(y, rows + 10);
  for (let i = 0; i < 10; i++) assert(got[i] === y0[i], `element ${i} before the offset changed`);
  for (let r = 0; r < rows; r++) {
    let ref = y0[10 + r] + b[r];
    for (let k = 0; k < cols; k++) ref += w.values[r * cols + k] * x[k];
    assert(Math.abs(got[10 + r] - ref) < 1e-4, `row ${r}: got ${got[10 + r]}, want ${ref}`);
  }
});

test('matvec 2D dispatch (more than 65535 workgroups)', async () => {
  const cols = 8; // 4 threads per row -> 64 rows per workgroup
  const rows = 65535 * 64 + 77;
  const rand = rng(11);
  const w = randomWeights(rows * cols, 'bf16', rand);
  const x = randomVec(cols, rand);
  const y = zeros(rows);
  const grid = K.matvecGroups(rows, cols);
  assert(grid[1] === 2, `expected a 2-row grid, got ${grid}`);
  await dispatch(K.matvec({ rows, K: cols, dtype: 'bf16' }), [buffer(w.bytes), buffer(x), y], ...grid);
  const got = await read(y, rows);
  for (const r of [0, 1, 65535 * 64 - 1, 65535 * 64, rows - 1]) {
    let ref = 0;
    for (let k = 0; k < cols; k++) ref += w.values[r * cols + k] * x[k];
    assert(Math.abs(got[r] - ref) < 1e-5, `row ${r}: got ${got[r]}, want ${ref}`);
  }
});

for (const dtype of ['bf16', 'f16', 'f32']) {
  test(`swiglu ${dtype}`, async () => {
    const I = 99;
    const cols = 128;
    const rand = rng(5);
    const w = randomWeights(2 * I * cols, dtype, rand);
    const x = randomVec(cols, rand);
    const y = zeros(I);
    await dispatch(K.swiglu({ I, K: cols, dtype }), [buffer(w.bytes), buffer(x), y], ...K.matvecGroups(I, cols));
    const got = await read(y, I);
    for (let r = 0; r < I; r++) {
      let g = 0;
      let u = 0;
      for (let k = 0; k < cols; k++) {
        g += w.values[r * cols + k] * x[k];
        u += w.values[(r + I) * cols + k] * x[k];
      }
      const ref = (g / (1 + Math.exp(-g))) * u;
      assert(Math.abs(got[r] - ref) < 1e-4 * (1 + Math.abs(ref)), `row ${r}: got ${got[r]}, want ${ref}`);
    }
  });
}

test('rmsnorm', async () => {
  const N = 576;
  const rand = rng(9);
  const x = randomVec(N, rand, 3);
  const g = randomVec(N, rand);
  const y = zeros(N);
  await dispatch(K.rmsnorm({ N, eps: 1e-5 }), [buffer(x), buffer(g), y], 1);
  const got = await read(y, N);
  let ms = 0;
  for (const v of x) ms += v * v;
  const inv = 1 / Math.sqrt(ms / N + 1e-5);
  const ref = Array.from(x, (v, i) => g[i] * v * inv);
  const { diff } = maxAbsDiff(got, ref);
  assert(diff < 1e-5, `max diff ${diff}`);
  return `max err ${diff.toExponential(1)}`;
});

test('rope + kv cache write (grouped-query layout)', async () => {
  const HQ = 6;
  const HKV = 2;
  const D = 64;
  const maxT = 16;
  const pos = 11;
  const rand = rng(21);
  const qkv0 = randomVec((HQ + 2 * HKV) * D, rand);
  const { cos, sin } = ropeTables(D, 10000, maxT);
  const qkv = buffer(qkv0);
  const kc = zeros(maxT * HKV * D);
  const vc = zeros(maxT * HKV * D);
  const r = K.ropeKv({ HQ, HKV, D });
  await dispatch(r.code, [uniform([pos, 0, 0, 0]), qkv, buffer(cos), buffer(sin), kc, vc], Math.ceil(r.total / 64));
  const q = await read(qkv, HQ * D);
  const k = await read(kc, maxT * HKV * D);
  const v = await read(vc, maxT * HKV * D);
  const half = D / 2;
  const rot = (head, out, outBase) => {
    for (let j = 0; j < half; j++) {
      const a = qkv0[head * D + j];
      const b = qkv0[head * D + j + half];
      const c = cos[pos * half + j];
      const s = sin[pos * half + j];
      assert(Math.abs(out[outBase + j] - (a * c - b * s)) < 1e-6, `head ${head} dim ${j}`);
      assert(Math.abs(out[outBase + j + half] - (b * c + a * s)) < 1e-6, `head ${head} dim ${j + half}`);
    }
  };
  for (let h = 0; h < HQ; h++) rot(h, q, h * D);
  for (let h = 0; h < HKV; h++) rot(HQ + h, k, pos * HKV * D + h * D);
  for (let i = 0; i < HKV * D; i++) assert(v[pos * HKV * D + i] === qkv0[(HQ + HKV) * D + i], `v ${i}`);
  assert(k.slice(0, pos * HKV * D).every((x) => x === 0), 'wrote outside position');
});

for (const [T, HQ, HKV, D] of [[1, 4, 4, 64], [300, 6, 2, 64], [77, 8, 2, 16], [40, 9, 3, 64]]) {
  test(`attention T=${T} heads ${HQ}/${HKV} d=${D}`, async () => {
    const maxT = 320;
    const rand = rng(T * HQ);
    const q = randomVec(HQ * D, rand);
    const kc = randomVec(maxT * HKV * D, rand);
    const vc = randomVec(maxT * HKV * D, rand);
    const out = zeros(HQ * D);
    const att = zeros(HQ * maxT);
    const code = K.attention({ HQ, HKV, D, maxT });
    await dispatch(code, [uniform([T - 1, 0, 0, 0]), buffer(q), buffer(kc), buffer(vc), att, out], HQ);
    const got = await read(out, HQ * D);
    const ref = new Float64Array(HQ * D);
    for (let h = 0; h < HQ; h++) {
      const kvh = Math.floor(h / (HQ / HKV));
      const s = new Float64Array(T);
      for (let t = 0; t < T; t++) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot += q[h * D + d] * kc[(t * HKV + kvh) * D + d];
        s[t] = dot / Math.sqrt(D);
      }
      const m = Math.max(...s);
      let z = 0;
      for (let t = 0; t < T; t++) z += (s[t] = Math.exp(s[t] - m));
      for (let d = 0; d < D; d++) {
        let o = 0;
        for (let t = 0; t < T; t++) o += s[t] * vc[(t * HKV + kvh) * D + d];
        ref[h * D + d] = o / z;
      }
    }
    const { diff } = maxAbsDiff(got, ref);
    assert(diff < 2e-5, `max diff ${diff}`);
    return `max err ${diff.toExponential(1)}`;
  });
}

for (const dtype of ['bf16', 'f16', 'f32']) {
  test(`embedding gather ${dtype}`, async () => {
    const H = 72;
    const V = 10;
    const rand = rng(33);
    const w = randomWeights(V * H, dtype, rand);
    const x = zeros(H);
    for (const row of [0, 7, 9]) {
      await dispatch(K.embed({ H, dtype }), [uniform([0, row, 0, 0]), buffer(w.bytes), x], Math.ceil(H / 64));
      const got = await read(x, H);
      for (let i = 0; i < H; i++) assert(got[i] === w.values[row * H + i], `row ${row} col ${i}`);
    }
  });
}

test('argmax (ties resolve to the lowest index)', async () => {
  const V = 49152;
  const rand = rng(44);
  const l = randomVec(V, rand);
  l[31337] = 50;
  l[40000] = 50;
  const out = zeros(1);
  await dispatch(K.argmax({ V }), [buffer(l), out], 1);
  const [idx] = await read(out, 1, Uint32Array);
  assert(idx === 31337, `got ${idx}`);
});

await run('WGSL kernels');
