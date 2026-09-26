import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseHeader, readSafetensors, toFloat32, halfToFloat } from '../../src/safetensors.js';

const file = new Uint8Array(readFileSync(new URL('../fixtures/tiny-llama-bf16/model.safetensors', import.meta.url)));

function streamOf(bytes, chunk) {
  let i = 0;
  return new ReadableStream({
    pull(c) {
      if (i >= bytes.length) return c.close();
      const n = typeof chunk === 'function' ? chunk() : chunk;
      c.enqueue(bytes.slice(i, i + n));
      i += n;
    },
  });
}

async function collect(stream) {
  const tensors = new Map();
  let header;
  let seen = 0;
  await readSafetensors(stream, {
    onHeader: (h) => (header = h),
    onTensor: (t, b) => tensors.set(t.name, b),
    onBytes: (n) => (seen = n),
  });
  return { header, tensors, seen };
}

test('streamed tensors are byte-identical for any chunking', async () => {
  const n = Number(new DataView(file.buffer).getBigUint64(0, true));
  const { tensors: ref } = await collect(streamOf(file, file.length));
  const header = parseHeader(file.subarray(8, 8 + n));
  assert.equal(ref.size, header.tensors.length);
  let s = 12345;
  const random = () => 1 + ((s = (s * 1103515245 + 12345) >>> 0) % 5000);
  for (const chunk of [1, 7, 4096, random]) {
    const { tensors, seen } = await collect(streamOf(file, chunk));
    assert.equal(seen, file.length);
    for (const t of header.tensors) {
      assert.deepEqual(tensors.get(t.name), file.subarray(8 + n + t.begin, 8 + n + t.end), `${t.name} (chunk ${chunk})`);
    }
  }
});

test('a truncated file is an error, not silent garbage', async () => {
  await assert.rejects(collect(streamOf(file.subarray(0, file.length - 100), 1000)), /ended inside|end of stream/);
});

test('bf16 / f16 / f32 conversion', () => {
  const bf = new Uint8Array(new Uint16Array([0x3f80, 0xc040, 0x0000]).buffer); // 1, -3, 0
  assert.deepEqual(Array.from(toFloat32(bf, 'BF16')), [1, -3, 0]);
  const h = new Uint8Array(new Uint16Array([0x3c00, 0xc000, 0x0001, 0x7bff]).buffer);
  assert.deepEqual(Array.from(toFloat32(h, 'F16')), [1, -2, 2 ** -24, 65504]);
  assert.ok(Number.isNaN(halfToFloat(0x7e00)));
  const f = new Uint8Array(new Float32Array([0.5, -8]).buffer);
  assert.deepEqual(Array.from(toFloat32(f, 'F32')), [0.5, -8]);
});
