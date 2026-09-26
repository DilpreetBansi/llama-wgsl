import { test } from 'node:test';
import assert from 'node:assert/strict';
import { argmax, makeRng, sample, topK } from '../../src/sampler.js';

const logits = (n, seed = 1) => {
  const r = makeRng(seed);
  return Float32Array.from({ length: n }, () => r() * 10 - 5);
};

test('topK agrees with a full sort, ties broken by lower index', () => {
  const v = logits(5000);
  v[10] = v[20] = 99;
  const ref = Array.from(v.keys()).sort((a, b) => v[b] - v[a] || a - b).slice(0, 50);
  assert.deepEqual(topK(v, 50), ref);
  assert.deepEqual(topK(v, 2), [10, 20]);
});

test('temperature 0 is greedy', () => {
  const v = logits(1000);
  assert.equal(sample(v.slice(), { temperature: 0 }), argmax(v));
});

test('same seed, same tokens', () => {
  const v = logits(1000);
  const run = (seed) => {
    const rng = makeRng(seed);
    return Array.from({ length: 20 }, () => sample(v.slice(), { temperature: 1, topK: 50, topP: 0.95 }, rng));
  };
  assert.deepEqual(run(42), run(42));
  assert.notDeepEqual(run(42), run(43));
});

test('top-p keeps only the nucleus', () => {
  const v = new Float32Array(100).fill(-20);
  v[3] = 5; // ~0.73 of the mass
  v[7] = 4; // ~0.27
  const rng = makeRng(1);
  const seen = new Set();
  for (let i = 0; i < 300; i++) seen.add(sample(v.slice(), { temperature: 1, topK: 100, topP: 0.5 }, rng));
  assert.deepEqual([...seen], [3]);
  for (let i = 0; i < 300; i++) seen.add(sample(v.slice(), { temperature: 1, topK: 100, topP: 0.99 }, rng));
  assert.deepEqual([...seen].sort((a, b) => a - b), [3, 7]);
});

test('repetition penalty follows transformers (divide positive, multiply negative)', () => {
  const v = Float32Array.from([2, -2, 1]);
  sample(v, { temperature: 0, repetitionPenalty: 2, history: [0, 1, 1] });
  assert.deepEqual(Array.from(v), [1, -4, 1]);
});
