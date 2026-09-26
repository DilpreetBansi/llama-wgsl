import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Tokenizer, StreamDecoder } from '../../src/tokenizer.js';

const fixture = (f) => new URL(`../fixtures/${f}`, import.meta.url);
const tok = new Tokenizer(JSON.parse(readFileSync(fixture('smollm2-tokenizer.json'), 'utf8')));
const cases = JSON.parse(readFileSync(fixture('tokenizer_cases.json'), 'utf8'));

test('encode matches Hugging Face tokenizers on every case', () => {
  for (const { text, ids } of cases) assert.deepEqual(tok.encode(text), ids, JSON.stringify(text));
});

test('decode round-trips every case', () => {
  for (const { text } of cases) assert.equal(tok.decode(tok.encode(text), { skipSpecial: false }), text);
});

test('special tokens are split out and can be skipped when decoding', () => {
  const ids = tok.encode('<|im_start|>user\nhi<|im_end|>');
  assert.equal(ids[0], 1);
  assert.equal(ids.at(-1), 2);
  assert.equal(tok.decode(ids), 'user\nhi');
});

test('streaming decode holds back split UTF-8 characters', () => {
  const text = 'emoji 🙂👍🏽 and 中文 done';
  const sd = new StreamDecoder(tok);
  const pieces = tok.encode(text).map((id) => sd.push(id));
  assert.ok(pieces.every((p) => !p.includes('�')), 'no replacement characters mid-stream');
  assert.equal(pieces.join('') + sd.flush(), text);
});
