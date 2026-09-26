import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { chatML } from '../../src/chat.js';

const ref = JSON.parse(readFileSync(new URL('../fixtures/smollm2-135m-ref.json', import.meta.url), 'utf8'));

test("chatML reproduces SmolLM2's chat template (transformers apply_chat_template)", () => {
  const system = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face';
  assert.equal(chatML(ref.messages, { system }), ref.prompt_text);
});

test('an explicit system message replaces the default', () => {
  const s = chatML([{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'Hi' }], { system: 'default' });
  assert.equal(s, '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n');
});
