// Real-checkpoint check: SmolLM2-135M-Instruct against transformers (tools/make_reference.py).
// Loads through the same streaming loader the demo uses; the runner mounts the checkpoint
// directory at /models/smollm2-135m (SMOLLM_DIR=... npm run test:smollm).
import { createDevice } from '../../src/model.js';
import { loadModel } from '../../src/loader.js';
import { chatML } from '../../src/chat.js';
import { assert, maxAbsDiff, run, test } from './harness.js';

const ref = await (await fetch('../fixtures/smollm2-135m-ref.json')).json();
const refLast = new Float32Array(await (await fetch('../fixtures/smollm2-135m-last-logits.bin')).arrayBuffer());
const { device, adapter } = await createDevice();
let model, tokenizer;

test('load SmolLM2-135M-Instruct (streamed safetensors -> GPU)', async () => {
  const t0 = performance.now();
  ({ model, tokenizer } = await loadModel('/models/smollm2-135m', { device, cache: false, maxSeqLen: 1024 }));
  const mb = (b) => (b / 2 ** 20).toFixed(0);
  return `${((performance.now() - t0) / 1000).toFixed(1)} s, weights ${mb(model.bytes.weights)} MiB, ` +
    `KV cache ${mb(model.bytes.kv)} MiB, ${model.dispatchesPerToken} dispatches/token, adapter ${adapter.info?.architecture || '?'}`;
});

test('chat template + tokenizer reproduce the transformers prompt ids', async () => {
  const text = chatML(ref.messages, { system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' });
  assert(text === ref.prompt_text, 'chat template text differs');
  const ids = tokenizer.encode(text);
  assert(ids.join(',') === ref.prompt_ids.join(','), `ids differ:\n${ids}\n${ref.prompt_ids}`);
  return `${ids.length} tokens`;
});

test('logits after the prompt match transformers', async () => {
  const t0 = performance.now();
  const { result } = await model.prefill(ref.prompt_ids);
  const ms = performance.now() - t0;
  const { diff, at } = maxAbsDiff(result, refLast);
  let top = 0;
  for (let i = 1; i < result.length; i++) if (result[i] > result[top]) top = i;
  assert(diff < 2e-3, `max |diff| ${diff} at token ${at}`);
  assert(top === ref.greedy[0], `top token ${top} != ${ref.greedy[0]}`);
  return `max |Δlogit| ${diff.toExponential(1)} (logit range ${Math.min(...refLast).toFixed(1)}..${Math.max(...refLast).toFixed(1)}), ` +
    `prefill ${ref.prompt_ids.length} tok in ${ms.toFixed(0)} ms`;
});

test('greedy continuation is token-for-token identical', async () => {
  let { result: next } = await model.prefill(ref.prompt_ids, { greedy: true });
  const out = [next];
  const t0 = performance.now();
  while (out.length < ref.greedy.length) out.push((next = await model.forwardGreedy(next)));
  const secs = (performance.now() - t0) / 1000;
  const same = out.filter((t, i) => t === ref.greedy[i]).length;
  assert(same === ref.greedy.length, `first mismatch at ${out.findIndex((t, i) => t !== ref.greedy[i])}: ${tokenizer.decode(out)}`);
  return `${same}/${ref.greedy.length} identical: "${tokenizer.decode(out)}"; decode ${((out.length - 1) / secs).toFixed(1)} tok/s`;
});

await run('SmolLM2-135M-Instruct vs transformers');
