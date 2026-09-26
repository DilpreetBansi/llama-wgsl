// End-to-end parity with Hugging Face transformers on tiny random checkpoints
// (tools/make_tiny_model.py writes the checkpoints and the reference outputs).
import { createDevice, LlamaModel, parseConfig } from '../../src/model.js';
import { assert, maxAbsDiff, run, test } from './harness.js';

const { device } = await createDevice();
const TOL = 1e-4;

async function load(name) {
  const base = `../fixtures/${name}`;
  const cfg = parseConfig(await (await fetch(`${base}/config.json`)).json());
  const ref = await (await fetch(`${base}/ref.json`)).json();
  const logits = new Float32Array(await (await fetch(`${base}/ref_logits.bin`)).arrayBuffer());
  const model = new LlamaModel(device, cfg, { maxSeqLen: 128 });
  await model.loadWeights((await fetch(`${base}/model.safetensors`)).body);
  return { model, ref, logits, V: cfg.V };
}

for (const name of ['tiny-llama-bf16', 'tiny-llama-f32-untied', 'tiny-qwen2-f16']) {
  test(`${name}: logits at every position`, async () => {
    const { model, ref, logits, V } = await load(name);
    let worst = 0;
    for (let t = 0; t < ref.T; t++) {
      const got = await model.forward(ref.tokens[t]);
      const { diff, at } = maxAbsDiff(got, logits.subarray(t * V, (t + 1) * V));
      assert(diff < TOL, `position ${t}, token ${at}: |diff| = ${diff}`);
      worst = Math.max(worst, diff);
    }
    model.destroy();
    return `max |Δlogit| ${worst.toExponential(1)} over ${ref.T} positions`;
  });

  test(`${name}: greedy decoding matches generate()`, async () => {
    const { model, ref } = await load(name);
    let { result: next } = await model.prefill(ref.greedy_prompt, { greedy: true });
    const out = [next];
    while (out.length < ref.greedy.length) out.push((next = await model.forwardGreedy(next)));
    model.destroy();
    const same = out.filter((t, i) => t === ref.greedy[i]).length;
    assert(same === ref.greedy.length, `got ${out.join(',')}\nwant ${ref.greedy.join(',')}`);
    return `${same}/${ref.greedy.length} tokens identical`;
  });

  test(`${name}: prefix-cached prefill equals a full recompute`, async () => {
    const { model, ref, logits, V } = await load(name);
    const want = (t) => logits.subarray(t * V, (t + 1) * V);
    let r = await model.prefill(ref.tokens.slice(0, 30));
    assert(r.reused === 0 && r.computed === 30, `first prefill reused ${r.reused}`);
    assert(maxAbsDiff(r.result, want(29)).diff < TOL, 'logits after 30 tokens');
    r = await model.prefill(ref.tokens.slice(0, 20)); // shrink: only the last token is recomputed
    assert(r.reused === 19 && r.computed === 1, `shrink reused ${r.reused}`);
    assert(maxAbsDiff(r.result, want(19)).diff < TOL, 'logits after shrinking to 20 tokens');
    const edited = ref.tokens.slice(0, 40);
    r = await model.prefill(edited); // grow again: tokens 20..39 are computed
    assert(r.reused === 20 && r.computed === 20, `grow reused ${r.reused}`);
    assert(maxAbsDiff(r.result, want(39)).diff < TOL, 'logits after growing to 40 tokens');
    model.destroy();
  });
}

await run('parity with transformers');
