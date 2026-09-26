"""Reference outputs for a real checkpoint, used by tests/browser/smollm.test.js.

Runs HuggingFaceTB/SmolLM2-135M-Instruct in float32 with transformers and records, for a short
chat prompt: the prompt token ids (checks the JS tokenizer and chat template), the logits after
the prompt, and a 32-token greedy continuation.

Usage: python tools/make_reference.py [model_dir_or_repo] [out_dir]
"""

import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
MESSAGES = [{"role": "user", "content": "Explain what a GPU is to a ten-year-old, in two sentences."}]
NEW_TOKENS = 32

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
text = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids

with torch.no_grad():
    last = model(ids).logits[0, -1].float().numpy()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False,
                         attention_mask=torch.ones_like(ids), eos_token_id=None, pad_token_id=tok.eos_token_id)
    secs = time.perf_counter() - t0
greedy = out[0, ids.shape[1]:].tolist()

os.makedirs(OUT, exist_ok=True)
last.astype("<f4").tofile(os.path.join(OUT, "smollm2-135m-last-logits.bin"))
with open(os.path.join(OUT, "smollm2-135m-ref.json"), "w") as f:
    json.dump({
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "messages": MESSAGES,
        "prompt_text": text,
        "prompt_ids": ids[0].tolist(),
        "greedy": greedy,
        "greedy_text": tok.decode(greedy),
        "torch_cpu_tokens_per_s": NEW_TOKENS / secs,
    }, f, indent=1)
print(f"{ids.shape[1]} prompt tokens; greedy: {tok.decode(greedy)!r}; torch CPU {NEW_TOKENS / secs:.1f} tok/s")
