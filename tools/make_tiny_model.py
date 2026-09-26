"""Build tiny random checkpoints plus reference outputs from Hugging Face transformers.

The browser parity test (tests/browser/parity.html) loads each checkpoint into llama-wgsl, runs
the same token sequence, and compares every position's logits and a greedy continuation with
what transformers produced here. Each variant exercises a different code path:

  tiny-llama-bf16        bf16 weights, tied embeddings, grouped-query attention (4 q / 2 kv heads)
  tiny-llama-f32-untied  f32 weights, separate lm_head, multi-head attention, o_proj bias
  tiny-qwen2-f16         f16 weights, Qwen2 q/k/v biases, 8 q / 2 kv heads, head_dim 16

Usage: python tools/make_tiny_model.py [out_dir]   (default: tests/fixtures)
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

VOCAB, HIDDEN, INTER, LAYERS, SEQ = 384, 128, 352, 3, 48

VARIANTS = {
    "tiny-llama-bf16": dict(arch="llama", dtype=torch.bfloat16, tie=True, heads=4, kv=2, bias=False),
    "tiny-llama-f32-untied": dict(arch="llama", dtype=torch.float32, tie=False, heads=4, kv=4, bias=True),
    "tiny-qwen2-f16": dict(arch="qwen2", dtype=torch.float16, tie=True, heads=8, kv=2, bias=None),
}


def build(name, v, out_root):
    torch.manual_seed(len(name))
    common = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=v["heads"],
        num_key_value_heads=v["kv"],
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=v["tie"],
        initializer_range=0.08,  # larger than default so attention is far from uniform
        bos_token_id=1,
        eos_token_id=2,
    )
    if v["arch"] == "llama":
        cfg = LlamaConfig(**common, rope_theta=50000.0, attention_bias=v["bias"])
        model = LlamaForCausalLM(cfg)
    else:
        cfg = Qwen2Config(**common, rope_theta=1000000.0)
        model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))  # exercise the RMSNorm gain
            elif n.endswith(".bias"):
                p.copy_(0.1 * torch.randn_like(p))

    out = os.path.join(out_root, name)
    model.to(v["dtype"]).save_pretrained(out, safe_serialization=True)
    for extra in ("generation_config.json",):
        path = os.path.join(out, extra)
        if os.path.exists(path):
            os.remove(path)

    # Reference: the saved weights upcast to f32, exactly what the WebGPU engine computes with.
    ref = type(model).from_pretrained(out, dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(7)
    tokens = torch.randint(3, VOCAB, (1, SEQ), generator=g)
    with torch.no_grad():
        logits = ref(tokens).logits[0].float().numpy()
        prompt = tokens[:, :6]
        greedy = ref.generate(prompt, max_new_tokens=24, do_sample=False, eos_token_id=None,
                              pad_token_id=0, attention_mask=torch.ones_like(prompt))[0, 6:]
    logits.astype("<f4").tofile(os.path.join(out, "ref_logits.bin"))
    with open(os.path.join(out, "ref.json"), "w") as f:
        json.dump({"tokens": tokens[0].tolist(), "T": SEQ, "V": VOCAB,
                   "greedy_prompt": prompt[0].tolist(), "greedy": greedy.tolist(),
                   "transformers": __import__("transformers").__version__}, f)
    size = os.path.getsize(os.path.join(out, "model.safetensors"))
    print(f"{name}: {size/1024:.0f} KiB, logits range [{logits.min():.2f}, {logits.max():.2f}]")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    for name, v in VARIANTS.items():
        build(name, v, root)
