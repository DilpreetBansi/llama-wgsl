# Test fixtures

| File | What it is | Made by |
|---|---|---|
| `tiny-*/` | Tiny random checkpoints (`config.json`, `model.safetensors`) with reference logits for 48 tokens (`ref_logits.bin`, f32 little-endian, `[48, vocab]`) and a greedy continuation (`ref.json`) | `python tools/make_tiny_model.py` |
| `smollm2-135m-ref.json`, `smollm2-135m-last-logits.bin` | SmolLM2-135M-Instruct: chat prompt ids, logits after the prompt, 32-token greedy continuation | `python tools/make_reference.py` |
| `smollm2-tokenizer.json` | The SmolLM2 tokenizer, from [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (Apache-2.0) | downloaded |
| `tokenizer_cases.json` | Strings and the ids Hugging Face `tokenizers` gives them | `python tools/make_tokenizer_cases.py` |

References were produced with transformers 5.17 and PyTorch 2.14 (CPU, float32).
