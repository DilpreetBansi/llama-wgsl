"""Write tests/fixtures/tokenizer_cases.json: strings and the ids Hugging Face `tokenizers` produces for them."""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

tok_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/smollm2-tokenizer.json"
tok = Tokenizer.from_file(tok_path)
cases = [
    "Hello, world!", "The capital of France is Paris.", "  leading and   multiple   spaces ",
    "Numbers: 12345 and 3.14159, year 2026.", "tabs\tand\nnewlines\n\n\nend", "emoji 🙂👍🏽 and accents: café, naïve, Zürich",
    "中文字符和日本語のテキスト", "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "<|im_start|>user\nWhat is a KV cache?<|im_end|>\n<|im_start|>assistant\n", "it's they're we've I'm you'll he'd",
    "URL: https://example.com/path?q=1&x=2#frag", "   ", "", "a", "!!!???...", "x" * 50,
    "Mixed123Digits456And789Letters", "email@domain.co.uk", " non-breaking spaces", "résumé — em dash – en dash",
]
out = [{"text": c, "ids": tok.encode(c, add_special_tokens=False).ids} for c in cases]
Path("tests/fixtures/tokenizer_cases.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(len(out), "cases")
