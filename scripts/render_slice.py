"""Apply a fitted Jacobian lens and render interactive slice pages.

Usage:
    uv run python scripts/render_slice.py --prompt-slug ascii-face
    uv run python scripts/render_slice.py --prompt "Your own text here"
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import torch
import transformers

import jlens
from jlens.examples import EXAMPLES, resolve_prompt
from jlens.vis import build_page, compute_slice

GLOSS_PATH = Path("assets/qwen_gloss.json.gz")


def load_gloss() -> dict[int, str]:
    if not GLOSS_PATH.exists():
        return {}
    with gzip.open(GLOSS_PATH, "rt", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lens", default="out/jacobian_lens.pt")
    parser.add_argument("--prompt-slug", default=None, help="slug from jlens.examples.EXAMPLES")
    parser.add_argument("--prompt", default=None, help="raw prompt text")
    parser.add_argument(
        "--out-dir", default="out/slices", help="directory for rendered HTML pages"
    )
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args()

    if (args.prompt_slug is None) == (args.prompt is None):
        parser.error("pass exactly one of --prompt-slug / --prompt")

    jlens.configure_logging()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(args.lens)
    print(f"lens loaded: {lens.n_layers} layers, d_model={lens.d_model}")

    if args.prompt_slug is not None:
        example = next(e for e in EXAMPLES if e.slug == args.prompt_slug)
        prompt = resolve_prompt(example, tokenizer)
        title = example.section
        description = example.description
    else:
        prompt = args.prompt
        title = "Custom prompt"
        description = ""

    print(f"computing slice ({len(tokenizer.encode(prompt))} tokens) ...")
    slice_data = compute_slice(model, lens, prompt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.prompt_slug or "custom"
    page, _, _ = build_page(
        slice_data,
        prompt,
        title=f"{title} — {args.model}",
        description=description,
        mode="embed",
        alt_token=load_gloss(),
    )
    page_path = out_dir / f"{slug}.html"
    page_path.write_text(page, encoding="utf-8")
    print(f"wrote {page_path}")

    if args.open:
        subprocess.run(["open", str(page_path)], check=True)


if __name__ == "__main__":
    main()
