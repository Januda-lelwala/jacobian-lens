"""Fit a Jacobian lens on a small HuggingFace model (Mac/MPS friendly).

Usage:
    uv run python scripts/fit_lens.py --n-prompts 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--out", default="out/jacobian_lens.pt")
    parser.add_argument("--checkpoint", default="out/ckpt.pt")
    args = parser.parse_args()

    jlens.configure_logging()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"loading {args.model} on {args.device} ({args.dtype}) ...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype
    ).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Gradients are only needed w.r.t. residual activations; freezing params
    # lets ActivationRecorder root the autograd graph at the first source
    # layer and skips all weight-gradient computation (roughly halves cost).
    for param in hf_model.parameters():
        param.requires_grad_(False)

    model = jlens.from_hf(hf_model, tokenizer)
    prompts = load_wikitext_prompts(args.n_prompts)
    print(f"fitting on {len(prompts)} prompts, max_seq_len={args.max_seq_len}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint,
    )
    elapsed = time.time() - start

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.out)
    print(f"done in {elapsed:.0f}s ({elapsed / len(prompts):.1f}s/prompt)")
    print(f"lens saved to {args.out}")


if __name__ == "__main__":
    main()
