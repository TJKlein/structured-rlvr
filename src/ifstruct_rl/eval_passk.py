"""T=1.0 pass@k on the frozen IFStruct probe (plan D19, 128-prompt subset)."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from ifstruct_rl.data import load_test_examples, select_probe
from ifstruct_rl.generate import generate_k, load_model
from ifstruct_rl.passk import pass_at_k
from ifstruct_rl.score import score_example


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--n", type=int, default=128, help="Probe size")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/pass8_probe128.json")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN is unset; public downloads may still work")

    examples = select_probe(load_test_examples(), n=args.n)
    print(
        f"model={args.model} probe={len(examples)} k={args.k} "
        f"T={args.temperature} max_new_tokens={args.max_new_tokens}"
    )
    model, tokenizer = load_model(args.model)

    records = []
    t0 = time.perf_counter()
    for i, example in enumerate(examples, start=1):
        step_t = time.perf_counter()
        gens = generate_k(
            model,
            tokenizer,
            example.prompt,
            k=args.k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + int(example.seed),
        )
        samples = []
        n_pass = 0
        for gen in gens:
            scored = score_example(example, gen.text)
            if scored["passed"]:
                n_pass += 1
            samples.append(
                {
                    "response": gen.text,
                    "n_tokens": gen.n_tokens,
                    "truncated": gen.truncated,
                    "passed": scored["passed"],
                    "errors": scored["errors"][:8],
                }
            )
        rec = {
            "seed": example.seed,
            "entity_type": example.entity_type,
            "output_format": example.output_format,
            "n_pass": n_pass,
            "k": args.k,
            "pass_at_1": pass_at_k(args.k, n_pass, 1),
            "pass_at_k": pass_at_k(args.k, n_pass, args.k),
            "latency_ms": (time.perf_counter() - step_t) * 1000.0,
            "samples": samples,
        }
        records.append(rec)
        print(
            f"[{i}/{len(examples)}] seed={example.seed} {example.output_format} "
            f"{n_pass}/{args.k} pass@1={rec['pass_at_1']:.3f} "
            f"{rec['latency_ms']:.0f}ms",
            flush=True,
        )

    summary = _summary(records, args)
    summary["elapsed_s"] = time.perf_counter() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"wrote {summary_path}")


def _summary(records: list[dict], args: argparse.Namespace) -> dict:
    n = len(records)
    by_format: dict[str, dict] = {}
    for fmt in ("json", "yaml"):
        sub = [r for r in records if r["output_format"] == fmt]
        if sub:
            by_format[fmt] = {
                "n": len(sub),
                "pass_at_1": sum(r["pass_at_1"] for r in sub) / len(sub),
                "pass_at_k": sum(r["pass_at_k"] for r in sub) / len(sub),
            }
    return {
        "model": args.model,
        "n": n,
        "k": args.k,
        "temperature": args.temperature,
        "seed": args.seed,
        "pass_at_1": sum(r["pass_at_1"] for r in records) / n if n else 0.0,
        "pass_at_k": sum(r["pass_at_k"] for r in records) / n if n else 0.0,
        "any_pass_frac": sum(r["n_pass"] > 0 for r in records) / n if n else 0.0,
        "mean_n_pass": sum(r["n_pass"] for r in records) / n if n else 0.0,
        "by_format": by_format,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "pass@1 is the unbiased estimator c/k at T=1.0 (not greedy). "
            "Compare to results/baseline_probe128.json greedy pass_rate."
        ),
    }


if __name__ == "__main__":
    main()
