"""128-prompt IFStruct baseline for LFM2.5-350M (plan D37 probe)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from ifstruct_rl.data import load_test_examples, select_probe
from ifstruct_rl.generate import generate_one, load_model
from ifstruct_rl.score import score_example


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--n", type=int, default=128, help="Probe size (default 128)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--out", default="results/baseline_probe128.json")
    parser.add_argument("--smoke", type=int, default=0, help="If >0, only run this many prompts")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN is unset; public downloads may still work")

    examples = select_probe(load_test_examples(), n=args.n)
    if args.smoke:
        examples = examples[: args.smoke]
        print(f"smoke: {len(examples)} prompts")

    print(f"model={args.model} probe={len(examples)} max_new_tokens={args.max_new_tokens}")
    model, tokenizer = load_model(args.model)

    records = []
    t0 = time.perf_counter()
    for i, example in enumerate(examples, start=1):
        step_t = time.perf_counter()
        gen = generate_one(
            model,
            tokenizer,
            example.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        scored = score_example(example, gen.text)
        rec = {
            "seed": example.seed,
            "entity_type": example.entity_type,
            "output_format": example.output_format,
            "require_wrapper_key": example.require_wrapper_key,
            "require_code_block": example.require_code_block,
            "require_no_commentary": example.require_no_commentary,
            "prompt": example.prompt,
            "response": gen.text,
            "token_logprobs": gen.token_logprobs,
            "n_tokens": gen.n_tokens,
            "truncated": gen.truncated,
            "latency_ms": (time.perf_counter() - step_t) * 1000.0,
            **scored,
        }
        records.append(rec)
        mark = "PASS" if scored["passed"] else "FAIL"
        print(
            f"[{i}/{len(examples)}] seed={example.seed} {example.output_format} "
            f"{mark} tokens={gen.n_tokens} {rec['latency_ms']:.0f}ms",
            flush=True,
        )
        if not scored["passed"] and scored["errors"]:
            print(f"    {scored['errors'][0][:160]}", flush=True)

    summary = _summary(records, args.model, time.perf_counter() - t0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": records}
    out.write_text(json.dumps(payload, indent=2))
    summary_path = out.with_suffix(".summary.json")
    if summary_path == out:
        summary_path = out.parent / f"{out.stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"wrote {summary_path}")


def _summary(records: list[dict], model: str, elapsed_s: float) -> dict:
    n = len(records)
    n_pass = sum(1 for r in records if r["passed"])
    errors = Counter()
    for rec in records:
        for err in rec.get("errors") or []:
            key = err.split(":")[0][:80]
            errors[key] += 1
    by_format = {}
    for fmt in ("json", "yaml"):
        sub = [r for r in records if r["output_format"] == fmt]
        if sub:
            by_format[fmt] = {
                "passed": sum(1 for r in sub if r["passed"]),
                "total": len(sub),
                "pass_rate": sum(1 for r in sub if r["passed"]) / len(sub),
            }
    return {
        "model": model,
        "n": n,
        "passed": n_pass,
        "pass_rate": n_pass / n if n else 0.0,
        "elapsed_s": elapsed_s,
        "mean_tokens": statistics.mean(r["n_tokens"] for r in records) if records else 0,
        "truncated_frac": sum(1 for r in records if r["truncated"]) / n if n else 0,
        "by_format": by_format,
        "top_errors": errors.most_common(15),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Expected pass_rate ~0.21 on full IFStruct; probe-128 may differ by a few points.",
    }


if __name__ == "__main__":
    main()
