"""Cookbook-vs-official disagreement on stored greedy evals.

hack = combined cookbook reward > 0.8 AND official validate_response failed.
Cookbook json_format is JSON-oriented; YAML rows still go through field/schema terms.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ifstruct_rl.data import load_test_examples
from ifstruct_rl.rewards.combined import cookbook_components


D20_NEEDLES = (
    "extraneous field",
    "Unclosed code block",
    "Trailing content",
    "text outside",
    "duplicate",
)


def _schema_fields_count(schema: dict) -> int:
    items = schema.get("items") if schema.get("type") == "array" else schema
    if not isinstance(items, dict):
        return 0
    props = items.get("properties") or {}
    return len(props)


def _d20(errors: list[str]) -> bool:
    blob = " | ".join(errors)
    return any(n in blob for n in D20_NEEDLES)


def analyze(eval_path: Path) -> dict:
    payload = json.loads(eval_path.read_text())
    records = payload["records"]
    by_seed = {ex.seed: ex for ex in load_test_examples()}
    rows = []
    for rec in records:
        ex = by_seed.get(int(rec["seed"]))
        if ex is None:
            continue
        schema = ex.json_schema
        scores = cookbook_components(
            rec["response"],
            wants_fence=bool(ex.require_code_block),
            schema_fields_count=_schema_fields_count(schema),
            schema_str=json.dumps(schema),
        )
        official = bool(rec["passed"])
        high = scores["combined"] > 0.8
        rows.append(
            {
                "seed": rec["seed"],
                "output_format": rec["output_format"],
                "official_pass": official,
                "cookbook_combined": scores["combined"],
                "cookbook_high": high,
                "hack": high and not official,
                "d20_error": _d20(rec.get("errors") or []),
                "first_error": (rec.get("errors") or [""])[0][:120],
            }
        )
    n = len(rows)
    n_high = sum(r["cookbook_high"] for r in rows)
    n_hack = sum(r["hack"] for r in rows)
    json_rows = [r for r in rows if r["output_format"] == "json"]
    n_high_json = sum(r["cookbook_high"] for r in json_rows)
    n_hack_json = sum(r["hack"] for r in json_rows)
    fail = [r for r in rows if not r["official_pass"]]
    return {
        "n": n,
        "official_pass_rate": sum(r["official_pass"] for r in rows) / n if n else 0.0,
        "cookbook_high_n": n_high,
        "hack_n": n_hack,
        "hack_rate_given_cookbook_high": n_hack / n_high if n_high else None,
        "json_n": len(json_rows),
        "json_hack_rate_given_cookbook_high": (
            n_hack_json / n_high_json if n_high_json else None
        ),
        "official_fail_n": len(fail),
        "official_fail_d20_frac": (
            sum(r["d20_error"] for r in fail) / len(fail) if fail else None
        ),
        "top_fail_errors": Counter(r["first_error"] for r in fail).most_common(8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evals", nargs="+", help="eval JSON files with records[]")
    parser.add_argument("--out", default="artifacts/hack_rate.json")
    args = parser.parse_args()
    report = {Path(p).stem: analyze(Path(p)) for p in args.evals}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
