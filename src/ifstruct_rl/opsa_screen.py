"""Cheap OPSA screen: do IFStruct failures look like low-logprob tokens?

This is not OPSA training. OPSA (Ding & Zhang, arXiv:2608.31046, Eq. 5) samples
on-policy, updates only the lowest-20% logp tokens, and assigns entropy-adaptive
advantages in [-1.0, -0.5]. We only check whether that mechanism could even see
the IFStruct error modes.

Go: failed completions have a clearly lower lowest-20% logp (gap >= 0.3 nats)
AND a higher fraction of tokens below TAIL_LOGP. Then a 40-step OPSA train is
worth it.

No-go: failures are as confident as (or more than) passes. OPSA would then
preserve the high-p extra keys / wrong wrappers (paper Fig. 6d) and skip them.

Logprobs here come from greedy decode (the policy mode). OPSA trains on sampled
rollouts; greedy is the conservative test of "is the wrong answer already the
head of the distribution?"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


LOWEST_FRAC = 0.20  # paper S_lowest20
TAIL_LOGP = -2.0  # tokens less likely than e^-2 ≈ 0.135; diagnostic, not in paper
GAP_NATS = 0.3


def _lowest_frac_mean(logps: list[float], frac: float = LOWEST_FRAC) -> float:
    k = max(1, math.ceil(frac * len(logps)))
    return statistics.mean(sorted(logps)[:k])


def _bucket(records: list[dict], pred) -> dict:
    sub = [r for r in records if pred(r)]
    if not sub:
        return {"count": 0}
    with_lp = [r for r in sub if r.get("token_logprobs")]
    means = [statistics.mean(r["token_logprobs"]) for r in with_lp]
    lowest20 = [_lowest_frac_mean(r["token_logprobs"]) for r in with_lp]
    tails = [
        sum(1 for p in r["token_logprobs"] if p < TAIL_LOGP) / len(r["token_logprobs"])
        for r in with_lp
    ]
    return {
        "count": len(sub),
        "pass_rate": sum(1 for r in sub if r["passed"]) / len(sub),
        "mean_of_mean_logp": statistics.mean(means) if means else None,
        "mean_of_lowest20_logp": statistics.mean(lowest20) if lowest20 else None,
        "mean_frac_tail": statistics.mean(tails) if tails else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-results", default="results/baseline_probe128.json")
    parser.add_argument("--out", default="results/opsa_screen.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.from_results).read_text())
    records = payload["records"]

    report = {
        "source": args.from_results,
        "n": len(records),
        "passed": _bucket(records, lambda r: r["passed"]),
        "failed": _bucket(records, lambda r: not r["passed"]),
        "json": _bucket(records, lambda r: r["output_format"] == "json"),
        "yaml": _bucket(records, lambda r: r["output_format"] == "yaml"),
        "escaping_entity": _bucket(
            records, lambda r: "escaping" in r.get("entity_type", "")
        ),
        "decision_rule": {
            "go": (
                "failed.mean_of_lowest20_logp is clearly lower than passed "
                f"(gap >= {GAP_NATS} nats) AND failed.mean_frac_tail is higher. "
                "Then a 40-step OPSA train is worth it."
            ),
            "nogo": (
                "failed lowest-20% logp >= passed, or failed tail mass is not higher. "
                "Skip OPSA training. Canonical OPSA (DripNowhy/On-Policy-Self-Adaptation) "
                "selects the lowest 20% actor logp tokens and assigns entropy-adaptive "
                "advantages in [-1.0, -0.5]; it does not use a task reward."
            ),
        },
    }

    p = report["passed"].get("mean_of_lowest20_logp")
    f = report["failed"].get("mean_of_lowest20_logp")
    p_tail = report["passed"].get("mean_frac_tail")
    f_tail = report["failed"].get("mean_frac_tail")
    if p is not None and f is not None:
        gap = p - f  # failed more negative → positive gap
        report["mean_logp_gap_passed_minus_failed"] = (
            (report["passed"].get("mean_of_mean_logp") or 0)
            - (report["failed"].get("mean_of_mean_logp") or 0)
        )
        report["lowest20_gap_passed_minus_failed"] = gap
        go_gap = gap >= GAP_NATS
        go_tail = (
            p_tail is not None and f_tail is not None and f_tail > p_tail
        )
        report["recommendation"] = (
            "GO_OPSA_SCREEN_TRAIN" if (go_gap and go_tail) else "NOGO_SKIP_OPSA"
        )
    else:
        report["recommendation"] = "INSUFFICIENT_DATA"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    print("recommendation:", report["recommendation"])


if __name__ == "__main__":
    main()
