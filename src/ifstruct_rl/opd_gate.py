"""Go/no-go for on-policy distillation: teacher must beat cookbook GRPO on the exam.

Fu et al. (arXiv:2609.04172) need a teacher with new capability. Same-family
size-up often teaches nothing at token KL. We only consider OPD if
``LiquidAI/LFM2.5-1.2B-Instruct`` is at least 3 points above A0 on the official
128-probe. This script does not train OPD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


A0_PROBE_PASS_RATE = 45 / 128
MARGIN = 0.03  # 3 points, same bar as the rest of the IFStruct comparisons


def decide(teacher_pass_rate: float, a0_pass_rate: float = A0_PROBE_PASS_RATE) -> dict:
    delta = teacher_pass_rate - a0_pass_rate
    go = delta >= MARGIN
    return {
        "teacher": "LiquidAI/LFM2.5-1.2B-Instruct",
        "teacher_pass_rate": teacher_pass_rate,
        "a0_pass_rate": a0_pass_rate,
        "delta": delta,
        "margin": MARGIN,
        "recommendation": "consider_opd" if go else "skip_opd",
        "reason": (
            "teacher is ≥3 pts above cookbook GRPO on the official probe"
            if go
            else "teacher is not clearly above cookbook GRPO; OPD would distill a weaker or tied exam policy"
        ),
    }


def _rate_from_eval(path: Path) -> float:
    data = json.loads(path.read_text())
    if "summary" in data and "pass_rate" in data["summary"]:
        return float(data["summary"]["pass_rate"])
    if "pass_rate" in data:
        return float(data["pass_rate"])
    records = data.get("records") or []
    if not records:
        raise ValueError(f"no pass_rate in {path}")
    return sum(bool(r["passed"]) for r in records) / len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_json", help="1.2B instruct probe eval JSON or summary sidecar")
    parser.add_argument("--out", default="artifacts/opd_gate.json")
    args = parser.parse_args()
    path = Path(args.eval_json)
    rate = _rate_from_eval(path)
    report = decide(rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
