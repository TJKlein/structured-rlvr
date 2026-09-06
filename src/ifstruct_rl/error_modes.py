"""Tag official IFStruct errors and turn them into SDPO 'mind the gap' hints.

After the 2,000-prompt eval, ``python -m ifstruct_rl.error_modes`` counts leftover
failure modes. SDPO's self-teacher then sees a standing announcement (the
known gaps) plus a hint for this attempt's errors — not only the raw checker
dump.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# First match order is stable; an attempt can hit several modes.
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("schema_dump", ("expected array, got dict", "expected object, got dict")),
    (
        "wrapper",
        (
            "Expected bare list",
            "Expected wrapped object",
            "Expected top-level key",
        ),
    ),
    ("extra_keys", ("extraneous field", "additional properties", "unexpected")),
    ("missing_field", ("required field", "is a required property", "required property")),
    ("enum", ("not in allowed values",)),
    (
        "item_count",
        ("array has", "items, maximum", "items, minimum", "minItems", "maxItems"),
    ),
    ("bounds", ("greater than maximum", "less than minimum")),
    (
        "fence",
        (
            "Unclosed code block",
            "Expected JSON output, got YAML",
            "Expected YAML output, got JSON",
            "Cookbook format",
        ),
    ),
    (
        "parse",
        (
            "JSON parse",
            "YAML parsing",
            "No valid JSON",
            "No valid YAML",
            "got NoneType",
            "no parseable JSON",
            "no object to check",
        ),
    ),
]

HINTS: dict[str, str] = {
    "schema_dump": "Mind the gap: write the data instance, not a JSON Schema wrapper.",
    "wrapper": "Mind the gap: bare list vs wrapped object — match the asked top-level shape.",
    "extra_keys": "Mind the gap: extra keys fail the exam even when a cheap schema score looks fine.",
    "missing_field": "Mind the gap: every required field must be present.",
    "enum": "Mind the gap: enums are closed lists; do not invent a nearby word.",
    "item_count": "Mind the gap: item count is a hard range, not a suggestion.",
    "bounds": "Mind the gap: numeric min/max are checked.",
    "fence": "Mind the gap: JSON vs YAML fencing must match the asked format.",
    "parse": "Mind the gap: the body must parse as the asked format.",
    "other": "Mind the gap: the official checker is stricter than the train scores.",
}

# Probe leftovers until the 2,000-row catalog overwrites them.
DEFAULT_STANDING = (
    "schema_dump",
    "wrapper",
    "extra_keys",
    "enum",
    "item_count",
    "fence",
)


def classify(errors: list[str] | None) -> list[str]:
    blob = " ".join(errors or [])
    low = blob.lower()
    hits = [mode for mode, needles in RULES if any(n.lower() in low for n in needles)]
    return hits or (["other"] if blob.strip() else [])


def attempt_hints(errors: list[str] | None) -> str:
    modes = classify(errors)
    if not modes:
        return ""
    return "\n".join(HINTS[m] for m in modes if m in HINTS)


def standing_announcement(modes: list[str] | None = None) -> str:
    use = list(modes) if modes else list(DEFAULT_STANDING)
    lines = [HINTS[m] for m in use if m in HINTS]
    return "Mind the gap.\n" + "\n".join(lines)


def load_catalog(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return list(DEFAULT_STANDING)
    data = json.loads(path.read_text())
    modes = data.get("standing_modes") or []
    return [m for m in modes if m in HINTS] or list(DEFAULT_STANDING)


def _records(payload: dict) -> list[dict]:
    if isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload.get("summary"), dict) and "records" in payload:
        return payload["records"]
    return []


def analyze_eval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    recs = _records(payload)
    fail = [r for r in recs if not r.get("passed")]
    mode_counts: Counter[str] = Counter()
    by_format: dict[str, Counter[str]] = {"json": Counter(), "yaml": Counter()}
    for rec in fail:
        modes = classify(rec.get("errors") or [])
        mode_counts.update(modes)
        fmt = rec.get("output_format")
        if fmt in by_format:
            by_format[fmt].update(modes)
    standing = [m for m, _n in mode_counts.most_common(6)]
    return {
        "source": str(path),
        "n": len(recs),
        "fail_n": len(fail),
        "mode_counts": dict(mode_counts),
        "json_fail_modes": dict(by_format["json"]),
        "yaml_fail_modes": dict(by_format["yaml"]),
        "standing_modes": standing or list(DEFAULT_STANDING),
        "standing_text": standing_announcement(standing or list(DEFAULT_STANDING)),
    }


def merge_catalogs(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    for rep in reports:
        total.update(rep.get("mode_counts") or {})
    standing = [m for m, _n in total.most_common(6)] or list(DEFAULT_STANDING)
    return {
        "standing_modes": standing,
        "standing_text": standing_announcement(standing),
        "combined_counts": dict(total),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evals", nargs="+", help="eval JSON with records[]")
    parser.add_argument("--out", default="artifacts/mind_the_gap.json")
    args = parser.parse_args()
    reports = []
    for raw in args.evals:
        path = Path(raw)
        if not path.exists():
            print(f"skip missing {path}")
            continue
        reports.append(analyze_eval(path))
    catalog = merge_catalogs(reports) if reports else {
        "standing_modes": list(DEFAULT_STANDING),
        "standing_text": standing_announcement(),
        "combined_counts": {},
        "reports": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2))
    print(json.dumps(catalog, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
