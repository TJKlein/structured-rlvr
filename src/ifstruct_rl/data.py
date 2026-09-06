"""Load IFStruct test rows and the frozen 128-prompt probe."""

from __future__ import annotations

from typing import Any, Protocol


PROBE_N = 128


class HasSeed(Protocol):
    seed: int


def _row_to_example(row: dict[str, Any]):
    from ifstruct.dataset import IfStructExample

    schema = row["json_schema"]
    if isinstance(schema, str):
        import json

        schema = json.loads(schema)
    count = row["top_level_count"]
    if isinstance(count, str):
        import json

        count = json.loads(count)
    return IfStructExample(
        seed=int(row.get("seed", row["doc_id"])),
        entity_type=str(row["entity_type"]),
        prompt=str(row["prompt"]),
        json_schema=schema,
        top_level_count=count,
        top_level_key=row.get("top_level_key"),
        require_wrapper_key=bool(row["require_wrapper_key"]),
        require_code_block=bool(row["require_code_block"]),
        require_no_commentary=bool(row["require_no_commentary"]),
        output_format=str(row["output_format"]),
    )


def load_test_examples():
    from datasets import load_dataset

    ds = load_dataset("LiquidAI/ifstruct-v1.0", split="test")
    return [_row_to_example(dict(row)) for row in ds]


def select_probe(examples: list[HasSeed], n: int = PROBE_N) -> list[HasSeed]:
    """Plan D37: even seeds 0, 2, ..., 254 (128 prompts). Fallback: first n even seeds."""
    even = sorted(
        (ex for ex in examples if ex.seed % 2 == 0),
        key=lambda ex: ex.seed,
    )
    targeted = [ex for ex in even if 0 <= ex.seed <= 254]
    if len(targeted) >= n:
        return targeted[:n]
    if len(even) >= n:
        return even[:n]
    return sorted(examples, key=lambda ex: ex.seed)[:n]
