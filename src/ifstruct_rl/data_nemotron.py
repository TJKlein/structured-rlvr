"""Nemotron structured-output rows, filtered and augmented like the Liquid cookbook."""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset, load_dataset
from jsonschema import Draft7Validator

DATASET_ID = "nvidia/Nemotron-RL-instruction_following-structured_outputs"
TRAIN_SAMPLES = 1000
MAX_PROMPT_CHARS = 6000


def adapt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["responses_create_params"]["input"],
        "wants_fence": False,
    }


def prompt_fits(row: dict[str, Any]) -> bool:
    return sum(len(message["content"]) for message in row["prompt"]) <= MAX_PROMPT_CHARS


def schema_is_valid(row: dict[str, Any]) -> bool:
    if '"$ref"' in row["schema_str"]:
        return False
    try:
        Draft7Validator.check_schema(json.loads(row["schema_str"]))
    except Exception:
        return False
    return True


def augment(row: dict[str, Any], idx: int) -> dict[str, Any]:
    variant = idx % 5
    if variant > 2:
        return row
    messages = [dict(message) for message in row["prompt"]]
    if variant == 2:
        n = 2 + idx % 3
        array_schema = {
            "type": "array",
            "items": json.loads(row["schema_str"]),
            "minItems": n,
            "maxItems": n,
        }
        messages[-1]["content"] += f"\n\nReturn a JSON array containing exactly {n} of these objects."
        return {**row, "prompt": messages, "schema_str": json.dumps(array_schema)}
    messages[-1]["content"] += "\n\nReturn the output inside a fenced code block."
    return {**row, "prompt": messages, "wants_fence": True}


def load_nemotron_cookbook(n: int = TRAIN_SAMPLES) -> Dataset:
    ds = load_dataset(DATASET_ID, split=f"train[:{n}]")
    ds = ds.map(adapt_row, remove_columns=["responses_create_params"])
    ds = ds.filter(prompt_fits)
    ds = ds.filter(schema_is_valid)
    ds = ds.map(augment, with_indices=True)
    return ds
