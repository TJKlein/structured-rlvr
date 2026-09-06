"""Liquid cookbook GRPO rewards (faithful port of grpo_with_trl_ifstruct.ipynb)."""

from __future__ import annotations

import json
import re

from jsonschema import Draft7Validator


def extract_json(text: str):
    """Parse a completion as raw JSON or as the first valid fenced JSON block.

    Returns (obj, form) with form in {"direct", "fenced", None}.
    """
    if text.startswith(("{", "[")) and text.endswith(("}", "]")):
        try:
            return json.loads(text), "direct"
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip()), "fenced"
        except json.JSONDecodeError:
            continue
    return None, None


def json_format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for completion, fence_requested in zip(completions, kwargs["wants_fence"]):
        _, form = extract_json(completion[-1]["content"].strip())
        requested = "fenced" if fence_requested else "direct"
        rewards.append(0.0 if form is None else 1.0 if form == requested else 0.2)
    return rewards


def field_count_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for completion, expected_count in zip(completions, kwargs["schema_fields_count"]):
        expected_count = int(expected_count)
        obj, _ = extract_json(completion[-1]["content"].strip())
        if isinstance(obj, list):
            dict_items = [item for item in obj if isinstance(item, dict)]
            actual_count = (
                sum(len(item) for item in dict_items) / len(dict_items) if dict_items else None
            )
        elif isinstance(obj, dict):
            actual_count = len(obj)
        else:
            actual_count = None
        if actual_count is None:
            rewards.append(0.0)
        elif expected_count == 0:
            rewards.append(1.0 if actual_count == 0 else 0.0)
        else:
            rewards.append(max(0.0, 1.0 - abs(actual_count - expected_count) / expected_count))
    return rewards


def count_schema_checks(schema) -> int:
    constraint_keywords = (
        "enum",
        "pattern",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "additionalProperties",
    )
    if isinstance(schema, list):
        return sum(count_schema_checks(item) for item in schema)
    if not isinstance(schema, dict):
        return 0
    properties = schema.get("properties")
    required = schema.get("required")
    count = len(properties) if isinstance(properties, dict) else 0
    count += len(required) if isinstance(required, list) else 0
    count += sum(1 for keyword in constraint_keywords if keyword in schema)
    count += sum(
        count_schema_checks(value) for value in schema.values() if isinstance(value, (dict, list))
    )
    return count


def schema_validation_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for completion, schema_text in zip(completions, kwargs["schema_str"]):
        schema = json.loads(schema_text)
        obj, _ = extract_json(completion[-1]["content"].strip())
        if not isinstance(obj, (dict, list)):
            rewards.append(0.0)
            continue
        if (schema.get("type") == "array") != isinstance(obj, list):
            coverage = 0.0
        elif isinstance(obj, dict):
            required = schema.get("required", [])
            coverage = sum(key in obj for key in required) / len(required) if required else 1.0
        elif not obj:
            coverage = 0.0
        else:
            n_min = schema.get("minItems")
            coverage = min(len(obj), n_min) / n_min if n_min else 1.0
            items = schema.get("items")
            required = items.get("required", []) if isinstance(items, dict) else []
            if required:
                coverage *= sum(
                    sum(key in item for key in required) for item in obj if isinstance(item, dict)
                ) / (len(required) * len(obj))
        errors = list(Draft7Validator(schema).iter_errors(obj))
        num_checks = count_schema_checks(schema)
        if isinstance(obj, list) and schema.get("minItems"):
            num_checks += (schema["minItems"] - 1) * count_schema_checks(schema.get("items"))
        partial = coverage * max(0.0, 1.0 - len(errors) / max(1, num_checks))
        rewards.append(0.75 * partial + (0.25 if not errors else 0.0))
    return rewards


REWARD_FUNCS = [json_format_reward, field_count_reward, schema_validation_reward]
REWARD_WEIGHTS = [1.0, 0.5, 2.0]
