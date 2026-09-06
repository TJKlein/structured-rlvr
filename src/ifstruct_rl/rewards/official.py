"""A2 rewards: official IFStruct validator, graded.

``reward_funcs = [format_or_shape, dense_schema, official_binary]``
``reward_weights = [1.0, 1.0, 2.0]``
"""

from __future__ import annotations

import json
from typing import Any

from ifstruct.validator import validate_response


def _content(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def _schema(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def _count(raw: Any):
    if isinstance(raw, (int, list)):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _key(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in {"1", "true", "yes"}
    return bool(raw)


def score_completion(text: str, **spec) -> dict:
    result = validate_response(
        response=text,
        json_schema=_schema(spec.get("json_schema", spec.get("json_schema_str"))),
        top_level_count=_count(spec.get("top_level_count", spec.get("top_level_count_str"))),
        require_no_commentary=_bool(spec["require_no_commentary"]),
        output_format=str(spec["output_format"]),
        top_level_key=_key(spec.get("top_level_key")),
        require_wrapper_key=_bool(spec["require_wrapper_key"]),
        require_code_block=_bool(spec["require_code_block"]),
    )
    return {
        "passed": result.passed,
        "score": result.score,
        "errors": result.errors,
        "details": result.details,
    }


SPEC_KEYS = (
    "json_schema",
    "json_schema_str",
    "top_level_count",
    "top_level_count_str",
    "top_level_key",
    "require_wrapper_key",
    "require_code_block",
    "require_no_commentary",
    "output_format",
)


def _row_spec(kwargs: dict, index: int) -> dict:
    out = {}
    for key in SPEC_KEYS:
        if key not in kwargs:
            continue
        value = kwargs[key]
        try:
            out[key] = value[index]
        except Exception:
            out[key] = value
    return out


def _is_shape_error(err: str) -> bool:
    """Wrap / count / fence / commentary / parse — not schema field errors."""
    if err.startswith("extraneous field") or err.startswith("required field"):
        return False
    if err.startswith("expected ") and "got" in err:
        return False
    needles = (
        "code block",
        "text outside",
        "Expected wrapped object",
        "Expected bare list",
        "Expected top-level key",
        "Expected ",
        "Unclosed code block",
        "JSON parse error",
        "YAML parsing error",
        "No valid JSON",
        "No valid YAML",
        "Trailing content",
        "JSON/flow style",
    )
    return any(n in err for n in needles)


def format_or_shape_reward(completions, **kwargs) -> list[float]:
    out = []
    for i, completion in enumerate(completions):
        scored = score_completion(_content(completion), **_row_spec(kwargs, i))
        if scored["passed"]:
            out.append(1.0)
        elif any(_is_shape_error(err) for err in scored["errors"]):
            out.append(0.0)
        else:
            out.append(1.0)
    return out


def dense_schema_reward(completions, **kwargs) -> list[float]:
    out = []
    for i, completion in enumerate(completions):
        scored = score_completion(_content(completion), **_row_spec(kwargs, i))
        ratio = scored["details"].get("schema_match_ratio")
        out.append(float(ratio) if ratio is not None else 0.0)
    return out


def official_binary_reward(completions, **kwargs) -> list[float]:
    out = []
    for i, completion in enumerate(completions):
        scored = score_completion(_content(completion), **_row_spec(kwargs, i))
        out.append(1.0 if scored["passed"] else 0.0)
    return out


REWARD_FUNCS = [format_or_shape_reward, dense_schema_reward, official_binary_reward]
REWARD_WEIGHTS = [1.0, 1.0, 2.0]
