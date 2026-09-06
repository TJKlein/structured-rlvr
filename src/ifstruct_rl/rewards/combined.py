"""Cookbook combined reward helper (format 1.0 + field 0.5 + schema 2.0)."""

from __future__ import annotations

from ifstruct_rl.rewards.cookbook import (
    REWARD_WEIGHTS,
    field_count_reward,
    json_format_reward,
    schema_validation_reward,
)


def as_chat(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def cookbook_components(text: str, *, wants_fence, schema_fields_count, schema_str) -> dict[str, float]:
    chat = [as_chat(text)]
    fmt = json_format_reward(chat, wants_fence=[wants_fence])[0]
    fields = field_count_reward(chat, schema_fields_count=[schema_fields_count])[0]
    schema = schema_validation_reward(chat, schema_str=[schema_str])[0]
    combined = (
        REWARD_WEIGHTS[0] * fmt + REWARD_WEIGHTS[1] * fields + REWARD_WEIGHTS[2] * schema
    )
    return {
        "json_format": fmt,
        "field_count": fields,
        "schema_validation": schema,
        "combined": combined,
    }
