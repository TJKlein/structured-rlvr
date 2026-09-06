"""IFStruct-compatible training generator. Entity types are ``train__*`` only.

Template-only v1 (no LLM rewrite cache). Format 50/50, wrapper 50/50,
commentary 50/50, fence ~0.63, mixed item counts, 30% escaping strings.
"""

from __future__ import annotations

import json
import random
from typing import Any

from datasets import Dataset

TEST_ENTITY_PREFIX = "test__"
TRAIN_PREFIX = "train__"

# Held-out names. Must never collide with LiquidAI/ifstruct-v1.0 entity_type.
ENTITY_SPECS: list[tuple[str, str, list[tuple[str, dict[str, Any], bool]]]] = [
    (
        "warehouse_slot",
        "warehouse slot records",
        [
            ("aisle", {"type": "string"}, True),
            ("bin_code", {"type": "string"}, True),
            ("sku", {"type": "string"}, True),
            ("qty", {"type": "integer", "minimum": 0, "maximum": 9999}, True),
            ("frozen", {"type": "boolean"}, False),
        ],
    ),
    (
        "lab_sample_vial",
        "lab sample vials",
        [
            ("vial_id", {"type": "string"}, True),
            ("assay", {"type": "string", "enum": ["pcr", "elisa", "gcms", "culture"]}, True),
            ("volume_ul", {"type": "number", "minimum": 0.1, "maximum": 2000}, True),
            ("capped", {"type": "boolean"}, True),
        ],
    ),
    (
        "playlist_track",
        "playlist tracks",
        [
            ("title", {"type": "string"}, True),
            ("artist", {"type": "string"}, True),
            ("bpm", {"type": "integer", "minimum": 40, "maximum": 240}, False),
            ("explicit", {"type": "boolean"}, True),
        ],
    ),
    (
        "parking_permit",
        "parking permits",
        [
            ("plate", {"type": "string"}, True),
            ("zone", {"type": "string", "enum": ["A", "B", "C", "visitor"]}, True),
            ("hours", {"type": "integer", "minimum": 1, "maximum": 72}, True),
            ("ev_only", {"type": "boolean"}, False),
        ],
    ),
    (
        "coffee_order",
        "coffee orders",
        [
            ("drink", {"type": "string"}, True),
            ("size", {"type": "string", "enum": ["small", "medium", "large"]}, True),
            ("shots", {"type": "integer", "minimum": 1, "maximum": 4}, True),
            ("oat_milk", {"type": "boolean"}, True),
        ],
    ),
    (
        "garden_plot",
        "community garden plots",
        [
            ("plot_id", {"type": "string"}, True),
            ("crop", {"type": "string"}, True),
            ("rows", {"type": "integer", "minimum": 1, "maximum": 20}, True),
            ("needs_water", {"type": "boolean"}, True),
        ],
    ),
    (
        "radio_segment",
        "radio segments",
        [
            ("show_name", {"type": "string"}, True),
            ("seconds", {"type": "integer", "minimum": 15, "maximum": 900}, True),
            ("live", {"type": "boolean"}, True),
            ("desk", {"type": "string", "enum": ["news", "music", "sports", "traffic"]}, False),
        ],
    ),
    (
        "museum_loan",
        "museum loan objects",
        [
            ("accession", {"type": "string"}, True),
            ("title", {"type": "string"}, True),
            ("year", {"type": "integer", "minimum": 1000, "maximum": 2026}, False),
            ("fragile", {"type": "boolean"}, True),
        ],
    ),
    (
        "bike_share_trip",
        "bike-share trips",
        [
            ("bike_id", {"type": "string"}, True),
            ("start_dock", {"type": "string"}, True),
            ("minutes", {"type": "integer", "minimum": 1, "maximum": 240}, True),
            ("ebike", {"type": "boolean"}, True),
        ],
    ),
    (
        "library_hold",
        "library holds",
        [
            ("barcode", {"type": "string"}, True),
            ("title", {"type": "string"}, True),
            ("branch", {"type": "string"}, True),
            ("days_waiting", {"type": "integer", "minimum": 0, "maximum": 90}, True),
        ],
    ),
    (
        "weather_station_ping",
        "weather station pings",
        [
            ("station", {"type": "string"}, True),
            ("celsius", {"type": "number", "minimum": -80, "maximum": 60}, True),
            ("humidity_pct", {"type": "integer", "minimum": 0, "maximum": 100}, True),
            ("raining", {"type": "boolean"}, True),
        ],
    ),
    (
        "chess_round",
        "chess round cards",
        [
            ("white", {"type": "string"}, True),
            ("black", {"type": "string"}, True),
            ("result", {"type": "string", "enum": ["1-0", "0-1", "1/2-1/2", "*"]}, True),
            ("board", {"type": "integer", "minimum": 1, "maximum": 64}, True),
        ],
    ),
    (
        "bakery_batch",
        "bakery batches",
        [
            ("item", {"type": "string"}, True),
            ("dozen", {"type": "integer", "minimum": 1, "maximum": 40}, True),
            ("oven_c", {"type": "integer", "minimum": 140, "maximum": 260}, True),
            ("gluten_free", {"type": "boolean"}, False),
        ],
    ),
    (
        "drone_flight_log",
        "drone flight logs",
        [
            ("tail", {"type": "string"}, True),
            ("altitude_m", {"type": "number", "minimum": 1, "maximum": 400}, True),
            ("minutes", {"type": "integer", "minimum": 1, "maximum": 45}, True),
            ("night", {"type": "boolean"}, True),
        ],
    ),
    (
        "yoga_class",
        "yoga class listings",
        [
            ("style", {"type": "string", "enum": ["hatha", "vinyasa", "yin", "ashtanga"]}, True),
            ("room", {"type": "string"}, True),
            ("capacity", {"type": "integer", "minimum": 4, "maximum": 40}, True),
            ("heated", {"type": "boolean"}, True),
        ],
    ),
    (
        "stamp_auction_lot",
        "stamp auction lots",
        [
            ("lot_no", {"type": "string"}, True),
            ("country", {"type": "string"}, True),
            ("year", {"type": "integer", "minimum": 1840, "maximum": 2020}, True),
            ("mint", {"type": "boolean"}, True),
        ],
    ),
    (
        "ferry_sailing",
        "ferry sailings",
        [
            ("route", {"type": "string"}, True),
            ("depart", {"type": "string"}, True),
            ("vehicles", {"type": "integer", "minimum": 0, "maximum": 200}, True),
            ("cancelled", {"type": "boolean"}, True),
        ],
    ),
    (
        "soil_core",
        "soil core samples",
        [
            ("core_id", {"type": "string"}, True),
            ("depth_cm", {"type": "number", "minimum": 1, "maximum": 200}, True),
            ("ph", {"type": "number", "minimum": 3.0, "maximum": 10.0}, True),
            ("organic", {"type": "boolean"}, False),
        ],
    ),
    (
        "podcast_ad_break",
        "podcast ad breaks",
        [
            ("show", {"type": "string"}, True),
            ("seconds", {"type": "integer", "minimum": 5, "maximum": 120}, True),
            ("pre_roll", {"type": "boolean"}, True),
            ("host_read", {"type": "boolean"}, True),
        ],
    ),
    (
        "climbing_route",
        "climbing routes",
        [
            ("name", {"type": "string"}, True),
            ("grade", {"type": "string"}, True),
            ("bolts", {"type": "integer", "minimum": 0, "maximum": 30}, True),
            ("trad", {"type": "boolean"}, True),
        ],
    ),
]


WRAPPER_KEYS = [
    "items",
    "records",
    "entries",
    "rows",
    "payload",
    "results",
    "listing",
]


def _entity_type(slug: str) -> str:
    return f"{TRAIN_PREFIX}{slug}"


def _item_schema(fields: list[tuple[str, dict[str, Any], bool]], escaping: bool) -> dict[str, Any]:
    properties = {name: dict(schema) for name, schema, _req in fields}
    if escaping:
        properties["notes"] = {"type": "string"}
        required = [name for name, _schema, req in fields if req] + ["notes"]
    else:
        required = [name for name, _schema, req in fields if req]
    return {"type": "object", "properties": properties, "required": required}


def _array_schema(item: dict[str, Any], count) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": item}
    if isinstance(count, int):
        schema["minItems"] = count
        schema["maxItems"] = count
    else:
        schema["minItems"] = count[0]
        schema["maxItems"] = count[1]
    return schema


def _field_bullets(item: dict[str, Any]) -> str:
    lines = []
    for name, schema in item["properties"].items():
        req = "required" if name in item.get("required", []) else "optional"
        extra = ""
        if "enum" in schema:
            extra = f" enum={schema['enum']}"
        elif "minimum" in schema or "maximum" in schema:
            extra = f" bounds={schema.get('minimum')}/{schema.get('maximum')}"
        lines.append(f"- `{name}`: {schema.get('type')} ({req}){extra}")
    return "\n".join(lines)


def _render_prompt(
    *,
    label: str,
    output_format: str,
    require_wrapper: bool,
    top_level_key: str | None,
    count,
    require_code_block: bool,
    require_no_commentary: bool,
    item: dict[str, Any],
    json_schema: dict[str, Any],
    presentation: str,
    escaping: bool,
) -> str:
    if isinstance(count, int):
        count_txt = f"exactly {count}"
    else:
        count_txt = f"between {count[0]} and {count[1]} inclusive"
    shape = (
        f"wrap the list in an object under the key `{top_level_key}`"
        if require_wrapper
        else "emit a bare array at the root, not wrapped in an object"
    )
    fence = (
        "put the entire payload in a fenced code block"
        if require_code_block
        else "do not use a fenced code block; emit the payload only"
    )
    commentary = (
        "no commentary, no preface, nothing outside the payload"
        if require_no_commentary
        else "a short comment alongside the payload is fine"
    )
    fmt = "JSON" if output_format == "json" else "block-style YAML (not JSON stuffed in a yaml fence)"
    escape_bit = (
        " One string field must be multiline and include quotes and a newline."
        if escaping
        else ""
    )

    if presentation == "raw_json_schema":
        return (
            f"Generate {count_txt} {label}. Output format: {output_format}. {shape}. "
            f"{fence}. {commentary}.{escape_bit}\n\n"
            f"JSON Schema for the array (after unwrapping if wrapped):\n"
            f"{json.dumps(json_schema, indent=2)}"
        )
    if presentation == "bullet_paths":
        return (
            f"Need {label}, {fmt}. Count: {count_txt}. {shape}. {fence}. {commentary}.{escape_bit}\n\n"
            f"Each item:\n{_field_bullets(item)}"
        )
    if presentation == "hedged_bare":
        tease = top_level_key or "records"
        return (
            f"need some {label} generated. let me spec this out. format is {output_format}. "
            f"{fence}. root should be a bare array — actually wait, keyed under `{tease}:`? no. "
            f"keep it a bare array, not wrapped in an object. {commentary}.\n\n"
            f"count: {count_txt}. each item:\n{_field_bullets(item)}\n"
            f"{escape_bit} do not invent extra keys."
        )
    # chat_prose
    return (
        f"need some {label} generated, keep it practical. let me spec this out.\n\n"
        f"format is {output_format}. {fence}. {shape}. {commentary}.\n\n"
        f"count: {count_txt}. each item is an object with:\n{_field_bullets(item)}\n"
        f"{escape_bit} do not invent extra keys."
    )


def _fields_for_slug(slug: str) -> tuple[str, str, list]:
    for name, label, fields in ENTITY_SPECS:
        if name == slug:
            return name, label, fields
    raise KeyError(f"unknown entity slug {slug!r}")


def build_row(
    *,
    slug: str,
    output_format: str,
    require_wrapper: bool,
    top_level_key: str | None,
    count: int | list[int],
    require_code_block: bool,
    require_no_commentary: bool,
    escaping: bool,
    presentation: str,
    coverage_mode: str = "",
) -> dict[str, Any]:
    _name, label, fields = _fields_for_slug(slug)
    entity_type = _entity_type(slug)
    assert not entity_type.startswith(TEST_ENTITY_PREFIX)
    item = _item_schema(fields, escaping)
    json_schema = _array_schema(item, count)
    prompt = _render_prompt(
        label=label,
        output_format=output_format,
        require_wrapper=require_wrapper,
        top_level_key=top_level_key,
        count=count,
        require_code_block=require_code_block,
        require_no_commentary=require_no_commentary,
        item=item,
        json_schema=json_schema,
        presentation=presentation,
        escaping=escaping,
    )
    row = {
        "prompt": [{"role": "user", "content": prompt}],
        "json_schema_str": json.dumps(json_schema),
        "top_level_count_str": json.dumps(count),
        "top_level_key": top_level_key or "",
        "require_wrapper_key": require_wrapper,
        "require_code_block": require_code_block,
        "require_no_commentary": require_no_commentary,
        "output_format": output_format,
        "entity_type": entity_type,
        "presentation": presentation,
    }
    if coverage_mode:
        row["coverage_mode"] = coverage_mode
    return row


def generate_row(rng: random.Random, idx: int) -> dict[str, Any]:
    slug, _label, _fields = ENTITY_SPECS[idx % len(ENTITY_SPECS)]
    require_wrapper = rng.random() < 0.5
    top_level_key = rng.choice(WRAPPER_KEYS) if require_wrapper else None
    if rng.random() < 0.5:
        count: int | list[int] = rng.randint(2, 5)
    else:
        lo = rng.randint(2, 4)
        count = [lo, lo + rng.randint(1, 2)]
    return build_row(
        slug=slug,
        output_format=rng.choice(["json", "yaml"]),
        require_wrapper=require_wrapper,
        top_level_key=top_level_key,
        count=count,
        require_code_block=rng.random() < 0.63,
        require_no_commentary=rng.random() < 0.5,
        escaping=rng.random() < 0.30,
        presentation=rng.choice(["chat_prose", "bullet_paths", "raw_json_schema"]),
    )


def difficulty_score(row: dict[str, Any]) -> float:
    """Within-source rank: harder / more exam-like rows later in the epoch."""
    score = 0.0
    if row["output_format"] == "yaml":
        score += 1.0
    if row["require_wrapper_key"]:
        score += 0.5
    if row["require_code_block"]:
        score += 0.3
    if row["require_no_commentary"]:
        score += 0.5
    if row.get("presentation") == "chat_prose":
        score += 0.4
    schema = json.loads(row["json_schema_str"])
    items = schema.get("items") or {}
    required = items.get("required") or []
    props = items.get("properties") or {}
    score += 0.1 * len(required)
    if "notes" in props:
        score += 0.4
    count = json.loads(row["top_level_count_str"])
    if isinstance(count, list):
        score += 0.2 + 0.05 * count[1]
    else:
        score += 0.05 * int(count)
    return score


def load_generator_dataset(
    n: int = 600,
    seed: int = 0,
    order: str = "shuffle",
) -> Dataset:
    rng = random.Random(seed)
    rows = [generate_row(rng, i) for i in range(n)]
    types = {row["entity_type"] for row in rows}
    if any(t.startswith(TEST_ENTITY_PREFIX) for t in types):
        raise RuntimeError("generator leaked a test__ entity type")
    if order == "curriculum":
        # Easy first, hard last.
        rows = sorted(rows, key=difficulty_score)
    elif order != "shuffle":
        raise ValueError(f"unknown generator order {order!r}")
    return Dataset.from_list(rows)
