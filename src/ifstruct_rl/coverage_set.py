"""Sixteen ``train__*`` prompts that cover remaining IFStruct error modes.

Fu et al. (arXiv:2609.04172): OPD is data-overfed — a handful of queries cover
most training states. Sparse GRPO still needs those states to look like the exam.
This set is the coverage, not the volume: schema dump, wrapper vs list, YAML
fence, enums, extra keys, item count. Never uses ``test__*`` entity types.
"""

from __future__ import annotations

from datasets import Dataset

from ifstruct_rl.generator import TEST_ENTITY_PREFIX, TRAIN_PREFIX, build_row


# Frozen 16. Each row is one exam-like state, not a random draw from 600.
COVERAGE_SPECS: list[dict] = [
    {
        "slug": "warehouse_slot",
        "output_format": "json",
        "require_wrapper": False,
        "top_level_key": None,
        "count": 3,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "raw_json_schema",
        "coverage_mode": "schema_dump_bare",
    },
    {
        "slug": "garden_plot",
        "output_format": "json",
        "require_wrapper": True,
        "top_level_key": "records",
        "count": 4,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "raw_json_schema",
        "coverage_mode": "schema_dump_wrapped",
    },
    {
        "slug": "playlist_track",
        "output_format": "yaml",
        "require_wrapper": False,
        "top_level_key": None,
        "count": [3, 4],
        "require_code_block": True,
        "require_no_commentary": False,
        "escaping": False,
        "presentation": "hedged_bare",
        "coverage_mode": "hedged_wrapper",
    },
    {
        "slug": "bike_share_trip",
        "output_format": "yaml",
        "require_wrapper": True,
        "top_level_key": "listing",
        "count": 2,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "chat_prose",
        "coverage_mode": "yaml_wrapper",
    },
    {
        "slug": "coffee_order",
        "output_format": "json",
        "require_wrapper": False,
        "top_level_key": None,
        "count": 3,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "bullet_paths",
        "coverage_mode": "enum_json",
    },
    {
        "slug": "yoga_class",
        "output_format": "yaml",
        "require_wrapper": False,
        "top_level_key": None,
        "count": [2, 3],
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "chat_prose",
        "coverage_mode": "enum_yaml",
    },
    {
        "slug": "parking_permit",
        "output_format": "json",
        "require_wrapper": True,
        "top_level_key": "payload",
        "count": 3,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "bullet_paths",
        "coverage_mode": "extra_keys_json",
    },
    {
        "slug": "chess_round",
        "output_format": "yaml",
        "require_wrapper": False,
        "top_level_key": None,
        "count": 4,
        "require_code_block": True,
        "require_no_commentary": False,
        "escaping": False,
        "presentation": "chat_prose",
        "coverage_mode": "extra_keys_yaml",
    },
    {
        "slug": "library_hold",
        "output_format": "json",
        "require_wrapper": False,
        "top_level_key": None,
        "count": [2, 3],
        "require_code_block": False,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "bullet_paths",
        "coverage_mode": "item_range_json",
    },
    {
        "slug": "ferry_sailing",
        "output_format": "yaml",
        "require_wrapper": True,
        "top_level_key": "rows",
        "count": 3,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "chat_prose",
        "coverage_mode": "item_exact_yaml",
    },
    {
        "slug": "lab_sample_vial",
        "output_format": "yaml",
        "require_wrapper": False,
        "top_level_key": None,
        "count": 2,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": True,
        "presentation": "chat_prose",
        "coverage_mode": "yaml_fence_escaping",
    },
    {
        "slug": "radio_segment",
        "output_format": "json",
        "require_wrapper": True,
        "top_level_key": "items",
        "count": 5,
        "require_code_block": False,
        "require_no_commentary": False,
        "escaping": False,
        "presentation": "bullet_paths",
        "coverage_mode": "no_fence_json",
    },
    {
        "slug": "bakery_batch",
        "output_format": "yaml",
        "require_wrapper": False,
        "top_level_key": None,
        "count": [2, 4],
        "require_code_block": False,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "hedged_bare",
        "coverage_mode": "yaml_no_fence",
    },
    {
        "slug": "drone_flight_log",
        "output_format": "json",
        "require_wrapper": True,
        "top_level_key": "results",
        "count": 2,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": False,
        "presentation": "chat_prose",
        "coverage_mode": "json_wrapper",
    },
    {
        "slug": "soil_core",
        "output_format": "yaml",
        "require_wrapper": True,
        "top_level_key": "entries",
        "count": 3,
        "require_code_block": True,
        "require_no_commentary": False,
        "escaping": True,
        "presentation": "raw_json_schema",
        "coverage_mode": "yaml_schema",
    },
    {
        "slug": "stamp_auction_lot",
        "output_format": "json",
        "require_wrapper": False,
        "top_level_key": None,
        "count": 4,
        "require_code_block": True,
        "require_no_commentary": True,
        "escaping": True,
        "presentation": "chat_prose",
        "coverage_mode": "extra_keys_escaping",
    },
]


def load_coverage_dataset() -> Dataset:
    rows = [build_row(**spec) for spec in COVERAGE_SPECS]
    types = {row["entity_type"] for row in rows}
    if any(t.startswith(TEST_ENTITY_PREFIX) for t in types):
        raise RuntimeError("coverage set leaked a test__ entity type")
    if not all(t.startswith(TRAIN_PREFIX) for t in types):
        raise RuntimeError("coverage set missing train__ prefix")
    if len(rows) != 16:
        raise RuntimeError(f"coverage set must be 16 rows, got {len(rows)}")
    return Dataset.from_list(rows)
