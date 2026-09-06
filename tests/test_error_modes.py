from ifstruct_rl.error_modes import (
    attempt_hints,
    classify,
    standing_announcement,
)
from ifstruct_rl.sdpo import build_teacher_prompts, teacher_feedback_blocks


def test_classify_schema_dump():
    assert classify(["expected array, got dict"]) == ["schema_dump"]


def test_classify_wrapper_and_enum():
    modes = classify(
        [
            "Expected bare list, got wrapped object with key 'chapters'",
            "'reflective' not in allowed values ['draft']",
        ]
    )
    assert "wrapper" in modes
    assert "enum" in modes


def test_attempt_hints_name_the_gap():
    text = attempt_hints(["extraneous field 'nope'"])
    assert "Mind the gap" in text
    assert "extra keys" in text.lower() or "Extra keys" in text


def test_standing_lists_defaults():
    text = standing_announcement()
    assert text.startswith("Mind the gap.")
    assert "JSON Schema" in text or "schema" in text.lower()


def test_sdpo_teacher_prompt_has_standing_gap():
    out = build_teacher_prompts(["User: emit JSON"], ["expected array, got dict"])
    assert "[Mind the gap]" in out[0]
    assert "Mind the gap:" in out[0]
    assert "[IFStruct checker]" in out[0]


def test_fail_block_adds_specific_hint():
    blocks = teacher_feedback_blocks(
        passed=[False] * 8,
        errors=[["expected array, got dict"]] * 8,
        completions=["{}"] * 8,
        num_generations=8,
    )
    assert "Mind the gap: write the data instance" in blocks[0]
