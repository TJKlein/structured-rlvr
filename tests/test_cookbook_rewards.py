from ifstruct_rl.rewards.cookbook import (
    field_count_reward,
    json_format_reward,
    schema_validation_reward,
)


def as_chat(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def test_json_format_direct_vs_fenced():
    empty = "{}"
    fenced = 'Here is the JSON: ```json\n{"unexpected": true}\n```'
    spam = "`" * 64
    assert json_format_reward([as_chat(empty)], wants_fence=[False]) == [1.0]
    assert json_format_reward([as_chat(empty)], wants_fence=[True]) == [0.2]
    assert json_format_reward([as_chat(fenced)], wants_fence=[True]) == [1.0]
    assert json_format_reward([as_chat(fenced)], wants_fence=[False]) == [0.2]
    assert json_format_reward([as_chat(spam)], wants_fence=[False]) == [0.0]


def test_field_count_exact_and_miss():
    obj = as_chat('{"a": 1, "b": 2}')
    assert field_count_reward([obj], schema_fields_count=["2"]) == [1.0]
    miss = field_count_reward([obj], schema_fields_count=["4"])[0]
    assert abs(miss - 0.5) < 1e-9
    assert field_count_reward([as_chat("not json")], schema_fields_count=["2"]) == [0.0]


def test_schema_validation_empty_object_not_full_credit():
    schema = '{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}'
    score = schema_validation_reward([as_chat("{}")], schema_str=[schema])[0]
    assert 0.0 <= score < 1.0
    good = schema_validation_reward([as_chat('{"a": 1}')], schema_str=[schema])[0]
    assert good == 1.0
