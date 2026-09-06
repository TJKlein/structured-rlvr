from ifstruct_rl.rewards.official import (
    dense_schema_reward,
    format_or_shape_reward,
    official_binary_reward,
)


def as_chat(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    },
    "minItems": 2,
    "maxItems": 2,
}


def spec(**overrides):
    base = dict(
        json_schema_str=[__import__("json").dumps(SCHEMA)],
        top_level_count_str=["2"],
        top_level_key=[""],
        require_wrapper_key=[False],
        require_code_block=[False],
        require_no_commentary=[True],
        output_format=["json"],
    )
    base.update(overrides)
    return base


def test_official_pass_is_one():
    text = '[{"title": "a"}, {"title": "b"}]'
    kw = spec()
    assert official_binary_reward([as_chat(text)], **kw) == [1.0]
    assert format_or_shape_reward([as_chat(text)], **kw) == [1.0]
    assert dense_schema_reward([as_chat(text)], **kw)[0] == 1.0


def test_extra_key_fails_official_keeps_shape():
    text = '[{"title": "a", "nope": true}, {"title": "b"}]'
    kw = spec()
    assert official_binary_reward([as_chat(text)], **kw) == [0.0]
    assert format_or_shape_reward([as_chat(text)], **kw) == [1.0]
    assert 0.0 < dense_schema_reward([as_chat(text)], **kw)[0] < 1.0


def test_wrong_wrapper_is_shape_zero():
    text = '{"items": [{"title": "a"}, {"title": "b"}]}'
    kw = spec()
    assert official_binary_reward([as_chat(text)], **kw) == [0.0]
    assert format_or_shape_reward([as_chat(text)], **kw) == [0.0]
