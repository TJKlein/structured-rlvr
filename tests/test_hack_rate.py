from ifstruct_rl.hack_rate import _d20, _schema_fields_count


def test_d20_tags_extra_keys_and_fences():
    assert _d20(["extraneous field 'notes'"])
    assert _d20(["Unclosed code block"])
    assert not _d20(["required field missing"])
    assert not _d20(["expected array, got dict"])


def test_schema_fields_count_object_and_array():
    obj = {"type": "object", "properties": {"a": {}, "b": {}}}
    arr = {"type": "array", "items": {"type": "object", "properties": {"a": {}}}}
    assert _schema_fields_count(obj) == 2
    assert _schema_fields_count(arr) == 1
