from ifstruct_rl.generator import (
    TEST_ENTITY_PREFIX,
    TRAIN_PREFIX,
    difficulty_score,
    load_generator_dataset,
)


def test_generator_never_uses_test_entity_types():
    ds = load_generator_dataset(n=80, seed=0)
    types = set(ds["entity_type"])
    assert types
    assert all(t.startswith(TRAIN_PREFIX) for t in types)
    assert all(not t.startswith(TEST_ENTITY_PREFIX) for t in types)
    assert "json" in set(ds["output_format"]) and "yaml" in set(ds["output_format"])
    assert len(ds) == 80
    assert "json_schema_str" in ds.column_names
    assert isinstance(ds[0]["prompt"], list)


def test_curriculum_orders_easy_before_hard():
    shuffled = load_generator_dataset(n=120, seed=0, order="shuffle")
    ordered = load_generator_dataset(n=120, seed=0, order="curriculum")
    first = [difficulty_score(ordered[i]) for i in range(40)]
    last = [difficulty_score(ordered[i]) for i in range(80, 120)]
    assert sum(first) / len(first) < sum(last) / len(last)
    assert list(shuffled["output_format"]) != list(ordered["output_format"])
