from ifstruct_rl.coverage_set import COVERAGE_SPECS, load_coverage_dataset
from ifstruct_rl.generator import TEST_ENTITY_PREFIX, TRAIN_PREFIX
from ifstruct_rl.opd_gate import A0_PROBE_PASS_RATE, decide


def test_coverage_set_is_sixteen_held_out_train_types():
    ds = load_coverage_dataset()
    assert len(ds) == 16
    assert len(COVERAGE_SPECS) == 16
    types = set(ds["entity_type"])
    assert all(t.startswith(TRAIN_PREFIX) for t in types)
    assert all(not t.startswith(TEST_ENTITY_PREFIX) for t in types)
    assert "json" in set(ds["output_format"]) and "yaml" in set(ds["output_format"])
    assert True in set(ds["require_wrapper_key"]) and False in set(ds["require_wrapper_key"])
    modes = set(ds["coverage_mode"])
    assert "schema_dump_bare" in modes
    assert "hedged_wrapper" in modes
    assert "enum_yaml" in modes
    prompts = " ".join(row["prompt"][0]["content"] for row in ds)
    assert "JSON Schema" in prompts
    assert "keyed under" in prompts


def test_opd_gate_requires_three_point_margin():
    skip = decide(A0_PROBE_PASS_RATE)
    assert skip["recommendation"] == "skip_opd"
    skip2 = decide(A0_PROBE_PASS_RATE + 0.02)
    assert skip2["recommendation"] == "skip_opd"
    go = decide(A0_PROBE_PASS_RATE + 0.03)
    assert go["recommendation"] == "consider_opd"
