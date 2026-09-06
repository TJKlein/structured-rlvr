from types import SimpleNamespace

from ifstruct_rl.data import PROBE_N, select_probe


def test_probe_even_seeds_0_to_254():
    examples = [SimpleNamespace(seed=i) for i in range(2000)]
    probe = select_probe(examples, n=PROBE_N)
    assert len(probe) == 128
    assert [p.seed for p in probe] == list(range(0, 256, 2))
