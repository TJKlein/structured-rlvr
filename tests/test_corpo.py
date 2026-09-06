import inspect

import torch
from ifstruct_rl.corpo import corpo_advantages, patched_method_source


def test_corpo_never_boosts_subthreshold():
    # Group of 8: one extra-key-ish 1.8, rest unparseable 0.2. Mean=0.4, T=2.0.
    rewards = torch.tensor([1.8] + [0.2] * 7)
    adv = corpo_advantages(rewards, num_generations=8, r_min_correct=2.0)
    assert (adv > 0).sum() == 0
    assert adv[0] < 0  # 1.8 - 2.0
    assert torch.allclose(adv[1:], torch.full((7,), 0.2 - 2.0))


def test_corpo_recovers_grpo_when_mean_above_threshold():
    rewards = torch.tensor([3.0, 2.5, 2.2, 2.8, 3.1, 2.4, 2.6, 2.7])
    mean = rewards.mean()
    assert mean >= 2.0
    adv = corpo_advantages(rewards, num_generations=8, r_min_correct=2.0)
    assert torch.allclose(adv, rewards - mean, atol=1e-6)


def test_grpo_would_boost_best_fail_corpo_does_not():
    rewards = torch.tensor([1.8] + [0.2] * 7)
    grpo = rewards - rewards.mean()
    assert grpo[0] > 0
    corpo = corpo_advantages(rewards, num_generations=8, r_min_correct=2.0)
    assert corpo[0] < 0


def test_trl_patch_compiles():
    import pytest

    pytest.importorskip("trl")
    from trl.trainer.grpo_trainer import GRPOTrainer

    src = inspect.getsource(GRPOTrainer._generate_and_score_completions)
    patched = patched_method_source(src)
    compile(patched, "<corpo_patch>", "exec")
    assert "corpo_advantages(" in patched
    assert "advantages = rewards - mean_grouped_rewards" not in patched
