import torch

from ifstruct_rl.sdpo import (
    build_teacher_prompts,
    format_error_text,
    mix_advantages,
    teacher_feedback_blocks,
)


def test_format_error_truncates():
    long = ["x" * 1000]
    text = format_error_text(long)
    assert len(text) <= 400
    assert text.endswith("...")


def test_empty_errors_have_fallback():
    assert "rejected" in format_error_text([])


def test_feedback_uses_sibling_only_on_fails():
    passed = [True, False, False, False, False, False, False, False]
    errors = [[], ["expected array, got dict"]] + [[]] * 6
    completions = ["[1]", "{bad}"] + ["x"] * 6
    blocks = teacher_feedback_blocks(
        passed=passed,
        errors=errors,
        completions=completions,
        num_generations=8,
    )
    assert len(blocks) == 8
    assert "accepted" in blocks[0]
    assert "expected array, got dict" in blocks[1]
    assert "A correct attempt" in blocks[1]
    assert "[1]" in blocks[1]
    assert "A correct attempt" not in blocks[0]


def test_all_fail_has_no_demo():
    passed = [False] * 8
    errors = [["Expected bare list, got wrapped object with key 'chapters'"]] * 8
    completions = ["chapters: []"] * 8
    blocks = teacher_feedback_blocks(
        passed=passed,
        errors=errors,
        completions=completions,
        num_generations=8,
    )
    assert all("correct attempt" not in b for b in blocks)
    assert "bare list" in blocks[0]


def test_teacher_prompt_appends_checker_block():
    prompts = ["User: emit JSON"]
    feedback = ["expected array, got dict"]
    out = build_teacher_prompts(prompts, feedback)
    assert out[0].startswith("User: emit JSON")
    assert "[IFStruct checker]" in out[0]
    assert "expected array, got dict" in out[0]
    assert "The attempt follows" in out[0]


def test_mix_lambda_one_is_broadcast_grpo():
    grpo = torch.tensor([1.0, -0.5])
    teacher = torch.zeros(2, 4)
    student = torch.ones(2, 4)
    mask = torch.ones(2, 4)
    mixed = mix_advantages(grpo, teacher, student, mask, lam=1.0)
    assert mixed.shape == (2, 4)
    assert torch.allclose(mixed[0], torch.ones(4))
    assert torch.allclose(mixed[1], torch.full((4,), -0.5))


def test_mix_lambda_zero_is_sdpo_gap():
    grpo = torch.tensor([99.0, 99.0])
    teacher = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    student = torch.zeros(2, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    mixed = mix_advantages(grpo, teacher, student, mask, lam=0.0, clamp=5.0)
    assert mixed[0, 1].item() == 1.0
    assert mixed[0, 3].item() == 0.0  # padded
    assert torch.allclose(mixed[1], torch.zeros(4))


def test_hybrid_is_weighted_sum():
    grpo = torch.tensor([2.0])
    teacher = torch.zeros(1, 2)
    student = torch.zeros(1, 2)
    mask = torch.ones(1, 2)
    mixed = mix_advantages(grpo, teacher, student, mask, lam=0.9)
    assert torch.allclose(mixed, torch.full((1, 2), 1.8))
