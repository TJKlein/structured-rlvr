"""CoRPO baseline clip (Garg et al., arXiv:2511.04439).

A = R - max(mean_group(R), R_min_correct). Failed / sub-threshold completions
never get a positive advantage, even if they beat a weak group mean. That is
the extra-key failure mode on IFStruct: GRPO can reinforce parseable junk.
"""

from __future__ import annotations

import inspect
import textwrap
import types

import torch


def corpo_advantages(
    rewards: torch.Tensor,
    *,
    num_generations: int,
    r_min_correct: float,
) -> torch.Tensor:
    """Vectorized CoRPO advantages for a packed (batch * G,) reward tensor."""
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    baseline = torch.clamp(mean, min=r_min_correct)
    return (grouped - baseline).reshape_as(rewards)


def patched_method_source(src: str) -> str:
    """Swap GRPO's group-mean baseline for CoRPO. Keep this a single line so
    indent of the surrounding TRL method does not matter after dedent."""
    src = textwrap.dedent(src)
    old = "advantages = rewards - mean_grouped_rewards"
    if old not in src:
        raise RuntimeError(
            "TRL GRPOTrainer no longer uses `advantages = rewards - mean_grouped_rewards`; "
            "refit the CoRPO patch."
        )
    new = (
        "advantages = corpo_advantages("
        "rewards, num_generations=self.num_generations, "
        "r_min_correct=float(self.r_min_correct))"
    )
    return src.replace(old, new, 1)


def patch_grpo_trainer(trainer, r_min_correct: float) -> None:
    """Replace GRPO's group-mean baseline with CoRPO inside TRL 1.7.x."""
    trainer.r_min_correct = float(r_min_correct)
    method = trainer._generate_and_score_completions
    patched_src = patched_method_source(inspect.getsource(method))
    ns: dict = {}
    globs = dict(method.__globals__)
    globs["corpo_advantages"] = corpo_advantages
    exec(compile(patched_src, "<corpo_patch>", "exec"), globs, ns)
    trainer._generate_and_score_completions = types.MethodType(
        ns["_generate_and_score_completions"], trainer
    )
