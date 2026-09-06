"""Hybrid SDPO+GRPO (Baecher et al., arXiv:2601.20802) on official IFStruct errors.

GRPO still scores the whole completion. SDPO then re-reads those same tokens
after concatenating checker errors and, when present, a successful sibling.
The dense term is log π_teacher(y_t | x, f, y_<t) − log π_student(y_t | x, y_<t).

350M is below the paper's self-teaching scale, so the default is λ=0.9
(mostly GRPO). No external teacher. No EMA in this v1.
"""

from __future__ import annotations

from typing import Any

import torch

MAX_ERROR_CHARS = 400
TEACHER_TAIL = (
    "\n\n[IFStruct checker]\n{feedback}\n[End checker. The attempt follows.]\n"
)


def format_error_text(errors: list[str] | None) -> str:
    blob = "; ".join(e.strip() for e in (errors or []) if e.strip())
    blob = " ".join(blob.split())
    if not blob:
        return "The checker rejected this attempt."
    if len(blob) > MAX_ERROR_CHARS:
        return blob[: MAX_ERROR_CHARS - 3] + "..."
    return blob


def teacher_feedback_blocks(
    *,
    passed: list[bool],
    errors: list[list[str]],
    completions: list[str],
    num_generations: int,
) -> list[str]:
    """One retrospective block per rollout, grouped like TRL (prompt-major, G each)."""
    n = len(passed)
    if n == 0:
        return []
    if n % num_generations != 0:
        raise ValueError(f"batch {n} is not divisible by G={num_generations}")
    out: list[str] = []
    for start in range(0, n, num_generations):
        sl = slice(start, start + num_generations)
        sib = None
        for ok, text in zip(passed[sl], completions[sl], strict=True):
            if ok:
                sib = text
                break
        for ok, err in zip(passed[sl], errors[sl], strict=True):
            if ok:
                block = "The checker accepted this attempt."
            else:
                block = format_error_text(err)
                if sib is not None:
                    demo = sib.strip()
                    if len(demo) > 800:
                        demo = demo[:797] + "..."
                    block = f"{block}\nA correct attempt for this task:\n{demo}"
            out.append(block)
    return out


def build_teacher_prompts(prompts: list[str], feedback: list[str]) -> list[str]:
    if len(prompts) != len(feedback):
        raise ValueError("prompts and feedback length mismatch")
    return [p.rstrip() + TEACHER_TAIL.format(feedback=f) for p, f in zip(prompts, feedback)]


def mix_advantages(
    grpo_adv: torch.Tensor,
    teacher_logps: torch.Tensor,
    student_logps: torch.Tensor,
    mask: torch.Tensor,
    *,
    lam: float,
    clamp: float = 5.0,
) -> torch.Tensor:
    """λ · A_GRPO (broadcast) + (1−λ) · A_SDPO. Shape (B, T)."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    sdpo = (teacher_logps - student_logps).clamp(-clamp, clamp) * mask
    return lam * grpo_adv.unsqueeze(1) + (1.0 - lam) * sdpo


def score_from_row(text: str, row: dict[str, Any]) -> dict:
    from ifstruct_rl.rewards.official import SPEC_KEYS, score_completion

    spec = {key: row[key] for key in SPEC_KEYS if key in row}
    return score_completion(text, **spec)


def sdpo_trainer_class():
    """GRPOTrainer subclass. Import TRL only when training."""
    from trl import GRPOTrainer

    try:
        from trl.extras.profiling import profiling_decorator
    except ImportError:

        def profiling_decorator(fn):
            return fn

    class _SdpoGRPOTrainer(GRPOTrainer):
        """TRL 1.7 GRPOTrainer with (B, T) hybrid advantages after scoring."""

        sdpo_lambda: float = 0.9

        @profiling_decorator
        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            lam = float(getattr(self, "sdpo_lambda", 0.9))
            if lam >= 1.0:
                return output
            output["advantages"] = self._sdpo_mixed_advantages(output, inputs)
            return output

        def _sdpo_mixed_advantages(self, output: dict, inputs: list[dict]) -> torch.Tensor:
            prompt_ids = output["prompt_ids"]
            prompt_mask = output["prompt_mask"]
            completion_ids = output["completion_ids"]
            completion_mask = output["completion_mask"]
            grpo_adv = output["advantages"]
            device = completion_ids.device
            g = int(self.num_generations)
            b = completion_ids.size(0)
            if b % g != 0:
                raise RuntimeError(f"SDPO needs full groups; batch={b} G={g}")
            if grpo_adv.dim() != 1:
                grpo_adv = grpo_adv.reshape(b)
            if len(inputs) != b:
                if len(inputs) * g == b:
                    inputs = [row for row in inputs for _ in range(g)]
                else:
                    raise RuntimeError(
                        f"SDPO inputs ({len(inputs)}) vs completions ({b}) vs G={g}"
                    )

            tok = self.processing_class
            prompts = tok.batch_decode(prompt_ids, skip_special_tokens=True)
            completions = tok.batch_decode(completion_ids, skip_special_tokens=True)
            passed: list[bool] = []
            errors: list[list[str]] = []
            for text, row in zip(completions, inputs, strict=True):
                scored = score_from_row(text, row)
                passed.append(bool(scored["passed"]))
                errors.append(list(scored.get("errors") or []))

            feedback = teacher_feedback_blocks(
                passed=passed,
                errors=errors,
                completions=completions,
                num_generations=g,
            )
            teacher_prompts = build_teacher_prompts(prompts, feedback)

            unwrap = self.accelerator.unwrap_model(self.model)
            was_training = unwrap.training
            unwrap.eval()
            try:
                with torch.inference_mode():
                    student = output.get("old_per_token_logps")
                    if student is None:
                        student = self._sdpo_logps(
                            unwrap,
                            torch.cat([prompt_ids, completion_ids], dim=1),
                            torch.cat([prompt_mask, completion_mask], dim=1),
                            completion_ids.size(1),
                        )
                    teacher = self._sdpo_teacher_logps(
                        unwrap,
                        teacher_prompts,
                        completion_ids,
                        completion_mask,
                    )
            finally:
                if was_training:
                    unwrap.train()

            mixed = mix_advantages(
                grpo_adv.to(device=device, dtype=student.dtype),
                teacher,
                student,
                completion_mask,
                lam=float(getattr(self, "sdpo_lambda", 0.9)),
            )
            gap = ((teacher - student).abs() * completion_mask).sum() / completion_mask.sum().clamp(
                min=1
            )
            train_metrics = self._metrics.setdefault("train", {})
            train_metrics.setdefault("sdpo/abs_logp_gap", []).append(
                self.accelerator.gather(gap).nanmean().item()
            )
            n_pass = gap.new_tensor(float(sum(passed)) / max(len(passed), 1))
            train_metrics.setdefault("sdpo/official_pass_frac", []).append(
                self.accelerator.gather(n_pass).nanmean().item()
            )
            return mixed

        def _sdpo_teacher_logps(self, model, teacher_prompts, completion_ids, completion_mask):
            tok = self.processing_class
            max_prompt = int(getattr(self, "max_prompt_length", None) or 2048)
            prev_side = tok.padding_side
            tok.padding_side = "left"
            try:
                encoded = tok(
                    teacher_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt,
                    add_special_tokens=True,
                )
            finally:
                tok.padding_side = prev_side
            t_ids = encoded["input_ids"].to(completion_ids.device)
            t_mask = encoded["attention_mask"].to(completion_ids.device)
            full_ids = torch.cat([t_ids, completion_ids], dim=1)
            full_mask = torch.cat([t_mask, completion_mask], dim=1)
            return self._sdpo_logps(model, full_ids, full_mask, completion_ids.size(1))

        def _sdpo_logps(self, model, input_ids, attention_mask, logits_to_keep):
            logps, _, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=min(2, input_ids.size(0)),
            )
            return logps

    return _SdpoGRPOTrainer
