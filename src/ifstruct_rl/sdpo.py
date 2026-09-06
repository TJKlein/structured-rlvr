"""Hybrid SDPO+GRPO (Baecher et al., arXiv:2601.20802).

Two matched recipes, same LoRA / 100-step / G=8 envelope:

* ``cookbook`` — Nemotron + the three cheap train rewards; dense feedback
  is jsonschema / format text from that weak checker.
* ``official`` — ``train__*`` generator + official-shaped rewards; dense
  feedback is ``validate_response`` error strings.

GRPO still scores the whole completion. SDPO re-reads those same tokens
after concatenating checker text and, when present, a successful sibling.
λ=0.9 (mostly GRPO) because 350M is below the paper's self-teaching scale.
"""

from __future__ import annotations

from typing import Any

import torch

from ifstruct_rl.error_modes import attempt_hints, standing_announcement

MAX_ERROR_CHARS = 400
TEACHER_TAILS = {
    "official": (
        "\n\n[IFStruct checker]\n{feedback}\n[End checker. The attempt follows.]\n"
    ),
    "cookbook": (
        "\n\n[Cookbook checker]\n{feedback}\n[End checker. The attempt follows.]\n"
    ),
}


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
                hints = attempt_hints(err)
                if hints:
                    block = f"{block}\n{hints}"
                if sib is not None:
                    demo = sib.strip()
                    if len(demo) > 800:
                        demo = demo[:797] + "..."
                    block = f"{block}\nA correct attempt for this task:\n{demo}"
            out.append(block)
    return out


def build_teacher_prompts(
    prompts: list[str],
    feedback: list[str],
    *,
    recipe: str = "official",
    standing: str | None = None,
) -> list[str]:
    if len(prompts) != len(feedback):
        raise ValueError("prompts and feedback length mismatch")
    if recipe not in TEACHER_TAILS:
        raise ValueError(f"unknown sdpo recipe {recipe}")
    tail = TEACHER_TAILS[recipe]
    gap = standing if standing is not None else standing_announcement()
    out = []
    for prompt, fb in zip(prompts, feedback):
        out.append(
            prompt.rstrip()
            + f"\n\n[Mind the gap]\n{gap}\n"
            + tail.format(feedback=fb)
        )
    return out


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


def score_cookbook_row(text: str, row: dict[str, Any]) -> dict:
    """Weak-checker pass/fail + error strings. jsonschema is a core dep."""
    import json

    from jsonschema import Draft7Validator

    from ifstruct_rl.rewards.cookbook import extract_json

    errors: list[str] = []
    obj, form = extract_json(text.strip())
    wants_fence = bool(row.get("wants_fence"))
    requested = "fenced" if wants_fence else "direct"
    if form is None:
        errors.append("Cookbook format: no parseable JSON")
    elif form != requested:
        errors.append(f"Cookbook format: got {form}, wanted {requested}")
    schema_raw = row.get("schema_str")
    if schema_raw:
        schema = json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
        if not isinstance(obj, (dict, list)):
            errors.append("Cookbook schema: no object to check")
        else:
            for err in Draft7Validator(schema).iter_errors(obj):
                msg = err.message.strip()
                if msg:
                    errors.append(f"Cookbook schema: {msg}")
    return {"passed": not errors, "errors": errors}


def score_from_row(text: str, row: dict[str, Any], recipe: str = "official") -> dict:
    if recipe == "cookbook":
        return score_cookbook_row(text, row)
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
        sdpo_recipe: str = "official"

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
            recipe = str(getattr(self, "sdpo_recipe", "official"))
            passed: list[bool] = []
            errors: list[list[str]] = []
            for text, row in zip(completions, inputs, strict=True):
                scored = score_from_row(text, row, recipe=recipe)
                passed.append(bool(scored["passed"]))
                errors.append(list(scored.get("errors") or []))

            feedback = teacher_feedback_blocks(
                passed=passed,
                errors=errors,
                completions=completions,
                num_generations=g,
            )
            teacher_prompts = build_teacher_prompts(
                prompts,
                feedback,
                recipe=recipe,
                standing=self._sdpo_standing(recipe),
            )

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
            train_metrics.setdefault("sdpo/env_pass_frac", []).append(
                self.accelerator.gather(n_pass).nanmean().item()
            )
            return mixed

        def _sdpo_standing(self, recipe: str) -> str:
            from pathlib import Path

            from ifstruct_rl.error_modes import load_catalog, standing_announcement

            specific = Path(f"artifacts/mind_the_gap_{recipe}.json")
            fallback = Path("artifacts/mind_the_gap.json")
            return standing_announcement(load_catalog(specific if specific.exists() else fallback))

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
