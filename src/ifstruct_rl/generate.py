"""Local greedy generation with per-token logprobs (for the OPSA screen)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class Generation:
    text: str
    token_logprobs: list[float]
    n_tokens: int
    truncated: bool


def load_model(model_id: str, dtype: str = "auto") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    torch_dtype = dtype
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _prompt_text(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


@torch.inference_mode()
def generate_k(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    k: int = 8,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    seed: int | None = None,
    messages: list[dict] | None = None,
) -> list[Generation]:
    """Sample ``k`` completions at ``temperature`` (no logprobs; used for pass@k)."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if messages is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = _prompt_text(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=1.0 if do_sample else None,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id,
    )
    eos = tokenizer.eos_token_id
    gens: list[Generation] = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_len:]
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != eos:
            keep = gen_ids != tokenizer.pad_token_id
            gen_ids = gen_ids[keep]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        truncated = bool(gen_ids.numel() >= max_new_tokens) and (
            eos is None or int(gen_ids[-1]) != int(eos)
        )
        gens.append(
            Generation(
                text=decoded,
                token_logprobs=[],
                n_tokens=int(gen_ids.numel()),
                truncated=truncated,
            )
        )
    return gens


@torch.inference_mode()
def generate_one(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 2048,
) -> Generation:
    text = _prompt_text(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out.sequences[0, prompt_len:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    logprobs: list[float] = []
    for step, logits in enumerate(out.scores):
        if step >= gen_ids.shape[0]:
            break
        log_softmax = torch.log_softmax(logits[0].float(), dim=-1)
        logprobs.append(float(log_softmax[int(gen_ids[step])]))
    eos = tokenizer.eos_token_id
    truncated = bool(gen_ids.numel() >= max_new_tokens) and (
        eos is None or int(gen_ids[-1]) != int(eos)
    )
    return Generation(
        text=decoded,
        token_logprobs=logprobs,
        n_tokens=int(gen_ids.numel()),
        truncated=truncated,
    )
