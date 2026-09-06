"""RAFT: generate G=8 on Nemotron, keep combined cookbook reward >= R_min, SFT.

Same ~4,000-rollout budget as A0. No policy gradient. Judge is only a filter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from ifstruct_rl.data_nemotron import load_nemotron_cookbook
from ifstruct_rl.generate import generate_k, load_model
from ifstruct_rl.rewards.combined import cookbook_components


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--output-dir", default="runs/raft-lfm350-cookbook-seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sft-epochs", type=float, default=1.0)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)

    train_ds = load_nemotron_cookbook(n=args.train_samples)
    print(f"nemotron rows after cookbook filters: {len(train_ds)}")

    model, tokenizer = load_model(args.model)
    kept: list[dict] = []
    n_rollouts = 0
    for i, row in enumerate(train_ds):
        gens = generate_k(
            model,
            tokenizer,
            prompt="",
            k=args.k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + i,
            messages=list(row["prompt"]),
        )
        n_rollouts += len(gens)
        for gen in gens:
            scores = cookbook_components(
                gen.text,
                wants_fence=row["wants_fence"],
                schema_fields_count=row["schema_fields_count"],
                schema_str=row["schema_str"],
            )
            if scores["combined"] >= args.r_min:
                kept.append(
                    {
                        "messages": list(row["prompt"])
                        + [{"role": "assistant", "content": gen.text}],
                        "combined": scores["combined"],
                        **{k: scores[k] for k in ("json_format", "field_count", "schema_validation")},
                    }
                )
        if (i + 1) % 25 == 0 or i + 1 == len(train_ds):
            print(
                f"[{i + 1}/{len(train_ds)}] rollouts={n_rollouts} kept={len(kept)}",
                flush=True,
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_prompts": len(train_ds),
        "n_rollouts": n_rollouts,
        "n_kept": len(kept),
        "keep_rate": len(kept) / n_rollouts if n_rollouts else 0.0,
        "r_min": args.r_min,
        "k": args.k,
        "temperature": args.temperature,
    }
    (out_dir / "raft_filter.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not kept:
        raise SystemExit("RAFT kept 0 completions; lower --r-min")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sft_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    sft_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
        ],
    )
    if sft_tok.pad_token is None:
        sft_tok.pad_token = sft_tok.eos_token
    sft_model = get_peft_model(sft_model, lora_config)
    sft_rows = [{"messages": row["messages"]} for row in kept]
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.sft_epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_steps=500,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        max_length=2048,
    )
    trainer = SFTTrainer(
        model=sft_model,
        args=sft_args,
        train_dataset=Dataset.from_list(sft_rows),
        processing_class=sft_tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    sft_tok.save_pretrained(str(out_dir))
    merged_dir = f"{out_dir}-merged"
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    sft_tok.save_pretrained(merged_dir)
    print(f"RAFT adapter {out_dir}; merged {merged_dir}")


if __name__ == "__main__":
    main()
