"""Hybrid SDPO+GRPO (Baecher et al., arXiv:2601.20802).

Two recipes, same LoRA / 100 steps / G=8 / λ=0.9:

* cookbook — Nemotron + cheap train rewards (A0 envelope)
* official — train__* generator + official-shaped rewards (A2 envelope)
"""

from __future__ import annotations

import argparse
import os

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig

from ifstruct_rl.sdpo import sdpo_trainer_class


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--recipe",
        choices=("cookbook", "official"),
        default="official",
        help="cookbook = A0 data+rewards; official = A2 data+rewards",
    )
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument(
        "--data-source",
        choices=("generator", "coverage"),
        default="generator",
        help="Only used with --recipe official",
    )
    parser.add_argument(
        "--sdpo-lambda",
        type=float,
        default=0.9,
        help="Blend: λ·A_GRPO + (1−λ)·A_SDPO. Paper uses ~0.9 on weak models.",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN is unset")
    if not 0.0 <= args.sdpo_lambda <= 1.0:
        raise SystemExit("--sdpo-lambda must be in [0, 1]")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)

    if args.recipe == "cookbook":
        from ifstruct_rl.data_nemotron import load_nemotron_cookbook
        from ifstruct_rl.rewards.cookbook import REWARD_FUNCS, REWARD_WEIGHTS

        n = 1000 if args.train_samples is None else args.train_samples
        train_ds = load_nemotron_cookbook(n=n)
        default_dir = f"runs/sdpo-lfm350-cookbook-seed{args.seed}"
        print(f"recipe=cookbook nemotron rows={len(train_ds)}")
    else:
        from ifstruct_rl.coverage_set import load_coverage_dataset
        from ifstruct_rl.generator import load_generator_dataset
        from ifstruct_rl.rewards.official import REWARD_FUNCS, REWARD_WEIGHTS

        n = 600 if args.train_samples is None else args.train_samples
        if args.data_source == "coverage":
            train_ds = load_coverage_dataset()
            print(f"recipe=official coverage rows={len(train_ds)}")
        else:
            train_ds = load_generator_dataset(n=n, seed=args.seed)
            print(f"recipe=official generator rows={len(train_ds)}")
        default_dir = f"runs/sdpo-lfm350-official-seed{args.seed}"

    output_dir = args.output_dir or default_dir

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        warmup_steps=max(1, int(0.1 * args.max_steps)),
        lr_scheduler_type="cosine",
        max_steps=args.max_steps,
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        max_completion_length=1024,
        mask_truncated_completions=False,
        num_generations=8,
        temperature=1.1,
        reward_weights=REWARD_WEIGHTS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        steps_per_generation=2,
        beta=0.01,
        logging_steps=1,
        save_steps=args.max_steps,
        report_to="none",
    )
    Trainer = sdpo_trainer_class()
    trainer = Trainer(
        model=model,
        reward_funcs=list(REWARD_FUNCS),
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.sdpo_lambda = args.sdpo_lambda
    trainer.sdpo_recipe = args.recipe
    print(f"SDPO+GRPO recipe={args.recipe} λ={args.sdpo_lambda} (1=pure GRPO)")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    merged_dir = f"{output_dir}-merged"
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"adapter saved to {output_dir}")
    print(f"merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
