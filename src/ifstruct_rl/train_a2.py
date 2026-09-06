"""A2: official-validator GRPO on generator data (not Nemotron, not the test set).

Same 100-step LoRA envelope as A0 so A2 vs A0 is reward+data, not a new trainer.
"""

from __future__ import annotations

import argparse
import os

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from ifstruct_rl.generator import load_generator_dataset
from ifstruct_rl.rewards.official import REWARD_FUNCS, REWARD_WEIGHTS


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--output-dir", default="runs/a2-lfm350-official-seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument(
        "--data-order",
        choices=("shuffle", "curriculum"),
        default="shuffle",
        help="shuffle = default; curriculum = easy→hard within the generator",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "constant_with_warmup"),
        default="cosine",
        help="constant_with_warmup keeps LR alive when hard rows arrive",
    )
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN is unset")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)

    train_ds = load_generator_dataset(
        n=args.train_samples,
        seed=args.seed,
        order=args.data_order,
    )
    print(f"generator rows: {len(train_ds)} order={args.data_order} lr={args.lr_scheduler}")
    print(f"entity types: {sorted(set(train_ds['entity_type']))}")

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
        output_dir=args.output_dir,
        learning_rate=5e-5,
        warmup_steps=max(1, int(0.1 * args.max_steps)),
        lr_scheduler_type=args.lr_scheduler,
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
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=list(REWARD_FUNCS),
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    merged_dir = f"{args.output_dir}-merged"
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"adapter saved to {args.output_dir}")
    print(f"merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
