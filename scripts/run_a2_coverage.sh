#!/usr/bin/env bash
# Official-validator GRPO on the 16-prompt coverage set, 300 steps.
# Same LoRA envelope as cookbook GRPO; constant LR so later steps still learn.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED="${1:-0}"
python -m ifstruct_rl.train_a2 \
  --model LiquidAI/LFM2.5-350M \
  --output-dir "runs/a2-lfm350-coverage-seed${SEED}" \
  --seed "${SEED}" \
  --max-steps 300 \
  --data-source coverage \
  --lr-scheduler constant_with_warmup
