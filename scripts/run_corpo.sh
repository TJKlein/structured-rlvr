#!/usr/bin/env bash
# CoRPO 1-seed appendix. Same A0 envelope, clipped GRPO baseline. Run AFTER A0.
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
python -m ifstruct_rl.train_a0 \
  --algorithm corpo \
  --r-min 2.0 \
  --model LiquidAI/LFM2.5-350M \
  --output-dir "runs/corpo-lfm350-cookbook-seed${SEED}" \
  --seed "${SEED}" \
  --max-steps 100
