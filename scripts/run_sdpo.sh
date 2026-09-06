#!/usr/bin/env bash
# Hybrid SDPO+GRPO on the A2 envelope (official errors as rich feedback).
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
LAMBDA="${2:-0.9}"
python -m ifstruct_rl.train_sdpo \
  --model LiquidAI/LFM2.5-350M \
  --output-dir "runs/sdpo-lfm350-official-seed${SEED}" \
  --seed "${SEED}" \
  --max-steps 100 \
  --sdpo-lambda "${LAMBDA}"
