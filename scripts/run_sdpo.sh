#!/usr/bin/env bash
# Hybrid SDPO+GRPO. Usage: run_sdpo.sh [cookbook|official] [seed] [lambda]
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

RECIPE="${1:-official}"
SEED="${2:-0}"
LAMBDA="${3:-0.9}"
python -m ifstruct_rl.train_sdpo \
  --recipe "${RECIPE}" \
  --model LiquidAI/LFM2.5-350M \
  --output-dir "runs/sdpo-lfm350-${RECIPE}-seed${SEED}" \
  --seed "${SEED}" \
  --max-steps 100 \
  --sdpo-lambda "${LAMBDA}"
