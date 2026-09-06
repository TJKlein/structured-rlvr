#!/usr/bin/env bash
# Full first milestone: 128-prompt IFStruct baseline + OPSA go/no-go screen.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p results
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"

python -m ifstruct_rl.eval_baseline \
  --model LiquidAI/LFM2.5-350M \
  --n 128 \
  --max-new-tokens 2048 \
  --out results/baseline_probe128.json

python -m ifstruct_rl.opsa_screen \
  --from-results results/baseline_probe128.json \
  --out results/opsa_screen.json
