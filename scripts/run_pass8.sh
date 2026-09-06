#!/usr/bin/env bash
# T=1.0 pass@8 on the frozen 128-prompt IFStruct probe.
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

python -m ifstruct_rl.eval_passk \
  --model LiquidAI/LFM2.5-350M \
  --n 128 \
  --k 8 \
  --temperature 1.0 \
  --max-new-tokens 2048 \
  --seed 0 \
  --out results/pass8_probe128.json
