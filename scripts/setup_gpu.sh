#!/usr/bin/env bash
# Run on a CUDA GPU after cloning. Reuses the image's PyTorch if present.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN in .env (do not commit it)"
  exit 1
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA torch not found. Need a CUDA-enabled PyTorch install.")
print("gpu", torch.cuda.get_device_name(0))
PY

python -m pip install -U pip
python -m pip install -e .

echo "setup ok. next: bash scripts/run_baseline.sh"
echo "optional smoke (2 prompts): python -m ifstruct_rl.eval_baseline --smoke 2 --out results/smoke.json"
