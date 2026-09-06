# structured-rlvr

Short reinforcement-learning-from-verifier-rewards (RLVR) recipes for **structured output**. The first experiment is instruction-following on [IFStruct](https://github.com/Liquid4All/ifstruct) with `LiquidAI/LFM2.5-350M`.

This repository is meant to grow. Later experiments can live under `artifacts/<name>/` with the same eval loop.

## IFStruct / LFM2.5-350M (1 seed, 128-prompt probe)

All pass rates use the **official** IFStruct `validate_response` checker, greedy decode unless noted. The public 128-prompt probe is even seeds `0..254` (6.4% of the 2,000-row test set). Treat these as directional; a full 2,000-prompt table is the number to cite later.

| Run | Official greedy | JSON | YAML | pass@8 (T=1.0) |
|---|---:|---:|---:|---:|
| Base `LFM2.5-350M` | 27/128 (**21.1%**) | 16.4% | 25.4% | 32.8% |
| A0 — cookbook GRPO (Nemotron, looser train rewards) | 45/128 (**35.2%**) | 39.3% | 31.3% | 44.5% |
| CoRPO (`R_min=2.0`), same envelope | 39/128 (**30.5%**) | 27.9% | 32.8% | — |
| RAFT (filter then SFT), same ~4k rollouts | 20/128 (**15.6%**) | 26.2% | 6.0% | — |

OPSA screen on the base greedy run: **skip**. Failures are as confident as (or more than) passes on the lowest-20% token logprobs; the typical error is extra keys / wrong wrappers, not a low-likelihood tail.

A2 (official-validator GRPO on a held-out `train__*` generator) and a curriculum-ordered follow-up are in the code (`scripts/run_a2.sh`, `scripts/run_a2_cma.sh`). Probe numbers for those runs will be added under `artifacts/ifstruct-lfm350/` when evals finish.

Compact JSON for the table: [`artifacts/ifstruct-lfm350/`](artifacts/ifstruct-lfm350/). Merged weights are not in git.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"   # train extra pulls trl==1.7.1 and peft
cp .env.example .env        # set HF_TOKEN; never commit .env
```

CPU tests:

```bash
PYTHONPATH=src pytest -q
```

GPU (CUDA PyTorch already installed):

```bash
bash scripts/setup_gpu.sh
bash scripts/run_baseline.sh          # 128-probe greedy + OPSA screen
bash scripts/run_a0.sh                # cookbook GRPO
bash scripts/run_a2.sh                # official-validator GRPO
```

`HF_TOKEN` is only needed to download models/datasets. It must not be committed.

## Layout

```
src/ifstruct_rl/     # eval, generator, rewards, trainers
scripts/             # one-command runs
artifacts/<exp>/     # compact metrics (no generations, no weights)
tests/
```

## License

Apache-2.0. IFStruct and Liquid cookbook remain their authors' work; this repo reproduces and measures them.
