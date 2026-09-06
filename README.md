# structured-rlvr

Short reinforcement-learning-from-verifier-rewards (RLVR) recipes for **structured output**. The first experiment is instruction-following on [IFStruct](https://github.com/Liquid4All/ifstruct) with `LiquidAI/LFM2.5-350M`.

This repository is meant to grow. Later experiments can live under `artifacts/<name>/` with the same eval loop.

## IFStruct / LFM2.5-350M (1 seed, 128-prompt probe)

All pass rates use the **official** IFStruct `validate_response` checker, greedy decode unless noted. The public 128-prompt probe is even seeds `0..254` (6.4% of the 2,000-row test set). Treat these as directional; a full 2,000-prompt table is queued and will replace the probe as the number to cite.

| Run | Official greedy | JSON | YAML | pass@8 (T=1.0) |
|---|---:|---:|---:|---:|
| Base `LFM2.5-350M` | 27/128 (**21.1%**) | 16.4% | 25.4% | 32.8% |
| A0 — cookbook GRPO (Nemotron, looser train rewards) | 45/128 (**35.2%**) | 39.3% | 31.3% | 44.5% |
| A2 — official-validator GRPO (`train__*` generator) | 42/128 (**32.8%**) | 34.4% | 31.3% | **49.2%** |
| CoRPO (`R_min=2.0`), same envelope | 39/128 (**30.5%**) | 27.9% | 32.8% | — |
| RAFT (filter then SFT), same ~4k rollouts | 20/128 (**15.6%**) | 26.2% | 6.0% | — |

Matching the official checker at train time on a clean homemade generator **lost greedy to the cookbook** (32.8% vs 35.2%) and **won pass@8** (49.2% vs 44.5%). That points at prompt distribution (Nemotron + the public JSON/YAML mix), not a missing RL loss.

A0 = Liquid's [public GRPO cookbook](https://huggingface.co/blog/grpo-with-trl-ifstruct): NVIDIA Nemotron structured-output practice set, three cheap train rewards, official IFStruct exam at test. A2 keeps the same LoRA / GRPO envelope (r=16, G=8, 100 steps, T=1.1, β=0.01) and swaps in official-shaped train rewards plus a held-out `train__*` prompt generator.

OPSA screen on the base greedy run: **skip**. Failures are as confident as (or more than) passes on the lowest-20% token logprobs; the typical error is extra keys / wrong wrappers, not a low-likelihood tail.

Compact JSON: [`artifacts/ifstruct-lfm350/`](artifacts/ifstruct-lfm350/). Merged weights are not in git.

### Cookbook vs official (“hack rate”)

A generation is a **hack** if the cookbook combined reward is `> 0.8` and official `validate_response` still fails. On the 128-probe:

| Run | cookbook-high | of those, official fail | hack rate |
|---|---:|---:|---:|
| Base | 25 | 16 | **64%** |
| A0 cookbook GRPO | 47 | 26 | **55%** |
| A2 official GRPO | 37 | 20 | 54% |

More than half of A0 generations that would score well on the public train checker still fail the exam. Remaining errors are mostly missing required fields, wrapper vs bare-list shape, and extra keys — not forgotten fences. Script: `python -m ifstruct_rl.hack_rate`.

### Three probe examples (greedy, same seed)

Truncated. Full snippets: [`artifacts/ifstruct-lfm350/examples_probe128.json`](artifacts/ifstruct-lfm350/examples_probe128.json).

**1. JSON array vs schema dump (seed 18, customer email threads).** Prompt wants a JSON array of threads.

- Base / A2 emit `{"type": "array", "items": [...]}` — a JSON Schema wrapper. Official: `expected array, got dict`.
- A0 emits a bare JSON array and **passes**.

**2. YAML bare list (seed 0, short-story chapters).** Prompt says keep a bare array, not wrapped under `chapters:`.

- Base / A0 wrap under `chapters:`.
- A2 emits a bare list (right shape) then fails an enum: `tone: reflective` is not in the allowed values.

**3. YAML item count (seed 36, repro-steps batches).** Prompt asks for 2–3 batches.

- Base emits 1 item (`array has 1 items, minimum is 2`).
- A0 and A2 both **pass**.

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
