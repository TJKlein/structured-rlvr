# structured-rlvr

Short reinforcement-learning-from-verifier-rewards (RLVR) recipes for **structured output**. The first experiment is [IFStruct](https://github.com/Liquid4All/ifstruct) instruction-following on [`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M).

**Result.** Liquid’s public 100-step [GRPO cookbook](https://huggingface.co/blog/grpo-with-trl-ifstruct) trains a looser checker than official IFStruct `validate_response`. Matching the official checker on a held-out generator did not beat that cookbook on greedy decode, but it did raise pass@8.

## Results (LFM2.5-350M, 1 seed)

Every number below uses official `validate_response`. Greedy decode unless noted. The public probe is 128 even seeds `0..254` (6.4% of the 2,000-row test set). Treat the probe as directional; a full-set greedy table will replace it when it lands.

| Run | Official greedy | JSON | YAML | pass@8 (T=1.0) |
|---|---:|---:|---:|---:|
| Base `LFM2.5-350M` | 27/128 (**21.1%**) | 16.4% | 25.4% | 32.8% |
| Cookbook GRPO (A0) | 45/128 (**35.2%**) | 39.3% | 31.3% | 44.5% |
| Official-validator GRPO (A2) | 42/128 (**32.8%**) | 34.4% | 31.3% | **49.2%** |
| CoRPO (`R_min=2.0`) | 39/128 (**30.5%**) | 27.9% | 32.8% | — |
| RAFT (filter then SFT) | 20/128 (**15.6%**) | 26.2% | 6.0% | — |

A0 follows the cookbook envelope: Nemotron structured-output prompts, three cheap train rewards, official exam at test. A2 keeps the same LoRA / GRPO setup (r=16, G=8, 100 steps, T=1.1, β=0.01) and swaps in official-shaped train rewards plus a held-out `train__*` prompt generator. CoRPO and RAFT use the same ~4,000-rollout budget as A0. An OPSA screen on the base greedy run is a skip: failures are as confident as passes on the lowest-20% token logprobs.

Compact metrics: [`artifacts/ifstruct-lfm350/`](artifacts/ifstruct-lfm350/). Merged weights are not in git.

### Train checker vs exam

A generation is a **hack** if the cookbook combined reward is `> 0.8` and official `validate_response` still fails.

| Run | Cookbook-high | Of those, official fail | Hack rate |
|---|---:|---:|---:|
| Base | 25 | 16 | **64%** |
| Cookbook GRPO (A0) | 47 | 26 | **55%** |
| Official-validator GRPO (A2) | 37 | 20 | 54% |

More than half of A0 generations that would score well on the public train checker still fail the exam. Remaining errors are mostly missing required fields, wrapper vs bare-list shape, and extra keys — not forgotten fences. Recompute with `python -m ifstruct_rl.hack_rate`.

### Examples (greedy, same seed)

Truncated. Snippets: [`artifacts/ifstruct-lfm350/examples_probe128.json`](artifacts/ifstruct-lfm350/examples_probe128.json).

1. **JSON array vs schema dump (seed 18).** Prompt wants a JSON array of customer-email threads. Base and A2 emit `{"type": "array", "items": [...]}` (a JSON Schema wrapper) and fail with `expected array, got dict`. A0 emits a bare JSON array and passes.
2. **YAML bare list (seed 0).** Prompt asks for a bare array of chapters, not wrapped under `chapters:`. Base and A0 wrap under `chapters:`. A2 emits a bare list, then fails an enum (`tone: reflective` is not allowed).
3. **YAML item count (seed 36).** Prompt asks for 2–3 repro-step batches. Base emits one item. A0 and A2 both pass.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"   # train extra pins trl==1.7.1 and peft
cp .env.example .env        # set HF_TOKEN; never commit .env
```

CPU tests:

```bash
pip install -e ".[dev]"
pytest -q
```

GPU (CUDA PyTorch already installed):

```bash
bash scripts/setup_gpu.sh
bash scripts/run_baseline.sh   # 128-probe greedy + OPSA screen
bash scripts/run_a0.sh         # cookbook GRPO
bash scripts/run_a2.sh         # official-validator GRPO
```

`HF_TOKEN` is only needed to download models and datasets.

## Layout

```
src/ifstruct_rl/     eval, generator, rewards, trainers
scripts/             one-command runs
artifacts/<exp>/     compact metrics (no generations, no weights)
tests/
```

## License

Apache-2.0. IFStruct and the Liquid cookbook remain their authors’ work; this repository reproduces and measures them.
