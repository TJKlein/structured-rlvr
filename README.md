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

A0 follows the cookbook envelope: Nemotron structured-output prompts, three cheap train rewards, official exam at test. A2 keeps the same LoRA / GRPO setup (r=16, G=8, 100 steps, T=1.1, β=0.01) and swaps in official-shaped train rewards plus a held-out `train__*` prompt generator. CoRPO and RAFT use the same ~4,000-rollout budget as A0. A curriculum-ordered A2 rerun (easy→hard, constant LR) scored 39/128 (**30.5%**) greedy and is not a win. An OPSA screen on the base greedy run is a skip: failures are as confident as passes on the lowest-20% token logprobs.

[Fu et al. (2609.04172)](https://arxiv.org/abs/2609.04172) show that on-policy distillation is data-overfed: a handful of queries cover most training states, and content-light templates nearly match real problems, because the teacher supplies a dense token signal. Sparse IFStruct GRPO is the other side of that split — Nemotron’s messier states beat a clean generator on greedy. `scripts/run_a2_coverage.sh` retrains A2 on 16 `train__*` prompts that target the remaining error modes (schema dump, wrapper vs list, YAML fence, enums, extra keys, item count) for 300 steps. On-policy distillation itself is gated: only if `LFM2.5-1.2B-Instruct` is at least 3 points above cookbook GRPO on the official probe (`python -m ifstruct_rl.opd_gate`). After that, `scripts/run_sdpo.sh cookbook` and `scripts/run_sdpo.sh official` are two 100-step hybrid [SDPO](https://arxiv.org/abs/2601.20802)+GRPO ablations (λ=0.9): cheap cookbook rewards vs official exam rewards, each with that checker’s own error text as the dense token term. 350M is below the scale where that self-teacher is known to help.

Compact metrics: [`artifacts/ifstruct-lfm350/`](artifacts/ifstruct-lfm350/). Merged weights are not in git.

![Figure 1](artifacts/ifstruct-lfm350/exam.png)

**Figure 1.** Official IFStruct pass on a 128-prompt probe (even seeds 0–254, one seed). **a**, Greedy decode (bars) and pass@8 at T = 1 (circles). Cookbook GRPO and official-validator GRPO are highlighted; CoRPO and RAFT share the same rollout budget. **b**, Greedy pass split by output format.

![Figure 2](artifacts/ifstruct-lfm350/train.png)

**Figure 2.** Training dynamics over 100 GRPO steps. **a**, Mean train-time reward (schema component for cookbook GRPO; official binary for official-validator GRPO). Thin traces are per-step means; thick traces are a 7-step moving average. **b**, Fraction of groups whose rewards have zero standard deviation.

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
bash scripts/run_a2_coverage.sh  # 16-prompt coverage set, 300 steps
bash scripts/run_sdpo.sh cookbook  # SDPO + cheap cookbook rewards
bash scripts/run_sdpo.sh official  # SDPO + official exam rewards

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
