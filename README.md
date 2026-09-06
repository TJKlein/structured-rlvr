# structured-rlvr

Short reinforcement-learning-from-verifier-rewards (RLVR) recipes for **structured output**. The first experiment is [IFStruct](https://github.com/Liquid4All/ifstruct) instruction-following on [`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M).

**Result.** Liquid’s public 100-step [GRPO cookbook](https://huggingface.co/blog/grpo-with-trl-ifstruct) trains a looser checker than official IFStruct `validate_response`. On the full 2,000-prompt official greedy exam, matching that checker on a held-out generator was about even with the cookbook (613/2000 vs 593/2000). That is 20 prompts, one seed, and both the data mix and the train rewards changed, so it is not a method win.

## Results (LFM2.5-350M, 1 seed)

Every number below uses official `validate_response`. Greedy decode via HF `generate` (`do_sample=False`, `max_new_tokens=2048`).

### Official greedy, full 2,000 prompts

| Run | Official greedy | JSON | YAML |
|---|---:|---:|---:|
| Base `LFM2.5-350M` | 417/2000 (**20.8%**) | 16.5% | 25.2% |
| Cookbook GRPO (A0) | 593/2000 (**29.6%**) | 31.0% | 28.3% |
| Official-validator GRPO (A2) | 613/2000 (**30.6%**) | 32.2% | 29.1% |

A0 follows the cookbook envelope: Nemotron structured-output prompts, three cheap train rewards, official exam at test. A2 keeps the same LoRA / GRPO setup (r=16, G=8, 100 steps, T=1.1, β=0.01) and swaps in official-shaped train rewards plus a held-out `train__*` prompt generator. YAML barely moved (~25% → 28–29%); almost all of the lift is JSON. Remaining A0 failures are mostly missing required fields, extra keys, and list-vs-schema-dump shape — not forgotten fences.

A0’s 29.6% is close to Liquid’s published 29.7%, but the decoder is different (HF `generate` vs llama.cpp), so do not read this as a rematch of the cookbook table.

### Probe (128 even seeds `0..254`) — pass@8 and other trainers

The 128-prompt slice is 6.4% of the test set. It flattered A0 on greedy (35.2% vs 32.8%) relative to the full set (29.6% vs 30.6%). Use it for pass@8, CoRPO, and RAFT, not as the A0 vs A2 greedy ranking.

| Run | Official greedy | JSON | YAML | pass@8 (T=1.0) |
|---|---:|---:|---:|---:|
| Base `LFM2.5-350M` | 27/128 (**21.1%**) | 16.4% | 25.4% | 32.8% |
| Cookbook GRPO (A0) | 45/128 (**35.2%**) | 39.3% | 31.3% | 44.5% |
| Official-validator GRPO (A2) | 42/128 (**32.8%**) | 34.4% | 31.3% | **49.2%** |
| CoRPO (`R_min=2.0`) | 39/128 (**30.5%**) | 27.9% | 32.8% | — |
| RAFT (filter then SFT) | 20/128 (**15.6%**) | 26.2% | 6.0% | — |

CoRPO and RAFT use the same ~4,000-rollout budget as A0. A curriculum-ordered A2 rerun (easy→hard, constant LR) scored 39/128 (**30.5%**) greedy and is not a win. An OPSA screen on the base greedy run is a skip: failures are as confident as passes on the lowest-20% token logprobs.

Compact metrics: [`artifacts/ifstruct-lfm350/`](artifacts/ifstruct-lfm350/). Merged weights are not in git.

![Figure 1](artifacts/ifstruct-lfm350/exam.png)

**Figure 1.** Official IFStruct pass, one seed. **a**, Greedy decode on the full 2,000-prompt split (bars) and pass@8 at T = 1 on the 128-prompt probe (circles). Cookbook GRPO and official-validator GRPO are highlighted. **b**, Full-set greedy pass split by output format.

![Figure 2](artifacts/ifstruct-lfm350/leftover.png)

**Figure 2.** Leftover official error mentions on the same 2,000 greedy generations (a completion can contribute more than one). Cookbook GRPO almost wipes fence failures. Extra keys and list-vs-schema-dump remain — those are not what the three cheap train rewards look at.

![Figure 3](artifacts/ifstruct-lfm350/train.png)

**Figure 3.** Training dynamics over 100 GRPO steps. **a**, Mean train-time reward (schema component for cookbook GRPO; official binary for official-validator GRPO). Thin traces are per-step means; thick traces are a 7-step moving average. **b**, Fraction of groups whose rewards have zero standard deviation.

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
