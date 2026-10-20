---
title: "DAPO Math: TitanRL's single-turn reference recipe (Qwen3-4B-Base, Math-Verify reward)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/rl/examples/dapo_math/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# DAPO Math: TitanRL's single-turn reference recipe (Qwen3-4B-Base, Math-Verify reward)

## Definition
DAPO Math is the reference experiment of the RL stack ([rl-post-training](rl-post-training.md)):
the top-level README's quick start now runs it instead of the old `alphabet_sort` smoke test. It
trains Qwen3-4B-Base with the DAPO loss on a filtered version of DAPO-Math-17k (the verifiable math
set released with the DAPO paper). Each episode is single-turn: user math problem -> one assistant
solution -> binary reward. The prompt asks for step-by-step reasoning ending in
`Answer: \boxed{...}`; the scorer takes the last boxed expression and checks it against the
reference answer with Math-Verify.

## In torchtitan (grounded)
- **Data.** [`DapoMathDataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/data.py#L85 "torchtitan/experiments/rl/examples/dapo_math/data#DapoMathDataset") loads
  the 12,643-row `hamishivi/DAPO-Math-17k-Processed_filtered` set (one user prompt plus a verifiable
  final answer per row) and wraps each problem in
  [`_MATH_PROMPT_TEMPLATE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/data.py#L17 "torchtitan/experiments/rl/examples/dapo_math/data#_MATH_PROMPT_TEMPLATE").
  [`AIME2025Dataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/data.py#L112 "torchtitan/experiments/rl/examples/dapo_math/data#AIME2025Dataset") supplies all 30
  AIME 2025 (I+II) problems for validation, unshuffled. Both are endless, resumable streams
  ([`_CyclingDataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/data.py#L36 "torchtitan/experiments/rl/examples/dapo_math/data#_CyclingDataset"): training
  reshuffles at each epoch boundary; `state_dict` captures RNG, order and position).
- **Environment.** [`DapoMathEnv`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/env.py#L21 "torchtitan/experiments/rl/examples/dapo_math/env#DapoMathEnv") is a
  `MessageEnv` that emits the problem as one user message and returns `done=True` after the first
  response.
- **Reward.** [`RewardMathVerify`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/rubric.py#L62 "torchtitan/experiments/rl/examples/dapo_math/rubric#RewardMathVerify")
  calls [`score_math_response`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/rubric.py#L36 "torchtitan/experiments/rl/examples/dapo_math/rubric#score_math_response"),
  which extracts the last complete `\boxed{...}`
  ([`_last_boxed_expression`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/rubric.py#L21 "torchtitan/experiments/rl/examples/dapo_math/rubric#_last_boxed_expression"))
  and returns 1.0 only if Math-Verify equates it with the ground truth. Malformed LaTeX scores 0
  rather than failing the loop. Math-Verify timeouts are disabled for now (a TODO in the code: its
  signal-based timeout needs the main thread, and rollouts run in worker threads).
- **Wiring.** [`DapoMathWorker`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/rollouter.py#L23 "torchtitan/experiments/rl/examples/dapo_math/rollouter#DapoMathWorker")
  is pure config on a `RolloutWorker`: the Math-Verify rubric (`error_reward=0.0`), the env, a
  one-turn `TokenEnv`, and an
  [`AdvantageEstimator`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/advantage.py#L22 "torchtitan/experiments/rl/rollout/advantage#AdvantageEstimator") with
  [`should_std_normalize`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/advantage.py#L32 "torchtitan/experiments/rl/rollout/advantage#AdvantageEstimator.Config.should_std_normalize")=False
  (group-mean-centered, not std-normalized, advantages: the Dr.GRPO mean baseline, which is also the
  class default).
  [`DapoMathRollouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/rollouter.py#L51 "torchtitan/experiments/rl/examples/dapo_math/rollouter#DapoMathRollouter") adds
  the train/validation datasets.
- **Recipes.** [`rl_dapo_qwen3_4b_math_8k`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/config_registry.py#L154 "torchtitan/experiments/rl/examples/dapo_math/config_registry#rl_dapo_qwen3_4b_math_8k")
  (default) and
  [`rl_dapo_qwen3_4b_math_32k`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/config_registry.py#L163 "torchtitan/experiments/rl/examples/dapo_math/config_registry#rl_dapo_qwen3_4b_math_32k")
  both call [`_qwen3_4b_dapo_math_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/dapo_math/config_registry.py#L46 "torchtitan/experiments/rl/examples/dapo_math/config_registry#_qwen3_4b_dapo_math_config"):

  | Setting | 8K | 32K |
  |---|---|---|
  | Response budget (generator `max_tokens`) | 8,192 | 32,768 |
  | Packing length (`max_total_tokens`) | 10,240 | 34,816 |
  | Prompt budget (README) | 2,048 | 2,048 |

  Shared: 150 optimizer steps on one 8-GPU node; one TP=2 trainer (2 GPUs) and six independent
  TP=1 vLLM generators
  ([`num_generators`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L327 "torchtitan/experiments/rl/controller#Controller.Config.num_generators")=6,
  routed by [`LeastLoadedRoutingStrategy`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/routing/strategies.py#L79 "torchtitan/experiments/rl/routing/strategies#LeastLoadedRoutingStrategy"));
  8 prompt groups x 16 completions per step; `target_offpolicy_steps=4` sizing the
  `RolloutGroupWorkBuffer`; constant LR `1e-6` (no warmup, `min_lr_factor=1.0`);
  [`DAPOLoss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/losses/dapo.py#L23 "torchtitan/experiments/rl/losses/dapo#DAPOLoss") with clip `[0.2, 0.28]` inside an
  8-chunk [`ChunkedLossWrapper`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/loss.py#L515 "torchtitan/components/loss#ChunkedLossWrapper"); an fp32 LM head via
  [`LMHeadCastConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/models/cast_linear.py#L63 "torchtitan/experiments/rl/models/cast_linear#LMHeadCastConverter") with a bf16
  forward everywhere else; varlen attention; `aot_eager` compile; Qwen3 renderer with thinking
  enabled; HF checkpoint load from `example_checkpoint/Qwen3-4B-Base`. The 32K variant has not been
  benchmarked.
- **Run.** `python -m torchtitan.experiments.rl.train --module dapo_math --config
  rl_dapo_qwen3_4b_math_8k` (`dapo_math` resolves through
  [`_supported_experiments`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/__init__.py#L7 "torchtitan/experiments/__init__#_supported_experiments")); setup installs
  the pinned `math-verify` from the recipe's `requirements.txt` and downloads Qwen3-4B-Base with
  `scripts/download_hf_assets.py`.

## Why it matters / when it applies
This is the recipe to start from for a single-turn, verifiable-reward task, and it is the workload
the README's other guides assume: the vLLM engine-metrics examples in [rl-metrics](rl-metrics.md)
and the Verifiers bridge ([verifiers-rl-integration](verifiers-rl-integration.md)), which reruns
this exact workload with only the rollout path replaced. The fp32 LM head keeps the vocabulary
logits precise while the rest runs in bf16; per the converter's docstring, trainer and generator
see the same fp32 -> bf16 -> fp32 cast chain, so it does not break trainer/generator agreement
([rl-batch-invariance](rl-batch-invariance.md)). The README's reference plots (Math-Verify reward
and mean response length over the 150-step 8K run) were generated on July 20, 2026; eval results
and the 32K curves are still a TODO.

**TPU applicability.** Like the rest of the in-tree RL stack, the recipe targets 8 CUDA GPUs (vLLM
CUDA graphs, FA3/FA4). The dataset, env and rubric classes are pure Python and device-agnostic; the
TPU RL port (`experiments/tpu/rl_nc_ray`, see [tpu-rl-with-ray](tpu-rl-with-ray.md)) ships no DAPO
Math recipe at this pin.

> [!warning] Doc vs code
> - The README's 2,048-token "prompt budget" is not a separate setting: the recipe only sets the
>   response budget and the packing length, and the prompt budget is their difference *(agent
>   inference from `_qwen3_4b_dapo_math_config`)*.
> - The reference plot assets are named `qwen3_4b_7k_*` while the recipe and README describe an 8K
>   response budget.

## Connections
- Code concepts: [loss-and-validation](../concepts/loss-and-validation.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [job-configuration](../concepts/job-configuration.md),
  [tensor-parallel](../concepts/tensor-parallel.md),
  [optimizer-and-lr-schedule](../concepts/optimizer-and-lr-schedule.md),
  [torch-compile-integration](../concepts/torch-compile-integration.md),
  [checkpointing](../concepts/checkpointing.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/rl/examples/dapo_math/*`, `experiments/rl/models/cast_linear`).
- Related doc-concepts: [rl-post-training](rl-post-training.md),
  [verifiers-rl-integration](verifiers-rl-integration.md),
  [search-r1-tool-use-rl](search-r1-tool-use-rl.md) (the multi-turn counterpart),
  [rl-metrics](rl-metrics.md), [qwen3-model](qwen3-model.md).

## Source
Extracted from `torchtitan/experiments/rl/examples/dapo_math/README.md` (kept in place).
