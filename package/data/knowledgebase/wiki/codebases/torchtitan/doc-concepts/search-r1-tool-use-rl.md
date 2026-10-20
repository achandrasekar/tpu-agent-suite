---
title: "Search-R1: multi-turn, tool-calling GRPO with an exact-match reward"
type: doc-concept
provenance: doc
source: torchtitan/experiments/rl/examples/search_r1/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Search-R1: multi-turn, tool-calling GRPO with an exact-match reward

## Definition
Search-R1 is the RL experiment's multi-turn, tool-using example. The policy has a `search` tool.
When it needs facts it emits a standard tool call, the environment runs dense retrieval (an e5
index over wiki-18 served at `http://127.0.0.1:8000/retrieve`) and returns the passages as a
`tool`-role message. The rollout continues until the model stops calling tools or runs out of
turns. The reward is exact match (EM) of the final answer against the golden answers on
NQ/HotpotQA. The README stresses that everything except the example folder and its config is the
framework's generic multi-turn rollouter and continuous-batching generator.

## In torchtitan (grounded)
- **Data.** [`SearchR1Dataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/data.py#L29 "torchtitan/experiments/rl/examples/search_r1/data#SearchR1Dataset") streams the
  `PeterJinGo/nq_hotpotqa_train` parquet into
  [`SearchR1Sample`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/data.py#L19 "torchtitan/experiments/rl/examples/search_r1/data#SearchR1Sample") (`question`,
  `golden_answers`).
- **Environment.** [`SearchR1Env`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/env.py#L113 "torchtitan/experiments/rl/examples/search_r1/env#SearchR1Env") is a
  [`MessageEnv`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/environment/message.py#L51 "torchtitan/experiments/rl/environment/message#MessageEnv"). It declares the `search`
  `ToolSpec`, reads the renderer-parsed `tool_calls` in
  [`step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/env.py#L149 "torchtitan/experiments/rl/examples/search_r1/env#SearchR1Env.step"), and calls the
  retriever ([`_search`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/env.py#L76 "torchtitan/experiments/rl/examples/search_r1/env#_search")). The turn budget is
  [`TokenEnv.Config.max_num_turns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/environment/token.py#L97 "torchtitan/experiments/rl/environment/token#TokenEnv.Config.max_num_turns").
  The worker config [`SearchR1Worker.Config.token_env`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/rollouter.py#L43 "torchtitan/experiments/rl/examples/search_r1/rollouter#SearchR1Worker.Config.token_env")
  sets it to 4, with
  [`max_rollout_tokens`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/environment/token.py#L89 "torchtitan/experiments/rl/environment/token#TokenEnv.Config.max_rollout_tokens")=3072.
- **Reward.** [`RewardExactMatch`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/rubric.py#L79 "torchtitan/experiments/rl/examples/search_r1/rubric#RewardExactMatch") is 0/1 EM
  by default. Two opt-in levers push against "closed-book" reward hacking (answering from
  memory without searching):
  [`no_search_penalty`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/rubric.py#L97 "torchtitan/experiments/rl/examples/search_r1/rubric#RewardExactMatch.Config.no_search_penalty")
  is subtracted from a correct answer that never searched, and
  [`retrieval_score`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/rubric.py#L102 "torchtitan/experiments/rl/examples/search_r1/rubric#RewardExactMatch.Config.retrieval_score")
  gives partial credit to a wrong answer whose search surfaced the gold answer. Truncated
  rollouts get
  [`truncation_reward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rubrics/rubric.py#L92 "torchtitan/experiments/rl/rubrics/rubric#Rubric.Config.truncation_reward")=0.
- **Wiring and recipes.** [`SearchR1Worker`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/rollouter.py#L21 "torchtitan/experiments/rl/examples/search_r1/rollouter#SearchR1Worker")
  holds the env, rubric and token budget, and
  [`SearchR1Rollouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/rollouter.py#L50 "torchtitan/experiments/rl/examples/search_r1/rollouter#SearchR1Rollouter")
  adds the datasets on top of
  [`Rollouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L43 "torchtitan/experiments/rl/rollout/rollouter#Rollouter"); all behavior is inherited.
  The recipes are
  [`rl_grpo_qwen3_1_7b_search_r1`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/config_registry.py#L63 "torchtitan/experiments/rl/examples/search_r1/config_registry#rl_grpo_qwen3_1_7b_search_r1"),
  [`rl_grpo_qwen3_8b_search_r1`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/config_registry.py#L138 "torchtitan/experiments/rl/examples/search_r1/config_registry#rl_grpo_qwen3_8b_search_r1"),
  a MoE perf variant,
  [`rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/config_registry.py#L173 "torchtitan/experiments/rl/examples/search_r1/config_registry#rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf"),
  and, new at this pin,
  [`rl_grpo_muse_glimmer_30b_search_r1`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/search_r1/config_registry.py#L273 "torchtitan/experiments/rl/examples/search_r1/config_registry#rl_grpo_muse_glimmer_30b_search_r1")
  (8 GPUs: a 6-GPU FSDP=3 x TP=2 trainer and a TP=2 generator; generator TP is capped at 2 by the
  model's 2 KV heads, and full activation checkpointing is required because Adam state allocated
  on the first step otherwise OOMs at step 2). Thinking is controlled only by the renderer's
  `enable_thinking` flag: the Qwen3 recipes pass
  `Qwen3RendererConfig(enable_thinking=False)` through
  [`RenderersLibraryConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/renderer.py#L33 "torchtitan/experiments/rl/renderer#RenderersLibraryConfig"), because
  chain-of-thought does not help short-answer EM and would use up the multi-turn token budget. The
  Muse Glimmer recipe uses the in-tree
  [`MuseGlimmerRendererConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/models/muse_glimmer/renderer.py#L72 "torchtitan/experiments/rl/models/muse_glimmer/renderer#MuseGlimmerRendererConfig")
  (harmony chat format, ATEM tool calls).
- **Checkpoint.** The README now documents the download step: `scripts/download_hf_assets.py`
  writes into a subdirectory named after the repo id's last component, which is exactly the
  `hf_assets_path` each recipe expects (for example `example_checkpoint/Muse-Glimmer-30B` or
  `example_checkpoint/Qwen3-1.7B`).

## Why it matters / when it applies
This is the template for any agentic RL task: implement a `MessageEnv` for the tool, a
`RewardFn` for the rubric, and a dataset. The framework handles multi-turn token accounting,
tool-message rendering and batching. The README reports validation EM rising from ~0.05 to
~0.41 (Qwen3-1.7B) and from ~0.26 to ~0.45 (Qwen3-8B) on the held-out NQ test split. The
retriever must run on spare GPUs before training starts.

**TPU applicability.** The recipe inherits the GPU-only runtime of the RL stack
([rl-post-training](rl-post-training.md)), and the retriever uses `--faiss_gpu`. The environment,
rubric and dataset classes are pure Python and device-agnostic. The TPU RL port
(`experiments/tpu/rl_nc_ray`) subclasses the same actors, but it ships no Search-R1 recipe.

**Minor doc drift:** the 8B results link points at `search_r1/config.py` at an older upstream
commit. At this pin the recipes live in `config_registry.py`.

## Connections
- Code concepts: [loss-and-validation](../concepts/loss-and-validation.md),
  [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [job-configuration](../concepts/job-configuration.md),
  [expert-parallel](../concepts/expert-parallel.md) (the 30B-A3B DeepEP perf recipe),
  [activation-checkpointing](../concepts/activation-checkpointing.md) (FullAC in the Muse Glimmer recipe),
  [state-dict-adapters](../concepts/state-dict-adapters.md) (Muse Glimmer HF layout on load).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/rl/examples/search_r1/*`, `experiments/rl/environment/*`).
- Related doc-concepts: [rl-post-training](rl-post-training.md),
  [rl-batch-invariance](rl-batch-invariance.md) (its cost table was measured on the 8B Search-R1 recipe),
  [dapo-math-rl-recipe](dapo-math-rl-recipe.md) (the single-turn counterpart),
  [tpu-rl-with-ray](tpu-rl-with-ray.md).

## Source
Extracted from `torchtitan/experiments/rl/examples/search_r1/README.md` (kept in place).
