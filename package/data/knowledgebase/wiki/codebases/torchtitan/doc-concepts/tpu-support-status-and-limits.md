---
title: "TPU support status and limits: README claims vs the registries"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# TPU support status and limits: README claims vs the registries

## Definition
The TPU README closes both of its launch sections with the same three caveats:
- "Currently, only the `llama3_tpu` configuration is supported."
- "You must limit `max_context_length` (formerly `seq_len`) to 128" on a single host, and to 512 on
  GKE, because "support for the full context length is coming soon".
- "Memory and Model Flops Utilization (MFU) metrics might be inaccurate in the current version."

This page records what those statements mean and how far the code has moved past them. An agent
choosing a model or sequence length for a TPU run should trust the registries over these notes.

## In torchtitan (grounded)
- **"`llama3_tpu`" is a package, not a config.** No registry function has that name. The string
  names the Llama3 TPU package (`torchtitan/experiments/tpu/llama3`, whose parallelizer docstring
  uses `model_name="llama3_tpu"`). That package offers
  [`llama3_1b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L139 "torchtitan/experiments/tpu/llama3/config_registry#llama3_1b"),
  [`llama3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L178 "torchtitan/experiments/tpu/llama3/config_registry#llama3_8b"),
  [`llama3_70b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L244 "torchtitan/experiments/tpu/llama3/config_registry#llama3_70b"),
  [`llama3_405b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L310 "torchtitan/experiments/tpu/llama3/config_registry#llama3_405b")
  and scan-lane variants such as
  [`llama3_405b_scan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L477 "torchtitan/experiments/tpu/llama3/config_registry#llama3_405b_scan").
- **More model families are registered.**
  [`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer")
  lives in `train.py`, which imports the model packages `deepseek_v3`, `flux`, `llama3`,
  `muse_glimmer` and `qwen3` "to trigger model registration". (`muse_glimmer` is new at this
  commit.) Examples: Qwen3 dense
  ([`qwen3_1_7b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L215 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_1_7b"),
  [`qwen3_8b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L375 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_8b")),
  Qwen3 MoE ([`qwen3_moe_30b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L557 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_moe_30b"),
  [`qwen3_moe_235b_scan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L950 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_moe_235b_scan")),
  Muse Glimmer
  ([`muse_glimmer_30b_fused_ce`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L248 "torchtitan/experiments/tpu/muse_glimmer/config_registry#muse_glimmer_30b_fused_ce")),
  and DeepSeek-V3 ([`deepseek_v3_debugmodel`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/deepseek_v3/config_registry.py#L28 "torchtitan/experiments/tpu/deepseek_v3/config_registry#deepseek_v3_debugmodel")).
  Flux is routed to the upstream Flux trainer and bypasses the TPU integration. Only `llama3`,
  `qwen3`, `deepseek_v3` and `muse_glimmer` can use the graph-trainer lane; any other model raises
  "is not currently supported with GraphTrainer".
- **Sequence lengths in the registries.** Upstream replaced `seq_len`, `local_batch_size` and
  `global_batch_size` with a context length and token budgets:
  [`max_context_length`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L51 "torchtitan/config/configs#TrainingConfig.max_context_length"),
  [`num_tokens_per_microbatch_per_dp_rank`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L37 "torchtitan/config/configs#TrainingConfig.num_tokens_per_microbatch_per_dp_rank")
  and
  [`num_tokens_per_train_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L43 "torchtitan/config/configs#TrainingConfig.num_tokens_per_train_step").
  The 128 cap is simply
  [`llama3_1b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L139 "torchtitan/experiments/tpu/llama3/config_registry#llama3_1b")'s
  default (`max_context_length=128` with a budget of 4 x 128 tokens). `llama3_8b`, `llama3_70b` and
  `llama3_405b` default to `max_context_length=8192` and use splash attention. Qwen3 recipes use
  2048 to 4096 (128 only in smoke/debug variants), DeepSeek-V3 uses 2048, and Muse Glimmer 30B
  uses 16384. The README's `--training.max_context_length=512` in the GKE example is a
  command-line override that shrinks `llama3_8b`, not a hardware limit. Because the token budget is
  not overridden with it, the per-rank batch grows to 16 sequences (see
  [tpu-multihost-launch](tpu-multihost-launch.md)).
- **Hard constraints that do exist.** The code does impose shape limits, just not the README's.
  With the Pallas fused loss, the token count of a microbatch must be a multiple of the loss
  kernel's `b_block_size` (default 1024), or
  [`pallas_cross_entropy_loss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/loss.py#L18 "torchtitan/experiments/tpu/loss#pallas_cross_entropy_loss")
  raises. Splash-attention block sizes in each recipe (for example 2048 in `llama3_8b`) have to
  fit the sequence length you choose.
- **MFU accounting.** [`TPUConfig.enable_mfu_benchmark`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L139 "torchtitan/experiments/tpu/tpu_job_config#TPUConfig.enable_mfu_benchmark")
  and [`get_maxtext_decoder_nparams_and_flops`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/mfu_flops.py#L17 "torchtitan/experiments/tpu/mfu_flops#get_maxtext_decoder_nparams_and_flops")
  give a MaxText-comparable FLOP count over a benchmark window. The generic MFU printed by the
  metrics processor depends on the peak-FLOPS table (see
  [profiling-and-tooling](../concepts/profiling-and-tooling.md)).

## Why it matters / when it applies
- **Don't pick a model from the README caveats.** At this commit the notes are stale on model
  coverage and sequence length. The [torchtpu-backend](../concepts/torchtpu-backend.md) page
  reaches the same conclusion. Choose from the config registries and the per-topology recipes
  (`experiments/tpu/recipes/`, covered on their own pages).
- **The metrics caveat still applies.** TPU peak-memory stats are not reset per window, and MFU
  depends on how the device name maps to peak FLOPS. Cross-check against an XProf trace before you
  quote numbers (see [tpu-profiling-tensorboard](tpu-profiling-tensorboard.md)).
- **Single-host size limits are real.** "Only small models can run on a single host" still holds.
  `llama3_1b` on `v6e-4` is the documented single-host case. Larger recipes assume multi-host
  FSDP (see [tpu-multihost-launch](tpu-multihost-launch.md)).

> [!warning] Doc vs code
> - The README says "only the `llama3_tpu` configuration is supported". No registry function has
>   that name, and five model packages are registered at this commit.
> - The 128 / 512 context caps are command-line choices, not limits in the code. The registries
>   default to up to 16384.

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md) (lanes, graph-trainer model
  coverage, README drift), [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md) (how registration
  works), [loss-and-validation](../concepts/loss-and-validation.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md),
  [profiling-and-tooling](../concepts/profiling-and-tooling.md).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/experiments/tpu/llama3/config_registry`,
  `torchtitan/experiments/tpu/qwen3/config_registry`, `torchtitan/experiments/tpu/train`).
- Related doc-concepts: [tpu-vm-provisioning](tpu-vm-provisioning.md),
  [tpu-multihost-launch](tpu-multihost-launch.md), [tpu-profiling-tensorboard](tpu-profiling-tensorboard.md),
  [tpu-training-recipes](tpu-training-recipes.md),
  [trillium-v6e-recipes](trillium-v6e-recipes.md), [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md).

## Source
Extracted from `torchtitan/experiments/tpu/README.md` (the "Notes" lists under "Run a TorchTitan
training job" and "Deploy the training workload"; kept in place).
