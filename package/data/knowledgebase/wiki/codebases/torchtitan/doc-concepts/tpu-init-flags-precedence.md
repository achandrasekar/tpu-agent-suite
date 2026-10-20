---
title: "XLA and libtpu flags as config: precedence, bundles and overrides"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# XLA and libtpu flags as config: precedence, bundles and overrides

## Definition
Launchers usually set `XLA_FLAGS` and `LIBTPU_INIT_ARGS` in the environment. The README makes them
"citizens of the TorchTitan training configuration" instead. A recipe declares them in the
`libtpu_init_flags` and `xla_flags` fields, and the flags are resolved in three layers:
1. **Config.** This is the whole set the recipe declares. Passing `--libtpu-init-flags=...` or
   `--xla-flags=...` on the command line **replaces** that set, and the empty string drops it.
2. **CLI extras.** `--extra-libtpu-init-flags` and `--extra-xla-flags` **add** flags the config does
   not set and **override** the ones it does. `'!flag'` drops a single flag, which then falls back
   to the libtpu/XLA default.
3. **Environment.** An exported `LIBTPU_INIT_ARGS` or `XLA_FLAGS` "has the last word".

Values are whitespace-separated `flag=value` entries. Groups of flags that only make sense together
are named bundles, composed with `merge_flags`.

## In torchtitan (grounded)
- **Fields.** [`BaseTPUTrainerConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L241 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig")
  carries four fields:
  [`libtpu_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L245 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.libtpu_init_flags"),
  [`xla_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L242 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.xla_flags"),
  [`extra_libtpu_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L251 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.extra_libtpu_init_flags")
  and [`extra_xla_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_job_config.py#L248 "torchtitan/experiments/tpu/tpu_job_config#BaseTPUTrainerConfig.extra_xla_flags").
  All four have type [`FlagMap`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L45 "torchtitan/experiments/tpu/tpu_flags#FlagMap").
  A dedicated type is needed because the CLI would otherwise take a dict field's options from its
  default keys, and then a config that declares flags "could only retune those -- never add one".
  The CLI rule [`_flag_map_rule`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L57 "torchtitan/experiments/tpu/tpu_flags#_flag_map_rule")
  parses the whole string with [`parse_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L87 "torchtitan/experiments/tpu/tpu_flags#parse_flags"),
  which strips leading dashes and turns `!name` into the
  [`REMOVE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L42 "torchtitan/experiments/tpu/tpu_flags#REMOVE")
  sentinel.
- **When flags are applied.** [`_start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/gmain.py#L23 "torchtitan/experiments/tpu/gmain#_start_trainer")
  calls [`apply_init_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L109 "torchtitan/experiments/tpu/tpu_flags#apply_init_flags")
  first, before topology bootstrap and before any TPU device exists. The reason is in its docstring:
  libtpu latches the variables "while the first program is compiled, so flags applied after that
  point are silently ignored".
- **The merge.** [`_apply_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L223 "torchtitan/experiments/tpu/tpu_flags#_apply_flags")
  computes `{**config, **extra}` and then resolves it under the current environment value with
  [`resolve_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L99 "torchtitan/experiments/tpu/tpu_flags#resolve_flags"),
  where the environment layer wins and `REMOVE` entries drop out. It writes the result back with
  [`format_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L82 "torchtitan/experiments/tpu/tpu_flags#format_flags").
  It logs the INFO/WARNING lines the README quotes ("Overriding ...", "Removing ...", "...
  already sets ... in the environment; keeping it"). If both the config and the extras are empty,
  it returns early and leaves the environment untouched.
- **Bundles.** [`merge_flags`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L72 "torchtitan/experiments/tpu/tpu_flags#merge_flags")
  merges bundles and then keyword overrides, with later arguments winning, and renders Python bools
  as lowercase `true`/`false`. The bundles, each with its rationale in a comment:
  - [`SPARSECORE_LATENCY_HIDING_SCHEDULER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L150 "torchtitan/experiments/tpu/tpu_flags#SPARSECORE_LATENCY_HIDING_SCHEDULER.SPARSECORE_LATENCY_HIDING_SCHEDULER")
    lets the layer scheduler overlap FSDP collectives with compute. It needs the SparseCore
    aggregator, so the two flags move together.
  - [`DUAL_SPARSECORE_REDUCE_SCATTER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L160 "torchtitan/experiments/tpu/tpu_flags#DUAL_SPARSECORE_REDUCE_SCATTER.DUAL_SPARSECORE_REDUCE_SCATTER")
    is optimal for ZeRO-2 or a dependency-ordered ZeRO-3 interleave. Re-evaluate it for ZeRO-3
    prefetch.
  - [`HIERARCHICAL_REDUCE_SCATTER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L169 "torchtitan/experiments/tpu/tpu_flags#HIERARCHICAL_REDUCE_SCATTER.HIERARCHICAL_REDUCE_SCATTER")
    needs libtpu 0.0.47 or newer.
  - [`SELECTIVE_ICI_RS_PIPELINING`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L178 "torchtitan/experiments/tpu/tpu_flags#SELECTIVE_ICI_RS_PIPELINING.SELECTIVE_ICI_RS_PIPELINING")
    pipelines small (<= 1024 KiB per shard) reduce-scatters across the scan backward loop's
    back-edge. Multi-MiB weight reductions stay inside their own iteration so VMEM/HBM is not
    double-buffered.
  - [`ICI_AG_PIPELINING`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L186 "torchtitan/experiments/tpu/tpu_flags#ICI_AG_PIPELINING.ICI_AG_PIPELINING")
    starts each layer's FSDP weight all-gather one scan iteration early. It costs extra HBM for the
    prefetched next-layer weights.
  - [`CONVOLUTION_UNFUSING`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L192 "torchtitan/experiments/tpu/tpu_flags#CONVOLUTION_UNFUSING.CONVOLUTION_UNFUSING")
    unfuses elementwise ops from MXU matmuls when fusion degrades tile selection.
  - [`SPARSECORE_COLLECTIVE_OFFLOAD`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L200 "torchtitan/experiments/tpu/tpu_flags#SPARSECORE_COLLECTIVE_OFFLOAD.SPARSECORE_COLLECTIVE_OFFLOAD")
    moves FSDP all-gather and reduce-scatter onto the SparseCores.
  - [`TPU7X_THROUGHPUT_P_STATE`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L211 "torchtitan/experiments/tpu/tpu_flags#TPU7X_THROUGHPUT_P_STATE.TPU7X_THROUGHPUT_P_STATE")
    sets `xla_tpu_dvfs_p_state=7`. It changes device frequency and power, not the graph.
  - [`DISABLE_ENHANCED_LAUNCH_BARRIER`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/tpu_flags.py#L218 "torchtitan/experiments/tpu/tpu_flags#DISABLE_ENHANCED_LAUNCH_BARRIER.DISABLE_ENHANCED_LAUNCH_BARRIER")
    works around distributed hangs. torch_tpu already forces this off for distributed runs when
    the flag is absent.
- **Where recipes use them.** Four registry families now declare libtpu flags: Qwen3-235B, the
  Llama3 scan lane, the Qwen3 dense TPU7x 4x4x4 configs and Muse Glimmer 30B. (Before this commit,
  only the Qwen3-235B family did.)
  - **Llama3 scan lane.** [`_on_scan_lane`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L76 "torchtitan/experiments/tpu/llama3/config_registry#_on_scan_lane"), used by the
    `llama3_*_scan` configs, merges
    [`_LLAMA3_SCAN_FLAGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L56 "torchtitan/experiments/tpu/llama3/config_registry#_LLAMA3_SCAN_FLAGS"). That set is the TPU7x p-state, a 128 MiB
    scoped-VMEM limit, no offloading copies to SparseCore, hierarchical reduce-scatter, both ICI
    pipelining bundles, the latency-hiding scheduler and convolution unfusing. The same helper also
    sets `xla_flags["xla_enable_hlo_sharding_v3"] = "false"`, because Shardy HLO sharding v3
    segfaults on the GraphTrainer.
  - **Qwen3 dense on TPU7x 4x4x4.** [`_QWEN3_DENSE_4X4X4_FLAGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L491 "torchtitan/experiments/tpu/qwen3/config_registry#_QWEN3_DENSE_4X4X4_FLAGS")
    is commented "The Qwen3-8B tpu7x-4x4x4 launch flags, carried by the 32B config". It holds the
    p-state, a 65472 KiB VMEM limit, the latency-hiding scheduler and hierarchical reduce-scatter.
    [`qwen3_32b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L499 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_32b") merges it.
  - **Muse Glimmer 30B.** [`_MUSE_GLIMMER_4X4X4_FLAGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/muse_glimmer/config_registry.py#L42 "torchtitan/experiments/tpu/muse_glimmer/config_registry#_MUSE_GLIMMER_4X4X4_FLAGS")
    is the same four-part composition.
  - **Qwen3-235B.** [`_Q235_SHARED_FLAGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L73 "torchtitan/experiments/tpu/qwen3/config_registry#_Q235_SHARED_FLAGS")
    holds the TPU7x p-state plus a 128 MiB scoped-VMEM limit.
    [`_Q235_COMPILED_OFFLOAD_FLAGS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L89 "torchtitan/experiments/tpu/qwen3/config_registry#_Q235_COMPILED_OFFLOAD_FLAGS")
    adds the latency-hiding and dual-SparseCore bundles and copy-insertion region analysis. Its
    comment records the MFU going from 18.528% to 19.490% in the A/B.
    [`_qwen3_moe_235b_fsdp_activation_offload`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L844 "torchtitan/experiments/tpu/qwen3/config_registry#_qwen3_moe_235b_fsdp_activation_offload")
    applies the compiled-offload set.
    [`qwen3_moe_235b_fsdp_activation_offload_4x4x4`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L903 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_moe_235b_fsdp_activation_offload_4x4x4")
    adds `HIERARCHICAL_REDUCE_SCATTER` on top.
    [`qwen3_moe_235b_scan`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/qwen3/config_registry.py#L950 "torchtitan/experiments/tpu/qwen3/config_registry#qwen3_moe_235b_scan")
    applies the shared set and adds hierarchical reduce-scatter, both ICI pipelining bundles, the
    latency-hiding scheduler and a raised backward layer-scheduler compute threshold. Its comment
    explains why the flags live in the config: "Left to the
    command line these get forgotten, and a scan step measured without them runs ~10% slow."

## Why it matters / when it applies
- **An exported variable silently beats the recipe.** The GKE example in the README itself sets
  `LIBTPU_INIT_ARGS` via `gcluster job submit --env`. Any flag in it overrides the recipe's value, with only a
  WARNING in the log. To tune a recipe, use `--extra-libtpu-init-flags` so that the change is part
  of the printed config.
- **`!flag` cannot remove an environment flag.** The environment layer is resolved last, so a
  `!flag` in the extras only drops a flag the config declared. This is a code observation.
- **Inspect the resolved set.** `--debug.print-config`
  ([`print_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L422 "torchtitan/config/configs#DebugConfig.print_config"))
  prints the full configuration with the flags. The README warns that the flags "may still be
  overridden by a LIBTPU_INIT_ARGS environment var". XLA options can also be overridden per compile
  inside torch_tpu (`TORCH_TPU_INTERNAL_XLA_OPTIONS`, the `custom_compiler_options` context
  manager).

> [!warning] Doc vs code
> The README's worked example cites "`qwen3_8b_tpu7x` (qwen3/config_registry.py#L534), which
> declares five flags including `xla_tpu_dvfs_p_state=7`". No such registry function exists at
> this commit. Line 534 now falls inside `qwen3_32b_4x8x8`, which sets no libtpu flags of its own.
> The closest current equivalent is `_QWEN3_DENSE_4X4X4_FLAGS` (the "Qwen3-8B tpu7x-4x4x4 launch
> flags"), which does include `xla_tpu_dvfs_p_state=7` but expands to seven flags. Treat the
> example commands as syntax illustrations. For real flag sets, read the registry constants above
> or the per-topology recipe READMEs.

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md) (where flag application sits
  in the launch sequence), [job-configuration](../concepts/job-configuration.md) (CLI parsing and
  custom tyro rules), [fsdp-sharding](../concepts/fsdp-sharding.md) (the collectives the bundles
  schedule).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/experiments/tpu/tpu_flags`,
  `torchtitan/experiments/tpu/tpu_job_config`, `torchtitan/experiments/tpu/qwen3/config_registry`,
  `torchtitan/experiments/tpu/llama3/config_registry`, `torchtitan/experiments/tpu/muse_glimmer/config_registry`).
- Related doc-concepts: [tpu-multihost-launch](tpu-multihost-launch.md),
  [tpu-vm-provisioning](tpu-vm-provisioning.md), [tpu-rl-with-ray](tpu-rl-with-ray.md) (which
  hard-codes its own `LIBTPU_INIT_ARGS` per actor), [tpu-training-recipes](tpu-training-recipes.md),
  [ironwood-tpu7x-recipes](ironwood-tpu7x-recipes.md) and [trillium-v6e-recipes](trillium-v6e-recipes.md)
  (per-topology flag sets).
- Hardware: [TPU7x](../../../hardware/tpu-7x.md), [TPU v6e](../../../hardware/tpu-v6e.md).

## Source
Extracted from `torchtitan/experiments/tpu/README.md` (section "Configuring XLA and libtpu flags",
with "Usage", "Flag bundling and merging flags" and "Logging"; kept in place).
