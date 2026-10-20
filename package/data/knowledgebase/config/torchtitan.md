---
slug: torchtitan
repo: https://github.com/google-pytorch/torchtitan
# Pin (main HEAD, 2026-10-15); also recorded in wiki/codebases/torchtitan/index.md `commit:`.
# `prepare`, `finalize` and `verify` all re-acquire this ref, so when ingesting from a
# developer working copy comment it out until the run is finished, then restore it.
ref: 033f131322a8b5b026ac53d734c689d664442bf8
acquire: clone
wiki_subdir: codebases
source_type: code
# Catalog tier: the symbol index (catalog/symbols.tsv, edges.tsv, index.md)
# instead of one page per module; `full` profile keeps signatures and doc lines in the rows
# because customers have no source on disk.
catalog: index
index_profile: full
index_shard_depth: 0
# Pure Python: no bazel_targets and no compile DB, unlike torch_tpu's C++ backend. The
# python indexer runs directly, so `prepare` has no cold bazel/scip-clang stage.
languages: [python]
index_shards: ["torchtitan/*"]
tests: ["tests/**/*.py"]
# No top-level README and no docs/ tree; the prose lives beside the code it documents
# (models, experiments, overrides, quantization, structured logging) plus the repo's own
# agent rules under .claude/, which read as design notes for the same subsystems.
docs:
  - "**/README*.md"
  - "torchtitan/components/quantization/*.md"
  - ".claude/CLAUDE.md"
  - ".claude/rules/*.md"
# The wiki ships without the source. Catalog links, OKF `sources` and change
# pages point at GitHub at the pinned commit; the local checkout (raw/code/torchtitan) is
# gitignored and excluded from the dist.
source_url: "https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8"
# Keep the shipped catalog focused: tests and vendored code keep a citeable page but drop
# the member body (coverage_collapse); nothing is excluded, so every citation resolves.
coverage_collapse: ["tests/**", "**/*_test.*", "**/test_*.py", "benchmarks/**"]
# Code withheld from the wiki (wikify >= 0.4.1 `source_exclude` / `symbol_exclude`) is listed
# in .wikify/torchtitan.exclude.yaml, which is never committed: re-ingest only with that file in
# place.
synthesis_focus: >-
  What a customer training a model on TPU with TorchTitan needs: how a job config selects
  a train spec and builds the mesh, how each parallelism axis (FSDP, TP, PP, CP, EP) is
  applied and composed, activation checkpointing and torch.compile integration, the
  checkpoint and state-dict-adapter path to and from Hugging Face, quantization and
  precision, the training loop and its metrics, and the TPU experiment backend.
agenda: subsystems
# The planner's own units duplicate the topic-named concepts below (its
# `components`/`distributed`/`core` umbrellas re-cover ground already split per
# mechanism), and one page per model is twelve pages of the same building blocks:
# attention, decoder, MoE, embedding, param init. Those fold into
# `model-definitions`, seeded at models/common, with per-model differences as
# sections. Coverage is unaffected - every module still gets a catalog row.
agenda_exclude:
  - torchtitan
  - torchtitan/components
  - torchtitan/distributed
  - torchtitan/hf_datasets
  - torchtitan/overrides
  - torchtitan/models/llama3
  - torchtitan/models/qwen3
  - torchtitan/models/qwen3_5
  - torchtitan/models/deepseek_v3
  - torchtitan/models/gpt_oss
  - torchtitan/models/flux
  - torchtitan/models/deepseek_v4
  - torchtitan/models/kimi_k2_7
  - torchtitan/models/kimi_k3
  - torchtitan/models/muse_glimmer
  - torchtitan/models/qwen3_6
  - torchtitan/models/qwen3_8
agenda_max: 24
---

# torchtitan - knowledge base ingest config

TorchTitan on TPU (the PyTorch-native distributed training platform: parallelism,
checkpointing, quantization and model definitions), ingested as the grounded code silo
behind [`wiki/codebases/torchtitan.md`](../wiki/codebases/torchtitan.md).

Pure Python, so `prepare` runs the python indexer alone - no bazel target list and no
compile database, unlike the torch_tpu ingest.

## Concepts

- **training-loop** - seeds: (subsystem: torchtitan/train.py)
- **trainer-lifecycle** - seeds: (subsystem: torchtitan/trainer.py)
- **job-configuration** - seeds: (subsystem: torchtitan/config)
- **parallel-dims-and-mesh** - seeds: `ParallelDims`, `build_mesh`, `world_mesh`
- **fsdp-sharding** - seeds: (subsystem: torchtitan/distributed/fsdp.py)
- **tensor-parallel** - seeds: `NoParallel`, `set_decoder_sharding_config`, `set_gqa_attention_sharding`, `set_dense_ffn_sharding`, `set_moe_sharding_config`, `colwise_config`, `rowwise_config`, `norm_config`, `dense_param_placement`, `_maybe_enable_async_tp`
- **pipeline-parallel** - seeds: (subsystem: torchtitan/distributed/pipeline_parallel.py)
- **context-parallel** - seeds: `prepare_context_parallel_input`, `cp_shard`, `ContextParallelTransform`, `CPInnerAttention`, `KVAllGatherCPFlexInnerAttention`, `UlyssesCPFlexInnerAttention`
- **expert-parallel** - seeds: (subsystem: torchtitan/distributed/minimal_async_ep)
- **activation-checkpointing** - seeds: (subsystem: torchtitan/distributed/activation_checkpoint.py)
- **dtensor-and-sharding-types** - seeds: (subsystem: torchtitan/distributed/spmd_types.py)
- **torch-compile-integration** - seeds: (subsystem: torchtitan/distributed/compile.py)
- **checkpointing** - seeds: (subsystem: torchtitan/components/checkpointer)
- **state-dict-adapters** - seeds: (subsystem: torchtitan/protocols/state_dict_adapter.py)
- **optimizer-and-lr-schedule** - seeds: (subsystem: torchtitan/components/optimizer)
- **loss-and-validation** - seeds: (subsystem: torchtitan/components/loss.py)
- **dataloading-and-tokenizer** - seeds: (subsystem: torchtitan/components/data)
- **quantization** - seeds: (subsystem: torchtitan/components/quantization)
- **lora** - seeds: (subsystem: torchtitan/components/lora.py)
- **train-spec-and-model-protocol** - seeds: (subsystem: torchtitan/protocols)
- **model-definitions** - seeds: (subsystem: torchtitan/models)
- **metrics-and-observability** - seeds: (subsystem: torchtitan/observability)
- **profiling-and-tooling** - seeds: (subsystem: torchtitan/tools)
- **torchtpu-backend** - seeds: `TPUTrainer`, `start_trainer`, `TPUConfig`, `TPUTrainerConfig`, `TpuGraphTrainer`, `TpuGraphTrainerConfig`, `run_distributed`, `maybe_init_distributed`, `apply_spmd_safe`, `AcceleratorDeviceType`, `PallasCrossEntropyLoss`, `pallas_cross_entropy_loss`, `wrap_model`, `ptd_checkpoint_wrapper_with_early_stop`, `JaxProfiler`, `StateValidatorCallback`

## Coverage note

`torchtitan/experiments/` - including `experiments/tpu`, the TPU backend - is indexed and
appears in the symbol catalog, but never becomes a concept page: wikify's
`discover.DEFAULT_EXCLUDES` classifies any `experiments/` path as non-library alongside
tests and examples, and that list is not configurable. For most repos that is right; here
it hides the TPU-specific code this knowledge base most wants. Report upstream; until then
the TPU backend gets its page through explicit symbol seeds (`torchtpu-backend` above):
`packet.resolve_seeds` matches against the whole graph with no library filter, so symbol
seeds reach `experiments/tpu` where a `(subsystem: ...)` seed cannot.
