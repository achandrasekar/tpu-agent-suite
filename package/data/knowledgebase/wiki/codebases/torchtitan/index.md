---
slug: torchtitan
commit: 033f131322a8b5b026ac53d734c689d664442bf8
scip_tool: scip-python
updated: 2026-10-15
okf_version: "0.2"
sources:
  - {resource: https://github.com/google-pytorch/torchtitan/tree/033f131322a8b5b026ac53d734c689d664442bf8, title: torchtitan @ 033f131322}
---

# torchtitan internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — TorchTitan, the PyTorch-native LLM training platform, as used on Cloud TPU: Python config trees select a model spec and build one device mesh, FSDP/TP/PP/CP/EP, activation checkpointing and per-block torch.compile are applied to a meta-device model, and the experiments/tpu backend runs the step on torch_tpu either as compiled blocks (TPUTrainer) or as one whole-step XLA graph (TpuGraphTrainer).

## Concepts (deep)
Grouped by the source area each page's citations point into.

### `torchtitan/distributed`
| Concept | Description | Status |
|---|---|---|
| [activation-checkpointing](concepts/activation-checkpointing.md) | How TorchTitan trades recompute for activation memory per transformer block: the four policies (FullAC, per-op SelectiveAC with its save list and every-other-matmul rule, RegionAC over model-declared torch_remat regions, compiler-driven MemoryBudgetAC), how the Trainer config selects them, where they are applied relative to compile and FSDP, and how the TPU recipes and the TPU graph trainer use them; read this before tuning activation memory. | fresh |
| [dtensor-and-sharding-types](concepts/dtensor-and-sharding-types.md) | How TorchTitan describes every parameter and activation layout once, as an spmd_types SpmdType of per-mesh-axis types (R, I, P, S(d), V + PartitionSpec), and how the two spmd_backend modes (partial_dtensor, spmd_types) turn that description into DTensor placements or into explicit per-axis collectives on local tensors via torchtitan/distributed/spmd_types.py; read this to understand placement errors, the spmd_types runtime mesh stack, or how a sharding plan maps to collectives. | fresh |
| [fsdp-sharding](concepts/fsdp-sharding.md) | How TorchTitan wraps a decoder with FSDP2 fully_shard per transformer block (FSDP or HSDP), resolves the storage mesh and DataParallelMeshDims under spmd_types, picks reshard-after-forward, handles tied embeddings and MoE experts (Shard(0) vs Shard(1), a separate efsdp mesh under EP, explicit prefetch), and why it disables FSDP's gradient division; read this to reason about FSDP memory/communication on TPU or to change the wrapping policy. | fresh |
| [parallel-dims-and-mesh](concepts/parallel-dims-and-mesh.md) | How ParallelDims validates the six parallelism degrees, builds one 1-D world mesh and unflattens it into dataloading, loss, dense and sparse views (pp, dp_replicate, fsdp/dp_shard, cp, tp, ep, efsdp), and how every parallelize function asks it for submeshes; read this before choosing mesh degrees for a TPU job or debugging a mesh-size assertion. | fresh |
| [pipeline-parallel](concepts/pipeline-parallel.md) | How TorchTitan cuts a decoder LLM into pipeline stages by module FQN, assigns (virtual) stages to PP ranks for looped and V-shaped schedules, applies the SPMD parallelisms per stage, and builds a torch.distributed.pipelining schedule whose microbatch count sets the bubble; read this before choosing pipeline_parallel_* knobs or wiring PP for a new model. | fresh |
| [torch-compile-integration](concepts/torch-compile-integration.md) | How TorchTitan applies torch.compile - one fullgraph compile per TransformerBlock, placed after activation checkpointing and before FSDP in every model's parallelize function, the async-TP Inductor setup it now owns, optional loss compile, and the regional_inductor scoop that keeps FlexInnerAttention on inductor under an aot_eager outer backend; read this to choose compile.backend/components (including backend=tpu on TPU) or to debug graph breaks, recompiles and backend errors. | fresh |

### `torchtitan/models/common`
| Concept | Description | Status |
|---|---|---|
| [context-parallel](concepts/context-parallel.md) | How TorchTitan enables context parallelism by swapping every inner attention for a CPInnerAttention kernel through ContextParallelTransform (FlexAttention with K/V all-gather, or Ulysses all-to-all onto heads), and how the model's preprocess_inputs shards tokens, labels, positions and block masks across the cp mesh axis with a head-tail or PTRR load balancer; read this before enabling context_parallel_degree for long-context training. | fresh |
| [model-definitions](concepts/model-definitions.md) | How TorchTitan models are assembled as fully-specified config trees from shared building blocks in torchtitan/models/common (configurable Linear, GQA attention with pluggable inner attention, SwiGLU FFN, MoE with RoutedExperts/GroupedExperts, RoPE, param_init), how a flavor function plus a seq_len becomes a built model, and how each model family differs (dense vs MoE, MLA, compressed sparse attention, hybrid linear attention, latent MoE, sliding-window/NoPE layers, diffusion); read this to understand or add a model, or to reason about the matmul shapes a model puts on the TPU MXU. | fresh |
| [tensor-parallel](concepts/tensor-parallel.md) | How TorchTitan applies Megatron-style tensor parallelism with optional sequence parallelism on the tp mesh axis: per-module ShardingConfig declarations built by decoder_sharding/moe_sharding helpers (colwise, rowwise, norm) and applied by Module.parallelize, the legacy NoParallel style, the compile-time async-TP gate, and the per-block collectives and memory it implies on TPU; read this before choosing a TP degree or debugging TP placements. | fresh |

### `torchtitan`
| Concept | Description | Status |
|---|---|---|
| [trainer-lifecycle](concepts/trainer-lifecycle.md) | The Trainer class and its Config: the order in which __init__ initializes distributed state, builds the model on the meta device, applies parallelism, materializes weights, and builds optimizer, dataloader, checkpointer and validator, plus the extension hooks subclasses (TPU, graph, Flux, fault-tolerant) override; read this before changing construction order or writing a trainer subclass. | fresh |
| [training-loop](concepts/training-loop.md) | How a TorchTitan run is launched (torchtitan.train main or the TPU start_trainer lane), how the step loop drives gradient accumulation, token-normalized loss, clipping, the optimizer and logging, and which TPU-specific variants replace parts of the step; read this to understand what happens per step and where host syncs and compilation boundaries are. | fresh |

### `torchtitan/components`
| Concept | Description | Status |
|---|---|---|
| [lora](concepts/lora.md) | How TorchTitan applies LoRA by rewriting the model config tree before the model is built (target Linear configs become dynamically generated LoRA subclasses, everything else becomes a freezing config), how adapters inherit tensor-parallel sharding, how LoRA composes with float8 quantization, and what it does and does not save in memory, optimizer state and checkpoints; read this before fine-tuning with adapters on TPU. | fresh |
| [loss-and-validation](concepts/loss-and-validation.md) | How TorchTitan computes the training and validation loss: the BaseLoss contract, sum reduction divided by a global valid-token count, fp32 upcast, vocab-parallel (loss-parallel) cross-entropy over tensor-parallel shards, the sequence-chunked loss that never materializes full logits, and which loss the TPU recipes actually select; read this before changing a loss, debugging loss numerics, or choosing a loss for a TPU run. | fresh |

### `torchtitan/protocols`
| Concept | Description | Status |
|---|---|---|
| [state-dict-adapters](concepts/state-dict-adapters.md) | How a model's StateDictAdapter renames, permutes, splits and merges weights between TorchTitan's native state dict and Hugging Face safetensors, how the checkpointer uses it to load HF weights directly into sharded DTensors and to export HF at the last step, and what breaks when a model has no adapter; read this before fine-tuning from an HF checkpoint on TPU or exporting a trained model back to HF. | fresh |
| [train-spec-and-model-protocol](concepts/train-spec-and-model-protocol.md) | The torchtitan/protocols contract between a model and the trainer: ModelSpec (the per-model bundle of config, max context length, and parallelize, pipelining, post-optimizer and state-dict-adapter hooks), BaseModel and its Config and preprocess_inputs hooks, the Module base class (Configurable nn.Module with param_init and init_states), and the declarative ShardingConfig that Module.parallelize turns into DTensor placements and forward-time redistributions; read this to add a model or change how its parameters and activations are sharded. | fresh |

### `torchtitan/components/checkpointer`
| Concept | Description | Status |
|---|---|---|
| [checkpointing](concepts/checkpointing.md) | How TorchTitan's checkpointer package (BaseCheckpointManager policies, the DCP CheckpointManager and the torch_checkpointing backend) saves and restores sharded training state with torch.distributed.checkpoint (DCP), when it saves, how the three async modes trade HBM-to-host blocking time for host RAM, how HF safetensors are loaded at startup and exported at the last step, and which config flags a TPU recipe sets; read this before resuming, fine-tuning from HF weights, or turning on async checkpointing. | fresh |

### `torchtitan/components/data`
| Concept | Description | Status |
|---|---|---|
| [dataloading-and-tokenizer](concepts/dataloading-and-tokenizer.md) | How TorchTitan turns a Hugging Face or JSONL corpus into fixed-size flat token batches: the GrainDataLoader / DatasetConfig composition (source -> filter/process -> mix/concat -> pack -> collate -> prefetch), per-DP-rank sharding and checkpointable Grain iterator state, concat-then-split and first-fit packing with per-document positions, padding masks and IGNORE_INDEX labels, the Hugging Face tokenizer's BOS/EOS handling, and what the flat [T] batch means on TPU; read this to add a dataset, debug data throughput or resume, or understand document masking. | fresh |

### `torchtitan/components/optimizer`
| Concept | Description | Status |
|---|---|---|
| [optimizer-and-lr-schedule](concepts/optimizer-and-lr-schedule.md) | How TorchTitan builds one or more Adam/AdamW optimizers per model part from regex param groups, how optimizer state is sized, sharded, FQN-flattened and checkpointed, what the fused/foreach/bf16-state implementations mean on TPU, how the MoE load-balancing step pre-hook works, and how the warmup-stable-decay LambdaLR schedule is computed and resumed; read this to size optimizer memory, change the optimizer or LR curve, or debug a resume. | fresh |

### `torchtitan/components/quantization`
| Concept | Description | Status |
|---|---|---|
| [quantization](concepts/quantization.md) | How TorchTitan enables low-precision (float8 rowwise, MXFP8, NVFP4) training by rewriting the model config tree before the model is built, which Linear and GroupedExperts modules get swapped, the hardware gates that decide whether a converter may run at all, the FSDP-managed MXFP8 weight cache, the MoE token-padding requirement, and what this means on Cloud TPU (only a TPU float8 subclass in the TPU experiment); read this before turning on float8/MXFP8/NVFP4 or porting a quantized recipe to TPU. | fresh |

### `torchtitan/config`
| Concept | Description | Status |
|---|---|---|
| [job-configuration](concepts/job-configuration.md) | How a TorchTitan job is described as a tree of Python dataclass configs (Configurable.Config), selected with --module/--config from a config_registry function, edited from the CLI by tyro, and rewritten by @override factories before anything is built; also the TrainingConfig/ParallelismConfig/CompileConfig knobs that matter for TPU. Read this to write or tune a recipe. | fresh |

### `torchtitan/distributed/minimal_async_ep`
| Concept | Description | Status |
|---|---|---|
| [expert-parallel](concepts/expert-parallel.md) | How TorchTitan's MinimalAsyncEP backend moves MoE routed tokens to expert-owner ranks and back: a count exchange through symmetric memory, device-side computation of every destination row, direct peer writes instead of an all-to-all plus permute, and hand-written backward custom ops; read this to understand MoE dispatch/combine communication, its capacity sizing and its (CUDA-only) constraints before choosing a moe_comm_backend. | fresh |

### `torchtitan/experiments/tpu`
| Concept | Description | Status |
|---|---|---|
| [torchtpu-backend](concepts/torchtpu-backend.md) | How torchtitan/experiments/tpu launches, configures, compiles and distributes a TorchTitan run on Cloud TPU through the torch_tpu runtime - the TPUTrainer and TpuGraphTrainer lanes, the TPUConfig knobs, per-process TPU topology bootstrap, the Pallas fused cross-entropy loss and the exact torch_tpu entry points it binds to; read this before choosing a compile lane, a memory policy or a launch recipe for TPU v6e or TPU7x. | fresh |

### `torchtitan/observability/structured_logger`
| Concept | Description | Status |
|---|---|---|
| [metrics-and-observability](concepts/metrics-and-observability.md) | How TorchTitan's observability subsystem (torchtitan/observability) stamps every training phase, marker and scalar with step, rank and task context, writes one JSONL file per rank and merges them into a Perfetto Gantt trace, and how the opt-in SDCReplayer replays a step's first forward/backward to detect silent data corruption; read this to instrument a TPU training loop, find where step time goes across ranks, plug in a custom log backend, or enable SDC checks. | fresh |

### `torchtitan/tools`
| Concept | Description | Status |
|---|---|---|
| [profiling-and-tooling](concepts/profiling-and-tooling.md) | How torchtitan/tools picks the device backend at import time and back-fills the device-module API on TPU and CPU, where the peak-FLOPS denominator for MFU comes from, how the Kineto/TPU profiler and memory-snapshot capture windows are scheduled, and the logging, GC and filesystem helpers the rest of TorchTitan depends on; read this before profiling a TPU run or trusting its MFU and memory numbers. | fresh |

## Doc-derived concepts
Concepts extracted from the project's own docs (README / `docs/`), grounded to the symbol catalog. The source docs stay in place.
- [checkpoint-conversion](doc-concepts/checkpoint-conversion.md)
- [config-system-conventions](doc-concepts/config-system-conventions.md)
- [dapo-math-rl-recipe](doc-concepts/dapo-math-rl-recipe.md)
- [dataset-mixing-and-packing](doc-concepts/dataset-mixing-and-packing.md)
- [deepseek-v3-model](doc-concepts/deepseek-v3-model.md)
- [deepseek-v4-model](doc-concepts/deepseek-v4-model.md)
- [distributed-conventions](doc-concepts/distributed-conventions.md)
- [ep-overlap-graph-chunking](doc-concepts/ep-overlap-graph-chunking.md)
- [experiments-policy](doc-concepts/experiments-policy.md)
- [flexshard-dist-muon](doc-concepts/flexshard-dist-muon.md)
- [float8-training](doc-concepts/float8-training.md)
- [flux-model](doc-concepts/flux-model.md)
- [full-configurations-and-recipes](doc-concepts/full-configurations-and-recipes.md)
- [fused-swiglu-and-helion-rope](doc-concepts/fused-swiglu-and-helion-rope.md)
- [grain-data-pipeline](doc-concepts/grain-data-pipeline.md)
- [graph-pp-pipeline-parallel](doc-concepts/graph-pp-pipeline-parallel.md)
- [graph-trainer-autoresearch](doc-concepts/graph-trainer-autoresearch.md)
- [graph-trainer-precompile](doc-concepts/graph-trainer-precompile.md)
- [graph-trainer](doc-concepts/graph-trainer.md)
- [ironwood-tpu7x-recipes](doc-concepts/ironwood-tpu7x-recipes.md)
- [jax-backend](doc-concepts/jax-backend.md)
- [kernel-overrides](doc-concepts/kernel-overrides.md)
- [kimi-k2-7-model](doc-concepts/kimi-k2-7-model.md)
- [kimi-k3-model](doc-concepts/kimi-k3-model.md)
- [llama3-model](doc-concepts/llama3-model.md)
- [long-context-training-recipe](doc-concepts/long-context-training-recipe.md)
- [model-authoring-conventions](doc-concepts/model-authoring-conventions.md)
- [model-config-transforms](doc-concepts/model-config-transforms.md)
- [model-registry](doc-concepts/model-registry.md)
- [muse-glimmer-model](doc-concepts/muse-glimmer-model.md)
- [mxfp8-training](doc-concepts/mxfp8-training.md)
- [nvfp4-training](doc-concepts/nvfp4-training.md)
- [qwen3-5-model](doc-concepts/qwen3-5-model.md)
- [qwen3-model](doc-concepts/qwen3-model.md)
- [repo-conventions](doc-concepts/repo-conventions.md)
- [rl-batch-invariance](doc-concepts/rl-batch-invariance.md)
- [rl-metrics](doc-concepts/rl-metrics.md)
- [rl-post-training](doc-concepts/rl-post-training.md)
- [search-r1-tool-use-rl](doc-concepts/search-r1-tool-use-rl.md)
- [structured-trace-logging](doc-concepts/structured-trace-logging.md)
- [testing-strategy](doc-concepts/testing-strategy.md)
- [torchax-backend](doc-concepts/torchax-backend.md)
- [torchft-fault-tolerance](doc-concepts/torchft-fault-tolerance.md)
- [torchtpu-wheel-and-container-image](doc-concepts/torchtpu-wheel-and-container-image.md)
- [tpu-init-flags-precedence](doc-concepts/tpu-init-flags-precedence.md)
- [tpu-multihost-launch](doc-concepts/tpu-multihost-launch.md)
- [tpu-profiling-tensorboard](doc-concepts/tpu-profiling-tensorboard.md)
- [tpu-recipe-scaling-patterns](doc-concepts/tpu-recipe-scaling-patterns.md)
- [tpu-rl-kuberay-deployment](doc-concepts/tpu-rl-kuberay-deployment.md)
- [tpu-rl-with-ray](doc-concepts/tpu-rl-with-ray.md)
- [tpu-support-status-and-limits](doc-concepts/tpu-support-status-and-limits.md)
- [tpu-training-recipes](doc-concepts/tpu-training-recipes.md)
- [tpu-vm-provisioning](doc-concepts/tpu-vm-provisioning.md)
- [transformers-modeling-backend](doc-concepts/transformers-modeling-backend.md)
- [trillium-v6e-recipes](doc-concepts/trillium-v6e-recipes.md)
- [verifiers-rl-integration](doc-concepts/verifiers-rl-integration.md)

## Changes
What changed between pins — pages affected and the commits behind them, newest first (see also [log.md](log.md)).
- [033f131322](changes/033f131322.md)

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); the **symbol
index** represents the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **13713** across 689 modules
- deep (concept pages): **843** (6.1%)
- index-only: **12870**
- represented total: **13713** (100.0%)
- classes represented: **1425/1425**

The catalog is the **symbol index**: [`catalog/index.md`](catalog/index.md) maps every module; `catalog/symbols/*.tsv` holds one row per symbol (anchor, path, line, kind, rank, hash, callers, citing pages) and `catalog/edges/*.tsv` the caller edges — grep them by anchor, never read them whole.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
