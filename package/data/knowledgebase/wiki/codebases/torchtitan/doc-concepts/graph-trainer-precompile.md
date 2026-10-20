---
title: "GraphTrainer pre-compile (compile-on-one-rank artifacts)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/graph_trainer/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# GraphTrainer pre-compile (compile-on-one-rank artifacts)

## Definition
The GraphTrainer README describes pre-compile as a two-step workflow. First, "compile AOT graphs
on a single GPU and save them to disk". Then "load them on all ranks during training - skipping
compilation entirely". The artifact is rank-agnostic because it is produced with
compile-on-one-rank (CooR). The README says artifacts are ephemeral: they are tied to the
PyTorch version, the CUDA version, the model architecture and the parallelism layout, and a
stale artifact is detected "via config fingerprinting" and raises when loaded.

## In torchtitan (grounded)
- **Producer.** [`main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile_main.py#L374 "torchtitan/experiments/graph_trainer/precompile_main#main")
  in `precompile_main.py` starts one process with a `fake` process group, sets
  `torch.distributed.config.compile_on_one_rank = True`, and traces through
  [`_precompile_aot_fx_trace`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile_main.py#L217 "torchtitan/experiments/graph_trainer/precompile_main#_precompile_aot_fx_trace").
  Its setup hard-codes `torch.device("cuda:0")`.
- **Artifact.** [`PrecompiledFxTraceArtifact`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile.py#L191 "torchtitan/experiments/graph_trainer/precompile#PrecompiledFxTraceArtifact")
  holds a pickled GraphModule (compiled Triton kernels from regional Inductor are baked in) plus
  the subclass layout metadata needed to unwrap DTensor inputs. It is written by
  [`precompile_fx_trace_save`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile.py#L308 "torchtitan/experiments/graph_trainer/precompile#precompile_fx_trace_save")
  through [`DiskStorageAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/storage.py#L49 "torchtitan/experiments/graph_trainer/storage#DiskStorageAdapter").
  [`_register_coor_ops`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile.py#L128 "torchtitan/experiments/graph_trainer/precompile#_register_coor_ops")
  registers the device-mesh opaque ops (for example `_runtime_compute_coordinate_on_dim`) that
  resolve rank-specific values at run time.
- **Consumer.** When `compile.precompile_artifact_dir` is set,
  [`GraphTrainer._load_precompiled_fx_trace`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/trainer.py#L217 "torchtitan/experiments/graph_trainer/trainer#GraphTrainer._load_precompiled_fx_trace")
  calls [`precompile_fx_trace_load`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile.py#L335 "torchtitan/experiments/graph_trainer/precompile#precompile_fx_trace_load")
  instead of tracing. [`construct_default_graph_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L420 "torchtitan/experiments/graph_trainer/passes#construct_default_graph_passes")
  then returns only [`cudagraph_pass`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/cudagraph.py#L261 "torchtitan/experiments/graph_trainer/cudagraph#cudagraph_pass"),
  because CUDA graphs have to be captured on each rank at run time.
- **Fingerprint.** [`compute_config_fingerprint`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile.py#L73 "torchtitan/experiments/graph_trainer/precompile#compute_config_fingerprint")
  hashes parameter/buffer names, shapes and dtypes, every public `ParallelDims` field, the
  compile mode, backend, pass list, memory policy and `full_recompute_save_ops` selectors,
  EP-overlap settings, `torch.__version__` and, when CUDA is
  present, the device capability.
  [`_validate_config_fingerprint`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/precompile.py#L145 "torchtitan/experiments/graph_trainer/precompile#_validate_config_fingerprint")
  raises when it does not match. The code has no separate CUDA-toolkit field. The README's "CUDA
  version" is only covered because the torch version string carries it (for example
  `+cu130`).

## Why it matters / when it applies
Compiling a large graph can take a long time, and every rank repeats it. Pre-compile moves that
cost to one process ahead of time, so every rank just loads the result. For multi-node runs the
README says to put the artifact directory on a shared filesystem. It also says the loader has
to run under `run_train_precompile.sh`, which passes `--virtual-local-rank` so that every
worker sees `cuda:0`, the device the artifact was compiled for. The README's composability
table still marks pre-compile as work in progress, and GraphPP cannot load artifacts yet. The
artifact is a Python pickle, so only load it from a trusted directory.

**On TPU.** [`TpuGraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer.py#L90 "torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer#TpuGraphTrainer")
takes the same `_load_precompiled_fx_trace` branch when `precompile_artifact_dir` is set, and
graph-split mode refuses a precompiled artifact. No in-tree tool produces one for TPU, though:
`precompile_main.py` is CUDA-only (`cuda:0`, Triton kernels in the artifact). TPU runs rely
instead on TorchTPU's own XLA executable caches, which a comment in the graph-split code says
"remain supported for every compiled child" (see [torchtpu-backend](../concepts/torchtpu-backend.md)).

## Connections
- Code concepts: [torch-compile-integration](../concepts/torch-compile-integration.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md).
- Module catalogs: [module map](../catalog/index.md) (`experiments/graph_trainer/precompile`,
  `experiments/graph_trainer/precompile_main`).
- Related doc-concepts: [graph-trainer](graph-trainer.md),
  [graph-pp-pipeline-parallel](graph-pp-pipeline-parallel.md).

## Source
Extracted from `torchtitan/experiments/graph_trainer/README.md` (kept in place).
