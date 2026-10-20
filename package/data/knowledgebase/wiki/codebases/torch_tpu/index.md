---
slug: torch_tpu
commit: 6d2a45754bdb52743883f75fbf753408345f6d1b
scip_tool: scip-python
updated: 2026-10-16
okf_version: "0.2"
sources:
  - {resource: https://github.com/google-pytorch/torch_tpu/tree/6d2a45754bdb52743883f75fbf753408345f6d1b, title: torch_tpu @ 6d2a45754b}
---

# torch_tpu internals wiki

Generated, grounded wiki. Start from a concept (or an area); drill into cited symbols.
The commit pin above is the single source version for every page in this silo.

**Start here → [Overview](overview.md)** — TorchTPU, the PyTorch backend for Cloud TPU (C++ core in csrc/, Python package in src/torch_tpu/): ATen kernels become deferred StableHLO graphs that are keyed, cached, optimized by an MLIR pass pipeline and compiled through XLA and PjRt, with torch.compile, dynamic-shape, distributed and Pallas custom-kernel paths on top.

## Areas
One page per top-level area: what it is for, the units inside it with their entry points, and the small units that have no page of their own.
- [csrc](areas/csrc.md) — Area csrc of torch_tpu: the C++ runtime layer under the op code - PjRt client and device state, the tpu_dist collectives backend, and the MLIR optimization passes run on every StableHLO module before compilation.

## Concepts (deep)
Grouped by the source area each page's citations point into.

### `csrc/common`
| Concept | Description | Status |
|---|---|---|
| [compilation-cache-key](concepts/compilation-cache-key.md) | What makes two TorchTPU programs "the same" for the compilation cache: the graph signature (op names, edges, dtypes, donated inputs, core pinning) split into a shapeless key and a dimensions key, the compile-options fingerprint, and the per-op parameter keys a kernel must declare; read this to understand or debug unexpected recompilations. | fresh |
| [compilation-cache](concepts/compilation-cache.md) | How compiled XLA executables are cached in memory (tier 1), in a host-local file cache shared across processes (tier 2), and in a persistent root across jobs (tier 3), how bounded- and unbounded-dynamic entries match, and which env vars and Python controls govern the cache; read this when a workload recompiles too often or when setting up a multi-host or restart-friendly job. | fresh |
| [compilation-pipeline](concepts/compilation-pipeline.md) | How a traversal's StableHLO module becomes a PjRt executable, how XLA compile options are resolved from defaults, env vars and the custom_compiler_options context, and why those options are fingerprinted into the cache key; read this to change compiler flags or to understand fast-compile versus fast-runtime mode. | fresh |
| [dtype-and-precision](concepts/dtype-and-precision.md) | Which PyTorch dtypes TorchTPU supports and how they map to StableHLO and XLA types, how float32 matmul precision (DEFAULT, HIGH, HIGHEST) and excess precision are controlled and keyed into the cache, what computation_dtype does inside a kernel, and how autocast policies are set for TPU; read this for numerics questions and "why is my float32 matmul bf16" surprises. | fresh |
| [error-handling](concepts/error-handling.md) | How TorchTPU reports errors as absl::Status through the TT_ macros, enriches them with the root op name and an optional C++ error trace, adapts XLA compiler and OOM errors, and translates a TtError into c10::Error subclasses so users see RuntimeError, IndexError or NotImplementedError; read this to interpret a TPU error message or to enable C++ stack traces. | fresh |

### `csrc/ops`
| Concept | Description | Status |
|---|---|---|
| [aten-kernel-authoring](concepts/aten-kernel-authoring.md) | How a TorchTPU ATen kernel is written: the TT_KERNEL macro that names the op, logs and type-checks arguments, computes parameter cache keys, captures Python context and translates errors, plus the UnaryOp, BinaryOp and TernaryOp helpers that turn a StableHLO builder into a complete out-of-place, in-place or out= kernel; read this before adding or debugging an op. | fresh |
| [python-context-and-op-names](concepts/python-context-and-op-names.md) | How each dispatched op is identified by an OpName enum kept on a thread-local stack, how the Python call chain and optional traceback are captured at dispatch and replayed as MLIR locations at build time, and how the root op name reaches error messages and profiles; read this to understand op attribution in xprof and error text, or the cost of enabling tracebacks. | fresh |
| [scan-and-custom-kernels](concepts/scan-and-custom-kernels.md) | How externally authored StableHLO kernels (such as Pallas exports) are registered, specialized to concrete shapes, partitioned from global to per-chip view and inlined into the current program, and how prefix scans are lowered either as StableHLO while loops or as native associative scans; read this to understand what happens under torch_tpu's custom-kernel API or a cumulative op. | fresh |
| [stablehlo-op-builder](concepts/stablehlo-op-builder.md) | How a kernel expresses its math as a StableHLO builder closure (nullary, unary, binary, N-ary or dynamic), how those closures are lifted into the generic MlirOpBuilder the deferred-op machinery calls, and the utility layer for constants, broadcasting, dtype queries and module serialization; read this to write or read a Build*Shlo function. | fresh |

### `csrc/eager`
| Concept | Description | Status |
|---|---|---|
| [deferred-execution-and-materialization](concepts/deferred-execution-and-materialization.md) | How a TPU tensor's data is a graph of deferred ops until something needs real bytes, and how materialization turns that graph into compiled XLA executions; read this before debugging "why did this op run now" or tuning eager-mode fusion. | fresh |
| [eager-op-dispatch](concepts/eager-op-dispatch.md) | How every TorchTPU ATen kernel funnels through DispatchOp, how scalars are promoted so executables stay reusable, and where the op name, cache keys and Python context are attached; read this to understand what a kernel must supply and why a scalar argument does or does not cause a recompile. | fresh |

### `csrc/distributed`
| Concept | Description | Status |
|---|---|---|
| [process-group-and-collectives](concepts/process-group-and-collectives.md) | How torch.distributed collectives on TPU are implemented by ProcessGroupTpu as deferred StableHLO collective ops over IOTA replica groups into a topology-ordered device assignment, how ranks map to global device ids through a node-local-multiplexed c10d store exchange, how subgroups, uneven all-to-all and cross-host send/recv work, and when a collective forces a graph split; read this to understand DDP/FSDP/MoE-style code on TPU and the "materialize collective tensors" behavior. | fresh |

### `csrc/internal/compile`
| Concept | Description | Status |
|---|---|---|
| [torch-compile-backend](concepts/torch-compile-backend.md) | How TorchTPU's TpuBackend plugs into torch.compile through AOTAutograd, traces the FX graph with placeholder tensors into StableHLO, compiles it (optionally asynchronously) through the shared compilation cache, executes it through a pre-bound C++ runner with graph-safe RNG state handling, lowers scan and associative_scan, and serializes artifacts; read this before compiling a model or debugging graph breaks and recompiles. | fresh |

### `csrc/internal/dynamism`
| Concept | Description | Status |
|---|---|---|
| [eager-dynamic-shapes](concepts/eager-dynamic-shapes.md) | How mark_dynamic annotates a tensor dimension with lower and upper bounds so that eager programs compile once at the bound and run at the logical size, how the bounded-dynamic cache entries with pad and slice adapter kernels serve later shapes, and what the pad/slice modules do; read this to cut eager recompilation for variable sequence lengths. | fresh |

### `csrc/internal/profiler`
| Concept | Description | Status |
|---|---|---|
| [profiler](concepts/profiler.md) | How torch_tpu profiles TPU programs: the profile context manager and start_trace/stop_trace API that drive an XLA profiler session, the Kineto plugin that registers TPU as a PrivateUse1 activity profiler so torch.profiler works (and merges TPU events into the Kineto trace), the TpuProfilerConfig options (host, device and Python tracer levels, run directory, worker rank, experimental options), output-path resolution, and the profiler server for on-demand captures; read this to capture and locate a TPU trace. | fresh |

### `csrc/internal/sync`
| Concept | Description | Status |
|---|---|---|
| [synchronization](concepts/synchronization.md) | What torch_tpu.synchronize does (materialize the whole device or some tensors, optionally waiting for execution), how its C++ fast paths dispatch, how it differs from .cpu() and from materialization alone, and the companion APIs to check materialization state and dump the pending computation as MLIR or Graphviz; read this for profiling barriers, debugging deferred graphs, and understanding "wait" semantics. | fresh |

### `csrc/mlir`
| Concept | Description | Status |
|---|---|---|
| [mlir-optimization-passes](concepts/mlir-optimization-passes.md) | How torch_tpu runs its own MLIR pass pipeline (CSE, canonicalize, the norm-weight reassociation rewrite, StableHLO target-independent optimization, symbol DCE) over every StableHLO module before XLA compiles it, how to dump per-pass IR and timing, and how to turn the rewrite off; read this when generated StableHLO differs from what an op builder emitted, when debugging a pass failure, or when adding a pass. | fresh |

### `csrc/ops/copy_from`
| Concept | Description | Status |
|---|---|---|
| [host-device-transfer](concepts/host-device-transfer.md) | How tensor.to('tpu'), .cpu(), copy_() and item() move data: the three copy directions, host-side dtype conversion, the non-blocking fast path, materialization on device-to-host, OOM translation, and the experimental batch transfer API with explicit futures; read this to reason about transfer cost, synchronization and pinned-memory style overlap. | fresh |

### `csrc/ops/fused_adagrad`
| Concept | Description | Status |
|---|---|---|
| [fused-optimizers](concepts/fused-optimizers.md) | How TorchTPU implements PyTorch's fused optimizer ops as single multi-tensor StableHLO programs that update every parameter and its moments in one executable, including AMP grad-scale unscaling and the branch-free found_inf revert, in-place buffer donation, and why a TPU-friendly AdamW subclass exists; read this when onboarding optimizer code or debugging optimizer step performance. | fresh |

### `csrc/ops/reductions`
| Concept | Description | Status |
|---|---|---|
| [reductions](concepts/reductions.md) | How reduction ops are lowered with one generic StableHLO reduce builder, how dims are canonicalized and keep-dim outputs reshaped, how mean, variance and standard deviation are derived from sum with reduction factors and Bessel correction warnings, and how reduction dtype rules and cache keys work; read this for reduction numerics or when adding a reduction op. | fresh |

### `csrc/ops/scaled_dot_product_attention`
| Concept | Description | Status |
|---|---|---|
| [scaled-dot-product-attention](concepts/scaled-dot-product-attention.md) | How torch.nn.functional.scaled_dot_product_attention is served on TPU: the backend choice between a C++-built Mosaic flash-attention custom kernel (routed through the cuDNN attention schema), a StableHLO "overrideable" implementation with half-precision softmax promotion, and PyTorch's math fallback, the conditions each requires, how tiling, padding, dropout, masks, GQA and output strides are handled, and the separate opt-in splash_sdpa Python wrapper; read this to know which attention path a model hits and why. | fresh |

### `csrc/ops/view_decomposition`
| Concept | Description | Status |
|---|---|---|
| [view-decomposition](concepts/view-decomposition.md) | How TorchTPU reproduces PyTorch's strided views (as_strided, transpose, slicing, unfold, view(dtype)) on immutable, stride-less XLA buffers by decomposing a strided layout into a sequence of StableHLO view primitives, and how in-place writes through a view are inverted back into a new base buffer; read this to understand view semantics, "zombie data", and why some in-place ops are rejected. | fresh |

### `csrc/pjrt`
| Concept | Description | Status |
|---|---|---|
| [pjrt-client](concepts/pjrt-client.md) | How TorchTPU lazily initializes the TPU PjRt plugin, resolves the single addressable device and global device count per process, guards against use in forked children, registers the profiler, and exposes the memcpy, raw-DMA and execute helpers every transfer and execution goes through; read this for device discovery, world-size checks, initialization errors and allocator statistics. | fresh |

### `src/torch_tpu/_internal/compile/dynamic`
| Concept | Description | Status |
|---|---|---|
| [dynamic-shapes-compiler](concepts/dynamic-shapes-compiler.md) | How the dynamic compiler turns SymInt dimensions into a small set of static executables compiled at 128-aligned bucket upper bounds, rewriting the FX graph with pad_real_to_static / set_dimension_logical_size ops, picking a bucket per call, solving output shapes symbolically, and compiling buckets asynchronously; read this to understand what dynamic shapes in the TPU backend do, their knobs (num_dynamic_buckets, bucket_size) and their recompilation caveats. | fresh |

### `src/torch_tpu/_internal/device`
| Concept | Description | Status |
|---|---|---|
| [device-backend-registration](concepts/device-backend-registration.md) | What import torch_tpu does: detect a TPU, rename the PrivateUse1 backend to "tpu", build the lazy torch.tpu public module over the Python device class (device count, streams, events, RNG, autocast, precision and eager-mode knobs, cache controls), register the Dynamo device interface, the distributed backend and profiler, and monkeypatch matmul precision and torch.compile; read this to understand device initialization, which torch.tpu.* calls are internal in favor of torch.accelerator, and what they actually do. | fresh |

### `src/torch_tpu/_internal/distributed`
| Concept | Description | Status |
|---|---|---|
| [distributed-launchers-and-spmd](concepts/distributed-launchers-and-spmd.md) | How a TorchTPU distributed job is launched (set_tpu_launch_env and the TORCH_TPU_SLICEBUILDER_ADDRESSES, TORCH_TPU_TOPOLOGY and NODE_RANK variables it derives for torchrun or any other launcher), how topology_aware_mesh maps a DeviceMesh onto the physical ICI/DCN layout, and how spmd_safe and the SPMD handshake decide whether collectives may fuse with surrounding work; read this to set up multi-chip, multi-host or multi-slice training. | fresh |

### `src/torch_tpu/_internal/pallas`
| Concept | Description | Status |
|---|---|---|
| [pallas-custom-kernels](concepts/pallas-custom-kernels.md) | How a JAX or Pallas function becomes a PyTorch op on TPU: the experimental jax_op decorator (and the deprecated custom_jax_kernel) export the jitted JAX function once, lower it per input signature to StableHLO at global (sharded) shapes, register it under a content fingerprint of the MLIR module, and call it as a torch.library custom op with a generated schema, a fake kernel, donation, container and enum arguments, and an optional rank-aligned Pallas mesh; read this to bring a Pallas kernel (tokamax, hand-written) into a PyTorch model. | fresh |

### `src/torch_tpu/_internal/utils`
| Concept | Description | Status |
|---|---|---|
| [hardware-introspection](concepts/hardware-introspection.md) | The torch_tpu utils subsystem: how TorchTPU discovers TPUs without initializing the runtime (PCI vendor and device ids to generation, count, topology, per-chip and per-device HBM and bf16 FLOP/s tables), how every public API is stamped with a lifecycle stage (experimental, stable, deprecated, internal) that warns or raises for user callers, the torch.compile-safe warning helper, and the op- and activation-level model tracers; read this for MFU denominators, topology strings, why a torch.tpu call warns or raises, and how to log what a model executes. | fresh |

## Doc-derived concepts
Concepts extracted from the project's own docs (README / `docs/`), grounded to the symbol catalog. The source docs stay in place.
- [api-stability-policy](doc-concepts/api-stability-policy.md)
- [async-micro-batching-and-deferred-loss](doc-concepts/async-micro-batching-and-deferred-loss.md)
- [aten-to-llo-lowering-path](doc-concepts/aten-to-llo-lowering-path.md)
- [building-and-installing](doc-concepts/building-and-installing.md)
- [common-onboarding-pitfalls](doc-concepts/common-onboarding-pitfalls.md)
- [cuda-to-tpu-api-mapping](doc-concepts/cuda-to-tpu-api-mapping.md)
- [debugging-ladder](doc-concepts/debugging-ladder.md)
- [distributed-launch-environment](doc-concepts/distributed-launch-environment.md)
- [docker-multistage-images](doc-concepts/docker-multistage-images.md)
- [eager-execution-modes](doc-concepts/eager-execution-modes.md)
- [environment-variables](doc-concepts/environment-variables.md)
- [error-anatomy-and-cpp-stacktraces](doc-concepts/error-anatomy-and-cpp-stacktraces.md)
- [examples-and-accelerator-api](doc-concepts/examples-and-accelerator-api.md)
- [excess-precision-and-out-dtype](doc-concepts/excess-precision-and-out-dtype.md)
- [extending-pytorch-on-tpu](doc-concepts/extending-pytorch-on-tpu.md)
- [feature-overview](doc-concepts/feature-overview.md)
- [fused-algebraic-rope](doc-concepts/fused-algebraic-rope.md)
- [gke-xpk-workloads](doc-concepts/gke-xpk-workloads.md)
- [jax-and-pallas-custom-ops](doc-concepts/jax-and-pallas-custom-ops.md)
- [launch-blocking-debug-eager](doc-concepts/launch-blocking-debug-eager.md)
- [multi-tier-compilation-cache](doc-concepts/multi-tier-compilation-cache.md)
- [mxu-precision-modes](doc-concepts/mxu-precision-modes.md)
- [numerical-parity-checking](doc-concepts/numerical-parity-checking.md)
- [one-bit-quantization-example](doc-concepts/one-bit-quantization-example.md)
- [pallas-kernel-autograd](doc-concepts/pallas-kernel-autograd.md)
- [pallas-ops-under-torch-compile](doc-concepts/pallas-ops-under-torch-compile.md)
- [porting-vit-walkthrough](doc-concepts/porting-vit-walkthrough.md)
- [reading-tpu-profiles](doc-concepts/reading-tpu-profiles.md)
- [simple-fsdp](doc-concepts/simple-fsdp.md)
- [single-trace-trainer](doc-concepts/single-trace-trainer.md)
- [step-scheduled-profiling-and-benchmarking](doc-concepts/step-scheduled-profiling-and-benchmarking.md)
- [torch-compile-graph-breaks](doc-concepts/torch-compile-graph-breaks.md)
- [torch-compile-tpu-backend](doc-concepts/torch-compile-tpu-backend.md)
- [torch-profiler-on-tpu](doc-concepts/torch-profiler-on-tpu.md)
- [torchtpu-lowering-stack](doc-concepts/torchtpu-lowering-stack.md)
- [tpu-debug-checks](doc-concepts/tpu-debug-checks.md)
- [tpu-device-onboarding](doc-concepts/tpu-device-onboarding.md)
- [tpu-dist-process-group](doc-concepts/tpu-dist-process-group.md)
- [tpu-hardware-model](doc-concepts/tpu-hardware-model.md)
- [tpu-profiler-config](doc-concepts/tpu-profiler-config.md)
- [tpu-training-loop-hygiene](doc-concepts/tpu-training-loop-hygiene.md)
- [user-guide-notebooks](doc-concepts/user-guide-notebooks.md)
- [whole-block-compile-fusion](doc-concepts/whole-block-compile-fusion.md)
- [xla-compilation-dumps](doc-concepts/xla-compilation-dumps.md)
- [zero-copy-views-vs-layout-copies](doc-concepts/zero-copy-views-vs-layout-copies.md)

## Changes
What changed between pins — pages affected and the commits behind them, newest first (see also [log.md](log.md)).
- [6d2a45754b](changes/6d2a45754b.md) — What changed in torch_tpu between 4c0c7c5711 and 6d2a45754b: the pages affected and the commits behind them, grouped by page.

## Coverage
Two tiers: **concept pages** explain mechanisms deeply (selective); the **symbol
index** represents the rest so the whole repo is navigable. Coverage is a
set-difference over the SCIP symbol table, not a graph walk — every documentable
symbol is enumerated and represented.

- documentable symbols: **19064** across 716 modules
- deep (concept pages): **926** (4.9%)
- index-only: **18138**
- represented total: **19064** (100.0%)
- classes represented: **1861/1861**

The catalog is the **symbol index**: [`catalog/index.md`](catalog/index.md) maps every module; `catalog/symbols/*.tsv` holds one row per symbol (anchor, path, line, kind, rank, hash, callers, citing pages) and `catalog/edges/*.tsv` the caller edges — grep them by anchor, never read them whole.

## Provenance
`extracted` = from SCIP / source. `inferred` = LLM judgment, treat as such.
Design-intent dynamics are labeled; none are runtime-measured (no L4 pass run).
Callers/callees are reference-scoped (SCIP has no call role), labeled "calls/refs".
