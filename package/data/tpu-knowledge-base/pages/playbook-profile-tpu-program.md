---
title: "Profile a TPU Program Against Its Roofline"
type: playbook
tags: [performance, compiler, verification]
created: 2026-09-04
updated: 2026-09-04
sources: 3
human_reviewed: true
aliases: ["roofline check", "xprof walkthrough"]
target_imports:
- tpu_profiling
---
# Profile a TPU Program Against Its Roofline

Capture a trace, read the HLO ops that dominate it, predict each one's time from the
roofline, and compare. This is the verification step every other optimization playbook
ends with; the book's own example matched a sharded FFW matmul to 0.4% and a
ReduceScatter to 3% (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).

## Applies when

- A JAX program on TPU runs slower than expected, or an optimization needs a before/after
  measurement. (PyTorch/torch_tpu capture is not yet ingested; the analysis steps 3–6
  apply to any XLA trace.)
- You can run the workload with `jax.profiler` and open TensorBoard/xprof or a Perfetto
  link (xprof (`index.py get codebase-xprof`)).

## Steps

1. **Capture** a steady-state region (skip compilation and the first step):
   ```python
   with jax.profiler.trace("/tmp/tensorboard"):
       out = step(...); out.block_until_ready()
   ```
   Signal: a `plugins/profile/<ts>` directory appears; TensorBoard shows a Trace Viewer tab
   (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).
2. **Find the repeated block.** In Trace Viewer (TPU:0), locate one layer's worth of XLA
   ops; label with `jax.named_scope` if the trace is unreadable. Signal: you can point at
   the attention ops and the MLP ops of one layer
   (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).
3. **Read the dominant ops.** Click each big fusion; record output shape, dtype, layout
   and memory space from its HLO line (`bf16[8,1024,16384]{2,1,0:T(8,128)(2,1)S(1)}`) and
   its inputs (hlo (`index.py get concept-hlo`)). Signal: you know the *local* shapes.
4. **Recover the global shapes and sharding** from local shapes and `replica_groups`
   (e.g. `[8,1024,8192]` local of `[32,1024,8192]` global = 4-way DP; group size 8 in an
   AllReduce = 8-way TP) (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).
5. **Predict.** For each matmul `T_math = 2·B·D·F / (N_chips · C)`; check the per-replica
   batch against critical-batch-size (`index.py get concept-critical-batch-size`) to know whether
   to expect `T_math` or `T_comms = bytes / W_hbm`. For each collective use
   collective-costs (`index.py get concept-collective-costs`) with the bidi ICI bandwidth of the
   generation (ici (`index.py get concept-ici`)). Book example: `2·32·1024·8192·32768 / (23e12·8)
   = 95.6 ms` predicted, 96 ms measured; a 128 MB ReduceScatter over one hop at 1.2e11 →
   1.1 ms predicted, 1.13 ms measured (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).
6. **Compare and classify** each dominant op *(thresholds and symptom mapping are agent heuristics, not from the book)*: within ~10–20% of roofline = done; matmul
   far slower = layout/tiling copies (retile ops in the trace), padding, or a comms-bound
   batch; a blocking collective bracketing matmuls = critical-path comms
   (collective-matmul (`index.py get concept-collective-matmul`)); large idle gaps = host/PCIe
   input or compilation (pcie (`index.py get concept-pcie`)).
7. **Memory tab**: confirm peak HBM and headroom; unexpected peaks point at activation
   checkpoints or an unsharded KV cache (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).

## Pitfalls

- Reading local shapes as global shapes (off by the sharding factor).
- Predicting with the wrong bandwidth: bidi ICI only with a wraparound; unidirectional
  otherwise (ici (`index.py get concept-ici`)).
- Auto-sharding surprises: a giant AllGather that should not be there — fix with
  `with_sharding_constraint` or Explicit mode (jax (`index.py get codebase-jax`)).
- Newer XLA narrows the naive/optimized gap (the book's exercise went from 184→67 ms to
  ~90→80 ms per layer) — always re-measure (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).

## Verification

Achieved MFU = model FLOPs ÷ (measured step time × N_chips × peak) against the target
(mfu (`index.py get concept-mfu`)); every dominant op annotated with predicted vs measured time.

## See also

- xprof (`index.py get codebase-xprof`), HLO (`index.py get concept-hlo`),
  Roofline analysis (`index.py get concept-roofline-analysis`),
  Static shapes & tracing (`index.py get playbook-static-shapes-and-tracing`) (compilation-bubble symptoms)

## Sources

- How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book) (`index.py get scaling-book-profiling`)
- How To Scale Your Model, Part 3: Sharded Matrices (scaling-book) (`index.py get scaling-book-sharding`)
- How To Scale Your Model, Part 1: Rooflines (scaling-book) (`index.py get scaling-book-roofline`)
