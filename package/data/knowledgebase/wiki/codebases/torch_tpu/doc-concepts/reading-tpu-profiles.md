---
title: "Reading a TPU profile: red flags and the ideal trace"
type: doc-concept
provenance: doc
source: docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md
sources: [docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md, docs/markdown/performance_profiling.md]
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Reading a TPU profile: red flags and the ideal trace

## Definition
The doc gives a short diagnostic table for xprof traces. Large gaps between TPU ops mean a
host-side bottleneck (data loading, Python overhead): prefetch data and move logic out of the
loop. Frequent short TPU bursts mean too many graph breaks and frequent recompilation: reduce
`.item()` / `.cpu()` calls and use `torch.compile`. Low MXU utilization means tensor
dimensions are not aligned to the 128 tile size: pad dimensions to multiples of 128. The ideal
profile shows long continuous TPU compute blocks (many ops fused into one execution), minimal
host-to-TPU gaps (data loading overlapped with compute) and MXU utilization above ~60%.

## In torch_tpu (grounded)
- **Short bursts** are the eager deferred-execution model made visible. Each
  [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp") records a deferred op; a host read such as
  `.item()` or `.cpu()` forces materialization, which compiles and runs whatever has been
  recorded so far as one executable. Many host reads give many small executables, and new
  graph shapes give compilation-cache misses. See
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md)
  and [compilation-cache](../concepts/compilation-cache.md).
- **`torch.compile`** removes the per-op host work: the `tpu` backend lowers the whole FX graph
  to one module ([torch-compile-backend](../concepts/torch-compile-backend.md)).
- **Gaps** can also be compilation: a cache miss compiles on the host before the device can run.
  The cache tiers that make the second run fast are on
  [multi-tier-compilation-cache](multi-tier-compilation-cache.md).
- **MXU alignment** is a property of the shapes you choose, not of TorchTPU code; the hardware
  facts (generation, cores, memory) that inform it are in
  [hardware-introspection](../concepts/hardware-introspection.md), and the MXU itself is described
  on [tpu-hardware-model](tpu-hardware-model.md). Varying shapes additionally
  cost recompilation; see [dynamic-shapes-compiler](../concepts/dynamic-shapes-compiler.md).

## Why it matters / when it applies
This is the triage step after capturing a profile: identify which of the three patterns
dominates before changing code. The CPU-only `torch.profiler` op table is the companion check
for CPU fallbacks (high `aten::copy_` / `aten::_to_copy` counts), described on
[debugging-ladder](debugging-ladder.md).

## Connections
- Code concepts: [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [torch-compile-backend](../concepts/torch-compile-backend.md),
  [compilation-cache](../concepts/compilation-cache.md),
  [dynamic-shapes-compiler](../concepts/dynamic-shapes-compiler.md),
  [hardware-introspection](../concepts/hardware-introspection.md),
  [profiler](../concepts/profiler.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/eager`).
- Related doc-concepts: [torch-profiler-on-tpu](torch-profiler-on-tpu.md),
  [tpu-profiler-config](tpu-profiler-config.md), [debugging-ladder](debugging-ladder.md),
  [aten-to-llo-lowering-path](aten-to-llo-lowering-path.md),
  [tpu-hardware-model](tpu-hardware-model.md).

## Source
Extracted from `docs/markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md`
(kept in place).
`docs/markdown/performance_profiling.md` is an older, unnumbered export of the same notebook with identical content (kept in place).
