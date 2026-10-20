---
title: "Asynchronous micro-batching and deferred loss reduction"
type: doc-concept
provenance: doc
source: examples/classical_ml/README.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Asynchronous micro-batching and deferred loss reduction

## Definition
The classical-ML recipe treats host-device synchronization as the main eager-mode anti-pattern:
calling `torch.accelerator.synchronize()` or `loss.item()` every step "drains the asynchronous
host-device execution queue and stalls pipelined TPU execution". Instead it groups steps into
tight micro-batch windows (`--micro_batches=8`), moves inputs with `non_blocking=True`, appends
`loss.detach()` to a Python list, and performs one on-device reduction
(`torch.stack(losses).sum().item()`) after each window, so the host synchronizes once per window
and no per-step eager addition kernels run between compiled graph executions.

## In torch_tpu (grounded)
- Eager ops are recorded, not run: [`DispatchOp`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/op_dispatcher.h#L312 "csrc/eager/op_dispatcher.h#DispatchOp") appends a deferred
  op and returns immediately, so the Python loop can run ahead of the device. Execution happens
  at materialization ([`Materialize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/materialize.cc#L812 "csrc/eager/materialize.cc#Materialize")), which `loss.item()` or a
  device-to-host copy forces.
- `.item()` and `.cpu()` end in [`CopyTpuToCpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/tpu_to_cpu.cc#L77 "csrc/ops/copy_from/tpu_to_cpu.cc#CopyTpuToCpu"); its
  `non_blocking` argument selects the non-blocking
  [`MaterializationBlockMode`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/materialize.h#L54 "csrc/eager/materialize.h#MaterializationBlockMode") instead of waiting for the result.
  Host-to-device `non_blocking=True` copies go through
  [`CopyCpuToTpuBuffer`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/cpu_to_tpu.cc#L38 "csrc/ops/copy_from/cpu_to_tpu.cc#CopyCpuToTpuBuffer").
- `torch.accelerator.synchronize()` reaches
  [`synchronize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L762 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.synchronize"), a full
  barrier: everything queued must finish.
- Keeping the per-step losses as detached device tensors means the eager graph between windows
  stays small; the single `stack(...).sum()` is one op instead of N scalar adds, each of which
  would otherwise be dispatched (and possibly compiled) separately.

## Why it matters / when it applies
- Applies to any TPU training loop, eager or `torch.compile`: logging a Python float every step
  serializes host and device. Log once per window, or keep metrics on device.
- In the recipe's measurements the warm step latency is about 21 ms in both modes once
  synchronization is limited to one per window; the remaining eager versus compiled difference is
  small (1.03x) for this tiny MLP.
- The same discipline matters when benchmarking: see
  [step-scheduled-profiling-and-benchmarking](step-scheduled-profiling-and-benchmarking.md).

## Connections
- Code concepts: [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [eager-op-dispatch](../concepts/eager-op-dispatch.md),
  [host-device-transfer](../concepts/host-device-transfer.md),
  [synchronization](../concepts/synchronization.md).
- Module catalogs: [module map](../catalog/index.md) (`csrc/eager`, `csrc/ops/copy_from`).
- Related doc-concepts: [eager-execution-modes](eager-execution-modes.md) (Strict versus
  DeferAndFuse dispatch), [tpu-training-loop-hygiene](tpu-training-loop-hygiene.md),
  [step-scheduled-profiling-and-benchmarking](step-scheduled-profiling-and-benchmarking.md),
  [whole-block-compile-fusion](whole-block-compile-fusion.md).

## Source
Extracted from `examples/classical_ml/README.md` (kept in place).
