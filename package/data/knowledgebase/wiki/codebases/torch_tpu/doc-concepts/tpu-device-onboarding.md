---
title: "Onboarding a PyTorch program to the tpu device"
type: doc-concept
provenance: doc
source: docs/markdown/1_basics_1_intro_to_torchtpu.md
sources:
  - docs/markdown/1_basics_1_intro_to_torchtpu.md
  - docs/markdown/1_basics_2_get_started_on_vm.md
  - docs/markdown/get_started_tpu.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# Onboarding a PyTorch program to the tpu device

## Definition
"Because TorchTPU uses PyTorch's autoload mechanism, importing `torch` in an environment where
`torch_tpu` is installed automatically registers the `"tpu"` device." From there, onboarding is
mostly a string change: `device="cuda"` becomes `device="tpu"` (a `torch.device("tpu")` object or
the bare `"tpu"` string both work), and `backend="nccl"` becomes `backend="tpu_dist"`. Tensors
move with `.to("tpu")` or are created directly with `device="tpu"`, compute runs on the TPU, and
`.cpu()` brings results back to the host. The VM guide installs into a Python 3.12 virtualenv on
a TPU VM, for example v6e, and has `pip` pull `torch_tpu`. It warns that the matching PyTorch 2.11
CPU wheel "will install automatically, do not install manually". It also notes that TPU v7 and
later are available only through GKE (see [gke-xpk-workloads](gke-xpk-workloads.md)).
The older unnumbered "Get started" page installs the same way and then lists optional extras:
`portpicker`, `marimo`, `scikit-learn` and `pandas` for the notebooks, `jax` for Pallas kernels
and `tensorboard` for xprof. It insists that initialization happen "before moving any tensors to
the TPU"; with autoload that is simply `import torch` on a host with TPUs attached.

## In torch_tpu (grounded)
- **Autoload.** The wheel declares a `torch.backends` entry point that points at
  [`load`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L379 "src/torch_tpu/_loader#load"). PyTorch calls it on `import torch` unless
  [`TORCH_DEVICE_BACKEND_AUTOLOAD`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L26 "src/torch_tpu/_loader#_AUTOLOAD_ENV_VAR._AUTOLOAD_ENV_VAR") is
  `0`. `load` initializes the device only when TPU devices are actually attached. On a machine
  with no TPU, the import succeeds and registers nothing.
- **Device registration.** [`_init_device_impl`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_loader.py#L206 "src/torch_tpu/_loader#_init_device_impl")
  installs the `torch.tpu` device module (through `torch._register_device_module`), registers the
  `tpu_dist`/`tpu` distributed backends, and registers the Kineto profiler backend. The public
  `torch.tpu.*` namespaces are declared as
  [`ModuleSpec`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/public_api.py#L55 "src/torch_tpu/_internal/public_api#ModuleSpec") entries that resolve lazily to
  internal modules. `torch.device("tpu")` itself only names the device. The import is what
  initializes it.
- **Host to device and back.** `.to("tpu")` and `.cpu()` both dispatch to
  [`AtenCopyFrom`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/copy_from_aten_kernels.cc#L232 "csrc/ops/copy_from/copy_from_aten_kernels.cc#AtenCopyFrom"), which routes on
  direction. The host-to-device path is [`CopyCpuToTpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/cpu_to_tpu.cc#L61 "csrc/ops/copy_from/cpu_to_tpu.cc#CopyCpuToTpu").
  The device-to-host path is [`CopyTpuToCpu`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/ops/copy_from/tpu_to_cpu.cc#L77 "csrc/ops/copy_from/tpu_to_cpu.cc#CopyTpuToCpu"), and
  it materializes the source first. In the deferred eager mode, that `.cpu()` call is the point
  where pending work is compiled and executed.

## Why it matters / when it applies
The first port of a model should change only device and backend strings, and run in the default
eager mode, before anything is tuned. Two behaviours surprise people coming from GPUs. First,
`.cpu()`, `.item()` and printing are synchronization points, so a loop that logs a scalar every
step serializes host and device. Second, a missing TPU makes the loader skip device
initialization, not fail loudly, so check `torch.tpu` and device placement if a script seems to
run on CPU. Do not install PyTorch on top of `torch_tpu`. The wheel pins a compatible
`torch` (2.11 or later, CPU build).

## Connections
- Code concepts: [device-backend-registration](../concepts/device-backend-registration.md),
  [host-device-transfer](../concepts/host-device-transfer.md),
  [deferred-execution-and-materialization](../concepts/deferred-execution-and-materialization.md),
  [hardware-introspection](../concepts/hardware-introspection.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu`,
  `src/torch_tpu/_internal`, `csrc/ops/copy_from`).
- Related doc-concepts: [eager-execution-modes](eager-execution-modes.md),
  [tpu-dist-process-group](tpu-dist-process-group.md),
  [torchtpu-lowering-stack](torchtpu-lowering-stack.md),
  [building-and-installing](building-and-installing.md),
  [gke-xpk-workloads](gke-xpk-workloads.md).

## Source
Extracted from `docs/markdown/1_basics_1_intro_to_torchtpu.md` and
`docs/markdown/1_basics_2_get_started_on_vm.md` (kept in place).
`docs/markdown/get_started_tpu.md` is an older, unnumbered variant of the VM guide; it adds the optional-utility install list and the note that the device must be initialized before tensors are moved to it (kept in place).
