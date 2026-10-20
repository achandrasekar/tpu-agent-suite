---
title: "torch.cuda to torch.tpu / torch.accelerator API mapping"
type: doc-concept
provenance: doc
source: docs/markdown/api_mapping.md
updated: 2026-10-13
generated: {by: wikify/0.4.1, at: 2026-10-13T22:01:39Z}
---
# torch.cuda to torch.tpu / torch.accelerator API mapping

## Definition
Once the backend is initialized (`torch.device("tpu")`), TorchTPU registers the Python module
`torch.tpu`, "designed as a drop-in replacement for the standard `torch.cuda` API". The mapping
guide's rule is: **where PyTorch has a device-agnostic `torch.accelerator` API, use it; use
`torch.tpu` only for what is TPU-specific.**

```
torch.cuda                      -> use instead
------------------------------- ------------------------------------------
is_available()                  torch.accelerator.is_available()
device_count()                  torch.accelerator.device_count()
current_device()                torch.accelerator.current_device_index()
set_device(rank)                torch.accelerator.set_device_index(rank)
device(rank)  (ctx manager)     torch.accelerator.device_index(rank)
synchronize()                   torch.accelerator.synchronize()
current_stream() / set_stream() torch.accelerator.current_stream() / set_stream()
initial_seed() / get_rng_state  torch.accelerator.random.*   (PyTorch >= 2.13)
is_initialized()                torch.tpu.is_initialized()
get_device_name/_properties()   torch.tpu.get_device_name/_properties()
manual_seed(_all)(seed)         torch.tpu.manual_seed(_all)(seed)
set_rng_state(state)            torch.tpu.set_rng_state(state)
default_stream()                torch.tpu.default_stream()
autocast("cuda", ...)           torch.amp.autocast("tpu", ...)
get_amp_supported_dtype()       torch.tpu.get_amp_supported_dtype()
```

The `torch.accelerator` rows need PyTorch 2.11 or later (2.13 for `torch.accelerator.random`).
On those versions the replaced `torch.tpu` functions "are internal to TorchTPU, and calling them
from your code raises an error". The guide also notes that streams are not needed: "the XLA
compiler and PjRt runtime manage execution ordering and hardware concurrency automatically".

## In torch_tpu (grounded)
- **Module and registration.** The import-time loader renames the PrivateUse1 backend to `tpu`
  and installs the device module from
  [`get_device_module`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L1123 "src/torch_tpu/_internal/device/_device_module#get_device_module")
  ([device-backend-registration](../concepts/device-backend-registration.md)). The exported names
  are listed in the public-API table (`src/torch_tpu/_internal/public_api.py`) and resolve to
  [`TpuDeviceModule`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L1006 "src/torch_tpu/_internal/device/_device_module#TpuDeviceModule"),
  a subclass of [`_DeviceModule`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L365 "src/torch_tpu/_internal/device/_device_module#_DeviceModule").
- **Why the old names raise.** Functions with a `torch.accelerator` replacement are decorated
  with [`internal_or_experimental`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/utils/annotations.py#L406 "src/torch_tpu/_internal/utils/annotations#internal_or_experimental"),
  with `internal_when` set to a PyTorch version check and `prefer` naming the replacement. On
  PyTorch 2.11 or later this covers
  [`device_count`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L779 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.device_count"),
  [`current_device`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L420 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.current_device"),
  [`set_device`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L456 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.set_device"),
  [`is_available`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L526 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.is_available"),
  [`synchronize`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L762 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.synchronize"),
  [`set_stream`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L744 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.set_stream")
  and [`current_stream`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L856 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.current_stream");
  on 2.13 or later it adds
  [`initial_seed`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L655 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.initial_seed")
  and [`get_rng_state`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L680 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.get_rng_state").
  Calls from PyTorch itself still work; calls from user code raise a `RuntimeError` that names
  the preferred API. Setting `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS=1` downgrades the error to a
  one-time warning, which the source comment intends for users moving off these APIs. Calls
  traced by `torch.compile` neither raise nor warn, because Dynamo cannot trace the caller
  lookup.
- **Seeding and RNG state.**
  [`manual_seed`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L608 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.manual_seed")
  and [`manual_seed_all`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L627 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.manual_seed_all")
  are no-ops on hosts without a TPU (so `torch.manual_seed()` keeps working on CPU-only hosts),
  and raise if TPU initialization fails rather than leave the device unseeded. The C++ side,
  [`SetManualSeedAll`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/csrc/eager/device_gen_impl.cc#L753 "csrc/eager/device_gen_impl.cc#SetManualSeedAll"),
  seeds the generator of every device this process sees.
  [`set_rng_state`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L702 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.set_rng_state")
  and `get_rng_state` accept only this process's own TPU
  ([`_rng_validate_device_index`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L154 "src/torch_tpu/_internal/device/_device_module#_rng_validate_device_index")),
  which fits the one-device-per-process model.
- **AMP.** [`get_amp_supported_dtype`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L891 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.get_amp_supported_dtype")
  returns the fixed list
  [`_AMP_SUPPORTED_DTYPES`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L51 "src/torch_tpu/_internal/device/_device_module#_AMP_SUPPORTED_DTYPES")
  (bf16, fp16, fp32, fp64, the fp8 variants, complex and bool, among others). A source comment
  says the list "was chosen arbitrarily", so it does not mean every listed dtype is fast. The
  module's autocast dtype defaults to bfloat16
  ([`_autocast_dtype`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L385 "src/torch_tpu/_internal/device/_device_module#_DeviceModule._autocast_dtype")).
- **Streams.** [`default_stream`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/device/_device_module.py#L874 "src/torch_tpu/_internal/device/_device_module#_DeviceModule.default_stream")
  returns a [`TpuStream`](https://github.com/google-pytorch/torch_tpu/blob/6d2a45754bdb52743883f75fbf753408345f6d1b/src/torch_tpu/_internal/stream/streams.py#L86 "src/torch_tpu/_internal/stream/streams#TpuStream")
  with stream id 0. Streams exist so CUDA-shaped code runs. Ordering comes from the deferred
  execution model ([synchronization](../concepts/synchronization.md)).

## Why it matters / when it applies
- When onboarding CUDA code, replace `torch.cuda.X` with `torch.accelerator.X` wherever it
  exists. That code stays device-agnostic and avoids the RuntimeError on PyTorch 2.11 and later.
  Reach for `torch.tpu` only for device name and properties, seeding, `set_rng_state`,
  `default_stream` and AMP dtype queries.
- On older PyTorch the `torch.tpu` spellings (`torch.tpu.device_count()`,
  `torch.tpu.synchronize()`) are the fallback.
- Drop explicit stream juggling: it does not buy concurrency on TPU. Use
  `torch.accelerator.synchronize()` only where you really need a host-visible barrier, such as
  timing, because it forces materialization
  ([eager-execution-modes](eager-execution-modes.md)).
- Use `torch.amp.autocast("tpu", dtype=torch.bfloat16)`. bf16 is the native TPU matmul input
  type ([dtype-and-precision](../concepts/dtype-and-precision.md)).

## Connections
- Code concepts: [device-backend-registration](../concepts/device-backend-registration.md),
  [hardware-introspection](../concepts/hardware-introspection.md),
  [synchronization](../concepts/synchronization.md),
  [dtype-and-precision](../concepts/dtype-and-precision.md).
- Module catalog: [catalog index](../catalog/index.md) (`src/torch_tpu/_internal/device`,
  `src/torch_tpu/_internal/utils`, `src/torch_tpu/_internal/stream`, `csrc/eager`).
- Related doc-concepts: [tpu-device-onboarding](tpu-device-onboarding.md),
  [examples-and-accelerator-api](examples-and-accelerator-api.md),
  [api-stability-policy](api-stability-policy.md),
  [tpu-dist-process-group](tpu-dist-process-group.md).

## Source
Extracted from `docs/markdown/api_mapping.md` (kept in place).
