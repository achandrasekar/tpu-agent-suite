---
title: "Single-host TorchTitan training on a Cloud TPU VM"
type: doc-concept
provenance: doc
source: torchtitan/experiments/tpu/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Single-host TorchTitan training on a Cloud TPU VM

## Definition
The TPU README's first path is a single TPU VM for "debugging and development" of small models.
You export `TPU_NAME`, `PROJECT_ID`, `ZONE`, `ACCELERATOR_TYPE` (for example `v6e-4`) and a matching
runtime `VERSION` (`v2-alpha-tpuv6e` for v6e). Then `gcloud compute tpus tpu-vm create` builds the VM
(`--reserved` for reservation capacity, `--spot` for a Spot VM) and `gcloud compute tpus tpu-vm ssh`
connects to it. After installing the stack (see
[torchtpu-wheel-and-container-image](torchtpu-wheel-and-container-image.md)), training is one
`torchrun` line with one process per chip:
`torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train --module=torchtitan.experiments.tpu.llama3 --config=llama3_1b ...`.
The README states that "Only small models can run on a single host".

## In torchtitan (grounded)
- **Entry point.** `-m torchtitan.experiments.tpu.train` ends in
  [`start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/train.py#L215 "torchtitan/experiments/tpu/train#start_trainer"),
  which `train.py` hands to [`handle_main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/gmain.py#L124 "torchtitan/experiments/tpu/gmain#handle_main").
  `--module`/`--config` name a registry function. The README's example is
  [`llama3_1b`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L139 "torchtitan/experiments/tpu/llama3/config_registry#llama3_1b"),
  which is the only Llama registry entry that already has `max_context_length=128`. Its token
  budget is `num_tokens_per_microbatch_per_dp_rank=4 * 128`, which is 4 sequences per rank, and it
  uses foreach AdamW. Its FSDP comes from the default parallelism config. The README's command-line
  `--training.*` flags override that config, so its `--training.max_context_length=128` changes
  nothing for this recipe.
- **Per-process topology bootstrap on a VM.** Each torchrun worker enters
  [`_start_trainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/gmain.py#L23 "torchtitan/experiments/tpu/gmain#_start_trainer").
  That function calls [`maybe_init_distributed`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/distributed_utils.py#L9 "torchtitan/experiments/tpu/distributed_utils#maybe_init_distributed")
  before the first TPU tensor exists. With `WORLD_SIZE > 1` and no preset
  `TORCH_TPU_SLICEBUILDER_ADDRESSES`, it delegates to the Cloud
  [`maybe_init_distributed`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/cloud_distributed_utils.py#L194 "torchtitan/experiments/tpu/cloud_distributed_utils#maybe_init_distributed").
  The GKE probe fails on a bare VM, so control falls to
  [`_maybe_init_distributed_on_tpu_vm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/cloud_distributed_utils.py#L136 "torchtitan/experiments/tpu/cloud_distributed_utils#_maybe_init_distributed_on_tpu_vm").
  That function counts local chips from PCI IDs with
  [`get_local_chips`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/local_tpu_device.py#L177 "torchtitan/experiments/tpu/local_tpu_device#get_local_chips")
  (chip type from [`TpuChip`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/local_tpu_device.py#L29 "torchtitan/experiments/tpu/local_tpu_device#TpuChip")).
  It raises if the chip count differs from `WORLD_SIZE`. It then sets `TPU_VISIBLE_CHIPS=LOCAL_RANK`
  and one-chip-per-host bounds, and looks up the host grid in
  [`_TPU_CHIP_TO_HOST_BOUNDS`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/cloud_distributed_utils.py#L25 "torchtitan/experiments/tpu/cloud_distributed_utils#_TPU_CHIP_TO_HOST_BOUNDS")
  (for example v6e with 4 chips -> `2,2,1`). Finally it exports
  `TORCH_TPU_SLICEBUILDER_ADDRESSES`/`TPU_PROCESS_ADDRESSES` as `localhost:10000...` and sets
  `TORCH_TPU_TOPOLOGY`.
- **Device check.** [`_verify_requested_device`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/gmain.py#L81 "torchtitan/experiments/tpu/gmain#_verify_requested_device")
  treats `--device` as an assertion, not a selector. To force CPU on a TPU VM, set
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before launching.

## Why it matters / when it applies
- **`--nproc_per_node` must equal the local chip count.** The VM bootstrap assumes one process per
  chip and refuses any other layout ("Assuming single-chip setup"). The code comments this path
  "only works for singlehost". Use [tpu-multihost-launch](tpu-multihost-launch.md) for anything
  larger.
- **Only some slice shapes are mapped.** `_TPU_CHIP_TO_HOST_BOUNDS` covers v5e, v5p, v6e and TPU7x
  with 1, 4 or 8 chips. Other chip counts on a VM have no host-bounds entry. This is a code
  observation; the README does not mention it.
- **Pick the runtime to match the hardware.** The README says to use "the correct version for your
  hardware" and to check the console Quotas page for zones with capacity. `--spot` VMs suit
  short debugging sessions.
- **Stale limits in the README.** The README adds that "only the `llama3_tpu` configuration is
  supported", that `max_context_length` (formerly `seq_len`) is capped at 128, and that memory/MFU
  metrics may be inaccurate. The 128 cap matches the `llama3_1b` default but not the other
  registries. See [tpu-support-status-and-limits](tpu-support-status-and-limits.md).

## Connections
- Code concepts: [torchtpu-backend](../concepts/torchtpu-backend.md) (launch flow, topology
  bootstrap), [job-configuration](../concepts/job-configuration.md) (`--module`/`--config`
  resolution), [fsdp-sharding](../concepts/fsdp-sharding.md),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md).
- Module catalogs: [module map](../catalog/index.md) (`torchtitan/experiments/tpu/cloud_distributed_utils`,
  `torchtitan/experiments/tpu/gmain`, `torchtitan/experiments/tpu/local_tpu_device`).
- Related doc-concepts: [torchtpu-wheel-and-container-image](torchtpu-wheel-and-container-image.md),
  [tpu-multihost-launch](tpu-multihost-launch.md), [tpu-init-flags-precedence](tpu-init-flags-precedence.md),
  [tpu-support-status-and-limits](tpu-support-status-and-limits.md).
- Runtime side: torch_tpu [distributed-launch-environment](../../torch_tpu/doc-concepts/distributed-launch-environment.md),
  [hardware-introspection](../../torch_tpu/concepts/hardware-introspection.md); hardware
  [TPU v6e](../../../hardware/tpu-v6e.md).

## Source
Extracted from `torchtitan/experiments/tpu/README.md` (sections "Single-host model training on a TPU
VM", "Run a TorchTitan training job"; kept in place).
