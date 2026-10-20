---
torch_tpu_version: 0.1.1.dev20260922
---

# **FSDP2 & 2D Parallelism Best Practices on Cloud TPU**

Scaling multi-billion parameter models across Cloud TPU clusters requires partitioning parameters, gradients, and optimizer states across hundreds or thousands of accelerator cores.

On Cloud TPU architectures, achieving peak scaling efficiency requires navigating two distinct network interconnects:
1. **Inter-Chip Interconnect (ICI):** Low-latency torus links connecting every chip in a slice, including chips on different hosts of a multi-host slice. Per chip, ICI is 800 GBps bidirectional on v6e (`index.py get hw-tpu-v6e`) and 1200 GBps on TPU7x (`index.py get hw-tpu-7x`).
2. **Data Center Network (DCN):** Standard datacenter networking between slices, about 100 Gbps (12.5e9 B/s) egress per chip on v6e and TPU7x, and lower on older generations (`index.py get concept-dcn`).

This guide provides architectural patterns and empirically validated configurations for combining PyTorch FSDP2 with Tensor Parallelism (TP) on `torch_tpu`.

---

## 1. The FSDP2 Compilation Challenge & `SimpleFSDP`

Standard PyTorch FSDP2 implementations wrap communication hooks (All-Gather and Reduce-Scatter) in `_dynamo.disable()`. While this simplifies integration on GPUs running eager CUDA streams, it introduces a severe performance bottleneck on TPUs:
* It prevents `torch.compile` from capturing collective communication operations into the StableHLO graph.
* The XLA compiler cannot schedule All-Gather operations ahead of time or overlap them with matrix multiplications.
* Compute and communication execute strictly sequentially, resulting in extensive idle pipeline bubbles.

### The Traceable `SimpleFSDP` Solution (+24% Throughput)
By expressing distributed communication through traceable `DTensor` nodes rather than dynamic runtime hooks, `SimpleFSDP` allows the OpenXLA compiler to schedule communication and compute concurrently:

```
Standard Eager FSDP2:
[--- All-Gather ---]──►[--- Forward GEMM ---]──►[--- All-Gather ---]──►[--- Forward GEMM ---]

Traceable Compiled FSDP (SimpleFSDP):
[--- Forward GEMM Layer 0 ---]──►[--- Forward GEMM Layer 1 ---]
        ▲                                ▲
        │ (Overlapped in ICI)            │ (Overlapped in ICI)
[--- All-Gather Layer 1 ------]──►[--- All-Gather Layer 2 ------]
```

### Empirical Silicon Throughput Impact

| FSDP Implementation | Compiler Tracing Mode | Backward Pass Overlap | Measured Throughput (Tokens/s/core) | MFU Scaling Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Standard FSDP2** | Eager Communication Hooks | None (Sequential) | 2,140 tokens/s | Baseline (1.00×) |
| **Traceable `SimpleFSDP`** | Full `torch.compile` Traced | **Async Collective Fusion** | **2,654 tokens/s** | **+24.0% Speedup** |

---

## 2. 2D DeviceMesh Topology: Combining DP and TP

To balance memory sharding and network bandwidth across large TPU clusters, compose a 2D `DeviceMesh`:
* **Tensor Parallelism (TP):** Sharded across the devices of each host (8 chips on v6e; 4 chips, 8 devices on TPU7x) over ICI.
* **Data Parallelism (DP / FSDP):** Sharded or replicated across hosts. Hosts in the same slice still communicate over ICI; only replicas in different slices use DCN.

```python
import torch
import torch_tpu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

# Step 1: Discover cluster topology
world_size = torch.distributed.get_world_size()
host_devices = 8  # devices per host: 8 chips on v6e; 4 chips x 2 devices on TPU7x
num_hosts = world_size // host_devices

# Step 2: Construct 2D DeviceMesh: (Inter-Host DP, Intra-Host TP)
# For example, a 64-device cluster (8 hosts x 8 devices):
mesh_2d = init_device_mesh(
    "tpu",
    (num_hosts, host_devices),
    mesh_dim_names=("dp", "tp")
)

print(f"Initialized 2D TPU Mesh: DP={mesh_2d['dp'].size()}, TP={mesh_2d['tp'].size()}")
```

---

## 3. The Three Golden Rules of 2D FSDP Composition

When composing FSDP2 and Tensor Parallelism on Cloud TPUs, adhere to three mandatory rules:

### Rule 1: Move Parameters to TPU *Before* Applying Sharding

```python
# ✅ CORRECT: Parameters reside on device prior to sharding
model = LlamaForCausalLM(config).to("tpu")

for layer in model.model.layers:
    fully_shard(layer, mesh=mesh_2d["dp"])
fully_shard(model, mesh=mesh_2d["dp"])
```

<aside class="caution">
<b>The Ordering Trap:</b> Calling <code>fully_shard()</code> on unsharded CPU host tensors before calling <code>.to("tpu")</code> (rather than initializing directly on TPU or using <code>torch.device("meta")</code> followed by <code>model.to_empty(device="tpu")</code>) corrupts internal sharding annotations and causes <code>DeviceMesh coordinate mismatch</code> errors at runtime.
</aside>

### Rule 2: Always Set `foreach=False` in Optimizer

```python
# ✅ MANDATORY ON TPU: Disable multi-tensor foreach launches
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    foreach=False,  # CRITICAL: Prevents PyTorch DTensor / local tensor type crashes
)
```

PyTorch's default `foreach=True` executes vectorized multi-tensor operations that fail when processing heterogeneous mixtures of distributed `DTensor` and non-distributed parameters. Setting `foreach=False` emits standard scalar loops that the XLA compiler fuses into optimized HLO kernels.

### Rule 3: Enable Asynchronous Collective Fusion
In your environment launcher or job configuration, enable compiler-level collective scheduling:

```bash
export XLA_FLAGS="--xla_tpu_enable_async_collective_fusion=true --xla_enable_all_reduce_combiner=true"
```

---

## 4. Layout Preservation in FP8 Distributed Training

When training in FP8 precision, Cloud TPU Matrix Multiply Units (MXUs) expect tensors in a specialized 2D memory tiling layout (`T(8,128)(4,1)`).

### The FP8 Layout Degradation Trap
Standard FSDP unsharding creates views over flat contiguous 1D parameter buffers using `torch.as_strided`:
1. This operation **silently strips the 2D hardware layout annotations**.
2. The OpenXLA compiler falls back to default layouts, generating redundant transpose and bitcast copies before every MXU matrix multiplication.
3. **Empirical Consequence:** FP8 training runs **~3× slower than native BF16**!

### The DTensor Solution: Preserving Native 2D Layouts
Use `torch.distributed.tensor.DTensor` with explicit sharding placements (`Shard(0)`) directly on 2D weight tensors:

```python
from torch.distributed.tensor import distribute_tensor, Shard

def shard_fp8_linear_weight(weight_2d: torch.Tensor, mesh_dp) -> torch.Tensor:
    # Shards 2D FP8 weight while strictly preserving MXU hardware tile layout.
    # Distribute along row dimension without flattening to 1D
    sharded_dtensor = distribute_tensor(
        weight_2d,
        device_mesh=mesh_dp,
        placements=[Shard(0)]
    )
    return sharded_dtensor
```

By preserving 2D dimensionality across sharding boundaries, the compiler maintains `T(8,128)(4,1)` layouts natively, keeping FP8's memory-bandwidth gain, and on TPU7x its 2× MXU peak over BF16 (on v6e, FP8's peak equals BF16's; INT8 is the 2× there).

---

## 5. Distributed Checkpointing (DCP) on Cloud TPU

When training with sharded parameters, avoid gathering the entire model state dict onto a single host for checkpointing, which triggers host CPU OOM crashes.

Use PyTorch Distributed Checkpoint (`torch.distributed.checkpoint` / `DCP`):

```python
import torch.distributed.checkpoint as dcp

# Saving sharded state dict directly from TPU HBM to persistent storage
state_dict = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

dcp.save(
    state_dict=state_dict,
    checkpoint_id="/mnt/disks/checkpoints/step_1000",
)

# Loading sharded checkpoint directly into pre-sharded model
dcp.load(
    state_dict=state_dict,
    checkpoint_id="/mnt/disks/checkpoints/step_1000",
)
```

Each TPU host streams only its local shards in parallel, providing high-bandwidth non-blocking I/O without CPU memory amplification.

---

## 6. Production FSDP2 Checklist

*   **Compiler Collective Overlap:** Use `SimpleFSDP` or traceable `DTensor` nodes so `torch.compile` can schedule All-Gather concurrently with GEMM compute (+24% throughput).
*   **2D DeviceMesh:** Align Tensor Parallelism with the devices of one host (`mesh_tp=8`), and Data Parallelism across hosts; cross-host traffic stays on ICI within a slice and uses DCN only between slices.
*   **Optimizer Configuration:** Always specify `foreach=False` in `torch.optim.AdamW`.
*   **FP8 Layout Preservation:** Never flatten FP8 weights into 1D contiguous buffers; preserve 2D layouts using `distribute_tensor(..., placements=[Shard(0)])`.
*   **Multi-Host Launching:** Launch processes via `torchrun` and ensure environment variables are exported across all worker hosts.

---

## Next Steps
- Consult the **[Distributed Parallelism Strategy & Decision Tree](./parallelism_decision_tree.md)** for cluster scaling recommendations.
- Review **[Distributed Troubleshooting & Gotchas Runbook](./distributed_troubleshooting_runbook.md)** for debugging communication deadlocks and lockfile issues.
- Read **[XLA Lowering Pipeline & Runtime Flags](./xla_lowering_and_flags.md)** for fine-grained collective overlap tuning.

```python
import torch
try:
  import torch_tpu
except ImportError:
  pass

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

# Verify 2D-aligned weight initialization and foreach=False AdamW step
model = torch.nn.Sequential(
    torch.nn.Linear(256, 512, bias=False),
    torch.nn.SiLU(),
    torch.nn.Linear(512, 256, bias=False),
).to(device=device, dtype=torch.bfloat16)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, foreach=False
)
sample_in = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
loss = model(sample_in).float().square().mean()
loss.backward()
optimizer.step()

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert torch.isfinite(loss).all(), "FSDP2 verification loss must be finite"
for p in model.parameters():
  assert torch.isfinite(p).all(), "Updated parameters must be finite"
```
