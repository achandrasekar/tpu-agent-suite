---
torch_tpu_version: 0.1.1.dev20260922
---

# **XLA Compiler Flags Playbook: Lowering, Fusion & Production Tuning**

Understanding the compilation and lowering pipeline in `torch_tpu` allows engineers to inspect compiler optimizations, diagnose graph breaks, and understand how high-level PyTorch tensor operations are translated into TPU vector machine instructions. This guide outlines the execution pipeline from Python code down to hardware execution, followed by the runtime compiler flag architecture.

---

## 1. The TorchTPU Compilation & Lowering Pipeline
When you invoke `torch.compile(model, backend="tpu")`, the code passes through five distinct compilation layers:

<div style="text-align: center; margin: 24px 0;">
  <img src="https://cloud.google.com/tpu/docs/images/xla_lowering_and_flags_pipeline.png" alt="TorchTPU Compilation and Lowering Pipeline Architecture with Compiler Flag Controls" style="max-width: 100%; border-radius: 8px; border: 1px solid #dadce0;" />
  <p style="font-size: 13px; color: #5f6368; margin-top: 8px;">
    <strong>Figure 1:</strong> Five-stage lowering pipeline from PyTorch frontend to physical TPU hardware execution, highlighting compiler intermediate representations and runtime flag injection points.
  </p>
</div>

### Stage 1: TorchDynamo (Graph Capture & Guarding)
TorchDynamo intercepts Python bytecode execution using PEP 523 hooks. It inspects Python frames, verifies tensor shape and dtype guards, and extracts computational subgraphs while leaving non-tensor Python control flow outside the graph boundary.

### Stage 2: AOTAutograd (Joint Graph Decomposition)
For training workloads, AOTAutograd captures the joint forward and backward computation graphs. It decomposes complex PyTorch operations into core ATen operators (`torch.ops.aten`), making data dependencies explicit for compiler scheduling.

### Stage 3: StableHLO Lowering
The ATen operators are converted into **StableHLO**—a backward-compatible, portable intermediate representation based on MLIR. StableHLO standardizes tensor layouts, broadcasting semantics, and custom calls.

### Stage 4: OpenXLA Compiler Optimizations
The OpenXLA compiler performs the heavy mathematical and physical optimizations:
- **Loop Fusion:** Collapsing adjacent pointwise and reduction kernels into single memory-resident loops.
- **Buffer Assignment:** Allocating tensors in on-chip Vector Memory (VMEM) or High Bandwidth Memory (HBM) and reusing memory addresses to minimize peak footprint.
- **Systolic Tiling:** Aligning matrix dimensions to the physical 256×256 MXU array structure.

### Stage 5: Low-Level Optimizer (LLO) Code Generation
LLO generates hardware binary code for the TPU cores, scheduling VPU vector load/store instructions, MXU systolic pipeline cycles, and asynchronous DMA transfers.

---

## 2. Runtime Flag Architecture & Propagation
The OpenXLA compiler and `libtpu` runtime behavior are controlled through environment variables configured prior to process startup:

- `LIBTPU_INIT_ARGS`: Passes low-level runtime flags directly to the Google TPU driver and hardware execution runtime.
- `XLA_FLAGS`: Passes optimization directives to the OpenXLA compiler passes (such as fusion limits and layout heuristics).

```bash
# Example environment configuration for TorchTPU
export LIBTPU_INIT_ARGS="--xla_tpu_detect_nan=true"
export XLA_FLAGS="--xla_tpu_enable_async_collective_fusion=true --xla_dump_to=/tmp/xla_dumps"
```

---

## 3. Flag Injection Channels: `LIBTPU_INIT_ARGS` vs. `XLA_FLAGS`

A frequent source of misconfiguration is passing compiler flags into runtime environment variables or vice versa. The OpenXLA stack divides configuration into two separate communication channels:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PyTorch TorchTPU Process                                                               │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ 1. Device Driver / Runtime Options       │ 2. Compiler Optimization Passes             │
│    (Passed via LIBTPU_INIT_ARGS)         │    (Passed via XLA_FLAGS)                   │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • Memory allocator pool limits           │ • Operator fusion heuristics                │
│ • Hardware interrupt trapping (NaN/Inf)  │ • Collective communication scheduling       │
│ • Inter-chip interconnect initialization │ • Buffer assignment & rematerialization     │
│ • libtpu hardware lockfile management    │ • StableHLO lowering & text dump flags      │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

<aside class="caution">
<b>The Silent Flag Ignored Trap:</b> Flags intended for the compiler (such as <code>--xla_dump_to</code> or <code>--xla_tpu_enable_async_collective_fusion</code>) are <b>silently ignored</b> if passed to <code>LIBTPU_INIT_ARGS</code>. Conversely, runtime driver flags (such as <code>--xla_tpu_detect_nan</code>) will cause crashes or be ignored if passed to <code>XLA_FLAGS</code>.
</aside>

---

## 4. Empirically Validated Production Flags Catalog

Below are compiler and runtime flags empirically validated to deliver measurable throughput improvements on Cloud TPU v6e and v7:

### A. Asynchronous Collective Communication & Fusion (Recommended for Multi-Host)
*   **`--xla_tpu_enable_async_collective_fusion=true`** (in `XLA_FLAGS`): Merges sequential and independent All-Reduce and All-Gather operations across the Inter-Chip Interconnect (ICI). Reduces discrete hardware synchronization handshakes by up to **28%** in multi-host FSDP and MoE workloads.
*   **`--xla_enable_all_reduce_combiner=true`** (in `XLA_FLAGS`): Coalesces small individual gradient tensor reductions into combined multi-megabyte transfers, avoiding network latency bottlenecks on fine-grained model layers.

### B. Compilation Cache Storage Tiers (High-Speed Restart)
*   **`TORCH_TPU_TIER2_COMPILATION_CACHE="/dev/shm/tpu_cache"`**: Stores serialized HLO compilations in Linux shared memory (RAM disk). Enables instant sub-millisecond graph retrieval across multiple worker processes on the same host.
*   **`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/mnt/disks/tpu_cache"`**: Directs long-term compilation caching to persistent local SSD or attached network storage, eliminating the 600ms+ compilation overhead on job restarts.

### C. Hardware Diagnostic Traps (Debugging Only)
*   **`--xla_tpu_detect_nan=true --xla_tpu_detect_inf=true`** (in `LIBTPU_INIT_ARGS`): Hardware-level instruction traps that immediately freeze execution and dump the offending HLO operator upon encountering non-finite floating-point values, with zero Python host synchronization penalty during healthy execution.

---

## 5. Inspecting Lowered HLO Graphs
To verify how the compiler transforms your model and to check for fused operations:

```python
import os
import torch
try:
  import torch_tpu
except ImportError:
  pass

# Configure XLA to dump intermediate HLO graphs to disk
os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text"

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

class SimpleModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(1024, 4096)
    self.fc2 = torch.nn.Linear(4096, 1024)

  def forward(self, x):
    return self.fc2(torch.nn.functional.silu(self.fc1(x)))

model = SimpleModel().to(device)
compiled_model = (
    torch.compile(model, backend="tpu", dynamic=False)
    if device.type == "tpu"
    else model
)

x = torch.randn(256, 1024, device=device)
out = compiled_model(x)

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert torch.isfinite(out).all(), "Compiled output must be finite"
print("HLO graphs written to /tmp/hlo_dumps")
```

Inspect the generated `.txt` files in `/tmp/hlo_dumps` to view:
- **`%fusion` clusters:** Verifying that linear layers, biases, and activations fused into single kernels.
- **Buffer allocation:** Inspecting whether intermediate tensors reside in VMEM or HBM.

---

## Next Steps
- Read the detailed tutorial on **[The XLA & StableHLO Lowering Pipeline](./xla_stablehlo_pipeline.md)** to inspect each translation layer with `utils.format_model` and compare eager vs. compiled lowering paths.
- Return to the **[TorchTPU Performance Optimization Guide Hub](./performance_guide.md)** for an overview of all empirical guidelines.
- Learn how to capture execution traces in the **Performance Profiling Guide** (`index.py get torch-tpu-perf-profiling`).
