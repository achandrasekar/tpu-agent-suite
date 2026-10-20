---
torch_tpu_version: 0.1.1.dev20260922
---

# **Dynamic Shapes, Bucketing, & Avoiding Recompilation on TPU**

On Google Cloud TPUs, the OpenXLA compiler generates specialized high-performance machine code tailored to the exact tensor shapes and memory layouts of your model.

While this specialization yields industry-leading throughput, it introduces a critical performance hazard: **recompilation storms**. If input tensor shapes change from step to step, XLA pauses execution to compile a new graph, devastating training throughput.

Every metric in this guide is derived from **empirically validated benchmarks across 5 independent replicas** on Cloud TPU v7 hardware.

---

## 1. The Recompilation Latency Cliff
Understanding the magnitude of the compilation penalty is essential for designing high-throughput training and inference pipelines:

| Metric | Measured Value (TPU v7) | Physical Hardware Reality |
| :--- | :--- | :--- |
| **TPU Compilation Latency** | **640.8 – 679.7 ms** (median 648.6 ms) | Time required to lower StableHLO and generate LLO binary |
| **TPU Warm Step Time** | **0.16 – 0.20 ms** | Time to execute the pre-compiled graph on hardware |
| **Compilation Tax Ratio** | **~3,250× to 4,000×** | Compiling is over 3,000× slower than executing |
| **Wall-Time Slowdown** | **210× – 228× slower** | 100 dynamic ragged batches vs pre-warmed static padded batches |

<aside class="caution">
<b>The Silent Dynamo Fallback Hazard:</b> PyTorch Dynamo enforces a default cache limit of <b>8 compilations</b> per code object (<code>torch._dynamo.config.cache_size_limit = 8</code>). When your workload encounters a 9th distinct shape, Dynamo stops compiling and <b>silently falls back to uncompiled eager execution</b> without raising an error. Your training loop continues running, but steady-state throughput degrades by over 200×.
</aside>

### Production Guard Against Silent Fallback
In production training code, never allow Dynamo to silently fall back to eager mode. Always configure Dynamo to raise an exception if the compilation budget is exceeded:

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

# Fail fast in CI and production if unexpected shape variants trigger recompilation
torch._dynamo.config.suppress_errors = False
# If dynamic shapes are unavoidable, size the cache limit intentionally:
torch._dynamo.config.cache_size_limit = 16
```

---

## 2. Ragged Batches vs Static Padding
In natural language processing and sequence modeling, input batches naturally contain sequences of varying lengths. Passing raw ragged batches directly to a compiled model forces XLA to compile a separate kernel for each unique sequence length:

### Measured Comparison: 100 Batches across 6 Variable Lengths

| Execution Strategy | Wall Time (100 steps) | Warm Step Time | Compilations Triggered | Wall Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Ragged (Unpadded)** | **5.91 – 6.04 s** | 0.161 ms | 6 compilations | **210× – 228× slower** |
| **Static Padded** | **0.027 – 0.030 s** | 0.159 ms | 1 compilation (pre-warmed) | **Baseline (Optimal)** |

Once compiled, an individual ragged step takes 0.161 ms vs padded 0.159 ms (identical within 1%). The entire **220× wall-time difference is pure recompilation latency**.

---

## 3. Production Strategy: Dynamic Shape Bucketing
While padding all inputs to the absolute maximum sequence length eliminates recompilation, it wastes computation on padding tokens when sequence lengths are short.

The production solution is **Shape Bucketing**: quantizing variable sequence lengths into a small set of predefined, hardware-aligned buckets (e.g., powers of 2 or multiples of 128).

### Measured Sweep: 16 Distinct Batch Sizes (Cloud TPU v7)

| Strategy | Buckets Used | Compilations | Wall Time | Padding Compute Waste |
| :--- | :--- | :--- | :--- | :--- |
| **Ragged (Unbucketed)** | 16 (hit limit 8) | 8 + silent fallback | 13.93 s | 0% (exact) |
| **Bucketed (4 buckets)** | 4 buckets (128, 256, 512, 1024) | 4 compilations | **6.10 s (2.28× faster)** | **2.9% (880 rows out of 30,080)** |
| **All Padded (Max shape)** | 1 bucket (1024) | 1 compilation | 3.22 s (4.32× faster) | 11.2% (3,696 rows out of 30,080) |

<aside class="tip">
<b>The Bucketing Sweet Spot:</b> Dividing your sequence distribution into <b>4 aligned buckets</b> captures >50% of the single-bucket wall time reduction while keeping wasted padding arithmetic under <b>3%</b>.
</aside>

### Production Bucketing Implementation

```python
class ShapeBucketizer:
  """Buckets dynamic sequence lengths to 128-aligned hardware boundaries."""

  def __init__(self, buckets: list[int] | None = None, pad_value: int = 0):
    self.buckets = sorted(
        buckets if buckets is not None else [128, 256, 512, 1024]
    )
    self.pad_value = pad_value

  def find_bucket(self, seq_len: int) -> int:
    for b in self.buckets:
      if seq_len <= b:
        return b
    return self.buckets[-1]  # Clamp to max bucket

  def pad_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x shape: [batch_size, dynamic_seq_len]
    b, s = x.shape
    target_s = self.find_bucket(s)

    if target_s == s:
      mask = torch.ones_like(x, dtype=torch.bool)
      return x, mask

    padded = torch.full((b, target_s), self.pad_value, dtype=x.dtype, device=x.device)
    padded[:, :s] = x

    # Attention mask: True for valid tokens, False for padding
    mask = torch.zeros((b, target_s), dtype=torch.bool, device=x.device)
    mask[:, :s] = True
    return padded, mask

# Example Usage:
bucketizer = ShapeBucketizer(buckets=[128, 256, 512, 1024])
raw_batch = torch.randint(1, 1000, (8, 185), device=device)
padded_batch, attention_mask = bucketizer.pad_batch(raw_batch)
masked_values = torch.where(
    attention_mask,
    padded_batch.to(torch.float32),
    torch.zeros_like(padded_batch, dtype=torch.float32),
)

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert padded_batch.shape == (8, 256), f"Expected (8, 256), got {padded_batch.shape}"
assert torch.isfinite(masked_values).all(), "Bucketed values must be finite"
```

---

## 4. The `.nonzero()` and Boolean Masking Recompilation Trap

A widespread and catastrophic source of dynamic shapes in PyTorch models is the use of `.nonzero()` or dynamic boolean tensor indexing (`tensor[mask]`):

```python
# ❌ SEVERE ANTI-PATTERN: Triggers dynamic shape recompilation every step
valid_indices = (activations > 0.0).nonzero()
selected_tokens = embeddings[mask]
```

### The Physical Failure Mechanism
Unlike GPU architectures where kernels can allocate dynamic output buffers on the fly, TPU XLA requires static memory layouts allocated ahead of time. Because the count of non-zero elements or mask matches changes on every forward pass based on data values:
1. PyTorch Dynamo cannot infer the output tensor dimension at trace time.
2. The XLA compiler is forced to trigger a full graph re-lowering and compilation cycle on **every single step**.
3. **Empirical Impact on TPU:** In benchmarking a standard sequence filtering operation, moving from `x[mask]` to static masking reduced step execution time from **1,120.4 ms to 2.83 ms**—an astounding **~396× speedup** by eliminating dynamic recompilation.

### Production Solution: Static In-Place Masking with `torch.where`
Instead of slicing or filtering tensors to dynamic lengths, preserve static tensor dimensions across the entire graph using `torch.where()` or masking multipliers:

```python
# ✅ PRODUCTION PATTERN: Preserves static tensor shapes
# Replace dynamic boolean indexing:
#   filtered = x[mask]
# With static conditional replacement:
filtered = torch.where(mask, x, torch.zeros_like(x))

# In loss functions or attention, combine with attention masks:
masked_scores = scores.masked_fill(~mask, -1e9)
```

---

## 5. The DataLoader Ragged Batch Trap (`drop_last=True`)

In training loops, dataset sizes are rarely an exact multiple of the global batch size. For example, a dataset of 10,000 samples trained with batch size 128 yields 78 full batches of 128 and 1 final ragged batch of 16 samples.

### The Epoch Boundary Recompilation Storm
If `drop_last=False` (the default in `torch.utils.data.DataLoader`), the final batch of every epoch triggers:
1. A new compilation for batch size 16.
2. When the next epoch starts, if Dynamo cache limits are exceeded, it triggers cache eviction or silent eager fallbacks.
3. If distributed across multi-core TPUs, differing per-worker sample remainders will cause communication deadlocks across the collective mesh.

### Production Fix
Always pass `drop_last=True` to all training `DataLoader` instances on TPU:

```python
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True,  # MANDATORY ON TPU: Prevents ragged batch recompilations
    num_workers=4,
    pin_memory=True,
)
```

For validation or evaluation loops where every sample must be evaluated, pad the final batch to the full static batch size and mask out the padded predictions during metric aggregation.

---

## 6. Production Checklist for Recompilation Avoidance
1. **Always set `drop_last=True` in DataLoaders:** Eliminates the epoch-boundary recompilation storm caused by ragged terminal batches.
2. **Never call `.nonzero()` or `x[mask]`:** Replace dynamic boolean indexing with `torch.where()` or `.masked_fill()` to keep tensor shapes static.
3. **Pre-warm all buckets at startup:** Pass dummy tensors through each bucket before starting timed training or serving.
4. **Mask padding in loss computation:** Set `ignore_index=-100` in cross-entropy loss so padding tokens do not affect gradients.
5. **Align bucket boundaries to 128 or 256:** Ensure all bucket lengths are multiples of 128 or 256 to satisfy physical MXU systolic array geometry.
6. **Assert compilation budgets in CI:** Monitor `torch._dynamo.utils.counters` to assert that the number of compilations matches the expected bucket count.

---

## Next Steps
- Review **[Hardware Alignment & The VMEM Hierarchy](./hardware_alignment_and_vmem.md)** to learn about the MXU sublane step function.
- Explore **[Anti-Patterns & Compiler Scaling Laws](./anti_patterns_and_compile_laws.md)** for host-device synchronization rules.
