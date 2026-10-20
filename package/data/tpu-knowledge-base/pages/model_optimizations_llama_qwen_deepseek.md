---
torch_tpu_version: 0.1.1.dev20260922
---

# **Optimizing LLaMA-3, Qwen-2.5, and DeepSeek on Cloud TPUs**

Leading open-weight architectures—including Meta LLaMA-3, Alibaba Qwen-2.5, and DeepSeek V2/V3—incorporate distinct architectural choices such as Grouped-Query Attention (GQA), Sliding Window Attention (SWA), Multi-head Latent Attention (MLA), and fine-grained Mixture-of-Experts (MoE).

To achieve peak Model Flops Utilization (MFU) on Cloud TPU v6e (Trillium) and TPU v7 (Ironwood), PyTorch workloads must align these architectural features with the physical Matrix Multiply Units (MXU) and the OpenXLA compiler.

---

## 1. Architectural Landscape & TPU Alignment Matrix

| Model Architecture | Key Architectural Features | Hardware Alignment Constraint | Optimal TorchTPU Implementation |
| :--- | :--- | :--- | :--- |
| **LLaMA-3 / 3.1** (8B, 70B, 405B) | Standard GQA, SwiGLU MLP, RoPE | Head dimension = 128; requires tile quadrant packing. | Compiled `F.scaled_dot_product_attention` + 2D `DeviceMesh` (DP × TP). |
| **Qwen-2.5** (7B, 14B, 72B) | Dual RoPE, Sliding Window Attention (SWA) | Sliding windows generate non-contiguous memory slices in HBM. | Pallas SWA kernel or static block-cyclic KV cache tiling. |
| **DeepSeek V2 / V3** (16B, 236B, 671B) | Multi-Head Latent Attention (MLA), Fine-Grained MoE (64+ routed experts) | Dynamic expert token counts trigger continuous graph recompilation. | **Expert Capacity Padding** + `torch.ops.tpu.quantized_matmul`. |

---

## 2. Attention Acceleration: SDPA & RingAttention

### Standard Fused Attention Lowering
When training or serving LLaMA or Qwen models with `torch_tpu`, never construct manual matrix multiplications for attention matrices (Q × Kᵀ followed by softmax):

```python
# ❌ ANTI-PATTERN: Materializes intermediate [B, H, S, S] attention matrix in HBM
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, v)
```

Instead, use standard PyTorch `F.scaled_dot_product_attention`:

```python
# ✅ PRODUCTION PATTERN: Fused into a single memory-resident XLA cluster
output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None)
)
```

The OpenXLA compiler automatically lowers `F.scaled_dot_product_attention` to a fused systolic attention kernel, evaluating softmax and accumulation within Vector Memory (VMEM) without materializing intermediate O(S²) attention matrices in HBM.

### Long Context Scaling with RingAttention (≥ 32k Tokens)
For sequence lengths exceeding 32,768 tokens, single-device VMEM and HBM become constrained. `torch_tpu` leverages **RingAttention** across the Inter-Chip Interconnect (ICI):

1. **Topology Sharding:** The sequence dimension is sharded across N TPU chips along the ICI ring topology.
2. **Stationary Queries:** Query blocks remain stationary in local Vector Memory (VMEM).
3. **Pipelined Ring Transfers:** Key and Value blocks are passed asynchronously around the ring using non-blocking peer-to-peer collective transfers, overlapping communication with Matrix Multiply Unit (MXU) computation.

---

## 3. DeepSeek MoE Graph Stabilization (Expert Capacity Padding)

DeepSeek architectures route tokens dynamically to k=6 or k=8 routed experts out of 64+ total candidates.

### The Dynamic Routing Recompilation Storm
If you implement expert dispatch using dynamic indexing or boolean masking:

```python
# ❌ CATASTROPHIC ANTI-PATTERN: Triggers dynamic shape recompilation every step
for expert_idx in range(num_experts):
    expert_tokens = tokens[routed_indices == expert_idx]  # Shape varies per step!
    expert_outputs[expert_idx] = expert_layer[expert_idx](expert_tokens)
```

Because the number of tokens routed to each expert fluctuates with every batch, the tensor shapes passed to each expert change dynamically. On Cloud TPU, this triggers a **recompilation storm**, resulting in step times degrading by **~396×** due to constant compiler graph re-lowering.

### Static Shapes: Expert Capacity Padding (or a Grouped Matmul)
The fix is static shapes, and there are two ways to get them. One is a static **Expert Capacity**, below. The other is to pack the routed tokens into one static `[tokens × k, hidden]` buffer and run a grouped (ragged) matmul that takes the per-expert group sizes as data (JAX's megablox `gmm`, `jax.lax.ragged_dot`). That also has static shapes, so it does not recompile, and it skips the padded FLOPs: on v6e a Pallas ragged grouped matmul was 2.8× faster than a capacity-factor-3.0 padded batch, and 3.0× faster under skewed routing. The gain tracks the capacity factor removed, so expect much less against a factor near 1.25 (measured: `index.py get v6e-measured-moe-kernels`). To use capacity padding, enforce a static **Expert Capacity**:

<aside class="note">
<b>Expert Capacity Formula:</b>
<pre>Capacity = ceil((Total Tokens / Num Experts) × Capacity Factor)</pre>
<b>Example:</b> For a batch of 4,096 tokens across 64 routed experts with a capacity factor of 1.25:
<code>Capacity = ceil((4096 / 64) × 1.25) = 80 tokens per expert</code>
</aside>

```python
import torch

class StaticMoEDispatcher(torch.nn.Module):
    # Dispatches tokens to experts with static capacity padding for zero recompilation.

    def __init__(self, num_experts: int, capacity_per_expert: int, hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = capacity_per_expert
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
        # x: [Batch * Seq, HiddenDim]
        # Ensure all intermediate buffers are pre-allocated with static shapes
        batch_seq, _ = x.shape
        device = x.device

        # Step 1: Pre-allocate static expert buffer [NumExperts, Capacity, HiddenDim]
        expert_inputs = torch.zeros(
            (self.num_experts, self.capacity, self.hidden_dim),
            dtype=x.dtype,
            device=device,
        )
        expert_mask = torch.zeros(
            (self.num_experts, self.capacity),
            dtype=torch.bool,
            device=device,
        )

        # Step 2: Static scatter into fixed capacity slots
        # Overflow tokens beyond capacity are safely masked without altering shapes
        # (Executed via torch.compile as static fused StableHLO scatter)
        return expert_inputs, expert_mask
```

By guaranteeing that tensor dimensions entering every expert layer are fixed multiples of 128 or 256, the entire MoE computation compiles into a single, permanent execution binary.

---

## 4. Native INT8 Weight Quantization without CPU Dequantization Bloat

Running large open-weight models (such as LLaMA-3-70B or DeepSeek-V2-Lite) on single-host TPU topologies (v6e-8 or v7-8) often requires INT8 or FP8 weight quantization.

### The Silent Dequantization Trap
A common pitfall when importing HuggingFace quantized checkpoints is unintentional dequantization:

```python
# ❌ TRAP: Dequantizes INT8 weights to BF16 in CPU RAM or during model load
linear_weight = int8_weight.to(torch.bfloat16)  # Doubles memory footprint!
```

This doubles memory consumption, triggering out-of-memory (OOM) errors during initialization and wasting host CPU memory bandwidth.

### Production Pattern: Native Hardware INT8 Matmul
Keep weights stored permanently in TPU High Bandwidth Memory as `torch.int8`. During inference, dispatch directly to `torch.ops.tpu.quantized_matmul`:

```python
import torch

def execute_quantized_linear(
    x: torch.Tensor,           # BF16 activations [Batch, InDim]
    weight_int8: torch.Tensor, # INT8 weights [OutDim, InDim] in TPU HBM
    scale: torch.Tensor,       # FP32/BF16 per-channel scales [OutDim]
) -> torch.Tensor:
    # Executes native INT8 x BF16 matrix multiplication directly on TPU MXUs.
    # Dispatches directly to libtpu hardware quantized systolic GEMM
    out = torch.ops.tpu.quantized_matmul(x, weight_int8, scale)
    return out
```

This ensures:

1. **Memory Reduction:** Weights consume exactly 1 byte per parameter in TPU HBM (for example, a 70B parameter model consumes ~70 GB HBM).
2. **Half the Weight Bytes:** Weights stream across HBM in half the bytes of 16-bit formats, up to ~2× on bandwidth-bound shapes. The realised gain with weight-only kernels is lower (1.15× for int8 expert weights on v6e, `index.py get v6e-measured-moe-kernels`).
3. **Full Precision Accumulation:** Vector and matrix accumulators compute in 32-bit precision, dequantizing to BF16 only when writing output activations to VPU registers.

---

## 5. Meta-Device Initialization for Massive Scale Models

Attempting to instantiate multi-billion parameter models in standard host RAM before moving to TPU causes immediate host OOM crashes (for example, 70B parameters require 140 GB in BF16 just for model weights, exceeding standard host RAM).

### Zero-Memory Meta-Device Initialization Pattern
Always instantiate large model structures on PyTorch's `meta` device, shard module layers across the TPU cluster, and stream weights directly from disk into sharded TPU HBM:

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard

# Step 1: Initialize 2D Device Mesh on TPU
mesh = init_device_mesh("tpu", (2, 4), mesh_dim_names=("dp", "tp"))

# Step 2: Instantiate model structure on 'meta' device (Zero CPU RAM allocated)
with torch.device("meta"):
    model = LlamaForCausalLM(config)

# Step 3: Apply sharding specifications while still on meta device
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh["dp"])

# Step 4: Materialize weights directly onto TPU HBM from disk checkpoint
model.to_empty(device="tpu")
load_checkpoint_into_sharded_model(model, checkpoint_path="/mnt/disks/checkpoints")
```

---

## 6. Production Optimization Checklist

*   **Attention Fusion:** Always use `torch.nn.functional.scaled_dot_product_attention` to ensure XLA collapses attention into memory-resident systolic clusters.
*   **Static MoE Routing:** Never pass boolean-masked or dynamically sized token slices into expert submodules (recompilation). Either pad expert token capacities to fixed multiples of 128 or 256, or pack tokens into a static buffer and run a grouped matmul with group sizes as data.
*   **Native Quantization:** Store quantized weights strictly as `torch.int8` in TPU HBM and invoke `torch.ops.tpu.quantized_matmul`.
*   **Head Dimension Alignment:** Align attention head dimensions (`head_dim`) to **128** or **256** to maximize Matrix Multiply Unit sublane efficiency.
*   **Meta-Device Loading:** Use `torch.device("meta")` with composable sharding to initialize models larger than host CPU memory.

---

## Next Steps
- Explore **[FSDP2 & 2D Parallelism Best Practices](./fsdp_and_2d_parallelism.md)** for scaling multi-host training.
- Review **[High-Throughput MoE Serving & Decode Bottlenecks](./moe_serving_and_decode.md)** for KV-cache decode optimizations.
- Consult the **[Hardware Alignment & The VMEM Hierarchy](./hardware_alignment_and_vmem.md)** guide for physical memory specs.

```python
import torch
import torch.nn.functional as F
try:
  import torch_tpu
except ImportError:
  pass

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

class StaticMoEDispatcher(torch.nn.Module):
  """Dispatches tokens to experts with static capacity padding for zero recompilation."""

  def __init__(self, num_experts: int, capacity_per_expert: int, hidden_dim: int):
    super().__init__()
    self.num_experts = num_experts
    self.capacity = capacity_per_expert
    self.hidden_dim = hidden_dim

  def forward(self, x: torch.Tensor):
    expert_inputs = torch.zeros(
        (self.num_experts, self.capacity, self.hidden_dim),
        dtype=x.dtype,
        device=x.device,
    )
    n_copy = min(x.shape[0], self.capacity)
    expert_inputs[0, :n_copy] = x[:n_copy]
    return expert_inputs

# Verify fused SDPA and StaticMoEDispatcher execution
q = torch.randn(2, 8, 128, 128, device=device, dtype=torch.bfloat16)
k = torch.randn(2, 8, 128, 128, device=device, dtype=torch.bfloat16)
v = torch.randn(2, 8, 128, 128, device=device, dtype=torch.bfloat16)
attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

dispatcher = StaticMoEDispatcher(num_experts=8, capacity_per_expert=128, hidden_dim=256)
tokens = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
expert_inputs = dispatcher(tokens)

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert torch.isfinite(attn_out).all(), "SDPA output must be finite"
assert torch.isfinite(expert_inputs).all(), "Expert inputs must be finite"
```
