---
references:
  - references/tpu-performance-optimization/concepts/torch-compile-backend.md
  - references/tpu-performance-optimization/concepts/dynamic-shapes-compiler.md
  - references/tpu-performance-optimization/concepts/scaled-dot-product-attention.md
delegates:
  - tpu-knowledge-base
---

# Native PyTorch TPU Performance Optimization Recipe

This recipe provides the structured hill-climbing recipe for optimizing **native PyTorch models on Cloud TPUs (for example, v5e, v5p, v6e Trillium, and v7x Ironwood)** when running without an inference engine like `vllm-torchtpu`.

> [!NOTE]
> **Model Engine Scope**: Models natively supported by `vllm-torchtpu` (listed in `VLLM_TORCHTPU_MODEL_SUPPORT.md`, the single source of truth for model engine support) should be deployed and tuned via `vllm-torchtpu`. All other models fall under this recipe for native PyTorch / `torch_tpu` execution.

It implements the **"Make it Run → Make it Fast"** lifecycle:
1. **Pre-condition**: The model is already functionally onboarded and numerically verified on `torch_tpu`.
2. **This Recipe**: Extracts maximum hardware throughput (MFU/TFLOPS, TTFT, TPS) through non-invasive structural, padding, flag, and graph optimizations.
3. **Cumulative Compounding Optimizations**: Optimization wins **must build on each other** rather than being evaluated in isolation or picking isolated wins. Once an arm is verified as a `win` (clearing the noise floor with numerical parity), keep it as the new baseline foundation and layer subsequent ladder phases on top of it. Because `evaluate_arm.py` computes `delta_pct` against the study's immutable Round 0 baseline, a `win` verdict only means the arm beats Round 0. A stacked arm replaces the current foundation **only if it also beats the previous best winning arm** on the primary metric (beyond the noise floor); otherwise keep the previous best arm as the foundation, even though the stacked arm still scores a `win` against Round 0.
4. **Resilience & Knowledge-Base Driven Diagnosis**: If any step or candidate arm fails, regresses, or crashes (e.g. recompilation thrashing, shape mismatch, OOM, or dynamo recompile limits): **DO NOT QUIT**. Immediately delegate the failure symptom to {{agent:tpu-knowledge-base}}, which searches the TPU playbooks and concept pages, and cite the URL it returns; read the references below; diagnose the root cause, adapt the implementation (e.g. static bucketing, step isolation, weight tying, or flag adjustments), and iterate until winning configurations are found.
5. **Inference Benchmark Sizing**: For inference, it should set max tokens to generate (`max_new_tokens` / generation length) as 50 to be meaningful. Measuring fewer tokens (e.g. 5–10 tokens) is dominated by prefill and dispatch warmup overhead, obscuring steady-state decode throughput; a 50-token horizon is required for representative latency and tokens/sec measurements.
6. **Boundary**: Custom Pallas/C++ kernel authoring from scratch is out of scope. The agent identifies kernel bottlenecks, checks for existing off-the-shelf kernels (e.g. Tokamax), or creates a clean handoff artifact for `tpu-kernel-development` and stops.

---

## The 5-Phase Hill-Climbing Ladder

When a performance round (`tpu-performance-optimization`) consults this recipe to advance an arm, it must follow this **strict hierarchical ladder**. Do not skip phases: macro graph/shape issues will completely dwarf micro-optimizations. Each verified win is retained and accumulated into subsequent phases.

```
Phase 1: Graph Breaks & Host CPU Sync (Eliminate recompilations & CPU stalls)
    ↓ (Gate: 0 graph breaks, 0 dynamic recompiles)
Phase 2: Shape & Padding Alignment (Align tensor dims to 128-byte TPU tiles)
    ↓ (Gate: All major GEMM dims aligned to 128)
Phase 3: XLA Compiler Flag Sweeps (Empirical sweeps for fusion & memory layout)
    ↓ (Gate: Verified latency win with numerical parity)
Phase 4: PyTorch Op Reformulations (Vectorize MoE routing & slow indexing)
    ↓ (Gate: Target op speedup without breaking XLA graph fusion)
Phase 5: Off-the-Shelf Kernels & tpu-kernel-development Handoff (Tokamax or tpu-kernel-development handoff)
```

---

### Phase 1: Fix Graph Breaks and CPU Roundtrips

* **Problem**: PyTorch pauses TPU execution to synchronize with the host CPU due to dynamic shapes, host-side control flow, or unsupported Python operations. Autoregressive decoding recompiles every step if tensor size grows (e.g., token-by-token dynamic concatenation).

* **Signs of Recompilation Thrashing**:
  - **Dynamo Limit Warnings**: Logs report `torch._dynamo hit config.recompile_limit (8)` or repeated `CompilationCache` misses in `pjrt` / OpenXLA stats.
  - **Severe Latency Asymmetry**: The initial iteration or warmup takes tens of seconds (e.g., >30–60s) due to repeated graph compilation, whereas steady-state iterations take <50ms.
  - **Low Compilation Cache Hit Rate**: Steady-state cache hit rate remains low (<90%) instead of the healthy >97% target.
  - **Diagnostic Tracing**: Enable `TORCH_LOGS="recompiles,graph_breaks"` to identify the exact guard failure line and guard reason (e.g. guards on mutated Python integer attributes or dynamic tensor shapes).

* **Common Techniques to Avoid Recompilation**:
  1. **Preallocate Static KV-Cache**: Replace dynamic `torch.cat` autoregressive KV concatenation with preallocated static tensor buffers (e.g., `transformers.cache_utils.StaticCache`) sized to fixed multiples of 128.
  2. **Compile Static Single-Step Forward, Not `model.generate()`**: Never wrap `model.generate()` directly with `torch.compile(backend="tpu", dynamic=False)`. Autoregressive token-by-token sequence growth triggers Dynamo shape guard failures on every step, forcing 50+ consecutive full-model XLA compilations. Instead, compile only the inner fixed-shape single-step forward pass (`(batch_size, 1)` token inputs) with static tensor buffers, leaving Python loop control, termination checks, and sampling outside the compiled graph.
  3. **Bypass Dynamic Masking Control Flow via Precomputed Masks**: Dynamic attention masking functions (such as sliding window or causal mask generators) often perform data-dependent Python control flow (e.g. `.item()`, `fast_all()`, or integer sequence length comparisons) that trigger Dynamo graph breaks. Precompute and materialize static 4D attention masks (e.g. shape `(B, 1, 1, max_cache_len)` or precomputed mask dictionaries) in eager mode and inject them into `attention_mask` to trigger early returns in masking preprocessing.
  4. **Eliminate Python Integer State Mutations in Dynamo Graphs**: Avoid reading or mutating Python integer counters (e.g., `cumulative_length_int` or integer bounds) inside Dynamo-traced functions. Use in-place tensor slice updates (e.g., `keys[:, :, cache_position] = key_states`) indexed by a tensor `cache_position` to preserve pure static tensor semantics.
  5. **Static Sequence Bucketing**: Replace unconstrained dynamic sequence lengths with fixed sequence buckets (e.g., 128, 256, 512, 1024) to ensure a finite, pre-warmed set of compiled XLA graphs.
  6. **Eliminate Host Synchronizations**: Remove `.item()`, `print()`, `.cpu()`, `.tolist()`, or conditional branching dependent on device tensor values in the generation loop. Keep tensors on-device throughout the decode loop and synchronize/extract values only after generation finishes.

* **Knowledge Base References**:
  - **Static Shapes Playbook**: {{data:tpu-knowledge-base/pages/playbook-static-shapes-and-tracing.md}} (static bucketing, step barriers, and cache hit targets).
  - **KV Cache Theory & Sizing**: {{data:tpu-knowledge-base/pages/kv-cache.md}} (sizing formulas, HBM bandwidth floors).
  - **Runtime Compatibility**: {{data:tpu-knowledge-base/pages/huggingface-runtime-compatibility.md}} (DynamicCache compatibility).
  - **`torch.compile` Backend**: {{reference:tpu-performance-optimization/concepts/torch-compile-backend.md}} (how the `tpu` backend traces, guards, and caches compiled graphs).
  - **Dynamic Shapes & Recompilation**: {{reference:tpu-performance-optimization/concepts/dynamic-shapes-compiler.md}} (shape guards, bounded dynamic compilation, and why growing shapes recompile).
  - **Attention Masks & SDPA Backend Selection**: {{reference:tpu-performance-optimization/concepts/scaled-dot-product-attention.md}} (how an explicit `attn_mask` selects the SDPA kernel path on TPU).

* **Canonical Implementation Pattern**:
```python
import torch
import torch_tpu
from transformers.cache_utils import StaticCache

# 1. Preallocate StaticCache in TPU HBM up to max sequence length (multiple of 128)
max_new_tokens = 50  # Set to 50 tokens for meaningful steady-state throughput measurement
prompt_len = input_ids.shape[-1]
max_cache_len = ((prompt_len + max_new_tokens + 127) // 128) * 128
past_key_values = StaticCache(
    config=model.config,
    max_batch_size=batch_size,
    max_cache_len=max_cache_len,
    device=device,
    dtype=model.dtype,
)

# 2. Prefill step (populate cache across prompt tokens)
cache_position = torch.arange(prompt_len, device=device)
position_ids = cache_position.unsqueeze(0)
with torch.no_grad():
  outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      cache_position=cache_position,
      use_cache=True,
  )
next_token = outputs.logits[:, -1:].argmax(dim=-1)

# 3. Compile the single-step forward pass (fixed shape (batch_size, 1))
@torch.compile(backend="tpu", dynamic=False)
def static_decode_step(
    input_ids, past_key_values, cache_position, position_ids, attention_mask=None
):
  return model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      cache_position=cache_position,
      use_cache=True,
  )

# 4. Single-token decode loop with zero host synchronizations
for step in range(max_new_tokens - 1):
  cache_position = torch.tensor([prompt_len + step], device=device)
  position_ids = cache_position.unsqueeze(0)
  with torch.no_grad():
    step_out = static_decode_step(
        next_token, past_key_values, cache_position, position_ids
    )
  next_token = step_out.logits[:, -1:].argmax(dim=-1)
```

* **Arm Mapping in Study State**:
  - `kind`: `host-critical-path` or `recompilation-padding`
  - `claim`: *"Preallocate static KV cache and compile single-step decode to eliminate token-by-token XLA recompilations"*
* **Admissibility Gate**: 0 graph breaks, steady-state execution with 0 recompile pauses after warmup.

---

### Phase 2: Fix Shape and Padding Gaps

* **Problem**: Tensor dimensions that are not multiples of 128 (or byte-aligned to TPU tiles) waste MXU compute and cause hidden XLA memory bloat through suboptimal automatic compiler padding.
* **Diagnostic Check**:
  - Inspect model configuration: embedding dimensions, hidden sizes, intermediate sizes (MLP), and number of heads $\times$ head dimensions.
  - Identify any dimension where `dim % 128 != 0`.
* **Optimization Actions**:
  1. Add manual padding to the PyTorch tensors to round dimensions up to 128-byte tile boundaries before entering linear projections or GEMM operations.
  2. Slice away the extra padding on the output before loss calculation or returning outputs.
* **Arm Mapping in Study State**:
  - `kind`: `recompilation-padding` or `memory-capacity`
  - `claim`: *"Pad intermediate projection dimension from D to D_padded (multiple of 128) to eliminate compiler memory bloat"*
* **Admissibility Gate**: Numerical parity verified on output after un-padding; no memory regression.

---

### Phase 3: Find Best Compiler Flags — XLA Flag Sweeps

* **Problem**: Default XLA compiler settings are conservative, leaving fusion and memory layout opportunities on the table.
* **Diagnostic Check**:
  - Ask {{agent:tpu-knowledge-base}} for recommended `LIBTPU_INIT_ARGS` and XLA flags for this model architecture family (Transformers, MoEs).
* **Optimization Actions**:
  1. Sweep candidate flags **one by one** (do not bundle multiple speculative flags in a single round):
     - Fusion optimizations (e.g. multi-output fusion, reduction fusion).
     - Memory layout and scheduling flags.
     - Fast math / precision flags where numerically safe.
  2. Measure TTFT and throughput (tokens/sec) against baseline.
  3. Validate logits/output numerical parity against the unflagged baseline.
* **Arm Mapping in Study State**:
  - `kind`: `kernel-block` or `scheduler-batching`
  - `claim`: *"Enable XLA flag <FLAG_NAME> to improve multi-output fusion in attention blocks"*
* **Admissibility Gate**: Latency improvement exceeds the protocol noise floor; outputs match within tolerance.

---

### Phase 4: Rewrite Slow PyTorch Ops for TPU

* **Problem**: Standard PyTorch idioms (e.g., dynamic masking, non-vectorized Python loops, dynamic scatter/gather in MoE token dispatch) run poorly on TPU memory subsystems.
* **Diagnostic Check**:
  - Inspect {{agent:tpu-diagnostics}} top slow ops for high HBM bandwidth or high device-time share without heavy FLOPs.
  - Check MoE gating and expert routing code.
* **Optimization Actions**:
  1. Query Knowledge Base for proven PyTorch-to-TPU vectorization patterns.
  2. Rewrite non-vectorized index select or loop constructs into vectorized, batched tensor operations.
  3. Verify that the new formulation allows XLA to fuse the op with adjacent layers rather than creating an isolated execution boundary.
* **Arm Mapping in Study State**:
  - `kind`: `kernel-block`
  - `claim`: *"Vectorize MoE expert dispatch routing to eliminate HBM gather overhead"*
* **Admissibility Gate**: Subgraph numerical parity verified; end-to-end latency improves.

---

### Phase 5: Off-the-Shelf Kernels & tpu-kernel-development Handoff

* **Boundary Principle**: The Hillclimbing Agent does **not** author custom Pallas/C++ kernels from scratch.
* **Procedure**:
  1. **Check Off-the-Shelf Library**:
     - Check if an existing high-performance kernel exists in Tokamax or official libraries (e.g. Splash Attention, PagedAttention).
     - If available, integrate and test.
  2. **End-to-End Gating Requirement**:
     - *Critical Warning*: A custom kernel may achieve a 20% microbenchmark win in isolation, but can result in an end-to-end regression if XLA cannot fuse adjacent ops across the kernel boundary.
     - Any kernel integration **must be evaluated on end-to-end model latency**, not isolated microbenchmarks. Roll back if end-to-end latency regresses.
  3. **tpu-kernel-development Handoff**:
     - If the bottleneck cannot be resolved by standard PyTorch code or off-the-shelf kernels, **stop tuning**.
     - Generate a structured handoff artifact `artifacts/tpu_kernel_development_handoff.json` containing:
       - Op name and source location.
       - Exact tensor shapes, strides, and dtypes.
       - Percentage of total device step time from XProf.
       - Path to the XProf trace (`.xplane.pb`).
* **Arm Mapping in Study State**:
  - `kind`: `kernel-block`
  - `status`: `blocked` (or `decided` with handoff)
  - `claim`: *"Hand off persistent attention bottleneck to the tpu-kernel-development agent with trace and shape specs"*
