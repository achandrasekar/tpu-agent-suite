---
name: megakernel
description: "TPU Pallas Megakernel Architecture, Weight Streaming, Pool Aliasing, In-Kernel Collectives, and vLLM-TorchTPU Integration"
scripts:
- scripts/tpu-kernel-development/megakernel_analyzer.py
- scripts/tpu-kernel-integrator/build_plugin.py
- scripts/tpu-kernel-integrator/inplace_harness.py
---

# TPU Pallas Megakernels

A **TPU Megakernel** collapses a multi-layer or full-model autoregressive decode step (or $K$-token speculative verification pass) into a single grid-less `pl.pallas_call` (`grid=()`, `pltpu.PrefetchScalarGridSpec`), keeping token activations resident in VMEM across projections and layers while overlapping HBM-to-VMEM weight streaming (`pltpu.make_async_copy`) and cross-chip ICI reductions (`pltpu.make_async_remote_copy`) with MXU/VPU computation.

## 1. Automated Architecture Analysis & Scaffolding (`megakernel_analyzer.py`)

Before planning or coding a megakernel for a new model or decoder block, run the megakernel architecture analyzer:

```bash
python3 {{script:tpu-kernel-development/megakernel_analyzer.py}} \
  --config <path_to_config.json_or_inline_json> \
  --tpu-version <v5e|v5p|v6e|v7x> \
  --tp-size <tp_degree> \
  --batch-buckets 16,32 \
  --op-slug <model_slug>_megakernel \
  --target "vllm.model_executor.models.<model>:<TargetModuleClass>" \
  --emit-scaffold <scaffold_dir> \
  --out <scaffold_dir>/megakernel_spec.json
```

`megakernel_analyzer.py` inspects the model's `config.json` (and optional `--model-source <model.py>`) and outputs:
- **Per-rank shard dimensions & 128-lane alignment**: Detects whether `head_dim == 64` and `q_heads_per_rank % 2 == 0` (`pack_two_heads_per_row: true`), and flags any unaligned per-rank dimensions.
- **Global Bank-Ring tile schedule**: Computes `(tk, tn, nk, nn, nt)` via `tile_hw(K, N, max_bank_k, max_bank_n)` for every projection (`attn_qkv`, `attn_o`, `lin_in_proj_qkvz`, `lin_out_proj`, `mlp_gate_up`, `mlp_down`, `lm_head`).
- **Pool-Aliased MoE schedule**: For routed MoE models (`num_routed_experts > 0`), sizes the `pool_alias` VMEM slots (`EXPERT_SLOTS`, `w13_slot_bytes`, `w2_slot_bytes`) and shared-expert fold.
- **Exact VMEM budget verification**: Sums persistent scratch, double-buffered KV slots, recurrent state scratch, weight streaming banks, `pool_alias` MoE pool, and ICI collective buffers against `vmem_limit_bytes`, auto-tuning `bank_count` / `max_bank_n` / `max_bank_k` to avoid `CompileTimeScopedVmemOom`.
- **In-kernel ICI collective plan**: Selects `xor_butterfly_allreduce`, `reduce_scatter_fp32_all_gather_bf16`, or `hierarchical_int16_shared_exponent`.
- **Scaffold files** (when `--emit-scaffold` is passed):
  - `pool_alias.py`: Raw VMEM byte-pool allocation (`pool_scratch_spec`) for heterogeneous views.
  - `megakernel.py`: Self-contained Pallas TPU megakernel exposing `computation`, `reference_computation`, and `tile_hw`.
  - `op_<slug>.py`: `tpu_agent_suite_kernels` vLLM plugin op module implementing `OP_CONTRACT` (`TARGET`, `OWNED_BUCKETS`, `eligible`, `kernel_fn`, `fallback_fn`, `fake`, `call`).
  - `case_<slug>.py`: `inplace_harness.py` verification case implementing `CASE_CONTRACT`.

---

## 2. Core Architectural Mechanics

### 2.1 Grid-Less `PrefetchScalarGridSpec` Execution
Instead of relying on an outer Pallas `grid=(...)` pipeline that cannot handle heterogeneous projection shapes or dynamic MoE routing, bind all HBM inputs and outputs with `pl.BlockSpec(memory_space=pltpu.HBM)` and drive all DMA transfers manually inside the kernel body:

```python
pl.pallas_call(
    _megakernel_body,
    out_shape=jax.ShapeDtypeStruct((m_dim, k_dim), x.dtype),
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(3)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[...],
    ),
    compiler_params=pltpu.CompilerParams(
        vmem_limit_bytes=VMEM_LIMIT_BYTES,
        disable_bounds_checks=True,
    ),
)
```

### 2.2 Global Bank-Ring HBM-to-VMEM Weight Streaming
Allocate `BANK_COUNT` fixed-shape VMEM tile slots `pltpu.VMEM((BANK_COUNT, MAX_BANK_N, MAX_BANK_K), dtype)` and `pltpu.SemaphoreType.DMA((BANK_COUNT,))`. Stream each matrix tile $(n, k)$ into slot `s = tile_idx % BANK_COUNT` using `pltpu.make_async_copy`, overlapping the next tile's HBM DMA with the current tile's `lax.dot_general(..., preferred_element_type=jnp.float32)`.

### 2.3 Raw VMEM Pool Aliasing (`pool_alias`)
When a megakernel combines projections of widely varying aspect ratios or multi-slot MoE expert weights (`w13` + `w2` + quantization scales), separate `pltpu.VMEM(...)` scratch buffers for every shape exhaust VMEM. Allocate a single flat raw byte pool (`pltpu.VMEM((words, 128), jnp.uint32)`) sized to the maximum concurrent phase footprint and alias disjoint stage views into it.

### 2.4 Two-Heads-Per-Row Packing for `head_dim == 64`
Because TPU VREGs and MXU lanes are 128 elements wide, a 64-wide attention head placed on the trailing axis wastes 50% of every vector register and MXU contraction. When `head_dim == 64` and `q_heads_per_rank` is even, reshape head tensors to `[nh // 2, 128]` in VMEM (`pack_two_heads_per_row: true`), apply RoPE on both packed 64-dim halves simultaneously, and mask cross-head attention interaction during the $Q K^T$ contraction.

### 2.5 In-Kernel ICI Collectives (`pltpu.make_async_remote_copy`)
To eliminate host barrier orchestration and HBM round-trips around tensor-parallel `AllReduce`:
- **Single-token decode (`TP <= 8`)**: Use a $\log_2(\text{TP})$-hop XOR butterfly (`device_id={'x': lax.bitwise_xor(idx, delta)}`) with ping-pong VMEM buffers and barrier/DMA semaphores.
- **Multi-token speculative verification (`K = 8..16`, `TP = 8`)**: Use a 1-hop `ReduceScatter` in `float32` (each rank accumulates $1/\text{TP}$ of the rows in FP32) followed by a 1-hop `AllGather` in `bfloat16`, reducing ICI payload by $33\%$.
- **Multi-host pods (`TP = 16` or `32`)**: Perform intra-group XOR reduction first, quantize partial sums to `int16` with a per-row shared `int32` exponent (`max_exp - 14`) for the cross-group remote DMA hop, and dequantize back to `float32` before the final intra-group `AllGather`.

### 2.6 In-VMEM Low-Bit Weight Dequantization
Stream quantized weights (`fp8_e4m3fn`, `mxfp4`, `nvfp4`) in their packed format from HBM into VMEM banks to multiply effective HBM bandwidth by $2\times\text{--}4\times$, and unpack/scale inside VMEM immediately prior to `lax.dot_general(..., preferred_element_type=jnp.float32)`.

---

## 3. End-to-End `vllm-torchtpu` Serving Integration

Once the Pallas megakernel (`megakernel.py`) passes numerical parity and benchmark speedups in the `tpu-kernel-development` loop:

1. **Assemble the `tpu_agent_suite_kernels` vLLM general plugin**:
   ```bash
   python3 {{script:tpu-kernel-integrator/build_plugin.py}} \
     --out-dir {{runs}}/<run_id>/artifacts/tpu-kernel-development/plugin \
     --op <op_slug> \
     --op-module <scaffold_dir>/op_<op_slug>.py \
     --kernel <best_kernel_artifact.py> \
     --tar {{runs}}/<run_id>/artifacts/tpu-kernel-development/plugin.tgz
   ```
2. **Verify In-Place Module Patching & `torch.compile(fullgraph=True)` Speedup**:
   Run `{{script:tpu-kernel-integrator/inplace_harness.py}}` on the target TPU with `--plugin-dir` and `--case <scaffold_dir>/case_<op_slug>.py` to confirm:
   - `eligible(module)` returns `None`.
   - Every owned decode/verify bucket in `OWNED_BUCKETS` matches the unpatched module within `ATOL`/`RTOL` and improves compiled latency (`block_speedup > 1.0`).
   - Prefill buckets ($\ge 64$) cleanly fall back to `fallback_fn` with zero graph breaks.
3. **Serve in `vllm-torchtpu`**:
   Unpack `plugin.tgz` onto `PYTHONPATH` in the serving container and set `TPU_AGENT_SUITE_KERNELS=<op_slug>` so `vllm.general_plugins` automatically registers and patches the megakernel into every worker process at startup.
