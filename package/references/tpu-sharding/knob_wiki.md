# tpu-sharding Knob Wiki

Welcome to the **tpu-sharding Knob Wiki**. This document serves as the comprehensive catalog of configuration knobs that the **Optimizer** subagent can tune.

Subagents read this catalog directly (`knob_wiki.md`) to discover allowed knobs,
default values, data types, and search ranges for their target framework.

---

## 🔍 Deep Dive & Reference Resources

The agent can pull more context and valid ranges for these flags via the following repositories and docs:

### 1. **XLA Flags Guidance**

* **Documentation:** [OpenXLA Guides](https://openxla.org/xla/flags_guidance)
* **Use when:** Finding candidate values for low-level `--extra_flags` or `LIBTPU_INIT_ARGS`.

### 2. **MaxText Codebase**

* **Location:** `https://github.com/AI-Hypercomputer/maxtext`
* **Reference for:** Tuning parameters in `maxtext_flags` (Training workflows). Check `base.yml` or specific model YAML files for defaults.

### 3. **vLLM / TPU Inference**

* **Location:** `https://github.com/vllm-project/tpu-inference` | [vLLM GitHub](https://github.com/vllm-project/vllm)
* **Reference for:** Serving flags, PagedAttention block sizes, custom flags like `MODEL_IMPL_TYPE` or `USE_BATCHED_RPA_KERNEL`.

### 4. **AI-Hypercomputer TPU Recipes**

*   **Repository:**
    [http://github.com/AI-Hypercomputer/tpu-recipes](http://github.com/AI-Hypercomputer/tpu-recipes)
*   **Reference for:** Optimized baseline shardings (data, tensor, expert
    parallelism), XLA flags, MaxText/vLLM flags, and other configs for similar
    models and architectural families.

### 🧠 Optimizer Tip:
When searching the codebase or documentation, use keywords and flag names exactly as defined here (e.g., `xla_tpu_enable_sparse_core`).

---

## 🏷️ Metadata Schema & Entry Formatting Rules

To maintain consistency and enable precise CLI filtering, every entry in this document MUST adhere to the following schema and format template:

### 1. Tag Taxonomy

* **`framework`**: Target execution framework (`MAXTEXT`, `VLLM`, `TPU_INFERENCE`, `XLA`).
* **`category`**: Subsystem domain (`SHARDING`, `BATCHING`, `ATTENTION`, `MEMORY`, `PRECISION`, `PROFILING`, `CHECKPOINTING`, `SPARSECORE`, `LATENCY_HIDING`, `COMPILATION`, `HARDWARE_CLOCK`, `LAUNCH_FLAGS`).
* **`type`**: Parameter data type (`INT`, `FLOAT`, `BOOL`, `STRING`, `ENUM`).
* **`scope`**: Role scope (default: `OPTIMIZER`).

### 2. Standardized Entry Template

```markdown
### ⚙️ KNOB-<CATEGORY>-<ID>: <Knob Name or Group Name>
* **Metadata**: `[framework: <FRAMEWORK>]` `[category: <CATEGORY>]` `[type: <TYPE>]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: <DEFAULT_VALUE>]` `[range: <RANGE_OR_CANDIDATES>]`
* **Description**: Detailed description of what the knob controls, its tuning behavior, and performance trade-offs.
```

---

## 🛠️ 1. MaxText Knobs (Training & Inference)

### ⚙️ KNOB-SHARD-001: ICI Expert Parallelism (`ici_expert_parallelism`)

* **Metadata**: `[framework: MAXTEXT]` `[category: SHARDING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 1]` `[range: 1..N_chips]`
* **Description**: Slices MoE expert routing across ICI mesh axes. Lower values minimize cross-chip `AllToAll` routing latency but increase per-chip memory.

---

### ⚙️ KNOB-SHARD-002: ICI FSDP Parallelism (`ici_fsdp_parallelism`)

* **Metadata**: `[framework: MAXTEXT]` `[category: SHARDING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: -1]` `[range: -1..N_chips]`
* **Description**: Slices parameters, gradients, and optimizer states across ICI mesh axes (`-1` auto-shards across all available chips).

---

### ⚙️ KNOB-SHARD-003: ICI Tensor & Data Parallelism (`ici_tensor_parallelism`, `ici_data_parallelism`)

* **Metadata**: `[framework: MAXTEXT]` `[category: SHARDING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 1]` `[range: 1..N_chips]`
* **Description**: Slices matrix multiplications (`ici_tensor_parallelism`) or data batches (`ici_data_parallelism`) across ICI mesh axes.

---

### ⚙️ KNOB-SHARD-004: Multi-Slice DCN Sharding (`dcn_tensor_parallelism`, `dcn_fsdp_parallelism`, `dcn_expert_parallelism`)

* **Metadata**: `[framework: MAXTEXT]` `[category: SHARDING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 1]` `[range: 1..N_slices]`
* **Description**: Configures inter-slice sharding axes across multi-slice Data Center Network (DCN) clusters.

---

### ⚙️ KNOB-SHARD-005: Physical Mesh Axis Splitting & Ring of Experts (`allow_split_physical_axes`, `use_ring_of_experts`, `shard_exp_on_fsdp`)

* **Metadata**: `[framework: MAXTEXT]` `[category: SHARDING]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: false]` `[range: true, false]`
* **Description**: Toggles physical mesh axis splitting (`allow_split_physical_axes`), ring-based topology for MoE expert communication (`use_ring_of_experts`), and sharding MoE expert weights along the FSDP axis (`shard_exp_on_fsdp`).

---

### ⚙️ KNOB-BATCH-001: Per-Device Batch Size (`per_device_batch_size`)

* **Metadata**: `[framework: MAXTEXT]` `[category: BATCHING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 12]` `[range: 1..256]`
* **Description**: Number of sequences processed per chip per forward pass step.

---

### ⚙️ KNOB-BATCH-002: Max Target Length (`max_target_length`)

* **Metadata**: `[framework: MAXTEXT]` `[category: BATCHING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 2048]` `[range: 128..131072]`
* **Description**: Maximum token context sequence length per batch item.

---

### ⚙️ KNOB-BATCH-003: Training Steps & Dataset (`steps`, `dataset_type`)

* **Metadata**: `[framework: MAXTEXT]` `[category: BATCHING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: steps=150001, dataset_type=tfds]` `[range: steps: 1..1000000; dataset_type: synthetic, tfds, hf, grain, olmo_grain]`
* **Description**: Sets total training iterations/steps per run (`steps`) and input dataset pipeline source (`dataset_type`).

---

### ⚙️ KNOB-BATCH-004: Ragged Buffer Alignment (`ragged_buffer_factor`, `use_ragged_sort`)

* **Metadata**: `[framework: MAXTEXT]` `[category: BATCHING]` `[type: FLOAT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: ragged_buffer_factor=-1.0, use_ragged_sort=false]` `[range: ragged_buffer_factor: 1.0..4.0; use_ragged_sort: true, false]`
* **Description**: Optimizes memory allocation multiplier and sorting kernels for dynamic/ragged sequence lengths in MoE models.

---

### ⚙️ KNOB-ATTN-001: Attention Backend Implementation (`attention`)

* **Metadata**: `[framework: MAXTEXT]` `[category: ATTENTION]` `[type: ENUM]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: autoselected]` `[range: autoselected, dot_product, flash, cudnn_flash_te, vllm_rpa, vllm_batched_rpa]`
* **Description**: Attention algorithm implementation kernel selection.

---

### ⚙️ KNOB-ATTN-002: Tokamax GMM Kernels (`use_tokamax_gmm`, `use_gmm_v2`, `merge_gating_gmm`)

* **Metadata**: `[framework: MAXTEXT]` `[category: ATTENTION]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: false]` `[range: true, false]`
* **Description**: Enables Tokamax high-performance General Matrix Multiply (GMM) kernels for MoE token routing.

---

### ⚙️ KNOB-ATTN-003: Splash Attention Scheduler (`use_splash_scheduler`, `sa_fuse_reciprocal`, `sa_use_base2_exp`, `dq_reduction_steps`)

* **Metadata**: `[framework: MAXTEXT]` `[category: ATTENTION]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: sa_fuse_reciprocal=true, sa_use_base2_exp=true, dq_reduction_steps=0]` `[range: true, false]`
* **Description**: Advanced kernel scheduler optimizations, reciprocal fusion, and exponential math mode selection for Splash attention execution.

---

### ⚙️ KNOB-ATTN-004: MoE Routing & Embedding Execution (`use_random_routing`, `use_custom_sort_vjp`, `use_iota_embed`)

* **Metadata**: `[framework: MAXTEXT]` `[category: ATTENTION]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: use_random_routing=false, use_custom_sort_vjp=true, use_iota_embed=false]` `[range: true, false]`
* **Description**: Toggles random routing fallback (`use_random_routing`), custom VJP sorting kernels (`use_custom_sort_vjp`), and iota sequence embedding lookups (`use_iota_embed`).

---

### ⚙️ KNOB-FRAMEWORK-001: JAX & Flax Framework/Init Flags (`enable_nnx`, `pure_nnx`, `skip_jax_distributed_system`)

* **Metadata**: `[framework: MAXTEXT]` `[category: COMPILATION]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: enable_nnx=true, pure_nnx=true, skip_jax_distributed_system=false]` `[range: true, false]`
* **Description**: Toggles Flax NNX backend (`enable_nnx`), pure NNX mode (`pure_nnx`), and skips JAX distributed system setup (`skip_jax_distributed_system`) on init.

---

### ⚙️ KNOB-MEM-001: Activation Rematerialization Policy (`remat_policy`)

* **Metadata**: `[framework: MAXTEXT]` `[category: MEMORY]` `[type: ENUM]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: full]` `[range: full, minimal, save_dot_except_mlp, minimal_with_context, save_dot_with_context_except_mlp, save_dot_except_mlpwi, save_qkv_proj, qkv_proj_offloaded, custom, minimal_offloaded, save_out_proj]`
* **Description**: Controls activation rematerialization trade-offs between compute overhead and VMEM memory pressure.

---

### ⚙️ KNOB-MEM-002: Layer Input Offloading (`decoder_layer_input`)

* **Metadata**: `[framework: MAXTEXT]` `[category: MEMORY]` `[type: ENUM]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: device]` `[range: device, offload]`
* **Description**: Controls offloading layer inputs to host/secondary memory vs keeping them on-device.

---

### ⚙️ KNOB-PREC-001: Precision Data Types (`dtype`, `mu_dtype`, `grad_dtype`, `weight_dtype`)

* **Metadata**: `[framework: MAXTEXT]` `[category: PRECISION]` `[type: ENUM]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: dtype=bfloat16, grad_dtype=float32, weight_dtype=float32]` `[range: bfloat16, float32, float16]`
* **Description**: Data types for model weights, optimizer state, and gradients.

---

### ⚙️ KNOB-PREC-002: Optimizer Choice (`opt_type`)

* **Metadata**: `[framework: MAXTEXT]` `[category: PRECISION]` `[type: ENUM]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: adamw]` `[range: adamw, sgd, adam_pax, muon]`
* **Description**: Optimization algorithm selection.

---

### ⚙️ KNOB-PROF-001: Profiler Tracing Flags (`profiler`, `enable_tpu_profiling_options`, `skip_first_n_steps_for_profiler`, `profiler_steps`)

* **Metadata**: `[framework: MAXTEXT]` `[category: PROFILING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: profiler=xplane, enable_tpu_profiling_options=true, skip_first_n_steps_for_profiler=1, profiler_steps=5]` `[range: profiler: xplane, nsys; enable_tpu_profiling_options: true, false; skip_first_n_steps_for_profiler: 0..N; profiler_steps: 1..100]`
* **Description**: Selects profiler engine (`xplane`), enables TPU trace capture (`enable_tpu_profiling_options`), sets warmup steps (`skip_first_n_steps_for_profiler`), and specifies step count (`profiler_steps`).

---

### ⚙️ KNOB-CHECK-001: Async Checkpointing (`async_checkpointing`, `enable_checkpointing`)

* **Metadata**: `[framework: MAXTEXT]` `[category: CHECKPOINTING]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: async_checkpointing=true]` `[range: true, false]`
* **Description**: Toggles non-blocking background checkpoint saves and overall checkpointing functionality.

---

## ⚡ 2. vLLM Serving Knobs

*Note: The vLLM serving configuration options listed here are mapped by the Executor backend to standard container start/launch CLI flags (e.g., `--tensor-parallel-size`) or environment variables, depending on the deployed server framework configuration templates.*

### ⚙️ KNOB-VLLM-001: Serving Parallelism (`tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`)

* **Metadata**: `[framework: VLLM]` `[category: SHARDING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 1]` `[range: 1..N_chips]`
* **Description**: Configures tensor parallelism (`tp`), pipeline parallelism (`pp`), and data parallelism (`dp`) across device groups.

---

### ⚙️ KNOB-VLLM-002: Geometry & Concurrency (`max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `input_len`, `output_len`)

* **Metadata**: `[framework: VLLM]` `[category: BATCHING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: max_num_seqs: 256]` `[range: 1..1024]`
* **Description**: Controls maximum context length (`max_model_len`), active sequence concurrency (`max_num_seqs`), prefill token budget (`max_num_batched_tokens`), and prompt/output limits (`input_len`, `output_len`).

---

### ⚙️ KNOB-VLLM-003: Benchmark Client Concurrency (`max_concurrency`, `num_prompts`)

* **Metadata**: `[framework: VLLM]` `[category: BATCHING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 128]` `[range: 1..1024]`
* **Description**: Number of concurrent client dispatch threads (`max_concurrency`) and total request prompts (`num_prompts`) sent by the client harness.

---

### ⚙️ KNOB-VLLM-004: Memory & KV-Cache Management (`gpu_memory_utilization`, `kv_cache_dtype`, `block_size`, `num_gpu_blocks_override`)

* **Metadata**: `[framework: VLLM]` `[category: MEMORY]` `[type: FLOAT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: utilization: 0.90, dtype: auto]` `[range: utilization: 0.50..0.95; dtype: auto, fp8, bfloat16; block_size: 16, 32]`
* **Description**: Controls HBM memory fraction (`gpu_memory_utilization`), KV cache quantization (`kv_cache_dtype`), PagedAttention block size (`block_size`), and block count override (`num_gpu_blocks_override`).

---

### ⚙️ KNOB-VLLM-005: Prefix Caching & Chunked Prefill (`enable_prefix_caching`, `enable_chunked_prefill`)

* **Metadata**: `[framework: VLLM]` `[category: ATTENTION]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: false]` `[range: true, false]`
* **Description**: Caches shared prompt prefixes to lower TTFT (`enable_prefix_caching`) and splits large prompt prefills into chunks (`enable_chunked_prefill`) to stabilize ITL latency.

---

## 🏎️ 3. TPU Inference Knobs (`tpu_inference`)

*Note: TPU Inference configuration options are mapped by the Executor backend to container environment variables or server launch command-line options, depending on whether they configure the model impl, compiler steps, or request speculation.*

### ⚙️ KNOB-TPUINF-001: Engine Parallelism (`tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`)

* **Metadata**: `[framework: TPU_INFERENCE]` `[category: SHARDING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 1]` `[range: 1..N_chips]`
* **Description**: Slices attention heads, layer blocks, and workers across TPU chip groups in `tpu-inference`.

---

### ⚙️ KNOB-TPUINF-002: Serving Implementation & Kernels (`MODEL_IMPL_TYPE`, `USE_BATCHED_RPA_KERNEL`, `ENABLE_PAGED_ATTENTION`)

* **Metadata**: `[framework: TPU_INFERENCE]` `[category: ATTENTION]` `[type: ENUM]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: MODEL_IMPL_TYPE: auto, USE_BATCHED_RPA_KERNEL: false]` `[range: MODEL_IMPL_TYPE: vllm, tpu_inference, jax_serving; kernels: true, false]`
* **Description**: Specifies target serving engine backend (`MODEL_IMPL_TYPE`), enables batched RoPE kernel acceleration (`USE_BATCHED_RPA_KERNEL`), and toggles PagedAttention on TPU.

---

### ⚙️ KNOB-TPUINF-003: Compilation & Speculation (`NUM_SPECULATIVE_TOKENS`, `MAX_PREFILL_PREDICT_LENGTH`)

* **Metadata**: `[framework: TPU_INFERENCE]` `[category: BATCHING]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: tokens: 0]` `[range: tokens: 0..8; predict_length: 128..131072]`
* **Description**: Number of speculative draft tokens generated per step (`NUM_SPECULATIVE_TOKENS`) and pre-compiled prefill length buckets (`MAX_PREFILL_PREDICT_LENGTH`).

---

## ⚙️ 4. XLA Compiler & `libtpu` Runtime Flags

### ⚙️ KNOB-XLA-001: SparseCore Collective Offloading (`xla_tpu_enable_sparse_core_collective_offload_all_gather`, `reduce_scatter`, `all_reduce`, `2d_all_gather`, `3d_all_gather`, `nd_reduce_scatter`, `v2`, `tracing`, `offloading_gather`, `offloading_copy`, `single_sparse_core`, `concurrent`, `aggregator`)

* **Metadata**: `[framework: XLA]` `[category: SPARSECORE]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: false]` `[range: true, false]`
* **Description**: Offloads AllGather, ReduceScatter, AllReduce, Gather, and Copy ops to SparseCore co-processors to keep Tensor Cores dedicated to matrix math.

---

### ⚙️ KNOB-XLA-002: SparseCore Latency Multipliers & LHS Queuing (`xla_tpu_sparse_core_all_gather_latency_multiplier`, `reduce_scatter_latency_multiplier`, `xla_tpu_enable_sparse_core_offload_queuing_in_lhs`)

* **Metadata**: `[framework: XLA]` `[category: SPARSECORE]` `[type: FLOAT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: 1.0]` `[range: 0.1..10.0]`
* **Description**: Configures latency weighting multipliers for SparseCore collectives and queues offloads on LHS operands.

---

### ⚙️ KNOB-XLA-003: Latency-Hiding & Async Schedulers (`xla_tpu_enable_latency_hiding_layer_scheduler`, `rerun_post_sc`, `multi_compute_overlap`, `dependent_collectives`, `3d_reduce_scatter_decomposer`, `xla_enable_async_all_gather`, `prefer_async_allgather`, `fuse_all_gather`)

* **Metadata**: `[framework: XLA]` `[category: LATENCY_HIDING]` `[type: BOOL]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: false]` `[range: true, false]`
* **Description**: Enables XLA layer scheduling, re-runs post-SC assignment, overlaps multi-compute streams, decomposes 3D collectives, and enforces non-blocking async AllGather execution.

---

### ⚙️ KNOB-XLA-004: Collective-Matmul Modes & Fusion Passes (`xla_tpu_all_gather_collective_matmul_mode`, `reduce_scatter_collective_matmul_mode`, `xla_jf_crs_combiner_threshold_in_bytes`, `producer_consumer_fusion`, `window_config`, `domain_passes`, `constant_table`, `barrier_removal`, `spmd_rng`)

* **Metadata**: `[framework: XLA]` `[category: COMPILATION]` `[type: STRING]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: post_spmd_conservative (when decomposed)]` `[range: modes: post_spmd_conservative, post_spmd_aggressive; thresholds: 0..65536]`
* **Description**: Controls matmul fusion timing relative to collectives, CRS combining thresholds, instruction fusion caps, domain passes, and SPMD RNG bit generator modes. *Note: Some low-level compiler optimization options are passed via JAX backend-extra options (e.g. `--xla_backend_extra_options=...`) and are documented in the compiler source code rather than public user guides.*

---

### ⚙️ KNOB-XLA-005: Hardware Clock & Memory Bounds (`xla_tpu_dvfs_p_state`, `xla_tpu_scheduler_percent_shared_memory_limit`, `xla_tpu_scoped_vmem_limit_kib`, `xla_tpu_bf16_emission_mode`, `xla_tpu_pcie_bandwidth_multiplier`)

* **Metadata**: `[framework: XLA]` `[category: HARDWARE_CLOCK]` `[type: INT]` `[scope: OPTIMIZER]`
* **Default / Range**: `[default: p_state: 0, percent: 100]` `[range: p_state: 0..7; percent: 80..150; emission: NATIVE_EMISSION]`
* **Description**: Sets TPU dynamic voltage and frequency scaling power state (`7` for max performance clock), instruction scheduling shared memory percent, vector memory bounds, bfloat16 emission, and PCIe bandwidth scaling.
