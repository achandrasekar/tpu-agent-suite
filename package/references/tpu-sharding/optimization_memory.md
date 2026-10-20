---
scripts:
  - scripts/tpu-sharding/query_tpu_sharding.py
---
# tpu-sharding Optimization Memory

Welcome to the **tpu-sharding Optimization Memory**. This document serves as the shared long-term memory for the tpu-sharding multi-agent system. It stores accumulated human expert rules, performance heuristics, compiler flag bundles, and automated agent takeaways.

tpu-sharding subagents query this optimization memory via the CLI utility script (`{{script:tpu-sharding/query_tpu_sharding.py}}`) to discover and retrieve relevant rules for their current role and task without diluting their LLM context window.

This packaged copy is the **seed**. At runtime each run works against its own copy at `{{runs}}/<run_id>/artifacts/optimization_memory.md`, created from this file on first use, so runs never mutate the shipped catalog.

---

## 🏷️ Metadata Schema & Entry Formatting Rules

Every entry in this document MUST adhere to the following schema and format template:

### Entry <ID>
* **Metadata**: `[score: <SCORE_VALUE>]` `[scope: <SCOPE>]` `[source: <SOURCE>]` `[status: <STATUS>]` `[framework: <FRAMEWORK>]` `[model: <MODEL>]` `[hardware: <HARDWARE>]` `[topic: <TOPIC_1>]` `[topic: <TOPIC_2>]`
* **Description**:
  **<Descriptive Title>**
  Empirical description of the rule, workaround, or flag bundle.
  
  Source Run: [<N>/report_<N>.md]({{runs}}/<run_id>/artifacts/...) *(Mandatory for all TPU_SHARDING_TAKEAWAY entries)*
  Source Reference: [Source Link/Doc](https://...) *(Recommended for OTHER_AGENT entries when derived from external docs)*

---



### Entry 3
* **Metadata**: `[score: 1]` `[scope: EXECUTOR]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[topic: SYNTAX_ERROR]` `[topic: ENVIRONMENT_SETUP]`
* **Description**:
  **Mosaic On-Device Bounds Checking (Correctness)**: Enables on-device out-of-bounds memory checks for Mosaic codegen to catch illegal memory access panics during execution.
  
  ```bash
  --extra_flags=xla_mosaic_on_device_checks=bounds
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 4
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: TPU_SHARDING_TAKEAWAY]` `[status: UNVERIFIED]` `[framework: MAXTEXT]` `[model: DEEPSEEK_V4]` `[topic: MOE]` `[topic: FSDP]` `[topic: SHARDING]`
* **Description**:
  **Pure FSDP vs. Expert Parallelism for MoE Models**: For MoE models (e.g. DeepSeek V4), high Expert Parallelism (e.g. `ici_expert_parallelism=8`) can bottleneck on cross-chip `AllToAll` routing latency. Switching to **Pure FSDP** (`ici_expert_parallelism=1`, `per_device_batch_size=1`) eliminates `AllToAll` routing latency across chips, yielding a **+15–25% boost in TFLOPS/s/device**.

---

### Entry 5
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[hardware: TPU]` `[topic: LATENCY_HIDING]` `[topic: SHARDING]` `[topic: XLA_FLAGS]`
* **Description**:
  **OpenXLA ICI AllGather Collective Pipelining Bundle**: Enables collective pipelining of ICI (Interchip-Interconnect) AllGather operations across loop iterations to overlap communication with computation.
  
  ```bash
  --extra_flags=xla_should_allow_loop_variant_parameter_in_chain=kEnabled
  --extra_flags=xla_should_add_loop_invariant_op_in_chain=kEnabled
  --extra_flags=xla_tpu_enable_ici_ag_pipelining=true
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 6
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[hardware: TPU_V5E]` `[topic: LATENCY_HIDING]` `[topic: XLA_FLAGS]`
* **Description**:
  **TPU v5e Asynchronous AllGather Collective Fusion Bundle**: Activates asynchronous AllGather operations and collective fusion specifically for TPU v5e architectures.
  
  ```bash
  --extra_flags=xla_enable_async_all_gather=kAuto
  --extra_flags=xla_tpu_enable_async_collective_fusion=true
  --extra_flags=xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 7
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[hardware: TPU_V5E]` `[topic: LATENCY_HIDING]` `[topic: XLA_FLAGS]`
* **Description**:
  **TPU v5e Asynchronous AllReduce Collective Fusion Bundle**: Activates asynchronous AllReduce operations and collective fusion on TPU v5e architectures.
  
  ```bash
  --extra_flags=xla_tpu_enable_async_collective_fusion=true
  --extra_flags=xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 8
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[framework: TPU_INFERENCE]` `[framework: VLLM]` `[topic: LATENCY_HIDING]` `[topic: VLLM_INFERENCE]` `[topic: XLA_FLAGS]`
* **Description**:
  **Latency-Bound Collective Threshold Tuning (Inference Workloads)**: Triggers latency-bound optimizations for small-sized (latency-bound) collective operations during inference workloads.
  * **Default**: `-1` (Disabled)
  * **Suggested Search Range**: `4 MB` to `16 MB` (`4194304` to `16777216` bytes)
  * **Candidate Range**: `[0, 9223372036854775807]`
  
  ```bash
  # Suggested tuning candidate example (e.g. 16MB threshold):
  --extra_flags=xla_all_gather_latency_bound_threshold_in_bytes=16777216
  --extra_flags=xla_all_reduce_latency_bound_threshold_in_bytes=16777216
  --extra_flags=xla_collective_permute_latency_bound_threshold_in_bytes=16777216
  --extra_flags=xla_all_to_all_latency_bound_threshold_in_bytes=16777216
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 9
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[topic: LATENCY_HIDING]` `[topic: MOE]` `[topic: XLA_FLAGS]`
* **Description**:
  **Asynchronous All-to-All & Collective Permute**: Enables non-blocking async All-to-All and collective permute rewrites for MoE token routing and sequence parallelism.
  
  ```bash
  --extra_flags=xla_tpu_enable_async_all_to_all=true
  --extra_flags=xla_enable_async_collective_permute=kAuto
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 10
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[topic: XLA_FLAGS]`
* **Description**:
  **Compute-Centric Dot Strength Reduction & Dot-Dot Fusion**: Rewrites non-compute intensive dot products into multiply+reduce ops and fuses producer-consumer dots to eliminate VMEM memory footprint.
  
  ```bash
  --extra_flags=xla_tpu_enable_dot_strength_reduction=true
  --extra_flags=xla_tpu_dot_dot_fusion=true
  --extra_flags=xla_jf_enable_multi_output_fusion=true
  --extra_flags=xla_tpu_scoped_vmem_limit_kib=16384
  --extra_flags=xla_tpu_async_copy_bandwidth_scaling_factor=1.0
  --extra_flags=xla_msa_enable_cross_program_prefetch_freeing=enabled
  --extra_flags=xla_tpu_msa_inefficient_use_to_copy_ratio=0.5
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 11
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[topic: OOM_RECOVERY]` `[topic: XLA_FLAGS]`
* **Description**:
  **HBM Memory Scheduler Rerun & Memory Optimization (OOM Recovery)**: When encountering HBM Out-Of-Memory (OOM) compilation or execution errors, rerun the latency-hiding scheduler with tighter memory limits (`rerun=5`), switch to the BRKGA memory scheduler (`kBrkga`), or disable reduce+broadcast fusions.
  
  ```bash
  --extra_flags=xla_latency_hiding_scheduler_rerun=5
  --extra_flags=xla_tpu_rwb_fusion=false
  --extra_flags=xla_memory_scheduler=kBrkga
  --extra_flags=xla_jf_spmd_threshold_for_windowed_einsum_mib=100
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 12
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[hardware: TPU]` `[topic: FSDP]` `[topic: XLA_FLAGS]`
* **Description**:
  **TPU Data-Parallel DCN AllReduce & Scan-Loop Pipelining**: Increases overlap opportunities for DCN (Data Center Networking) AllReduce in data-parallel topologies and pipelines AllGathers backwards through scan loops.
  
  ```bash
  --extra_flags=xla_tpu_enable_data_parallel_all_reduce_opt=true
  --extra_flags=xla_tpu_data_parallel_opt_different_sized_ops=true
  --extra_flags=xla_tpu_megacore_fusion_allow_ags=true
  --extra_flags=xla_tpu_enable_ag_backward_pipelining=true
  --extra_flags=xla_tpu_spmd_rng_bit_generator_unsafe=true
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)

---

### Entry 13
* **Metadata**: `[score: 1]` `[scope: OPTIMIZER]` `[source: OTHER_AGENT]` `[status: UNVERIFIED]` `[framework: XLA]` `[hardware: H100]` `[topic: LATENCY_HIDING]` `[topic: XLA_FLAGS]`
* **Description**:
  **GPU Collective Pipelining & SOL Latency Estimator Bundle**: Enables latency hiding schedulers and SOL analytical estimators for Hopper/Blackwell GPUs to overlap AllGather, AllReduce, and ReduceScatter with computation.
  
  ```bash
  --extra_flags=xla_gpu_enable_latency_hiding_scheduler=true
  --extra_flags=xla_gpu_enable_analytical_sol_latency_estimator=true
  --extra_flags=xla_gpu_pipeline_all_gather=on
  --extra_flags=xla_gpu_pipeline_all_reduce=on
  --extra_flags=xla_gpu_pipeline_reduce_scatter=on
  --extra_flags=xla_gpu_enable_while_loop_double_buffering=true
  ```
  
  Source Reference: [OpenXLA Flags Guidance](https://openxla.org/xla/flags_guidance)
