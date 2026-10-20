# Cluster Information
- **Cluster:** Shared multipod cluster

---

## Objective
Verify that PyTorch can successfully execute on the cluster's TPU hardware before beginning a model port. A simple node connectivity or health response is not sufficient: you must execute an actual matrix multiplication (`matmul`) on the TPU and verify numerical correctness against a CPU baseline.

All necessary cluster configuration details are available in your environment variables.

---

### Environment & Execution Constraints

1. **Container Image & Working Directory:**
   - Use the container image specified by `$TASK_DOCKER` (contains the PyTorch TPU runtime).
   - Place all scripts, manifests, and files sent into the container under `/app`.
   - **Do NOT use `/workspace`** (it is already occupied inside the image).

2. **TPU Node Selection:**
   - Use `$GKE_TPU_ACCELERATOR` and `$GKE_TPU_TOPOLOGY` as node labels to target a TPU node.
   - Use these values exactly as provided in the environment. Do not guess or modify them; an invalid pair causes workloads to hang indefinitely in `Pending` rather than failing fast.

3. **Job Scheduling & Kueue:**
   - If `USE_KUEUE=1`, the cluster admits workloads via Kueue rather than scheduling directly.
   - Deploy the Kubernetes Job in namespace `$GKE_NAMESPACE` (the namespace designates queue quota).
   - Label the Job: `kueue.x-k8s.io/queue-name: $GKE_QUEUE`.

4. **Scope Boundary:**
   - Do **NOT** stray into other clusters, projects, or zones.
   - Do **NOT** provision new clusters, nodes, or cloud resources.

---

### Output Requirements

Your final response **MUST start immediately** with the exact markdown table below (no preamble, greetings, or introductory text before it):

| field | value |
| ------------ | ----- |
| device_type | |
| accelerator | |
| max_abs_diff | |
| tolerance | |

#### Table Field Specifications
Ensure the table uses these exact row names, in this exact order, with only a single table:

- **`device_type`**: How the device was accessed (not where it ran). Must be one of:
  - `native-tpu`
  - `cpu-fallback`
  - `cpu-simulated`
- **`accelerator`**: The TPU chip generation detected from **inside** the container (via `TPU_ACCELERATOR_TYPE` or the runtime's device name), e.g., `tpu7x-8`.
  - **Constraint:** Do NOT infer this from the cluster or node name (the cluster name includes a chip generation string, which does not guarantee what actually executed).
  - Use `unknown` if the value cannot be read from inside the container.
- **`max_abs_diff`**: The maximum absolute difference between the TPU result and the CPU reference result for the matmul operation.
- **`tolerance`**: The numerical tolerance threshold used for comparison.

*(Note: If an existing check script outputs this table, copy the exact output rows rather than reconstructing them from memory.)*

---

### Post-Table Analysis & Citations

Directly below the table, provide:
1. **Metric Interpretation:** An explanation of what the numerical values demonstrate regarding runtime readiness.
2. **Unresolved Factors:** Any hardware or runtime attributes that could not be determined.
3. **Porting Gotchas:** Specific warnings or pitfalls that could impact the upcoming PyTorch-to-TPU port.
4. **Hardware Citations:**
   - Cite any factual claims about the hardware itself (chip specifications, architectural capabilities, or configuration) as footnotes linking to public documentation.
   - Do **not** cite your own benchmark measurements or inferences; clearly label those as measured or inferred.
   - If an external URL is inaccessible in your environment, explicitly name the exact document/source you would have cited.
