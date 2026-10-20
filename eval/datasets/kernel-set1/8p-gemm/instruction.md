The code below is a dense matmul (one of the examples from JaxBench (https://github.com/AI-Hypercomputer/accelerator-agents/blob/main/JAXBench/benchmark/level1/8p_GEMM/baseline.py). I would like you optimize the code for TPU on the most recent tpuv7. Make sure to use the full host.

Guidelines:
1. Write code locally on the shared mount at /app . Use the mount space to create a local git space and make a clean commit for every change.
2. GKE Deployment & Job Execution:
   - Write your own GKE YAML manifest (Kubernetes Job) for deployment onto the cluster.
   - Make it clear you are an agent-launched job by strictly following the job name convention `tpu-auto-8p-<something uniq>` (e.g. `tpu-auto-8p-$(date +%s)` or `tpu-auto-8p-<uuid>`).
   - Do NOT build a new docker image on every change. Keep code changes under 1MB and use the mount mechanism (e.g. package your script into a Kubernetes ConfigMap mounted at `/app`).
   - Target TPU nodes using information from your env -- `bash env | grep GKE` could help.
3. Use xprof for profiling and use gcloud ML diagnostics, if its available on the cluster. Your GCP cluster info is available within the enviroment.
4. Use `tpu-agent` (with `tpu-kernel-development`) as needed. Work out which of
   these are in your environment before you start rather than when you hit
   something you cannot answer; if neither is installed, do the work
   directly yourself.
5. I would like you to propose a solution that comes within 30% of roofline or better. Max out perf is great.

Reference code --------
"""Dense bf16 GEMM — Llama-70B FFN dimensions.

Baseline dense matrix multiplication at Llama-3.1-70B hidden-to-FFN scale.
A = (8192, 8192), B = (8192, 28672) — matches hidden_dim -> mlp_dim projection.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemm_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'dense_matmul',
    'M': 8192,
    'K': 8192,
    'N': 28672,
}

def create_inputs(dtype=jnp.bfloat16):
    """Returns (A, B) matrices."""
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    return A, B

def workload(A, B):
    """Dense matmul: C = A @ B"""
    return jnp.dot(A, B)

def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for_ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) *1000
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    flops = 2* M *K* N
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }

if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
