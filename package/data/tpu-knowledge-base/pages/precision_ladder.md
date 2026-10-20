---
torch_tpu_version: 0.1.1.dev20260922
---

# **The Cloud TPU Precision Ladder: FP32 Emulation to FP8 W8A8**

Selecting the optimal floating-point format is one of the highest-leverage performance decisions when training or serving deep learning models on Google Cloud TPUs. This guide walks through the physical hardware behavior of precision formats across TPU v6e and v7 (Ironwood), detailing when FP8 accelerates workloads and where it offers zero benefit.

---

## The Hardware Precision Ladder Overview
| Precision Level | Physical Hardware Execution | Memory Footprint (Relative) | Peak Hardware Throughput | Primary Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 (Single)** | MXU: 1 BF16 pass by default (DEFAULT); 3 passes at HIGH, 6 at HIGHEST. VPU: native 32-bit ALUs | 100% (4 bytes/param) | 1.0× (Baseline) | Master optimizer weights, loss calculation |
| **BF16 (BFloat16)** | Native MXU systolic array execution | 50% (2 bytes/param) | **~2.0× over FP32** | Default baseline for training &amp; inference |
| **FP8 (E4M3 / E5M2)** | Native 8-bit MXU execution on TPU7x; on v6e listed at the BF16 rate, most likely without a native FP8 path (inferred, see below) | **25% (1 byte/param)** | **2× over BF16 on TPU7x; 1× (same as BF16) on v6e** | Large GEMMs, LLM Prefill, Paged KV Cache |

---

## 1. The Hardware VPU vs. MXU Dichotomy
A critical architectural distinction on Google TPUs is the difference between the **Matrix Multiply Unit (MXU)** and the **Vector Processing Unit (VPU)**:

<aside class="important">
<b>The VPU Elementwise Reality:</b> FP8 provides <b>0% throughput gain</b> on VPU elementwise vector operations (such as ReLU, GELU, RMSNorm, or tensor additions). The physical VPU vector ALUs only operate on 32-bit floating-point datapaths. Casting elementwise tensors to FP8 simply adds redundant cast instructions without increasing ALU clock speed.
</aside>

### Where FP8 Delivers Real Hardware Gains

In contrast, the **MXU systolic arrays** run 8-bit dot products natively -- but which 8-bit type depends on the generation. [TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x) lists FP8 at 4,614 TFLOPs per chip, twice its 2,307 BF16. For v6e, the TPU7x doc's comparison table lists FP8 at 918, equal to its BF16, and the [TPU v6e](https://docs.cloud.google.com/tpu/docs/v6e) doc lists only BF16 (918 TFLOPs) and INT8 (1,836 TOPs). Neither says whether v6e executes FP8 natively. The inference here is that it does not, so FP8 runs at best at the BF16 rate: the same table lists v5p FP8 equal to its BF16 as well, and FP8 matmuls through torch_tpu measured 6–7× *slower* than BF16 at 2048–8192³ (measured: `index.py get v6e-measured-matmul-quantization`; the relayout behind it is in `index.py get xla-8bit-layout-copies`). On v6e the 2× 8-bit path is INT8, not FP8.

```
[FP8 Matrix A: 1 Byte] ──┐
                         ├──► [MXU 256x256 Systolic Multiplier] ──► [FP32 Accumulator]
[FP8 Matrix B: 1 Byte] ──┘         (2x raw throughput vs. BF16 on TPU7x)
```

1. **Compute Throughput:** On TPU7x, FP8 GEMMs have **2× the peak TFLOPS** of BF16 on the MXU. On TPU v6e FP8 is listed at the BF16 rate (918 TFLOPs), most likely without a native FP8 path (inferred above), so FP8 weights buy memory, not MXU throughput; INT8 (1,836 TOPs) is the 2× peak there.
2. **Memory Footprint:** Weighs 50% less than BF16 in High Bandwidth Memory (HBM), cutting KV cache memory consumption in half.
3. **Bandwidth Savings:** Memory-bandwidth-bound matrix operations (such as decode linear projections) move half the bytes over the HBM bus, so they can run faster, by less than 2× and by an amount that depends on the shape and how much of the step is memory-bound; this page has no measurement of it.

### The 240:1 FP8-MXU:FP32-VPU Law & Sub-Channel Block Scaling
On modern TPUs, the computational throughput disparity between the MXU and VPU informs how low-precision quantization schemes must be architected:

* **Hardware Theoretical Ceilings vs. Measured Reality on TPU v7**:
  * Peak FP8 MXU Compute: **4,614 TFLOP/s** per chip, two TensorCores, so **~2,307 TFLOP/s** per TensorCore ([TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x); 2,307 per chip is its BF16 figure).
  * Peak FP32 VPU Compute: **~9.6 TFLOP/s** per device (32-bit vector ALUs).
  * Theoretical peak compute throughput ratio: **240 : 1**.
  * **Empirical Silicon Measurement**: In benchmark evaluations on Cloud TPU v7 hardware, large GEMMs achieved **746.05 TFLOP/s** on MXU vs **0.86 TFLOP/s** on VPU chained vector ALUs, yielding an empirical compute ratio of **865.3 : 1**.

* **The VPU Microscaling Bottleneck**:
  * On Cloud TPU v7 and v6e generations, sub-channel block scaling (calculating dequantization scale factors per sub-block in formats like MXFP8 or MXFP4) executes entirely on the **VPU**, *not* the MXU.
  * When executing fine-grained sub-channel block sizes, the VPU introduces significant overhead computing scale factors before the MXU can execute the matrix multiplication.

#### Empirical Sub-Channel Block Size Sweep on TPU v7 Silicon:
Measured with a (4,096 × 4,096) × (4,096 × 4,096) compiled workload on TPU v7 (pure GEMM baseline = 354.26 µs):

| Quantization Block Size (B) | Scales per Row (K=4096) | VPU Dequant Latency | Combined Latency | VPU Overhead | VPU Overhead (%) |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **32 (Fine-grained)** | 128 blocks | 297.19 µs | 440.47 µs | +86.22 µs | **24.3% overhead** |
| **64** | 64 blocks | 286.43 µs | 426.76 µs | +72.51 µs | **20.5% overhead** |
| **128** | 32 blocks | 279.73 µs | 437.45 µs | +83.20 µs | **23.5% overhead** |
| **256** | 16 blocks | 287.89 µs | 436.99 µs | +82.73 µs | **23.4% overhead** |
| **512** | 8 blocks | 287.30 µs | 452.03 µs | +97.77 µs | **27.6% overhead** |
| **1024** | 4 blocks | 308.81 µs | 430.83 µs | +76.58 µs | **21.6% overhead** |
| **4096 (Per-Channel)** | 1 block | **195.69 µs** | **383.07 µs** | **+28.81 µs** | **8.1% overhead** |

<aside class="important">
<b>Sub-Channel Block Size Rule:</b> On Cloud TPU v7 and v6e, sub-channel block scaling imposes an ~80–95 µs VPU latency floor (20%–28% overhead) on matrix operations because scale expansion executes on the VPU. Standard per-channel scaling (<code>B = K</code>) reduces overhead to <b>8.1%</b>.
</aside>

## 2. Selecting the Right FP8 Format: E4M3 vs. E5M2
TPU hardware supports two distinct 8-bit floating point formats defined by the OCP (Open Compute Project) specification:

| Format | Bit Layout (S.E.M) | Dynamic Range | Precision (Mantissa) | Recommended Usage |
| :--- | :--- | :--- | :--- | :--- |
| **FP8 E4M3** | 1 sign, 4 exp, 3 mantissa | [-448, 448] | Higher (3 bits) | **Forward Pass:** Model weights and activation GEMMs. |
| **FP8 E5M2** | 1 sign, 5 exp, 2 mantissa | [-57,344, 57,344] | Lower (2 bits) | **Backward Pass:** Gradients, where wide dynamic range prevents underflow. |

### Dynamic Scaling Factors
Because 8-bit formats have limited dynamic range compared to BF16, you must apply **dynamic scaling factors**:

```text
X_fp8 = clamp(round(X * (scale / max(|X|))))
```

On TPU, these scaling factors are evaluated in the fused prologue/epilogue of the compiled GEMM, avoiding extra memory passes.

---

## 3. Code Example: Automatic Mixed Precision (AMP) with `torch_tpu`

```python
import torch
try:
  import torch_tpu
except ImportError:
  pass

# Configure device
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

class TransformerMLP(torch.nn.Module):
  def __init__(self, hidden_dim=1024, ffn_dim=4096):
    super().__init__()
    self.gate_proj = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
    self.up_proj = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
    self.down_proj = torch.nn.Linear(ffn_dim, hidden_dim, bias=False)

  def forward(self, x):
    # Best Practice: GEMMs execute in BF16/FP8 on MXU,
    # while SiLU executes with full precision in VPU registers
    gate = torch.nn.functional.silu(self.gate_proj(x))
    return self.down_proj(gate * self.up_proj(x))

model = TransformerMLP().to(device)
compiled_model = (
    torch.compile(model, backend="tpu", dynamic=False)
    if device.type == "tpu"
    else model
)

# Execute under AMP Autocast
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
  inputs = torch.randn(4, 512, 1024, device=device)
  outputs = compiled_model(inputs)

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert torch.isfinite(outputs).all(), "AMP outputs must be finite"
```

---

## Next Steps
- Read **[High-Throughput MoE Serving & Decode Bottlenecks](./moe_serving_and_decode.md)** to see how FP8 accelerates KV cache decode.
- Review **[XLA Lowering Pipeline & Runtime Flag Architecture](./xla_lowering_and_flags.md)** for compiler-level precision control.
- See **[Why FP8 runs slower than BF16 on v6e](./xla-8bit-layout-copies.md)** before quantizing weights to FP8 on v6e.
- See **[FP8 KV cache with ragged paged attention](./fp8-kv-cache-rpa.md)** for which FP8 format to ask vLLM for.
