---
scripts:
- scripts/roofline-analysis/roofline.py
---

# JAX Type Promotion & TPU Hardware Rules

This document outlines how JAX handles data types, specifically focusing on hardware-level accumulation behaviors on Google TPUs, and the standard JAX type promotion rules depending on the data type of the Left-Hand Side (LHS) and Right-Hand Side (RHS).

---

## 1. TPU Hardware Characteristics & MatMul Accumulation

TPUs are designed heavily around Matrix Multiply Units (MXUs). When running matrix multiplications (`jax.numpy.dot`, `matmul`, or Dense layers), the data type behavior at the hardware level is governed by **Accumulation Rules**, which is different from element-wise type promotion.

### MXU Accumulation Rules
The MXU computes products and sums them up (accumulates). To prevent overflow and maintain precision, the accumulation type is often wider than the input type:

LHS Dtype (Input) | RHS Dtype (Weight) | MXU Accumulation Dtype          | Default Output Dtype (HBM) | Notes
:---------------- | :----------------- | :------------------------------ | :------------------------- | :----
`bfloat16`        | `bfloat16`         | **`float32`**                   | **`float32`**              | The MXU natively computes in `fp32`. By default, the intermediate output is written to memory in the accumulation dtype (`float32`) to preserve precision, unless a global precision (e.g., `bf16`) is explicitly specified.
`float32`         | `float32`          | **`float32`**                   | `float32`                  | TPUs may implicitly downcast `fp32` inputs to `bf16` inside the MXU for speed, unless `precision=jax.lax.Precision.HIGHEST` is specified.
`int8`            | `int8`             | **`int32`**                     | `int32`                    | Integer matrix multiplications accumulate exactly in `int32` to avoid overflow.
`fp8` (or `w8a8`) | `fp8` (or `w8a8`)  | **`float32`** or **`bfloat16`** | **`bfloat16`**             | Inputs: 1B. Output: 2B (bf16) default.
`fp8` (or `w4a8`) | `fp4` (or `w4a8`)  | **`float32`** or **`bfloat16`** | **`bfloat16`**             | LHS: 1B, RHS: 0.5B. Output: 2B (bf16) default.

**Key Takeaway for TPUs:** 
Because JAX strongly favors speed on accelerators, operations on `float32` may run at `bfloat16` speeds on older TPUs (v2/v3) natively. On newer TPUs (v4/v5), `float32` is supported natively but `bfloat16` is much faster.

---

### Selection outputs are not promoted values — they are integers

`top-k`, `argmax`, `argsort`, `where` produce **positions, not values**, and nothing promotes
them: the width comes from the range being indexed, not from the input dtype. `int32`, i.e.
`ceil(log2(N))` bits rounded to a machine type. An op whose output width differs from its input
width must say so at the call site — `{{script:roofline-analysis/roofline.py}}` carries this as the `indexer_indices`
numerics site, because a helper that charges one dtype for a whole op gets one end wrong.

## 2. JAX Element-wise Type Promotion Rules (LHS vs RHS)

When performing binary operations (like addition `+`, multiplication `*`) between two arrays, JAX uses a lattice system (similar to `jnp.promote_types`) to determine the resulting data type. 

JAX generally **avoids implicit promotion to 64-bit types** (`float64`, `int64`) to remain accelerator-friendly.

### Type Promotion Table (Common Neural Network Dtypes)

| LHS Dtype | RHS Dtype | Resulting Dtype | Reason / Rule |
| :--- | :--- | :--- | :--- |
| `int8` | `int8` | `int8` | Same type, no promotion. |
| `int8` | `int32` | `int32` | Same kind (integer), wider bit-width wins. |
| `int32` | `bfloat16` | `bfloat16` | Float dominates Integer. (Width does not force upcast to fp32). |
| `int32` | `float32` | `float32` | Float dominates Integer. |
| `bfloat16` | `bfloat16` | `bfloat16` | Same type, no promotion. |
| `bfloat16` | `float16` | `float32` | Mixing different 16-bit float formats safely promotes to 32-bit to avoid precision loss. |
| `bfloat16` | `float32` | `float32` | Same kind (float), wider bit-width wins. |

### General Rules Summary:

1. **Same Kind, Different Width:** The wider bit-width always wins (e.g., `int8 + int32 = int32`).
2. **Different Kinds:** Floating-point types always dominate Integer types (e.g., `int32 + float32 = float32`).
3. **Weakly Typed Scalars:** If one side is a Python scalar (e.g., `1.0` or `2`), JAX will "weakly" promote it to match the JAX array's dtype to prevent unnecessary widening.
    * Example: `jnp.array(1.0, dtype=jnp.bfloat16) + 1.0` results in `bfloat16`, not `float64`.
