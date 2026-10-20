# Collective Operations Performance Estimation Schema

This document defines the usage of the `collective.py` library for estimating the latency and bandwidth of collective communication operations on TPUs (v3, v4, v5e, v5p, v6e, v7x) and custom mesh/torus topologies.

## Setup Requirements

Before running any Python script for roofline analysis calculations that involves collective communication:
1. Ensure the collective cost estimator library is available at `cost_estimator/collective.py`.
2. Import it in your script as:
```python
from cost_estimator import collective
```
3. **Always use** the collective cost estimator library (`collective.py`) to estimate collective latency rather than simplified manual formulas.

---

## 1. Machine Topology Specification (`collective.tpu`)

The `collective.tpu` factory creates a `Machine` instance representing the physical and logical interconnect configuration.

```python
machine = collective.tpu(
    gen='v7x',                     # TPU generation: 'v3', 'v4', 'v5p', 'v5e', 'v6e', 'v7x'
    slice_dims=(4, 4, 4),          # Slice shape (package grid)
    wraparound=None,               # Optional tuple of bools for per-axis wraparound (torus)
    d2d=True,                      # True for v7x dual-die view (prepends size-2 D2D axis)
    d2d_bw=None,                   # Optional D2D one-way BW [B/s] (default: 360 GB/s placeholder)
    d2d_hop_latency=None,          # Optional D2D per-hop latency [s] (default: 1us)
    ici_contention=None,           # Optional ICI contention factor with d2d=True (default: 2.0)
    **ici_overrides                # link_efficiency, hop_latency, step_overhead, launch_overhead
)
```

### 1.1 TPU v7x Dual-Die (D2D) vs. Package View

*   **Package View (`d2d=False`)**: Models the TPU as single-device packages with a 3D grid `(X, Y, Z)`. `n_devices = prod(slice_dims)`.
*   **Per-Die View (`d2d=True`)**: Models the 2 compute dies per package as distinct devices. The topology becomes 4D: `(2, X, Y, Z)`.
    *   **Axis 0 (D2D)**: Represents the high-speed intra-package Die-to-Die link (`size=2`, `wraparound=False`, `contention=1.0`).
    *   **Axes 1..3 (ICI)**: Represent external ICI mesh/torus links (`size=X, Y, Z`). Contention is automatically set to `2.0` because the two die layers share the same physical ICI SerDes links (halving effective per-die ICI bandwidth).
    *   **Device Count**: `n_devices = 2 * prod(slice_dims)`.

### 1.2 Wraparound (Torus vs. Mesh) Rules

*   **OCS Generations (`v4`, `v5p`, `v7x`)**: The Optical Circuit Switch (OCS) provides wraparound only for full $4 \times 4 \times 4$ cubes. If all dimensions are multiples of 4 (e.g. `(4, 4, 4)` or `(4, 8, 8)`), all ICI axes are tori (`wraparound=True`). If any dimension is not a multiple of 4 (e.g. `(2, 2, 4)`), all axes default to line/mesh (`wraparound=False`).
*   **Fixed 2D Generations (`v5e`, `v6e`)**: An axis is a torus only at full board width (16 chips).
*   **Explicit Override**: You can explicitly specify `wraparound=(True, False, True)` to override heuristics.

---

## 2. Collective Communication APIs

### 2.1 Reduce-Scatter (`reduce_scatter`)

Estimates Reduce-Scatter latency and returns an `Estimate` dataclass.

```python
est = collective.reduce_scatter(
    machine=machine,
    bytes_per_device=d_bytes,  # Per-device input operand size in Bytes (output is D/N)
    axes=axes_subset,          # Optional tuple of active axis indices (e.g. (1, 2) or (0,))
    algo="auto",               # "auto" (picks min), "sequential", or "concurrent"
    unidirectional=False,      # Restrict rings to unidirectional transfers
)
```

#### Multi-Axis Algorithms:
*   **`sequential`**: Dimension-ordered Reduce-Scatter passing axes one by one. The data volume shrinks by $1/X_i$ after each axis pass. The estimator evaluates all permutations to find the lowest-latency order (faster axes first).
*   **`concurrent`**: Splits data into $k$ shards and rotates the axis schedule across shards to saturate all active links simultaneously. On a 3D symmetric torus, bandwidth time approaches $\sim \frac{1}{3} \times$ sequential time.
*   **`auto`** (default): Evaluates both sequential and concurrent algorithms and selects the one with the lowest total latency.

### 2.2 All-Gather (`all_gather_time`)

All-Gather transfers the exact same data volume as Reduce-Scatter in reverse.

```python
latency_s = collective.all_gather_time(
    machine=machine,
    bytes_full=full_gathered_bytes,  # Full gathered tensor output size in Bytes
    axes=axes_subset,                # Optional active axis indices
    algo="auto",
)
```

### 2.3 All-Reduce (`all_reduce` / `all_reduce_time`)

All-Reduce is decomposed as a Reduce-Scatter phase followed by an All-Gather phase ($AR(D) = RS(D) + AG(D) = 2 \times RS(D)$, launch overhead included — two legs are two launches).

```python
est       = collective.all_reduce(machine, bytes_per_device=d_bytes, axes=axes_subset)
latency_s = collective.all_reduce_time(machine, d_bytes, axes=axes_subset)  # scalar
```

`all_gather` and `all_reduce` return an `Estimate`; the `_time` spellings return a float and are kept for existing callers.

### 2.4 All-to-All (`all_to_all`) — **not** a Reduce-Scatter variant

```python
est = collective.all_to_all(
    machine=machine,
    elems=tokens * topk * hidden,  # the FULL per-device payload, incl. its own shard
    dtype="fp8",
    axes=axes_subset,
    algo="sequential",             # "sequential" | "concurrent". NO "auto".
)
```

RS/AG are **egress**-bound: the payload shrinks or accumulates as it travels, so no chunk crosses more than one link per step. All-to-all is a permutation — a chunk bound for a device $d$ hops away traverses $d$ links — so it is **link (bisection)** bound:

$$T_{bw} = \frac{S \cdot X}{4 \cdot \text{lanes} \cdot B \cdot \eta}$$

against RS's $S(X-1)/(X \cdot \text{lanes} \cdot B \cdot \eta)$. Exact for even $X$; at $X=2$ it coincides with the egress bound. Costing an all-to-all with an RS proxy understates it by $X^2/(4(X-1))$ — **4.27× on a 16-wide axis**, and the reason two agents pricing the same EP group once came out 5.45× apart.

Two further differences from RS:

* **The payload is preserved across phases**, so per-axis terms *add* rather than telescope, and the total is independent of axis order.
* **`algo="auto"` is refused.** `concurrent` equals the bisection floor; reporting a floor for a dimension-ordered implementation understates it. Name the schedule. Which one XLA issues on TPU is **UNREAD** as of 2026-09-19 — `sequential` is the conservative default.

`all_to_all_lower_bound(machine, bytes, axes)` gives $\max(\text{egress}, \max_a \text{bisection}_a)$.

### 2.5 Dtypes — pass `elems` + `dtype`, not bytes

Wire bytes are not always `elems × itemsize`:

| | wire dtype |
| --- | --- |
| `all_gather`, `all_to_all` | **always** the operand dtype (a permutation cannot upcast). Passing `wire_dtype` raises. |
| `reduce_scatter`, `all_reduce` | may upcast — an implementation can accumulate in a wider type than the operand. `wire_dtype=` expresses it; defaults to `dtype`. |

```python
# fp8 operands summed in bf16 put 2x the bytes on the wire:
collective.all_reduce(m, elems=n, dtype="fp8", wire_dtype="bf16")
```

A caller that pre-multiplies `elems * itemsize` cannot express the second row and will be silently low. `DTYPE_BYTES` carries the names; a bare number means bytes/element. The legacy positional `bytes_per_device` still works on every collective.

**The two legs of an MoE exchange usually differ** (fp8 out, bf16 back) — issue them as two calls rather than averaging.

### 2.4 Theoretical Lower Bound (`reduce_scatter_lower_bound`)

Calculates the absolute hardware floor for Reduce-Scatter across the active axes:

```python
lb = collective.reduce_scatter_lower_bound(machine, bytes_per_device, axes=axes_subset)
# lb.t_bandwidth : per-chip egress bound
# lb.t_latency   : network diameter * per-hop alpha
# lb.t           : max(lb.t_bandwidth, lb.t_latency)
```

---

## 3. Sub-Mesh & Parallelism Mapping Examples

When a collective operation executes on a subset of axes (e.g. Tensor Parallelism, Data Parallelism, or Expert Parallelism), pass the `axes` parameter:

### Example: TPU v7x 4x4x4 with `d2d=True`
*   Topology: `m = collective.tpu("v7x", (4, 4, 4), d2d=True)`
*   Axes mapping:
    *   `axis 0`: D2D (intra-package, size 2)
    *   `axis 1`: ICI X (inter-package, size 4)
    *   `axis 2`: ICI Y (inter-package, size 4)
    *   `axis 3`: ICI Z (inter-package, size 4)

#### 1. Intra-Package Tensor Parallelism (`TP=2`, D2D only)
Collective communication runs solely across the 2 chiplets of each package:
```python
# axes=(0,) isolates the D2D axis
tp_latency_s = collective.all_reduce_time(m, bytes_per_device, axes=(0,))
```

#### 2. Inter-Package Data / Expert Parallelism (`DP` on ICI only)
Collective communication runs across the 64 packages via ICI (bypassing D2D):
```python
# axes=(1, 2, 3) isolates the 3D ICI mesh
dp_latency_s = collective.all_reduce_time(m, bytes_per_device, axes=(1, 2, 3))
```

#### 3. Global Mesh Collective (e.g. `DP=64, TP=2` across all 128 dies)
```python
# axes=None includes all axes (0, 1, 2, 3)
global_latency_s = collective.all_reduce_time(m, bytes_per_device)
```

---

## 4. Working with `Estimate` Output

The `Estimate` dataclass returned by `reduce_scatter` provides comprehensive metrics:

```python
est = collective.reduce_scatter(m, d_bytes)

print(f"Total time: {est.total * 1e3:.3f} ms")
print(f"  Latency component: {est.t_latency * 1e3:.3f} ms")
print(f"  Bandwidth component: {est.t_bandwidth * 1e3:.3f} ms")
print(f"  Algorithm: {est.algo}")
print(f"  AlgBW: {est.algbw / 1e9:.1f} GB/s, BusBW: {est.busbw / 1e9:.1f} GB/s")
print(f"  Bottleneck bound: {est.bound}")  # 'latency' or 'bandwidth'

# Human-readable breakdown of per-axis phases:
print(collective.describe(est))
```

---

## 5. Utilities

*   **`collective.parse_size(s: str) -> float`**: Parses strings like `"128MiB"`, `"1GiB"`, `"64MB"`, `"1024B"` into byte floats.
*   **`collective.fmt_bytes(bytes: float) -> str`**: Formats byte numbers to human-readable units (e.g. `"128MiB"`).
*   **`collective.fmt_time(seconds: float) -> str`**: Formats time in `us`, `ms`, or `s`.
*   **`collective.fmt_bw(bytes_per_sec: float) -> str`**: Formats bandwidth in `GB/s`.
*   **`collective.sweep(machine, sizes, axes=None, title="")`**: Prints a benchmark table comparing sequential, concurrent, and lower bound costs across tensor sizes.