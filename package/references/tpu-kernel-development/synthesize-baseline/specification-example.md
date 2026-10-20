# SwiGLU feed-forward block

The file a `specification` run names as its input. It holds prose, not code —
an agent synthesizes the baseline from it, and the user approves that baseline
before anything is optimized.

The description still arrives as a file the caller named, which is what keeps
this mode under the same rule as every other input type: the loop never starts
against something nobody pointed at. It also gives the approval gate a durable
thing to point back at, so a reviewer can diff what was asked for against what
was written.

## What to compute

```
y = (silu(x @ Wg) * (x @ Wu)) @ Wd
```

A single token-major batch. No bias, no dropout, no residual.

## Tensors

| name | role | shape | dtype |
| :--- | :--- | :--- | :--- |
| `x` | input | (4, 512, 2048) | bfloat16 |
| `Wg` | parameter | (2048, 5632) | bfloat16 |
| `Wu` | parameter | (2048, 5632) | bfloat16 |
| `Wd` | parameter | (5632, 2048) | bfloat16 |
| `y` | output | (4, 512, 2048) | bfloat16 |

These are the problem. Do not substitute a convenient size for any of them.

## Numerics

Accumulate every matmul in float32 and store bf16. `silu` is `x * sigmoid(x)`,
computed in float32.

Precision the baseline gives away silently becomes precision the Pallas kernel
is not required to keep, so this is part of the specification rather than an
implementation detail.

## Edge cases

Sequence length is fixed at 512 and divides evenly. No ragged tail, no masking.

## Known equivalent

LLaMA's `FeedForward`, with gate / up / down projections.
