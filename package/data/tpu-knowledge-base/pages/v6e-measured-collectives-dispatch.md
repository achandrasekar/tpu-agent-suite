<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: v6e-measured-collectives-dispatch
keyword: v6e-measured-collectives-dispatch
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-14
source_updated: 2026-10-13
generator: authored
---

# The time around my kernels on TPU v6e is too high: all-reduce cost by dtype and size, collective kinds, dispatch overhead and recompiles

Your kernels are fast but the step is not. Time goes to gradient all-reduces,
to the host waiting on the device after every small op, or to compiling a new
shape every time the input length changes. This page collects measurements
on v6e of those overheads: what an all-reduce
costs in bf16 against fp32 and how that changes with payload size, what
all-gather, all-to-all and all-reduce cost against one local pass over the
same buffer, how much queueing ops before a synchronise saves, and how long a
new shape takes to compile through `jax_op`. The collective experiments ran on
four v6e chips in a 2x2 mesh, one process per chip under `torchrun`, through
`torch.distributed` on torch_tpu. Before timing, each rank filled a probe with
a rank-dependent value and ran one real collective, and the run stopped unless
every rank got the correct cross-chip result. The traffic between chips was
real. The dispatch and recompile experiments ran on one chip.

Every number below is a ratio: the time with the baseline divided by the
time with the alternative. Above one, the alternative is faster by that
factor; below one, it is slower, and the number is the share of the
baseline's speed it keeps. In the collective sections the baseline is
sometimes a local pass over the buffer, which is not a substitute for a
collective. There the ratio is only a yardstick: the local pass takes that
fraction of the collective's time.

All collective figures come from one 2x2 slice inside one host, at the
software versions under Evidence. Treat them as what this slice and stack
did, not as interconnect properties of v6e.

## Measured

### Send large gradient buffers as bf16

The test was `dist.all_reduce(x, SUM)` across the four chips on a bf16 buffer
(baseline) against the same call on an fp32 buffer of the same element count.
The fp32 arm moves twice the bytes. That is the choice you face when you hold
a gradient buffer of a given length and pick the dtype to send it in.

| elements per rank | fp32 keeps this share of bf16's speed |
|---|---|
| 64 Ki  | 1.03[^1] |
| 1 Mi  | 0.743[^2] |
| 4 Mi  | 0.519[^3] |
| 16 Mi  | 0.51[^4] |
| 64 Mi  | 0.504[^5] |

At the smallest payload the dtype made no meaningful difference: the fp32
reduce ran at 1.03 of the bf16 one's speed.[^1]
From 4 Mi elements up, fp32 cost close to twice as much as bf16. The ratio was
0.504 at 64 Mi,[^5] which is
the byte ratio. The change happens between 64 Ki and 4 Mi elements. At
1 Mi, fp32 already ran at only 0.743 of bf16's
speed.[^2] The 256 Ki cell has no result: the two
sessions disagreed and could not tell the two dtypes apart. If you reduce
gradients in fp32 for accuracy, on this 2x2 in-host slice a buffer of a few
million elements or more paid the full doubling.

### A large all-reduce costs a fixed multiple of a local pass

The yardstick was `x.mul_(1.0)` on the same bf16 buffer, which reads and
writes every element once, against `dist.all_reduce(x, SUM)` across the four
chips. A ratio near one means the all-reduce costs about as much as touching
the buffer once. A ratio of one half would mean it costs two touches.

| elements per rank | local pass time / all-reduce time |
|---|---|
| 64 Ki  | 0.924[^6] |
| 4 Mi  | 0.38[^7] |
| 16 Mi  | 0.181[^8] |
| 64 Mi  | 0.185[^9] |

- **Small buffers.** At 64 Ki elements the all-reduce took about as long as
  the local pass (ratio 0.924).[^6]
  Both arms ran near the rate at which torch_tpu issues work (see the
  dispatch section below). So the true latency of a small collective was
  hidden behind dispatch cost and was never measured here. The 256 Ki and
  1 Mi cells have no result. At 1 Mi one host timed the all-reduce as slightly
  faster than the local pass and the other as slightly slower. At 256 Ki one
  session was inadmissible.
- **Large buffers.** From 16 Mi elements the ratio stopped moving:
  0.181 at 16 Mi[^8] and
  0.185 at 64 Mi.[^9] At those two
  sizes a bf16 all-reduce on this 2x2 mesh cost about 5.5 local passes over
  the same buffer. Two points are not enough to call it constant beyond
  64 Mi. The experiment had expected a tenfold cost at some payload, and an
  all-reduce costing at least one local pass at every payload; neither held.
- A second experiment repeated the all-reduce as a reference point and got
  the same picture: 0.877 at
  1 Mi,[^10] 0.39 at
  4 Mi[^11] and 0.181 at
  16 Mi.[^12]

### All-to-all and all-gather cost more than all-reduce per input element

The same yardstick, a local pass over the per-rank bf16 buffer, was run
against `dist.all_gather_into_tensor` and `dist.all_to_all_single` on the 2x2
mesh. These collectives do different jobs. Each row is the cost of one kind
in units of a local pass, not a choice between kinds. In particular, the
all-gather writes an output four times the size of its input, one slice from
each chip, so it moves more data than the other two.

| per-rank elements | all-reduce | all-gather | all-to-all |
|---|---|---|---|
| 1 Mi  | 0.877[^10] | 0.495[^13] | 0.254[^14] |
| 4 Mi  | 0.39[^11] | 0.149[^15] | 0.0696[^16] |
| 16 Mi  | 0.181[^12] | 0.0671[^17] | 0.0311[^18] |

(Each cell is local pass time divided by collective time. Smaller means the
collective is more expensive.)

- **All-to-all was the most expensive per element, which the bandwidth
  model does not predict.** At 16 Mi per-rank elements a local pass took
  only 0.0311 of the all-to-all's time:[^18] about 2.2 times an all-gather
  and 5.8 times an all-reduce of the same per-rank input. On a ring the
  bandwidth model puts all-to-all at a quarter of an all-gather and half an
  all-reduce (`index.py get concept-collective-costs`), so this all-to-all ran about
  9 to 12 times slower than the model. torch_tpu lowers an even-split
  `all_to_all_single` to one XLA all-to-all, so the cost is in the
  XLA/libtpu implementation on this slice, not an inherent property of the
  collective. Its share fell steadily as the buffer grew and had not levelled
  off by the largest size tested. If an expert-parallel or sequence-parallel
  layout puts an all-to-all on the critical path, measure it on your own
  slice and versions rather than budgeting it from the model.
- **All-gather** sits between the other two. Its cost also kept rising
  relative to the local pass across the sweep, from
  0.495 at 1 Mi[^13] to
  0.0671 at 16 Mi.[^17] Part of
  that is its fourfold larger output.
- **All-reduce** stopped getting relatively more expensive from 16 Mi elements, as in the previous section's longer sweep.

### Queue many small ops before you synchronise

On one chip, one small op (a 512 by 512 bf16 matmul) was issued d times
before a single blocking synchronise, and the time was divided by d. Every
arm does the same work per op. The baseline synchronises after every op.

- **Two ops per synchronise** instead of one:
  1.67× faster per op.[^19]
- **16 ops:** 3.89× faster.[^20]
- **64 ops:** 4.69× faster.[^21]
- **256 ops:** 4.98× faster.[^22]
- **1024 ops:** 5.08× faster.[^23]

For this op, a 512 by 512 matmul on one chip, most of the measured cost was
the host waiting for the device, and most of the saving is already there by
a depth of 64. Larger ops spend proportionally less time waiting, so expect
a smaller gain from queueing them. Going past 256 added
little. In a separate group with 256 as the baseline, 1024 was
1.04× faster.[^24] The
sweep ran shallow-to-deep and then deep-to-shallow, so this is not
the chip warming up over the sweep. In practice: no `.item()`, no printing
of a device tensor and no explicit synchronise inside the inner loop.
Synchronise once per step or less.

Queueing does not remove a per-op floor. Even at depth 1024, the op took far
longer than its arithmetic. This page gives no figure for that floor
because nothing here measured it on its own. It means that splitting work into many tiny
ops costs time, however deep the queue.

### A new shape compiles faster as a Pallas kernel, and then runs slower

On one chip, `y = 2x` was written as plain JAX for XLA (baseline) and as a
Pallas kernel with 128 by 128 blocks, both called through torch_tpu's
`jax_op`. Each arm was compiled for twenty shapes it had never seen. Compile
time is the first call minus the steady-state call.

- **Compile:** the Pallas version compiled
  2.12× faster on a new
  shape.[^25] This was nearly the same at base row counts
  of 1024 (2.13×)[^26] and 8192
  (2.11×).[^27]
- **Run:** the same Pallas kernel kept only
  0.136 of XLA's speed once
  compiled,[^28] at both sizes.

Both arms compute the same result, so this is a real trade-off for this
kernel. The Pallas arm was an untuned 128 by 128 elementwise kernel, so its
run-time loss says how much a careless block costs, not how a tuned Pallas
kernel compares with XLA. Do not read the compile result as advice to write
elementwise ops in Pallas: it pays off only when a shape is compiled many
times and run very few times, such as a search over kernel variants. For a
shape you compile once and run many times, the plain-JAX version won by a
wide margin here. Better still, bucket shapes so new compiles are rare
(`index.py get torch-tpu-avoiding-recompilation`).

## Believed, not measured

- The usual account of the dtype and size results: a collective costs a
  fixed setup plus a per-byte transfer. When the setup dominates, the dtype
  hardly matters. Once the transfer dominates, cost tracks bytes. The shape
  of the measurements fits that account, but neither term was measured
  on its own.
- The all-reduce's ratio levelling off is believed to mark the point where
  both arms are limited by bandwidth: interconnect for the collective, HBM
  for the local pass. Interconnect bandwidth was not measured directly.
- Why all-to-all costs so much more than the bandwidth model, and keeps
  getting relatively more expensive with size rather than levelling off like
  all-reduce, is not established. On a 2x2 the model already counts the
  two-hop routes, so extra hops do not explain it. Whether the same holds in
  pure JAX (`jax.lax.all_to_all` under `shard_map`), on larger slices or on
  other libtpu versions was not tested.
- The per-op floor that queueing does not remove may be host-side enqueue
  work in torch_tpu, a fixed device-side launch cost, or something in the
  runtime. The experiment did not separate these.
- The Pallas kernel's faster compile is believed to come from handing XLA a
  pre-lowered custom call with less to optimise. Its slower run time is
  believed to come from the untuned 128 by 128 block, which is small for a
  streaming elementwise op on v6e. Neither explanation was tested.

## Never measured

- More than four chips, any mesh other than 2x2, and multi-host or
  multi-slice collectives over DCN. Every collective here ran inside one host.
- Reduce-scatter, send/recv, and async collectives that overlap compute.
  Bucketing of many small gradient all-reduces into one large one was not
  measured either. The results here suggest it matters, but it was not run.
- The latency of a small collective. At the small end, all-reduce times
  were hidden behind dispatch cost, so there is no clean setup-cost
  figure.
- Collectives in fp8 or int8, and all-gather or all-to-all in fp32.
- Collectives written with `jax.shard_map` or SPMD sharding rather than
  `torch.distributed`.
- Dispatch overhead for ops other than one small matmul, on more than one
  chip, or with a CUDA-graph-style captured sequence.
- Compile time for any kernel other than a scale, for Pallas kernels with
  tuned blocks, and the effect of a persistent compilation cache across
  processes.
- End-to-end step time. These are isolated timings, and how much of a real
  training step each overhead accounts for is open.

## Evidence

Every figure here was measured on TPU v6e: the collectives on a 2x2 slice of four chips inside one host, the dispatch and compile experiments on one chip, in two sessions on separate hosts, each arm checked against a full-precision reference before timing. A ratio is the baseline's time divided by the alternative's: above one the alternative is faster. Software: jax 0.11.2, libtpu 0.0.46, torch 2.13.0, with kernels called from PyTorch through torch_tpu's `pallas.jax_op` unless the text says otherwise. The measurements come from an unpublished measurement log, exported 2026-10-13; each footnote gives the ratio, in brackets its range across the sessions, and the shapes it was measured at. Nothing here was measured on another chip, another software version, or more than one chip unless stated.

[^1]: fp32 against bf16 all-reduce, 64 Ki elements: ratio 1.03 [1.03, 1.03], 2 sessions; N = 65536.
[^2]: fp32 against bf16 all-reduce, 1 Mi elements: ratio 0.743 [0.735, 0.752], 2 sessions; N = 1,048,576.
[^3]: fp32 against bf16 all-reduce, 4 Mi elements: ratio 0.519 [0.519, 0.52], 2 sessions; N = 4,194,304.
[^4]: fp32 against bf16 all-reduce, 16 Mi elements: ratio 0.51 [0.51, 0.51], 2 sessions; N = 16,777,216.
[^5]: fp32 against bf16 all-reduce, 64 Mi elements: ratio 0.504 [0.504, 0.504], 2 sessions; N = 67,108,864.
[^6]: all-reduce against a local pass, 64 Ki elements: ratio 0.924 [0.918, 0.93], 2 sessions; N = 65536.
[^7]: all-reduce against a local pass, 4 Mi elements: ratio 0.38 [0.372, 0.387], 2 sessions; N = 4,194,304.
[^8]: all-reduce against a local pass, 16 Mi elements: ratio 0.181 [0.18, 0.182], 2 sessions; N = 16,777,216.
[^9]: all-reduce against a local pass, 64 Mi elements: ratio 0.185 [0.183, 0.187], 2 sessions; N = 67,108,864.
[^10]: all-reduce against a local pass, 1 Mi elements: ratio 0.877 [0.834, 0.921], 2 sessions; N = 1,048,576.
[^11]: all-reduce against a local pass, 4 Mi elements: ratio 0.39 [0.381, 0.399], 2 sessions; N = 4,194,304.
[^12]: all-reduce against a local pass, 16 Mi elements: ratio 0.181 [0.179, 0.183], 2 sessions; N = 16,777,216.
[^13]: all-gather against a local pass, 1 Mi elements: ratio 0.495 [0.461, 0.529], 2 sessions; N = 1,048,576.
[^14]: all-to-all against a local pass, 1 Mi elements: ratio 0.254 [0.247, 0.26], 2 sessions; N = 1,048,576.
[^15]: all-gather against a local pass, 4 Mi elements: ratio 0.149 [0.135, 0.164], 2 sessions; N = 4,194,304.
[^16]: all-to-all against a local pass, 4 Mi elements: ratio 0.0696 [0.068, 0.0711], 2 sessions; N = 4,194,304.
[^17]: all-gather against a local pass, 16 Mi elements: ratio 0.0671 [0.0593, 0.0749], 2 sessions; N = 16,777,216.
[^18]: all-to-all against a local pass, 16 Mi elements: ratio 0.0311 [0.0308, 0.0314], 2 sessions; N = 16,777,216.
[^19]: queue depth 2 against 1: ratio 1.67 [1.61, 1.69], 2 sessions; K = 512; M = 512; N = 512.
[^20]: queue depth 16 against 1: ratio 3.89 [3.83, 3.92], 2 sessions; K = 512; M = 512; N = 512.
[^21]: queue depth 64 against 1: ratio 4.69 [4.67, 4.74], 2 sessions; K = 512; M = 512; N = 512.
[^22]: queue depth 256 against 1: ratio 4.98 [4.96, 5.02], 2 sessions; K = 512; M = 512; N = 512.
[^23]: queue depth 1024 against 1: ratio 5.08 [5.05, 5.11], 2 sessions; K = 512; M = 512; N = 512.
[^24]: queue depth 1024 against 256: ratio 1.04 [1.02, 1.07], 2 sessions; K = 512; M = 512; N = 512.
[^25]: Pallas against plain-JAX compile time on a new shape: median ratio 2.12, cells 2.11 to 2.13, over 2 cells, 2 sessions each.
[^26]: compile time, base 1024 rows: ratio 2.13 [2.08, 2.18], 2 sessions; M_base = 1024; N = 1024; new_shapes = 20.
[^27]: compile time, base 8192 rows: ratio 2.11 [2.08, 2.14], 2 sessions; M_base = 8192; N = 1024; new_shapes = 20.
[^28]: Pallas against plain-JAX run time: median ratio 0.136, range [0.134, 0.137], over 2 cells, 2 sessions each.
