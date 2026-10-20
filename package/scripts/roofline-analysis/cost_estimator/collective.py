#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""Analytic alpha-beta model for estimating collective communication latency.

Analytic alpha-beta model for estimating reduce-scatter, all-gather,
all-reduce and all-to-all latency on TPU-style mesh/torus (1D-3D)
interconnects (ICI).

Core idea
=========
Decompose reduce-scatter into a composition of per-axis ring/line passes and
cost each pass with an alpha-beta model.

Notation:
  D      : per-device RS *input* bytes (XLA reduce-scatter operand size;
           the output is D/N)
  N      : number of participating devices = product of participating axis sizes
  B      : one-way peak bandwidth of a single link [bytes/s]
  eta    : link efficiency (achieved fraction of peak)
  alpha  : per-hop latency [s] (TPU ICI ~ 1us)
  X      : axis size

Ring reduce-scatter over one axis (S = data entering that phase):
  steps = X - 1
  T_lat = steps * (alpha + step_overhead)
  T_bw  = S * (X-1) / (X * lanes * B * eta)
    lanes = 2 : wraparound (torus) axis + bidirectional ring (links in both
                directions carry data simultaneously)
    lanes = 1 : line axis without wraparound -- the edge (corner) chip has only
                one link along the axis, so its egress bound makes the pass
                exactly 2x slower than a torus. Unidirectional rings are also 1.

Multi-axis strategies (e.g., all axes of a 3D torus):
  * sequential : pass axes one by one. Data shrinks by 1/X after each pass.
                 With equal per-axis effective BW the (X-1)/X terms telescope:
                 T_bw = D * (1 - 1/N) / (lanes * B * eta).
  * concurrent : split the data into k (= number of axes) shards and rotate the
                 axis order per shard. At any moment different shards use
                 different axes' links, so all links can be saturated at once.
                 For large D,
                 T_bw ~= D * (1 - 1/N) / (sum_a lanes_a * B * eta)
                 (approaches the lower bound below)
  * lower bound: egress bound  = D*(N-1)/N / (total per-chip egress BW)
                 latency bound = (network diameter) * alpha

All-to-all is a different cost model, not a reduce-scatter variant
==================================================================
RS/AG are *egress*-bound: the payload shrinks (RS) or accumulates (AG) as it
travels, so no chunk crosses more than one link per step and every link carries
the same load. All-to-all is a permutation -- a chunk bound for a device d hops
away must traverse d links -- so it is *link* (bisection) bound, and on a wide
axis that is far more expensive than egress. Costing it with an RS proxy is the
single most common way to get MoE expert parallelism wrong.

Ring all-to-all over one axis (S = the full per-device payload, all X
destination shards including this device's own):

  even torus : sum of hop-distances per source = X^2/4, so total hop-bytes
               = S*X^2/4 over 2X directed links -> S*X/8 per link.
  line       : the middle edge is cut once, not twice -> exactly 2x the torus.

  Both collapse to, with lanes as above,

      T_bw = S * X / (4 * lanes * B * eta)

  Exact for even X (odd X is overstated by X^2/(X^2-1), i.e. <1% by X=11), and
  at X=2 it coincides with the egress bound on both a torus and the D2D axis.
  Against an RS proxy on the same payload the ratio is X^2/(4*(X-1)) -- 4.3x at
  X=16 -- which is the size of the error this function exists to remove.

Multi-axis all-to-all:
  * The payload is *preserved* across phases (a permutation moves everything),
    unlike RS where it shrinks by 1/X. So the per-axis terms add rather than
    telescope, and the cost is independent of axis order.
  * sequential : T_bw = sum_a S * X_a / (4 * lanes_a * B_a * eta)
  * concurrent : shard the payload and rotate axis schedules so every axis's
                 links are busy at once -> T_bw ~= max_a (per-axis term). The
                 work on a given axis cannot be reduced, only overlapped with
                 other axes, so this equals the bisection bound.
  * lower bound: max(egress bound, max_a bisection_a); latency as for RS.

  ``concurrent`` *is* the floor, so ``algo="auto"`` deliberately does NOT pick
  the cheaper of the two here (see all_to_all's docstring).

Dtypes
======
Every collective takes ``elems`` + ``dtype`` rather than only raw bytes,
because the bytes actually on the wire are not always ``elems * itemsize``:

  * all-gather / all-to-all are permutations -- the wire dtype is always the
    operand dtype, and passing ``wire_dtype`` is an error.
  * reduce-scatter / all-reduce perform arithmetic, and an implementation may
    reduce in a wider accumulation type than the operand (fp8 operands summed
    in bf16 put 2x the bytes on the wire). ``wire_dtype`` expresses that; it
    defaults to ``dtype``.

A caller that pre-multiplies ``elems * itemsize`` into a byte count cannot
express the second case and will be silently low. The legacy positional
``bytes_per_device`` argument still works everywhere.

v7x dual-die (D2D) support:
  * Per-die view: treating the 2 dies of each package as separate devices turns
    the topology into (2, X, Y, Z), where the leading size-2 axis is the
    die-to-die link. tpu("v7x", (X, Y, Z), d2d=True).
    (With d2d=True, axis 0 = d2d and the ICI axes shift to indices 1..ndim.)
  * Contention (the key part): in the per-die view, each ICI axis has twice as
    many logical rings as physical link sets -- the ring in die layer 0 and the
    ring in die layer 1 share the same physical ICI links. Hence every ICI
    axis's effective BW is halved (Axis.contention=2, applied automatically
    with d2d=True; under SPMD both layers always run the same phase at the
    same time, so this holds for sequential and concurrent alike).
    The D2D axis itself has contention=1: one die pair per package-local link.
  * If, due to the internal design, other axes' ICI traffic *transits* the D2D
    link (SerDes distributed across the two dies), express that share by
    raising the D2D axis's contention or derating d2d_bw.
  * In the bandwidth-bound regime, the per-die view beats the per-package view
    roughly when B_d2d > sum_ici lanes_a * B_a / 2
    (full 3D torus: B_d2d > 3 * B_ici).

Caveats / out of scope:
  * Define axes from the *participating group's* point of view. Even if the
    physical axis is a torus, a subgroup of it (e.g., a TP group of 8 chips on
    a 16-wide axis) is a line -> wraparound=False.
  * v4/v5p/v7x get wraparound from the OCS only in units of full 4x4x4 cubes
    -> only slices whose axes are all multiples of 4 are tori (2x2x4 has no
    wraparound). v5e/v6e only when an axis is 16 wide.
  * Multi-slice (DCN) segments, contention with other traffic, and XLA's
    chunking / algorithm-selection details are out of scope. First-order
    corrections via launch_overhead / step_overhead / eta.
  * On-the-fly reduction (VPU add) is assumed to run at line rate (generally
    true on TPU).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any

# =============================================================================
# 0. Dtypes
# =============================================================================

#: Bytes per element. Keys are the spellings a numerics sheet, a `config.json`
#: or a vLLM flag is likely to use; a bare number is also accepted anywhere a
#: dtype is, and means bytes-per-element directly.
DTYPE_BYTES: dict[str, float] = {
    "fp4": 0.5,
    "e2m1": 0.5,
    "int4": 0.5,
    "fp8": 1.0,
    "e4m3": 1.0,
    "e5m2": 1.0,
    "int8": 1.0,
    "uint8": 1.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "f16": 2.0,
    "int16": 2.0,
    "fp32": 4.0,
    "f32": 4.0,
    "int32": 4.0,
    "i32": 4.0,
    "fp64": 8.0,
    "f64": 8.0,
    "int64": 8.0,
    "i64": 8.0,
}


def dtype_bytes(dtype: str | float) -> float:
    """Bytes per element for a dtype name, or a number taken as bytes/element."""
    if isinstance(dtype, (int, float)) and not isinstance(dtype, bool):
        if dtype <= 0:
            raise ValueError(f"bytes/element must be > 0, got {dtype}")
        return float(dtype)
    key = str(dtype).strip().lower()
    if key not in DTYPE_BYTES:
        raise ValueError(
            f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}, or pass a"
            " number meaning bytes/element")
    return DTYPE_BYTES[key]


def _resolve_payload(
    bytes_per_device: float | None,
    elems: float | None,
    dtype: str | float | None,
    wire_dtype: str | float | None,
    *,
    reducing: bool,
    what: str,
) -> tuple[float, str | None, str | None]:
    """Turns (bytes | elems+dtype) into wire bytes, plus the two dtype labels.

  `wire_dtype` is the dtype actually on the wire. It defaults to `dtype`, and
  may only differ for a *reducing* collective, where an implementation is free
  to accumulate in a wider type than the operand. A permutation (all-gather,
  all-to-all) cannot change dtype, so passing one there is an error rather than
  a silently ignored argument.

  Returns:
    (wire_bytes, dtype_label, wire_dtype_label).
  """
    if wire_dtype is not None and not reducing:
        raise ValueError(
            f"{what} is a permutation and cannot change dtype; drop wire_dtype"
        )
    if (bytes_per_device is None) == (elems is None):
        raise ValueError(
            f"{what}: pass exactly one of bytes_per_device=... or elems=... "
            "(with dtype=...)")
    if elems is not None:
        if dtype is None:
            raise ValueError(f"{what}: elems=... requires dtype=...")
        if elems < 0:
            raise ValueError(f"{what}: elems must be >= 0, got {elems}")
        wire = wire_dtype if wire_dtype is not None else dtype
        return (
            float(elems) * dtype_bytes(wire),
            str(dtype),
            str(wire),
        )
    if dtype is not None:
        raise ValueError(
            f"{what}: dtype=... describes elems=...; with bytes_per_device=... the"
            " conversion has already happened")
    return float(bytes_per_device), None, None  # type: ignore[arg-type]


# =============================================================================
# 1. Topology
# =============================================================================


@dataclass(frozen=True)
class Axis:
    """One mesh axis, from the RS *participating group's* point of view.

  Attributes:
    size: Number of devices along this axis.
    wraparound: Whether the axis has wraparound links (torus).
    link_bw: One-way peak BW of a single link on this axis [B/s]. None -> use
      the ici default. (For axes whose physical link type differs, e.g., the D2D
      axis.)
    contention: Number of logical rings sharing one physical link on this axis.
      Effective BW = link_bw / contention. ICI axes in the per-die view have
      2.0.
    hop_latency: Per-hop latency override for this axis [s]. None -> use the ici
      default.
    name: Display name (e.g., "d2d").
  """

    size: int
    wraparound: bool = True
    link_bw: float | None = None
    contention: float = 1.0
    hop_latency: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"axis size must be >= 1, got {self.size}")
        if self.contention < 1.0:
            raise ValueError(f"contention must be >= 1, got {self.contention}")


@dataclass(frozen=True)
class Interconnect:
    """Interconnect parameters for network modeling.

  Attributes:
    link_bw_oneway: Single link, one-way peak BW [bytes/s].
    hop_latency: Per-hop latency [s]. Generic default 1us; `tpu()` overrides it
      per generation where this repo has a pin (v7x: 1.3us).

      **This is hop latency only, and it is not the whole fixed cost.** Measured
      on j-484cd4c0 (GLM-5.2-FP8, 2x2x2, TP16) by solving the alpha-beta system
      across two same-group all-gathers of different size, the effective fixed
      cost per collective is **13.5us** -- identical (13.49 vs 13.52) whether
      fitted on mean or min times, so it is not peer skew. Hop latency accounts
      for 5.2us of that on a 4-axis slice; the remaining ~8.3us is launch and
      barrier cost. Set `launch_overhead=8.3e-6` to reproduce that cell. Left
      **Deliberately left at 1.0us** (2026-09-19) even though this repo pins v7x
      at 1.3us (tpu/V7X.md s2, tpu/V7X_2x2x2.md "Inter-Chip Hop Latency"):
      changing it rebases six v7x tests and every v7x number produced so far,
      for a gain that does not close the 2.6x gap to the measurement either way.
      To reproduce the measured cell, pass
      `hop_latency=1.3e-6, launch_overhead=8.3e-6`. Revisit deliberately: the
      three numbers (1.0us default, 1.3us pinned, 13.5us measured) are still
      unreconciled, and the answer belongs in one cited source that both this file
      and tpu/V7X.md refer to.
    link_efficiency: Achieved fraction of peak bandwidth (0.0 to 1.0).
    step_overhead: Software overhead per ring step [s].
    launch_overhead: Fixed cost per collective operation [s].
  """

    link_bw_oneway: float
    # Kept at 1us by decision (2026-09-19), NOT because it is the best estimate.
    # This repo pins v7x at 1.3us (tpu/V7X.md s2) and the measured fixed cost per
    # collective is 13.5us. Changing it rebases six v7x tests and every v7x number
    # produced so far. Full reasoning in this class's docstring.
    hop_latency: float = 1.0e-6
    link_efficiency: float = 1.0
    step_overhead: float = 0.0
    # TODO(revisit): ~8.3e-6 on v7x 2x2x2, fitted on j-484cd4c0 -- the launch and
    # barrier cost left over after hop latency. Defaulting it would close most of
    # the 2.25x underprice on small (decode-sized) collectives, but it is one
    # measurement on one cell. Left at 0; pass launch_overhead=8.3e-6 to use it.
    launch_overhead: float = 0.0


@dataclass(frozen=True)
class Machine:
    """Machine topology specification."""

    axes: tuple[Axis, ...]
    ici: Interconnect

    @property
    def n_devices(self) -> int:
        n = 1
        for a in self.axes:
            n *= a.size
        return n


@dataclass(frozen=True)
class _TpuGenInfo:
    ndim: int
    link_bw_oneway: float
    full_axis: int | None


# Public approximations (per jax-ml "How to Scale Your Model"; one-way B/s per
# link). Override via arguments if you have internal / measured numbers.
_TPU_GENS: dict[str, _TpuGenInfo] = {
    "v3": _TpuGenInfo(ndim=2, link_bw_oneway=1.0e11, full_axis=32),
    "v4": _TpuGenInfo(ndim=3, link_bw_oneway=4.5e10, full_axis=None),
    "v5p": _TpuGenInfo(ndim=3, link_bw_oneway=9.0e10, full_axis=None),
    "v5e": _TpuGenInfo(ndim=2, link_bw_oneway=4.5e10, full_axis=16),
    "v6e": _TpuGenInfo(ndim=2, link_bw_oneway=9.0e10, full_axis=16),
    "v7x": _TpuGenInfo(ndim=3, link_bw_oneway=9.0e10, full_axis=None),
}

# v7x is dual-die (2 dies per package); in the per-die view the dies are joined
# by a D2D link. There is no public number for D2D one-way BW, so we only ship
# an arbitrary placeholder (4x ICI). Always override with internal measurements
# via d2d_bw=... .
_D2D_BW_PLACEHOLDER: dict[str, float] = {
    "v7x": 4 * 9.0e10,
}


def tpu(
    gen: str,
    slice_dims: Sequence[int],
    wraparound: Sequence[bool] | None = None,
    d2d: bool = False,
    d2d_bw: float | None = None,
    d2d_hop_latency: float | None = None,
    ici_contention: float | None = None,
    **ici_overrides: Any,
) -> Machine:
    """Constructs a Machine for a given TPU generation and slice shape.

  E.g. tpu("v5p", (4, 4, 8)), tpu("v6e", (16, 16)),
  tpu("v7x", (4, 4, 4), d2d=True).

  Wraparound heuristic (when not specified):
    * v4/v5p/v7x : the OCS provides wraparound only for full 4x4x4 cubes
                   -> all axes are tori only when every axis is a multiple
                      of 4 (otherwise all axes are lines)
    * v3/v5e/v6e : fixed boards -> an axis is a torus only at full width
                   (16; 32 for v3)

  d2d=True (v7x per-die view):
    * slice_dims stays the *package* grid; a size-2 D2D axis is prepended.
      axis 0 = d2d, ICI axes are 1..ndim. n_devices = 2 * prod(slice_dims).
    * Applies contention (default 2) to every ICI axis -- the rings of the
      two die layers share the same physical ICI links (override via
      ici_contention).
    * If d2d_bw is not given, the placeholder (_D2D_BW_PLACEHOLDER) is used.
      Internal numbers strongly recommended.

  Args:
    gen: TPU generation ('v3', 'v4', 'v5p', 'v5e', 'v6e', 'v7x').
    slice_dims: Dimensions of the slice.
    wraparound: Per-axis boolean for wraparound. If None, inferred based on
      generation and slice dimensions.
    d2d: Whether to use the dual-die view (v7x only), prepending a size-2 D2D
      axis and applying contention to ICI axes.
    d2d_bw: One-way bandwidth of D2D links [bytes/s].
    d2d_hop_latency: Per-hop latency override for D2D links [s].
    ici_contention: Contention factor for ICI axes when d2d is True (default
      2.0).
    **ici_overrides: Additional keyword arguments forwarded to Interconnect.

  Returns:
    A Machine instance configured for the specified topology.
  """
    gen = gen.lower()
    if gen not in _TPU_GENS:
        raise ValueError(f"unknown gen {gen!r}; one of {sorted(_TPU_GENS)}")
    info = _TPU_GENS[gen]
    dims = tuple(int(d) for d in slice_dims)
    if len(dims) != info.ndim:
        raise ValueError(f"{gen} expects {info.ndim}D slice dims, got {dims}")

    if wraparound is None:
        if info.full_axis is None:  # OCS generations (v4/v5p/v7x)
            full_cubes = all(d >= 4 and d % 4 == 0 for d in dims)
            wraparound_tuple = tuple(full_cubes for _ in dims)
        else:  # fixed 2D generations
            wraparound_tuple = tuple(d == info.full_axis for d in dims)
    else:
        wraparound_tuple = tuple(bool(w) for w in wraparound)

    axes = tuple(Axis(d, w) for d, w in zip(dims, wraparound_tuple))

    if d2d:
        if d2d_bw is None:
            d2d_bw = _D2D_BW_PLACEHOLDER.get(gen)
            if d2d_bw is None:
                raise ValueError(
                    f"no D2D placeholder for {gen} -- specify d2d_bw explicitly"
                )
        c = 2.0 if ici_contention is None else float(ici_contention)
        axes = tuple(replace(a, contention=a.contention * c) for a in axes)
        axes = (Axis(
            2,
            wraparound=False,
            link_bw=float(d2d_bw),
            hop_latency=d2d_hop_latency,
            name="d2d",
        ), ) + axes

    ici = Interconnect(link_bw_oneway=info.link_bw_oneway, **ici_overrides)
    return Machine(axes=axes, ici=ici)


# =============================================================================
# 2. Cost model (alpha-beta, per-axis ring/line pass)
# =============================================================================


@dataclass(frozen=True)
class Phase:
    """Cost breakdown for a single axis communication phase.

  Attributes:
    axis_index: Index of the axis.
    axis: Axis configuration.
    bytes_in: Per-device data entering this phase [bytes].
    steps: Number of ring steps in this phase.
    t_latency: Latency component of time [s].
    t_bandwidth: Bandwidth component of time [s].
    bw_eff: Effective BW used by this phase [B/s] = lanes * B / contention *
      eta.
  """

    axis_index: int
    axis: Axis
    bytes_in: float
    steps: int
    t_latency: float
    t_bandwidth: float
    bw_eff: float = 0.0

    @property
    def t(self) -> float:
        return self.t_latency + self.t_bandwidth


@dataclass(frozen=True)
class Estimate:
    """Latency and bandwidth estimate for a collective operation.

  Attributes:
    algo: Algorithm name ('sequential', 'concurrent(k=...)', etc.).
    order: Axis order traversed.
    n_devices: Number of participating devices.
    bytes_in: Per-device input data volume [bytes].
    t_latency: Total latency component [s].
    t_bandwidth: Total bandwidth component [s].
    t_fixed: Fixed launch overhead [s].
    phases: Per-phase breakdown.
    dtype: Operand dtype, when the caller supplied elems+dtype. None when the
      caller passed raw bytes.
    wire_dtype: Dtype actually on the wire. Differs from `dtype` only on a
      reducing collective that accumulates in a wider type.
  """

    algo: str
    order: tuple[int, ...]
    n_devices: int
    bytes_in: float
    t_latency: float
    t_bandwidth: float
    t_fixed: float
    phases: tuple[Phase, ...]
    dtype: str | None = None
    wire_dtype: str | None = None

    @property
    def total(self) -> float:
        return self.t_latency + self.t_bandwidth + self.t_fixed

    @property
    def algbw(self) -> float:
        """nccl-tests convention: algbw = input size / time."""
        return self.bytes_in / self.total if self.total > 0 else float("inf")

    @property
    def busbw(self) -> float:
        """busbw = algbw * (N-1)/N.

    At the egress lower bound, busbw == total per-chip egress BW, so it can
    be compared directly against link specs.
    """
        n = self.n_devices
        return self.algbw * (n - 1) / n if n > 1 else 0.0

    @property
    def bound(self) -> str:
        return ("latency" if
                (self.t_latency +
                 self.t_fixed) >= self.t_bandwidth else "bandwidth")


@dataclass(frozen=True)
class LowerBound:
    """Theoretical lower bound on collective latency.

  Attributes:
    t_bandwidth: Per-chip egress bound [s].
    t_latency: Network diameter * alpha [s].
    n_devices: Number of participating devices.
    bytes_in: Per-device input data volume [bytes].
  """

    t_bandwidth: float
    t_latency: float
    n_devices: int
    bytes_in: float

    @property
    def t(self) -> float:
        return max(self.t_bandwidth, self.t_latency)


def _lanes(axis: Axis, unidirectional: bool = False) -> int:
    """Number of one-way links usable simultaneously along the axis.

  torus + bidirectional ring : 2 (links in both directions carry data)
  line (mesh) axis           : 1 (interior chips have 2 links, but the
                                  corner chip's single-link egress is the
                                  bottleneck)
  unidirectional ring        : 1
  """
    if axis.wraparound and not unidirectional:
        return 2
    return 1


def _link_bw_eff(axis: Axis, ici: Interconnect) -> float:
    """Effective BW of one link (one-way, single lane) = (link_bw) / contention * eta."""
    base = axis.link_bw if axis.link_bw is not None else ici.link_bw_oneway
    return base * ici.link_efficiency / axis.contention


def _hop_alpha(axis: Axis, ici: Interconnect) -> float:
    return axis.hop_latency if axis.hop_latency is not None else ici.hop_latency


def _ring_phase(
    bytes_in: float,
    axis_index: int,
    axis: Axis,
    ici: Interconnect,
    unidirectional: bool,
) -> Phase:
    x = axis.size
    if x == 1:
        return Phase(axis_index, axis, bytes_in, 0, 0.0, 0.0, 0.0)
    bw = _link_bw_eff(axis, ici) * _lanes(axis, unidirectional)
    steps = x - 1
    t_lat = steps * (_hop_alpha(axis, ici) + ici.step_overhead)
    t_bw = bytes_in * (x - 1) / (x * bw)
    return Phase(axis_index, axis, bytes_in, steps, t_lat, t_bw, bw)


def _run_order(
    bytes_in: float,
    machine: Machine,
    order: Sequence[int],
    unidirectional: bool,
) -> tuple[tuple[Phase, ...], float, float]:
    s = float(bytes_in)
    phases = []
    for i in order:
        ax = machine.axes[i]
        ph = _ring_phase(s, i, ax, machine.ici, unidirectional)
        phases.append(ph)
        s /= ax.size
    return (
        tuple(phases),
        sum(p.t_latency for p in phases),
        sum(p.t_bandwidth for p in phases),
    )


def _a2a_phase(
    bytes_in: float,
    axis_index: int,
    axis: Axis,
    ici: Interconnect,
    unidirectional: bool,
) -> Phase:
    """One axis of an all-to-all. Link-bound, not egress-bound -- see module doc.

  A chunk bound for a device d hops away traverses d links, so summing hop
  distances (X^2/4 on an even torus) and dividing by the directed link count
  gives S*X/(4*lanes*B) rather than RS's S*(X-1)/(X*lanes*B).
  """
    x = axis.size
    if x == 1:
        return Phase(axis_index, axis, bytes_in, 0, 0.0, 0.0, 0.0)
    bw = _link_bw_eff(axis, ici) * _lanes(axis, unidirectional)
    steps = x - 1
    t_lat = steps * (_hop_alpha(axis, ici) + ici.step_overhead)
    t_bw = bytes_in * x / (4.0 * bw)
    return Phase(axis_index, axis, bytes_in, steps, t_lat, t_bw, bw)


def _run_order_a2a(
    bytes_in: float,
    machine: Machine,
    order: Sequence[int],
    unidirectional: bool,
) -> tuple[tuple[Phase, ...], float, float]:
    """Like `_run_order`, but the payload is *preserved* across phases.

  An all-to-all is a permutation: every phase moves the whole payload, so the
  `s /= ax.size` of `_run_order` must not appear here. A consequence is that
  the total is independent of axis order.
  """
    phases = []
    for i in order:
        phases.append(
            _a2a_phase(bytes_in, i, machine.axes[i], machine.ici,
                       unidirectional))
    return (
        tuple(phases),
        sum(p.t_latency for p in phases),
        sum(p.t_bandwidth for p in phases),
    )


def _active_axes(machine: Machine,
                 axes: Sequence[int] | None) -> tuple[int, ...]:
    idxs = tuple(range(len(machine.axes))) if axes is None else tuple(axes)
    return tuple(i for i in idxs if machine.axes[i].size > 1)


def _group_size(machine: Machine, act: Sequence[int]) -> int:
    n = 1
    for i in act:
        n *= machine.axes[i].size
    return n


def reduce_scatter_sequential(
    machine: Machine,
    bytes_per_device: float,
    axes: Sequence[int] | None = None,
    order: Sequence[int] | None = None,
    unidirectional: bool = False,
) -> Estimate:
    """Dimension-ordered RS passing one axis at a time.

  With order=None, all permutations are tried and the minimum is taken. When
  per-axis effective BW differs (mixed torus/line etc.), visiting faster axes
  first wins -- the first phase carries the most data.

  Args:
    machine: Machine topology.
    bytes_per_device: Per-device input data volume in bytes.
    axes: Participating axis indices. None means all axes.
    order: Specific axis order to use. None evaluates all permutations.
    unidirectional: Whether to restrict rings to unidirectional links.

  Returns:
    An Estimate instance.
  """
    act = _active_axes(machine, axes)
    n = _group_size(machine, act)
    fixed = machine.ici.launch_overhead
    if not act:
        return Estimate("sequential", (), 1, bytes_per_device, 0.0, 0.0, fixed,
                        ())

    orders = ([tuple(order)] if order is not None else sorted(
        set(permutations(act))))
    best: Estimate | None = None
    for o in orders:
        phases, t_lat, t_bw = _run_order(bytes_per_device, machine, o,
                                         unidirectional)
        est = Estimate("sequential", o, n, bytes_per_device, t_lat, t_bw,
                       fixed, phases)
        if best is None or est.total < best.total:
            best = est
    assert best is not None
    return best


def reduce_scatter_concurrent(
    machine: Machine,
    bytes_per_device: float,
    axes: Sequence[int] | None = None,
    unidirectional: bool = False,
) -> Estimate:
    """Split the data into k (= number of axes) shards and rotate axis order.

  Shard sizes are allocated inversely proportional to each rotation's
  per-byte bandwidth cost so that all shards finish together (identical to
  an equal split when the axes are symmetric). On a symmetric torus,
  T_bw ~= D * (1 - 1/N) / (sum_a lanes_a * B * eta), i.e., it reaches the
  egress lower bound (~3x sequential in 3D). With strongly heterogeneous
  axis speeds (e.g., D2D + ICI), k rotations cannot keep the fast axis busy
  the whole time and a gap to the bound remains -- that gap is close to a
  real scheduling limit.

  Args:
    machine: Machine topology.
    bytes_per_device: Per-device input data volume in bytes.
    axes: Participating axis indices. None means all axes.
    unidirectional: Whether to restrict rings to unidirectional links.

  Returns:
    An Estimate instance.
  """
    act = _active_axes(machine, axes)
    k = len(act)
    if k <= 1:
        est = reduce_scatter_sequential(machine, bytes_per_device, axes, None,
                                        unidirectional)
        return replace(est, algo="concurrent(k=1)")
    n = _group_size(machine, act)
    t_lat = 0.0
    inv_costs = []  # per rotation: 1 / (per-byte bw cost)
    orders = []
    for j in range(k):
        o = act[j:] + act[:j]
        _, lat_j, bw_j = _run_order(1.0, machine, o, unidirectional)
        orders.append(o)
        inv_costs.append(1.0 / bw_j if bw_j > 0 else 0.0)
        t_lat = max(t_lat, lat_j)  # every rotation has the same total steps
    tot_inv = sum(inv_costs)
    t_bw = bytes_per_device / tot_inv if tot_inv > 0 else 0.0
    shard0 = (bytes_per_device * inv_costs[0] /
              tot_inv) if tot_inv > 0 else 0.0
    rep_phases, _, _ = _run_order(shard0, machine, orders[0], unidirectional)
    return Estimate(
        f"concurrent(k={k})",
        act,
        n,
        bytes_per_device,
        t_lat,
        t_bw,
        machine.ici.launch_overhead,
        rep_phases,
    )


def reduce_scatter_lower_bound(
    machine: Machine,
    bytes_per_device: float,
    axes: Sequence[int] | None = None,
) -> LowerBound:
    """A floor no algorithm can beat.

  * egress bound : each chip must send out at least (N-1)/N * D of its bytes
    (even accounting for in-flight reduction), divided by the chip's total
    egress BW (sum_a lanes_a * B * eta). Line axes count lanes=1 (the corner
    chip's single link).
  * latency bound: the farthest chip's contribution has to arrive ->
    diameter * alpha, with diameter = sum_a (X/2 if torus else X-1).

  Args:
    machine: Machine topology.
    bytes_per_device: Per-device input data volume in bytes.
    axes: Participating axis indices. None means all axes.

  Returns:
    A LowerBound instance.
  """
    act = _active_axes(machine, axes)
    n = _group_size(machine, act)
    ici = machine.ici
    if n <= 1:
        return LowerBound(0.0, 0.0, 1, bytes_per_device)
    egress = sum(
        _lanes(machine.axes[i]) * _link_bw_eff(machine.axes[i], ici)
        for i in act)
    t_bw = bytes_per_device * (n - 1) / n / egress
    t_lat = sum(
        ((machine.axes[i].size //
          2) if machine.axes[i].wraparound else (machine.axes[i].size - 1)) *
        _hop_alpha(machine.axes[i], ici) for i in act)
    return LowerBound(t_bw, t_lat, n, bytes_per_device)


def reduce_scatter(
    machine: Machine,
    bytes_per_device: float | None = None,
    axes: Sequence[int] | None = None,
    algo: str = "auto",
    unidirectional: bool = False,
    *,
    elems: float | None = None,
    dtype: str | float | None = None,
    wire_dtype: str | float | None = None,
) -> Estimate:
    """Estimates reduce-scatter latency.

  Args:
    machine: Machine topology.
    bytes_per_device: Per-device input data volume in bytes. Legacy form; pass
      elems + dtype instead where the dtype is known.
    axes: Participating axis indices. None means all axes.
    algo: 'auto' (min of sequential and concurrent) | 'sequential' |
      'concurrent'.
    unidirectional: Whether to restrict rings to unidirectional links.
    elems: Per-device input element count. Requires dtype.
    dtype: Operand dtype (see DTYPE_BYTES), or bytes/element as a number.
    wire_dtype: Dtype actually reduced on the wire, when the implementation
      accumulates in a wider type than the operand. Defaults to dtype.

  Returns:
    An Estimate instance.
  """
    d, dt, wdt = _resolve_payload(
        bytes_per_device,
        elems,
        dtype,
        wire_dtype,
        reducing=True,
        what="reduce_scatter",
    )
    if algo == "sequential":
        est = reduce_scatter_sequential(machine, d, axes, None, unidirectional)
    elif algo == "concurrent":
        est = reduce_scatter_concurrent(machine, d, axes, unidirectional)
    elif algo == "auto":
        est = min(
            (
                reduce_scatter_sequential(machine, d, axes, None,
                                          unidirectional),
                reduce_scatter_concurrent(machine, d, axes, unidirectional),
            ),
            key=lambda e: e.total,
        )
    else:
        raise ValueError(f"unknown algo {algo!r}")
    return replace(est, dtype=dt, wire_dtype=wdt)


def all_gather(
    machine: Machine,
    bytes_full: float | None = None,
    axes: Sequence[int] | None = None,
    algo: str = "auto",
    unidirectional: bool = False,
    *,
    elems_full: float | None = None,
    dtype: str | float | None = None,
    wire_dtype: str | float | None = None,
) -> Estimate:
    """AG moves the same traffic as RS in the opposite direction -> same cost.

  Args:
    machine: Machine topology.
    bytes_full: AG *output* size (= the full gathered size in bytes).
    axes: Participating axis indices. None means all axes.
    algo: As for reduce_scatter.
    unidirectional: Whether to restrict rings to unidirectional links.
    elems_full: Element count of the *gathered* result, i.e. already multiplied
      by the group size. Requires dtype.
    dtype: Operand dtype. An all-gather is a permutation, so this is also the
      wire dtype.
    wire_dtype: Accepted only to reject it with an explanation -- an all-gather
      cannot change dtype.

  Returns:
    An Estimate instance.
  """
    d, dt, _ = _resolve_payload(
        bytes_full,
        elems_full,
        dtype,
        wire_dtype,
        reducing=False,
        what="all_gather",
    )
    est = reduce_scatter(machine, d, axes, algo, unidirectional)
    return replace(est,
                   algo=f"all_gather({est.algo})",
                   dtype=dt,
                   wire_dtype=dt)


def all_reduce(
    machine: Machine,
    bytes_per_device: float | None = None,
    axes: Sequence[int] | None = None,
    algo: str = "auto",
    unidirectional: bool = False,
    *,
    elems: float | None = None,
    dtype: str | float | None = None,
    wire_dtype: str | float | None = None,
) -> Estimate:
    """AR(D) = RS(D) + AG(D) ~= 2 * RS(D).

  Both legs are charged at `wire_dtype`. An implementation that casts the
  reduced result back down before gathering pays less on the second leg; model
  that as an explicit reduce_scatter + all_gather pair rather than here.

  Args:
    machine: Machine topology.
    bytes_per_device: Per-device input data volume in bytes.
    axes: Participating axis indices. None means all axes.
    algo: As for reduce_scatter.
    unidirectional: Whether to restrict rings to unidirectional links.
    elems: Per-device input element count. Requires dtype.
    dtype: Operand dtype.
    wire_dtype: Dtype actually reduced on the wire. Defaults to dtype.

  Returns:
    An Estimate instance: exactly 2 x the reduce-scatter, launch overhead
    included, since the two legs are two collective launches.
  """
    rs = reduce_scatter(
        machine,
        bytes_per_device,
        axes,
        algo,
        unidirectional,
        elems=elems,
        dtype=dtype,
        wire_dtype=wire_dtype,
    )
    return replace(
        rs,
        algo=f"all_reduce({rs.algo})",
        t_latency=2.0 * rs.t_latency,
        t_bandwidth=2.0 * rs.t_bandwidth,
        t_fixed=2.0 * rs.t_fixed,
    )


def all_gather_time(machine: Machine, bytes_full: float, **kw: Any) -> float:
    """Scalar-returning wrapper over `all_gather`, kept for existing callers."""
    return all_gather(machine, bytes_full, **kw).total


def all_reduce_time(machine: Machine, bytes_per_device: float,
                    **kw: Any) -> float:
    """Scalar-returning wrapper over `all_reduce`, kept for existing callers."""
    return all_reduce(machine, bytes_per_device, **kw).total


# ----------------------------------------------------------------- all-to-all


def all_to_all_sequential(
    machine: Machine,
    bytes_per_device: float,
    axes: Sequence[int] | None = None,
    order: Sequence[int] | None = None,
    unidirectional: bool = False,
) -> Estimate:
    """Dimension-ordered all-to-all, one axis at a time.

  Unlike RS, the payload does not shrink between phases, so the total is
  independent of axis order and no permutation search is needed.

  Args:
    machine: Machine topology.
    bytes_per_device: The full per-device payload -- all N destination shards
      including this device's own.
    axes: Participating axis indices. None means all axes.
    order: Axis order. Cost-neutral; accepted so the phase breakdown can be
      reported in a chosen order.
    unidirectional: Whether to restrict rings to unidirectional links.

  Returns:
    An Estimate instance.
  """
    act = _active_axes(machine, axes)
    n = _group_size(machine, act)
    fixed = machine.ici.launch_overhead
    if not act:
        return Estimate("a2a-sequential", (), 1, bytes_per_device, 0, 0, fixed,
                        ())
    o = tuple(order) if order is not None else act
    phases, t_lat, t_bw = _run_order_a2a(bytes_per_device, machine, o,
                                         unidirectional)
    return Estimate("a2a-sequential", o, n, bytes_per_device, t_lat, t_bw,
                    fixed, phases)


def all_to_all_concurrent(
    machine: Machine,
    bytes_per_device: float,
    axes: Sequence[int] | None = None,
    unidirectional: bool = False,
) -> Estimate:
    """Shard the payload and rotate axis schedules so all axes are busy at once.

  The traffic a given axis must carry is fixed by the permutation and cannot be
  reduced -- only overlapped with the other axes. So this is `max` over the
  per-axis terms where `sequential` is `sum`, and it coincides with the
  bisection lower bound. **It is a floor, not a prediction**: see `all_to_all`.

  Args:
    machine: Machine topology.
    bytes_per_device: The full per-device payload.
    axes: Participating axis indices. None means all axes.
    unidirectional: Whether to restrict rings to unidirectional links.

  Returns:
    An Estimate instance.
  """
    act = _active_axes(machine, axes)
    if len(act) <= 1:
        est = all_to_all_sequential(machine, bytes_per_device, axes, None,
                                    unidirectional)
        return replace(est, algo="a2a-concurrent(k=1)")
    n = _group_size(machine, act)
    phases, t_lat, _ = _run_order_a2a(bytes_per_device, machine, act,
                                      unidirectional)
    t_bw = max(p.t_bandwidth for p in phases)
    return Estimate(
        f"a2a-concurrent(k={len(act)})",
        act,
        n,
        bytes_per_device,
        t_lat,
        t_bw,
        machine.ici.launch_overhead,
        phases,
    )


def all_to_all_lower_bound(
    machine: Machine,
    bytes_per_device: float,
    axes: Sequence[int] | None = None,
) -> LowerBound:
    """A floor no all-to-all algorithm can beat.

  Two bounds, whichever is tighter:
    * egress    : as for RS -- each chip must push out (N-1)/N of its payload.
    * bisection : cutting axis a in half forces N*D/4 across that cut's links,
      giving D*X_a/(4*lanes_a*B_a). Dominates egress for X_a > 2, which is why
      an RS proxy understates all-to-all so badly.

  Args:
    machine: Machine topology.
    bytes_per_device: The full per-device payload.
    axes: Participating axis indices. None means all axes.

  Returns:
    A LowerBound instance.
  """
    act = _active_axes(machine, axes)
    n = _group_size(machine, act)
    ici = machine.ici
    if n <= 1:
        return LowerBound(0.0, 0.0, 1, bytes_per_device)
    egress_bw = sum(
        _lanes(machine.axes[i]) * _link_bw_eff(machine.axes[i], ici)
        for i in act)
    t_egress = bytes_per_device * (n - 1) / n / egress_bw
    t_bisect = max(
        bytes_per_device * machine.axes[i].size /
        (4.0 * _lanes(machine.axes[i]) * _link_bw_eff(machine.axes[i], ici))
        for i in act)
    t_lat = sum(
        ((machine.axes[i].size //
          2) if machine.axes[i].wraparound else (machine.axes[i].size - 1)) *
        _hop_alpha(machine.axes[i], ici) for i in act)
    return LowerBound(max(t_egress, t_bisect), t_lat, n, bytes_per_device)


def all_to_all(
    machine: Machine,
    bytes_per_device: float | None = None,
    axes: Sequence[int] | None = None,
    algo: str = "sequential",
    unidirectional: bool = False,
    order: Sequence[int] | None = None,
    *,
    elems: float | None = None,
    dtype: str | float | None = None,
    wire_dtype: str | float | None = None,
) -> Estimate:
    """Estimates all-to-all latency.

  **`algo` defaults to 'sequential', and 'auto' is refused.** For RS, taking
  the cheaper of the two algorithms is a reasonable default. For all-to-all it
  is not: 'concurrent' equals the bisection floor, and reporting a floor for a
  dimension-ordered implementation understates it by the ratio between them --
  measured at 3x on one 16-rank EP group. Name the schedule you mean.

  Which schedule an XLA-compiled all-to-all actually issues on TPU is UNREAD as
  of 2026-09-19; until someone reads it, 'sequential' is the conservative pick.

  Args:
    machine: Machine topology.
    bytes_per_device: The full per-device payload -- all N destination shards
      including this device's own. For an MoE dispatch that is
      tokens * topk * hidden * itemsize.
    axes: Participating axis indices. None means all axes.
    algo: 'sequential' (dimension-ordered) | 'concurrent' (the floor).
    unidirectional: Whether to restrict rings to unidirectional links.
    order: Axis order for 'sequential'. Cost-neutral -- accepted so a phase
      breakdown can be reported in a chosen order.
    elems: Per-device payload in elements. Requires dtype.
    dtype: Operand dtype. An all-to-all is a permutation, so this is also the
      wire dtype. The two legs of an MoE exchange often differ (fp8 out, bf16
      back) -- call this twice rather than averaging them.
    wire_dtype: Accepted only to reject it with an explanation.

  Returns:
    An Estimate instance.
  """
    d, dt, _ = _resolve_payload(
        bytes_per_device,
        elems,
        dtype,
        wire_dtype,
        reducing=False,
        what="all_to_all",
    )
    if algo == "auto":
        raise ValueError(
            "all_to_all has no 'auto': 'concurrent' is the bisection floor, not a "
            "prediction. Pass algo='sequential' or algo='concurrent' explicitly."
        )
    if algo == "sequential":
        est = all_to_all_sequential(machine, d, axes, order, unidirectional)
    elif algo == "concurrent":
        if order is not None:
            raise ValueError("order applies to algo='sequential' only")
        est = all_to_all_concurrent(machine, d, axes, unidirectional)
    else:
        raise ValueError(f"unknown algo {algo!r}")
    return replace(est, dtype=dt, wire_dtype=dt)


def all_to_all_time(machine: Machine, bytes_per_device: float,
                    **kw: Any) -> float:
    """Scalar-returning wrapper over `all_to_all`."""
    return all_to_all(machine, bytes_per_device, **kw).total


# =============================================================================
# 3. Utilities / reporting
# =============================================================================

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*$")
_SIZE_MULT = {
    "": 1,
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}


def parse_size(s: str) -> float:
    """Parses a human-readable size string (e.g. '128MiB', '1GB') into bytes."""
    m = _SIZE_RE.match(str(s))
    if not m:
        raise ValueError(f"cannot parse size: {s!r}")
    val, unit = float(m.group(1)), m.group(2).upper()
    if unit in ("K", "M", "G", "T"):
        unit += "B"
    if unit in ("KI", "MI", "GI", "TI"):
        unit += "B"
    if unit not in _SIZE_MULT:
        raise ValueError(f"unknown size unit in {s!r}")
    return val * _SIZE_MULT[unit]


def fmt_bytes(x: float) -> str:
    for u, m in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if x >= m:
            v = x / m
            return f"{v:.0f}{u}" if v >= 10 or v == int(v) else f"{v:.1f}{u}"
    return f"{x:.0f}B"


def fmt_time(t: float) -> str:
    if t < 1e-3:
        return f"{t * 1e6:8.2f}us"
    if t < 1.0:
        return f"{t * 1e3:8.3f}ms"
    return f"{t:8.3f}s "


def fmt_bw(x: float) -> str:
    return f"{x / 1e9:7.1f}GB/s"


def describe(est: Estimate) -> str:
    """Produces a multi-line human-readable summary of an Estimate."""
    if est.dtype is None:
        dt = ""
    elif est.wire_dtype and est.wire_dtype != est.dtype:
        dt = f", dtype={est.dtype}->wire {est.wire_dtype}"
    else:
        dt = f", dtype={est.dtype}"
    lines = [
        f"[{est.algo}] N={est.n_devices}, D_in={fmt_bytes(est.bytes_in)}{dt}, "
        f"order={est.order}, bound={est.bound}",
        f"  total={fmt_time(est.total).strip()}  "
        f"(lat={fmt_time(est.t_latency).strip()}, "
        f"bw={fmt_time(est.t_bandwidth).strip()}, "
        f"fixed={fmt_time(est.t_fixed).strip()})  "
        f"algbw={fmt_bw(est.algbw).strip()}, busbw={fmt_bw(est.busbw).strip()}",
    ]
    for p in est.phases:
        kind = p.axis.name or ("torus" if p.axis.wraparound else "line")
        cont = f" c={p.axis.contention:g}" if p.axis.contention != 1.0 else ""
        lines.append(f"    axis{p.axis_index} X={p.axis.size:<3d} {kind:<5} "
                     f"S_in={fmt_bytes(p.bytes_in):>8} steps={p.steps:<3d} "
                     f"lat={fmt_time(p.t_latency).strip():>9} "
                     f"bw={fmt_time(p.t_bandwidth).strip():>9} "
                     f"eff={fmt_bw(p.bw_eff).strip()}{cont}")
    return "\n".join(lines)


def sweep(
    machine: Machine,
    sizes: Sequence[float],
    axes: Sequence[int] | None = None,
    title: str = "",
) -> None:
    """Prints a tabular summary of RS latency across data sizes."""
    act = _active_axes(machine, axes)

    def _tag(a: Axis) -> str:
        return "d" if a.name == "d2d" else ("t" if a.wraparound else "m")

    shape = "x".join(f"{machine.axes[i].size}{_tag(machine.axes[i])}"
                     for i in act)
    ici = machine.ici
    print(f"\n=== {title}")
    print(f"    RS group {shape} (t=torus, m=mesh/line, d=d2d), "
          f"B={ici.link_bw_oneway / 1e9:.0f}GB/s/link/dir, "
          f"alpha={ici.hop_latency * 1e6:.1f}us, eta={ici.link_efficiency}")
    if any(machine.axes[i].link_bw is not None
           or machine.axes[i].contention != 1.0 for i in act):
        parts = []
        for i in act:
            a = machine.axes[i]
            eff = _link_bw_eff(a, ici) * _lanes(a)
            cont = f"/c{a.contention:g}" if a.contention != 1.0 else ""
            parts.append(
                f"axis{i}({a.size}{_tag(a)}{cont})={eff / 1e9:.0f}GB/s")
        print("    per-axis eff bw (lanes incl.): " + ", ".join(parts))
    hdr = (f"{'size':>8} | {'sequential':>10} | {'concurrent':>10} | "
           f"{'best':>10} {'algo':<16} {'bound':<9} | {'busbw':>11} | "
           f"{'lower bnd':>10}")
    print(hdr)
    print("-" * len(hdr))
    for d in sizes:
        seq = reduce_scatter_sequential(machine, d, axes)
        conc = reduce_scatter_concurrent(machine, d, axes)
        best = seq if seq.total <= conc.total else conc
        lb = reduce_scatter_lower_bound(machine, d, axes)
        print(
            f"{fmt_bytes(d):>8} | {fmt_time(seq.total)} | {fmt_time(conc.total)} |"
            f" {fmt_time(best.total)} {best.algo:<16} {best.bound:<9} |"
            f" {fmt_bw(best.busbw)} | {fmt_time(lb.t)}")


def _demo() -> None:
    """Runs demonstration sweeps across multiple topologies."""
    kib, mib, gib = 2**10, 2**20, 2**30
    sizes = [32 * kib, 256 * kib, 2 * mib, 16 * mib, 128 * mib, 1 * gib]

    # 1) v5p 4x4x4 (full cube, all 3 axes tori) -- RS over all axes
    sweep(tpu("v5p", (4, 4, 4)),
          sizes,
          title="TPU v5p 4x4x4, RS over all 3 axes")

    # 2) v5p 8x8x8 -- scaling up
    sweep(tpu("v5p", (8, 8, 8)),
          sizes,
          title="TPU v5p 8x8x8, RS over all 3 axes")

    # 3) v6e 16x16 full pod (both axes tori)
    sweep(tpu("v6e", (16, 16)), sizes, title="TPU v6e 16x16 (full 2D torus)")

    # 4) v5e 8x16 -- one line axis (8), one torus axis (16): mixed case
    sweep(tpu("v5e", (8, 16)), sizes, title="TPU v5e 8x16 (8=line, 16=torus)")

    # 5) partial-axis RS: a TP group using only (axis1, axis2) of a v5p 4x4x8
    sweep(
        tpu("v5p", (4, 4, 8)),
        sizes,
        axes=(1, 2),
        title="TPU v5p 4x4x8, RS over axes (1,2) only (TP submesh)",
    )

    # 6) subgroup of a physical torus axis -> must be treated as a line
    #    (TP=8 on a 16-wide v6e axis)
    m = Machine((Axis(8, wraparound=False), ),
                Interconnect(link_bw_oneway=9.0e10))
    sweep(
        m,
        sizes,
        title="TP group of 8 chips on a v6e 16-axis -> 1D line of 8",
    )
    m = Machine((Axis(8, wraparound=True), ),
                Interconnect(link_bw_oneway=9.0e10))
    sweep(
        m,
        sizes,
        title="(compare) 1D ring of 8 with wraparound -- exactly 2x faster",
    )

    # 7) v7x dual-die: per-package view vs per-die view
    #    (D2D axis + ICI contention x2)
    sweep(
        tpu("v7x", (4, 4, 4)),
        sizes,
        title="TPU v7x 4x4x4, per-package view (no d2d)",
    )
    sweep(
        tpu("v7x", (4, 4, 4), d2d=True),
        sizes,
        title=("TPU v7x 4x4x4, per-die view: d2d axis(2) + ICI contention x2"
               " [d2d_bw=placeholder 360GB/s]"),
    )
    print("\n=== breakdown: v7x 4x4x4 per-die (d2d), D=256MiB ===")
    m = tpu("v7x", (4, 4, 4), d2d=True)
    print(describe(reduce_scatter_sequential(m, 256 * mib)))
    print(describe(reduce_scatter_concurrent(m, 256 * mib)))
    lb = reduce_scatter_lower_bound(m, 256 * mib)
    print(f"  lower bound: bw={fmt_time(lb.t_bandwidth).strip()}, "
          f"lat={fmt_time(lb.t_latency).strip()}")

    # 8) per-phase breakdown example
    print("\n=== breakdown example: v5p 4x4x8 all axes, D=256MiB ===")
    m = tpu("v5p", (4, 4, 8))
    print(describe(reduce_scatter_sequential(m, 256 * mib)))
    print(describe(reduce_scatter_concurrent(m, 256 * mib)))
    lb = reduce_scatter_lower_bound(m, 256 * mib)
    print(f"  lower bound: bw={fmt_time(lb.t_bandwidth).strip()}, "
          f"lat={fmt_time(lb.t_latency).strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(
        "TPU mesh/torus reduce-scatter latency model (runs a demo sweep when"
        " called without arguments)"))
    ap.add_argument("--gen", default=None, help="v3|v4|v5p|v5e|v6e|v7x")
    ap.add_argument("--slice", dest="slice_", default=None, help="e.g., 4x4x8")
    ap.add_argument(
        "--bytes",
        default="128MiB",
        help="per-device RS input size (e.g., 1GiB, 64MB, 262144)",
    )
    ap.add_argument(
        "--axes",
        default=None,
        help="participating axis indices (e.g., 0,1). Default: all axes",
    )
    ap.add_argument(
        "--algo",
        default="auto",
        choices=["auto", "sequential", "concurrent"],
    )
    ap.add_argument(
        "--wrap",
        default=None,
        help="per-axis wraparound override (e.g., 1,0,1)",
    )
    ap.add_argument(
        "--bw",
        type=float,
        default=None,
        help="link one-way BW override [B/s]",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="hop latency override [s]",
    )
    ap.add_argument("--eta", type=float, default=1.0, help="link efficiency")
    ap.add_argument("--uni", action="store_true", help="unidirectional ring")
    ap.add_argument(
        "--d2d",
        action="store_true",
        help=("v7x per-die view: prepend a size-2 d2d axis as axis 0 and apply"
              " contention to the ICI axes (ICI axes start at index 1)"),
    )
    ap.add_argument(
        "--d2d-bw",
        type=float,
        default=None,
        help="D2D link one-way BW [B/s] (placeholder if omitted)",
    )
    ap.add_argument(
        "--ici-contention",
        type=float,
        default=None,
        help="ICI axis contention with --d2d (default 2)",
    )
    args = ap.parse_args()

    if args.gen is None or args.slice_ is None:
        _demo()
        return

    dims = tuple(int(x) for x in args.slice_.lower().split("x"))
    wrap = None
    if args.wrap is not None:
        wrap = tuple(bool(int(x)) for x in args.wrap.split(","))
    kw = {"link_efficiency": args.eta}
    if args.alpha is not None:
        kw["hop_latency"] = args.alpha
    m = tpu(
        args.gen,
        dims,
        wraparound=wrap,
        d2d=args.d2d,
        d2d_bw=args.d2d_bw,
        ici_contention=args.ici_contention,
        **kw,
    )
    if args.bw is not None:
        # ICI base BW override (the d2d axis has an explicit link_bw and is
        # unaffected)
        m = replace(m, ici=replace(m.ici, link_bw_oneway=args.bw))
    axes = (None if args.axes is None else tuple(
        int(x) for x in args.axes.split(",")))

    d = parse_size(args.bytes)
    est = reduce_scatter(m,
                         d,
                         axes=axes,
                         algo=args.algo,
                         unidirectional=args.uni)
    lb = reduce_scatter_lower_bound(m, d, axes=axes)
    print(describe(est))
    print(f"lower bound: bw={fmt_time(lb.t_bandwidth).strip()}, "
          f"lat={fmt_time(lb.t_latency).strip()}")


if __name__ == "__main__":
    main()
