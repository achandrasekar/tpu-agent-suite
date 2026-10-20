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
"""Tests for collective operations cost estimator, focusing on TPU v7x.

    python3 cost_estimator/collective_test.py      # from scripts/roofline-analysis/

Ported off an internal test harness (`googletest` -> stdlib `unittest`, which it was already
written against) so it runs on a bare `python3` with no install.
`collective.py` imports only the standard library, so there is nothing else to set up.
"""

import os
import sys
import unittest

# Import `collective` as a sibling regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_estimator import \
    collective  # noqa: E402  (must follow the sys.path insert)

# A `CUSTOM_HW = collective.HardwareSpec(...)` constant sat here upstream, carrying the
# legacy dummy benchmark numbers (ICI 200 GB/s bidir, D2D 3200 GB/s bidir, d2d_size=2).
# `collective.py` has no `HardwareSpec` -- the two files came over at different versions --
# and nothing referenced the constant except the comment in
# `test_hierarchical_all_gather_legacy_numbers`, which builds its `Machine` inline. Dropped
# rather than reconstructed: reviving it would mean inventing an API the estimator does not have.


class CollectiveTest(unittest.TestCase):

    def test_axis_initialization_and_validation(self):
        axis = collective.Axis(size=4, wraparound=True)
        self.assertEqual(axis.size, 4)
        self.assertTrue(axis.wraparound)
        self.assertIsNone(axis.link_bw)
        self.assertEqual(axis.contention, 1.0)
        self.assertIsNone(axis.hop_latency)
        self.assertEqual(axis.name, "")

        with self.assertRaises(ValueError):
            collective.Axis(size=0)

        with self.assertRaises(ValueError):
            collective.Axis(size=4, contention=0.5)

    def test_interconnect_defaults(self):
        ici = collective.Interconnect(link_bw_oneway=9.0e10)
        self.assertEqual(ici.link_bw_oneway, 9.0e10)
        self.assertEqual(ici.hop_latency, 1.0e-6)
        self.assertEqual(ici.link_efficiency, 1.0)
        self.assertEqual(ici.step_overhead, 0.0)
        self.assertEqual(ici.launch_overhead, 0.0)

    def test_machine_device_count(self):
        axes_3d = (collective.Axis(4), collective.Axis(4), collective.Axis(4))
        m_3d = collective.Machine(
            axes=axes_3d, ici=collective.Interconnect(link_bw_oneway=9.0e10))
        self.assertEqual(m_3d.n_devices, 64)

        axes_4d = (
            collective.Axis(2),
            collective.Axis(4),
            collective.Axis(4),
            collective.Axis(4),
        )
        m_4d = collective.Machine(
            axes=axes_4d, ici=collective.Interconnect(link_bw_oneway=9.0e10))
        self.assertEqual(m_4d.n_devices, 128)

    def test_tpu_v7x_package_view_torus(self):
        # v7x 4x4x4 cube (package view): all axes are multiples of 4 -> all tori.
        m = collective.tpu("v7x", (4, 4, 4))
        self.assertEqual(len(m.axes), 3)
        self.assertEqual(m.n_devices, 64)
        self.assertTrue(all(a.wraparound for a in m.axes))
        self.assertTrue(all(a.contention == 1.0 for a in m.axes))
        self.assertEqual(m.ici.link_bw_oneway, 9.0e10)

    def test_tpu_v7x_package_view_mesh(self):
        # v7x 2x2x4 (package view): not a full 4x4x4 cube -> mesh (no wraparound).
        m = collective.tpu("v7x", (2, 2, 4))
        self.assertEqual(len(m.axes), 3)
        self.assertEqual(m.n_devices, 16)
        self.assertFalse(any(a.wraparound for a in m.axes))

    def test_tpu_v7x_per_die_view_d2d(self):
        # v7x 4x4x4 in per-die view (d2d=True):
        # Prepends a size-2 D2D axis and applies 2x contention to ICI axes.
        m = collective.tpu("v7x", (4, 4, 4), d2d=True)
        self.assertEqual(len(m.axes), 4)
        self.assertEqual(m.n_devices, 128)

        # Axis 0 is D2D
        d2d_axis = m.axes[0]
        self.assertEqual(d2d_axis.name, "d2d")
        self.assertEqual(d2d_axis.size, 2)
        self.assertFalse(d2d_axis.wraparound)
        self.assertEqual(d2d_axis.link_bw, 3.6e11)  # 4x ICI placeholder
        self.assertEqual(d2d_axis.contention, 1.0)

        # Axes 1..3 are ICI with contention=2.0
        for ici_axis in m.axes[1:]:
            self.assertEqual(ici_axis.size, 4)
            self.assertTrue(ici_axis.wraparound)
            self.assertIsNone(ici_axis.link_bw)
            self.assertEqual(ici_axis.contention, 2.0)

    def test_tpu_v7x_custom_d2d_parameters(self):
        m = collective.tpu(
            "v7x",
            (4, 4, 4),
            d2d=True,
            d2d_bw=6.0e11,
            d2d_hop_latency=5.0e-7,
            ici_contention=3.0,
        )
        self.assertEqual(m.axes[0].link_bw, 6.0e11)
        self.assertEqual(m.axes[0].hop_latency, 5.0e-7)
        for ici_axis in m.axes[1:]:
            self.assertEqual(ici_axis.contention, 3.0)

    def test_tpu_wraparound_override(self):
        m = collective.tpu("v7x", (2, 2, 4), wraparound=(True, False, True))
        self.assertEqual(tuple(a.wraparound for a in m.axes),
                         (True, False, True))

    def test_tpu_error_handling(self):
        with self.assertRaises(ValueError):
            collective.tpu("v99", (4, 4, 4))

        with self.assertRaises(ValueError):
            collective.tpu("v7x", (4, 4))  # v7x requires 3D slice

        with self.assertRaises(ValueError):
            collective.tpu("v4", (4, 4, 4),
                           d2d=True)  # v4 has no D2D placeholder

    def test_reduce_scatter_sequential_v7x_package_view(self):
        # v7x 4x4x4 (package view): 3D torus, link_bw=90 GB/s, 2 lanes per axis.
        # Total effective BW per axis = 180 GB/s.
        m = collective.tpu("v7x", (4, 4, 4))
        bytes_in = 128 * 1024 * 1024  # 128 MiB = 134,217,728 bytes
        est = collective.reduce_scatter_sequential(m, bytes_in)

        self.assertEqual(est.algo, "sequential")
        self.assertEqual(est.n_devices, 64)
        self.assertEqual(len(est.phases), 3)
        self.assertEqual(est.phases[0].bytes_in, bytes_in)
        self.assertEqual(est.phases[1].bytes_in, bytes_in / 4.0)
        self.assertEqual(est.phases[2].bytes_in, bytes_in / 16.0)

        # Phase breakdown calculations:
        # Phase 0: T_lat = 3 * 1us = 0.003 ms, T_bw = 134217728 * (3 / (4 * 180e9)) = 0.55924053 ms
        # Phase 1: T_lat = 3 * 1us = 0.003 ms, T_bw = 0.55924053 / 4 = 0.13981013 ms
        # Phase 2: T_lat = 3 * 1us = 0.003 ms, T_bw = 0.13981013 / 4 = 0.03495253 ms
        expected_lat_ms = 0.009
        expected_bw_ms = 0.73400320
        expected_total_ms = 0.74300320

        self.assertAlmostEqual(est.t_latency * 1e3, expected_lat_ms, places=6)
        self.assertAlmostEqual(est.t_bandwidth * 1e3, expected_bw_ms, places=6)
        self.assertAlmostEqual(est.total * 1e3, expected_total_ms, places=6)
        self.assertEqual(est.bound, "bandwidth")

    def test_reduce_scatter_sequential_v7x_per_die_view(self):
        # v7x 4x4x4 (per-die view, d2d=True): 4D mesh (2, 4, 4, 4).
        # Axis 0: D2D (size 2, line, 360 GB/s eff BW)
        # Axes 1..3: ICI (size 4, torus, contention=2 -> 90 GB/s eff BW)
        m = collective.tpu("v7x", (4, 4, 4), d2d=True)
        bytes_in = 256 * 1024 * 1024  # 256 MiB = 268,435,456 bytes
        est = collective.reduce_scatter_sequential(m, bytes_in)

        self.assertEqual(est.n_devices, 128)
        self.assertEqual(len(est.phases), 4)
        # Fastest D2D axis (axis 0) should be visited first in optimal order
        self.assertEqual(est.order[0], 0)

        # Phase 0 (D2D): T_lat = 1us = 0.001 ms, T_bw = 268435456 * (1 / (2 * 360e9)) = 0.37282702 ms
        # Phase 1 (ICI): T_lat = 3us = 0.003 ms, T_bw = 134217728 * (3 / (4 * 90e9)) = 1.11848107 ms
        # Phase 2 (ICI): T_lat = 3us = 0.003 ms, T_bw = 1.11848107 / 4 = 0.27962027 ms
        # Phase 3 (ICI): T_lat = 3us = 0.003 ms, T_bw = 0.27962027 / 4 = 0.06990507 ms
        expected_lat_ms = 0.010
        expected_bw_ms = 1.84083342
        expected_total_ms = 1.85083342

        self.assertAlmostEqual(est.t_latency * 1e3, expected_lat_ms, places=6)
        self.assertAlmostEqual(est.t_bandwidth * 1e3, expected_bw_ms, places=6)
        self.assertAlmostEqual(est.total * 1e3, expected_total_ms, places=6)

    def test_reduce_scatter_concurrent_v7x(self):
        # Concurrent 3-way rotation on symmetric 3D torus v7x (package view).
        m = collective.tpu("v7x", (4, 4, 4))
        bytes_in = 128 * 1024 * 1024  # 128 MiB
        est_conc = collective.reduce_scatter_concurrent(m, bytes_in)

        self.assertEqual(est_conc.algo, "concurrent(k=3)")
        self.assertEqual(est_conc.n_devices, 64)

        # On symmetric 3D torus, concurrent T_bw = sequential T_bw / 3 = 0.73400320 / 3 = 0.24466773 ms
        expected_lat_ms = 0.009
        expected_bw_ms = 0.24466773
        expected_total_ms = 0.25366773

        self.assertAlmostEqual(est_conc.t_latency * 1e3,
                               expected_lat_ms,
                               places=6)
        self.assertAlmostEqual(est_conc.t_bandwidth * 1e3,
                               expected_bw_ms,
                               places=6)
        self.assertAlmostEqual(est_conc.total * 1e3,
                               expected_total_ms,
                               places=6)

    def test_reduce_scatter_lower_bound_v7x(self):
        # Theoretical egress and diameter latency floor for v7x 4x4x4 (package view).
        m = collective.tpu("v7x", (4, 4, 4))
        bytes_in = 128 * 1024 * 1024
        lb = collective.reduce_scatter_lower_bound(m, bytes_in)

        self.assertEqual(lb.n_devices, 64)
        self.assertEqual(lb.bytes_in, bytes_in)

        # Egress BW = 3 axes * 2 lanes * 90 GB/s = 540 GB/s
        # T_bw = 134217728 * (63 / (64 * 540e9)) = 0.24466773 ms
        # T_lat = 3 * (4 / 2) * 1us = 0.006 ms
        expected_bw_ms = 0.24466773
        expected_lat_ms = 0.006
        expected_total_ms = 0.24466773  # max(T_bw, T_lat)

        self.assertAlmostEqual(lb.t_bandwidth * 1e3, expected_bw_ms, places=6)
        self.assertAlmostEqual(lb.t_latency * 1e3, expected_lat_ms, places=6)
        self.assertAlmostEqual(lb.t * 1e3, expected_total_ms, places=6)

    def test_reduce_scatter_submesh_v7x(self):
        m = collective.tpu("v7x", (4, 4, 8), d2d=True)
        bytes_in = 64 * 1024 * 1024  # 64 MiB = 67,108,864 bytes

        # RS on D2D axis only (axis 0): size 2, line, 360 GB/s
        # T_lat = 1us = 0.001 ms, T_bw = 67108864 * (1 / (2 * 360e9)) = 0.09320676 ms
        est_d2d = collective.reduce_scatter(m, bytes_in, axes=(0, ))
        self.assertEqual(est_d2d.n_devices, 2)
        self.assertEqual(len(est_d2d.phases), 1)
        self.assertAlmostEqual(est_d2d.t_latency * 1e3, 0.001, places=6)
        self.assertAlmostEqual(est_d2d.t_bandwidth * 1e3, 0.09320676, places=6)
        self.assertAlmostEqual(est_d2d.total * 1e3, 0.09420676, places=6)

        # RS on ICI axes only (axes 1, 2, 3)
        est_ici = collective.reduce_scatter(m, bytes_in, axes=(1, 2, 3))
        self.assertEqual(est_ici.n_devices, 128)
        self.assertEqual(len(est_ici.phases), 3)

    def test_tpuv6e_2d_torus_exact_latency(self):
        # TPU v6e 16x16 full pod (2D torus), link_bw=90 GB/s (180 GB/s per axis).
        m = collective.tpu("v6e", (16, 16))
        bytes_in = 128 * 1024 * 1024  # 128 MiB = 134,217,728 bytes

        # Sequential:
        # Phase 0: T_lat = 15us = 0.015 ms, T_bw = 134217728 * (15 / (16 * 180e9)) = 0.69905067 ms
        # Phase 1: T_lat = 15us = 0.015 ms, T_bw = 0.69905067 / 16 = 0.04369067 ms
        # Total seq T_lat = 0.030 ms, T_bw = 0.74274133 ms, total = 0.77274133 ms
        est_seq = collective.reduce_scatter_sequential(m, bytes_in)
        self.assertAlmostEqual(est_seq.t_latency * 1e3, 0.030, places=6)
        self.assertAlmostEqual(est_seq.t_bandwidth * 1e3, 0.74274133, places=6)
        self.assertAlmostEqual(est_seq.total * 1e3, 0.77274133, places=6)

        # Concurrent (k=2): T_bw = 0.74274133 / 2 = 0.37137067 ms, total = 0.40137067 ms
        est_conc = collective.reduce_scatter_concurrent(m, bytes_in)
        self.assertAlmostEqual(est_conc.t_latency * 1e3, 0.030, places=6)
        self.assertAlmostEqual(est_conc.t_bandwidth * 1e3,
                               0.37137067,
                               places=6)
        self.assertAlmostEqual(est_conc.total * 1e3, 0.40137067, places=6)

        # Lower bound:
        lb = collective.reduce_scatter_lower_bound(m, bytes_in)
        self.assertAlmostEqual(lb.t_latency * 1e3, 0.016, places=6)
        self.assertAlmostEqual(lb.t_bandwidth * 1e3, 0.37137067, places=6)
        self.assertAlmostEqual(lb.t * 1e3, 0.37137067, places=6)

    def test_tpuv5p_3d_torus_exact_latency(self):
        # TPU v5p 4x4x4 (3D torus), link_bw=90 GB/s.
        m = collective.tpu("v5p", (4, 4, 4))
        bytes_in = 256 * 1024 * 1024  # 256 MiB = 268,435,456 bytes

        # Sequential:
        # Phase 0: T_lat = 0.003 ms, T_bw = 268435456 * (3 / (4 * 180e9)) = 1.11848107 ms
        # Phase 1: T_lat = 0.003 ms, T_bw = 1.11848107 / 4 = 0.27962027 ms
        # Phase 2: T_lat = 0.003 ms, T_bw = 0.27962027 / 4 = 0.06990507 ms
        # Total seq T_lat = 0.009 ms, T_bw = 1.46800640 ms, total = 1.47700640 ms
        est_seq = collective.reduce_scatter_sequential(m, bytes_in)
        self.assertAlmostEqual(est_seq.t_latency * 1e3, 0.009, places=6)
        self.assertAlmostEqual(est_seq.t_bandwidth * 1e3, 1.46800640, places=6)
        self.assertAlmostEqual(est_seq.total * 1e3, 1.47700640, places=6)

        # Concurrent (k=3): T_bw = 1.46800640 / 3 = 0.48933547 ms, total = 0.49833547 ms
        est_conc = collective.reduce_scatter_concurrent(m, bytes_in)
        self.assertAlmostEqual(est_conc.t_latency * 1e3, 0.009, places=6)
        self.assertAlmostEqual(est_conc.t_bandwidth * 1e3,
                               0.48933547,
                               places=6)
        self.assertAlmostEqual(est_conc.total * 1e3, 0.49833547, places=6)

        # Lower bound:
        lb = collective.reduce_scatter_lower_bound(m, bytes_in)
        self.assertAlmostEqual(lb.t_latency * 1e3, 0.006, places=6)
        self.assertAlmostEqual(lb.t_bandwidth * 1e3, 0.48933547, places=6)
        self.assertAlmostEqual(lb.t * 1e3, 0.48933547, places=6)

    def test_1d_line_vs_torus_exact_latency(self):
        # 8 chips with 64 MiB (67,108,864 bytes), link_bw = 90 GB/s.
        d = 64 * 1024 * 1024
        # Line (lanes=1): T_lat = 7us = 0.007 ms, T_bw = 67108864 * (7 / (8 * 90e9)) = 0.65244729 ms
        m_line = collective.Machine(
            axes=(collective.Axis(8, wraparound=False), ),
            ici=collective.Interconnect(link_bw_oneway=9.0e10),
        )
        est_line = collective.reduce_scatter(m_line, d)
        self.assertAlmostEqual(est_line.t_latency * 1e3, 0.007, places=6)
        self.assertAlmostEqual(est_line.t_bandwidth * 1e3,
                               0.65244729,
                               places=6)
        self.assertAlmostEqual(est_line.total * 1e3, 0.65944729, places=6)

        # Torus (lanes=2): T_lat = 7us = 0.007 ms, T_bw = 67108864 * (7 / (8 * 180e9)) = 0.32622364 ms
        m_torus = collective.Machine(
            axes=(collective.Axis(8, wraparound=True), ),
            ici=collective.Interconnect(link_bw_oneway=9.0e10),
        )
        est_torus = collective.reduce_scatter(m_torus, d)
        self.assertAlmostEqual(est_torus.t_latency * 1e3, 0.007, places=6)
        self.assertAlmostEqual(est_torus.t_bandwidth * 1e3,
                               0.32622364,
                               places=6)
        self.assertAlmostEqual(est_torus.total * 1e3, 0.33322364, places=6)

    def test_all_gather_and_all_reduce_v7x(self):
        m = collective.tpu("v7x", (4, 4, 4))
        bytes_in = 64 * 1024 * 1024

        rs_est = collective.reduce_scatter(m, bytes_in)
        ag_time = collective.all_gather_time(m, bytes_in)
        ar_time = collective.all_reduce_time(m, bytes_in)

        self.assertEqual(ag_time, rs_est.total)
        self.assertEqual(ar_time, 2.0 * rs_est.total)

    def test_other_tpu_generations(self):
        m_v3 = collective.tpu("v3", (16, 32))
        self.assertEqual(m_v3.n_devices, 512)

        m_v4 = collective.tpu("v4", (4, 4, 4))
        self.assertEqual(m_v4.n_devices, 64)

        m_v5p = collective.tpu("v5p", (4, 4, 8))
        self.assertEqual(m_v5p.n_devices, 128)

        m_v5e = collective.tpu("v5e", (8, 16))
        self.assertEqual(m_v5e.n_devices, 128)

        m_v6e = collective.tpu("v6e", (16, 16))
        self.assertEqual(m_v6e.n_devices, 256)

    def test_utility_functions(self):
        self.assertEqual(collective.parse_size("128MiB"), 128 * 2**20)
        self.assertEqual(collective.parse_size("1GB"), 1.0e9)
        self.assertEqual(collective.parse_size("256KB"), 256.0e3)
        self.assertEqual(collective.parse_size("512KiB"), 512 * 2**10)
        self.assertEqual(collective.parse_size("1024B"), 1024.0)

        with self.assertRaises(ValueError):
            collective.parse_size("invalid")

        with self.assertRaises(ValueError):
            collective.parse_size("128XYZ")

        self.assertEqual(collective.fmt_bytes(1024), "1KiB")
        self.assertIn("us", collective.fmt_time(1.0e-6))
        self.assertIn("GB/s", collective.fmt_bw(1.0e11))

        m = collective.tpu("v7x", (4, 4, 4))
        est = collective.reduce_scatter(m, 128 * 1024 * 1024)
        desc = collective.describe(est)
        self.assertIn("N=64", desc)
        self.assertIn("axis0", desc)

    def test_single_device_trivial_topology(self):
        m_single = collective.Machine(
            axes=(collective.Axis(1), ),
            ici=collective.Interconnect(link_bw_oneway=9.0e10,
                                        launch_overhead=5.0e-6),
        )
        bytes_in = 128 * 1024 * 1024
        est = collective.reduce_scatter(m_single, bytes_in)
        self.assertEqual(est.n_devices, 1)
        self.assertEqual(est.t_latency, 0.0)
        self.assertEqual(est.t_bandwidth, 0.0)
        self.assertEqual(est.t_fixed, 5.0e-6)
        self.assertEqual(est.total, 5.0e-6)
        self.assertEqual(collective.all_gather_time(m_single, bytes_in),
                         5.0e-6)
        self.assertEqual(collective.all_reduce_time(m_single, bytes_in),
                         10.0e-6)

    def test_zero_message_size(self):
        m = collective.tpu("v7x", (4, 4, 4))
        est = collective.reduce_scatter(m, 0.0)
        self.assertEqual(est.bytes_in, 0.0)
        self.assertEqual(est.t_bandwidth, 0.0)
        self.assertAlmostEqual(est.t_latency * 1e3, 0.009,
                               places=6)  # 9us for 3 axes * 3 steps

    def test_message_size_linearity_in_bandwidth_regime(self):
        m = collective.tpu("v7x", (4, 4, 4))
        bytes_1 = 128 * 1024 * 1024
        bytes_2 = 256 * 1024 * 1024
        est1 = collective.reduce_scatter_sequential(m, bytes_1)
        est2 = collective.reduce_scatter_sequential(m, bytes_2)
        # T_bw(256 MiB) = 2 * T_bw(128 MiB) = 2 * 0.73400320 ms = 1.46800640 ms
        self.assertAlmostEqual(est1.t_bandwidth * 1e3, 0.73400320, places=6)
        self.assertAlmostEqual(est2.t_bandwidth * 1e3, 1.46800640, places=6)
        self.assertAlmostEqual(est2.t_bandwidth,
                               2.0 * est1.t_bandwidth,
                               places=7)

    def test_unidirectional_ring_slowdown(self):
        # A single torus axis with bidirectional lanes=2 vs unidirectional lanes=1
        m = collective.Machine(
            axes=(collective.Axis(8, wraparound=True), ),
            ici=collective.Interconnect(link_bw_oneway=9.0e10),
        )
        bytes_in = 128 * 1024 * 1024
        est_bi = collective.reduce_scatter_sequential(m,
                                                      bytes_in,
                                                      unidirectional=False)
        est_uni = collective.reduce_scatter_sequential(m,
                                                       bytes_in,
                                                       unidirectional=True)
        # Bidirectional (2 lanes, 180 GB/s): T_bw = 134217728 * (7 / (8 * 180e9)) = 0.65244729 ms
        # Unidirectional (1 lane, 90 GB/s): T_bw = 134217728 * (7 / (8 * 90e9)) = 1.30489458 ms
        self.assertAlmostEqual(est_bi.t_bandwidth * 1e3, 0.65244729, places=6)
        self.assertAlmostEqual(est_uni.t_bandwidth * 1e3, 1.30489458, places=6)
        self.assertAlmostEqual(est_uni.t_bandwidth,
                               2.0 * est_bi.t_bandwidth,
                               places=7)

    def test_custom_interconnect_overrides(self):
        # Overrides: hop_latency = 2us, step_overhead = 0.1us, launch_overhead = 3us, efficiency = 0.8
        m = collective.tpu(
            "v7x",
            (4, 4, 4),
            link_efficiency=0.8,
            hop_latency=2.0e-6,
            step_overhead=1.0e-7,
            launch_overhead=3.0e-6,
        )
        bytes_in = 128 * 1024 * 1024
        est = collective.reduce_scatter_sequential(m, bytes_in)

        # T_lat = 3 axes * 3 steps * (2us + 0.1us) = 18.9us = 0.0189 ms
        # T_bw = 0.73400320 / 0.8 = 0.91750400 ms
        # T_fixed = 0.003 ms
        # Total = 0.0189 + 0.91750400 + 0.003 = 0.93940400 ms
        self.assertAlmostEqual(est.t_latency * 1e3, 0.0189, places=6)
        self.assertAlmostEqual(est.t_bandwidth * 1e3, 0.91750400, places=6)
        self.assertAlmostEqual(est.t_fixed * 1e3, 0.003, places=6)
        self.assertAlmostEqual(est.total * 1e3, 0.93940400, places=6)

    def test_asymmetric_mesh_and_line_topologies(self):
        # v5e: 8 is line (not 16), 16 is torus
        m_v5e = collective.tpu("v5e", (8, 16))
        self.assertFalse(m_v5e.axes[0].wraparound)
        self.assertTrue(m_v5e.axes[1].wraparound)

        # v4: (2, 2, 4) is not a full 4x4x4 cube -> all lines
        m_v4_mesh = collective.tpu("v4", (2, 2, 4))
        self.assertFalse(any(a.wraparound for a in m_v4_mesh.axes))

    def test_reduce_scatter_invalid_algorithm(self):
        m = collective.tpu("v7x", (4, 4, 4))
        with self.assertRaises(ValueError):
            collective.reduce_scatter(m, 1024, algo="invalid_algo")

    def test_ring_asymptotic_scaling_factor(self):
        ici = collective.Interconnect(link_bw_oneway=1.0e11)
        m_p2 = collective.Machine(axes=(collective.Axis(2, wraparound=True), ),
                                  ici=ici)
        m_p4 = collective.Machine(axes=(collective.Axis(4, wraparound=True), ),
                                  ici=ici)
        m_p8 = collective.Machine(axes=(collective.Axis(8, wraparound=True), ),
                                  ici=ici)

        d = 100 * 1024 * 1024  # 104,857,600 bytes
        est_p2 = collective.reduce_scatter_sequential(m_p2, d)
        est_p4 = collective.reduce_scatter_sequential(m_p4, d)
        est_p8 = collective.reduce_scatter_sequential(m_p8, d)

        # T_bw = D * (p-1)/(p * 200 GB/s)
        # p=2: 104857600 * 0.5 / 200e9 = 0.26214400 ms
        # p=4: 104857600 * 0.75 / 200e9 = 0.39321600 ms
        # p=8: 104857600 * 0.875 / 200e9 = 0.45875200 ms
        self.assertAlmostEqual(est_p2.t_bandwidth * 1e3, 0.26214400, places=6)
        self.assertAlmostEqual(est_p4.t_bandwidth * 1e3, 0.39321600, places=6)
        self.assertAlmostEqual(est_p8.t_bandwidth * 1e3, 0.45875200, places=6)

    def test_tpuv7x_d2d_1gb_all_gather(self):
        # p = 2 (2 dies in package), m_bytes = 1 GB (1e9 bytes), d2d_bw = 600 GB/s oneway (1200 GB/s bidir).
        m = collective.Machine(
            axes=(collective.Axis(size=2,
                                  wraparound=False,
                                  link_bw=600e9,
                                  name="d2d"), ),
            ici=collective.Interconnect(link_bw_oneway=90e9, hop_latency=1e-6),
        )
        est = collective.reduce_scatter(m, 1e9)
        # T_bw = 1e9 * (1 / (2 * 600e9)) * 1000 = 0.83333333 ms
        # T_lat = (2 - 1) * 1us = 0.001 ms
        expected_bw_ms = 0.83333333
        expected_lat_ms = 0.001
        expected_total_ms = 0.83433333

        self.assertAlmostEqual(est.t_bandwidth * 1e3, expected_bw_ms, places=4)
        self.assertAlmostEqual(est.t_latency * 1e3, expected_lat_ms, places=4)
        self.assertAlmostEqual(est.total * 1e3, expected_total_ms, places=4)

        # all_gather_time should match reduce_scatter total time
        ag_time = collective.all_gather_time(m, 1e9)
        self.assertAlmostEqual(ag_time * 1e3, expected_total_ms, places=4)

    def test_tpuv7x_all_gather_p4_exact_latency(self):
        # p = 4 devices on a ring (torus, 2 lanes), m_bytes = 1 GB (1e9 bytes), link_bw = 400 GB/s oneway (800 GB/s bidir).
        m = collective.Machine(
            axes=(collective.Axis(size=4, wraparound=True), ),
            ici=collective.Interconnect(link_bw_oneway=400e9,
                                        hop_latency=1e-6),
        )
        est = collective.reduce_scatter(m, 1e9)
        # T_bw = 1e9 * (3 / (4 * 800e9)) * 1000 = 0.9375 ms
        # T_lat = 3 * 1us = 0.003 ms
        expected_bw_ms = 0.9375
        expected_lat_ms = 0.003
        expected_total_ms = 0.9405

        self.assertAlmostEqual(est.t_bandwidth * 1e3, expected_bw_ms, places=4)
        self.assertAlmostEqual(est.t_latency * 1e3, expected_lat_ms, places=4)
        self.assertAlmostEqual(est.total * 1e3, expected_total_ms, places=4)

    def test_hierarchical_all_gather_legacy_numbers(self):
        # p = (4, 2): axis 0 = outer ICI (p=4, torus), axis 1 = inner D2D (p=2, line)
        # m_bytes = 8 * 8192 * 4096 = 268,435,456 bytes (256 MiB fp8)
        # CUSTOM_HW: ici_bw_bidir = 200 GB/s (contention=2 -> 50 GB/s oneway * 2 lanes = 100 GB/s eff)
        #            d2d_bw_bidir = 3200 GB/s (1600 GB/s oneway * 1 lane = 1600 GB/s eff)
        m = collective.Machine(
            axes=(
                collective.Axis(size=4,
                                wraparound=True,
                                link_bw=100e9,
                                contention=2.0),
                collective.Axis(size=2,
                                wraparound=False,
                                link_bw=1600e9,
                                contention=1.0),
            ),
            ici=collective.Interconnect(link_bw_oneway=100e9, hop_latency=0.0),
        )
        m_bytes = 8 * 8192 * 4096
        # Sequential visiting D2D (size 2) first then ICI (size 4):
        # D2D phase: 268435456 * (1 / (2 * 1600e9)) * 1000 = 0.08388608 ms
        # ICI phase: 134217728 * (3 / (4 * 100e9)) * 1000 = 1.00663296 ms
        # Total = 0.08388608 + 1.00663296 = 1.09051904 ms
        est = collective.reduce_scatter_sequential(m, m_bytes)
        expected_latency_ms = 1.09051904
        self.assertAlmostEqual(est.t_bandwidth * 1e3,
                               expected_latency_ms,
                               places=6)

    def test_hierarchical_all_gather_pipelined_legacy_numbers(self):
        # p = (4, 2), m_bytes = 8 * 8192 * 4096 (256 MiB fp8)
        # Pipelined chunking across 4 chunks:
        # D2D chunk: lat_d2d = (m_bytes / 4) * (1 / (2 * 1600e9)) * 1000 = 0.02097152 ms
        # ICI chunk: lat_ici = ((m_bytes / 4) / 2) * (3 / (4 * 100e9)) * 1000 = 0.25165824 ms
        # Total pipelined = 3 * max(lat) + sum(lat) = 1.02760448 ms (~1.0276 ms)
        m_bytes = 8 * 8192 * 4096
        lat_d2d = (m_bytes / 4) * (1 / (2 * 1600e9)) * 1000
        lat_ici = ((m_bytes / 4) / 2) * (3 / (4 * 100e9)) * 1000
        pipelined_lat = 3 * max(lat_d2d, lat_ici) + lat_d2d + lat_ici
        expected_latency_ms = 1.0276
        self.assertAlmostEqual(pipelined_lat, expected_latency_ms, places=3)

    def test_hierarchical_all_gather_with_inactive_ici_axis(self):
        # p = (2, 1, 2) on v7x should yield the same latency as p = (2, 2) on v7x
        m_bytes = 8 * 8192 * 4096
        m_inactive = collective.Machine(
            axes=(
                collective.Axis(size=2, wraparound=False,
                                link_bw=600e9),  # d2d
                collective.Axis(size=2,
                                wraparound=True,
                                link_bw=90e9,
                                contention=2.0),
                collective.Axis(size=1,
                                wraparound=True,
                                link_bw=90e9,
                                contention=2.0),
                collective.Axis(size=2,
                                wraparound=True,
                                link_bw=90e9,
                                contention=2.0),
            ),
            ici=collective.Interconnect(link_bw_oneway=90e9),
        )
        m_ref = collective.Machine(
            axes=(
                collective.Axis(size=2, wraparound=False,
                                link_bw=600e9),  # d2d
                collective.Axis(size=2,
                                wraparound=True,
                                link_bw=90e9,
                                contention=2.0),
                collective.Axis(size=2,
                                wraparound=True,
                                link_bw=90e9,
                                contention=2.0),
            ),
            ici=collective.Interconnect(link_bw_oneway=90e9),
        )
        lat_inactive = collective.reduce_scatter(m_inactive, m_bytes).total
        lat_reference = collective.reduce_scatter(m_ref, m_bytes).total
        self.assertEqual(lat_inactive, lat_reference)

    def test_hierarchical_all_gather_with_inactive_d2d_axis(self):
        # p = (2, 4, 1) on v7x should yield the same latency as p = (2, 4)
        # on a monolithic hardware spec with ICI bandwidth = 90 GB/s bidir (45 GB/s oneway).
        m_bytes = 8 * 8192 * 4096
        m_inactive_d2d = collective.Machine(
            axes=(
                collective.Axis(size=1, wraparound=False,
                                link_bw=600e9),  # inactive d2d
                collective.Axis(size=2,
                                wraparound=True,
                                link_bw=90e9,
                                contention=2.0),
                collective.Axis(size=4,
                                wraparound=True,
                                link_bw=90e9,
                                contention=2.0),
            ),
            ici=collective.Interconnect(link_bw_oneway=90e9),
        )
        m_monolithic_ref = collective.Machine(
            axes=(
                collective.Axis(size=2,
                                wraparound=True,
                                link_bw=45e9,
                                contention=1.0),
                collective.Axis(size=4,
                                wraparound=True,
                                link_bw=45e9,
                                contention=1.0),
            ),
            ici=collective.Interconnect(link_bw_oneway=45e9),
        )
        lat_inactive = collective.reduce_scatter(m_inactive_d2d, m_bytes).total
        lat_reference = collective.reduce_scatter(m_monolithic_ref,
                                                  m_bytes).total
        self.assertEqual(lat_inactive, lat_reference)


class DtypeTest(unittest.TestCase):
    """elems + dtype, and the upcast that only a reducing collective may do."""

    def setUp(self):
        self.m = collective.tpu("v7x", (2, 2, 2), d2d=True)

    def test_dtype_bytes_names_and_numbers(self):
        self.assertEqual(collective.dtype_bytes("bf16"), 2.0)
        self.assertEqual(collective.dtype_bytes("FP8"), 1.0)
        self.assertEqual(collective.dtype_bytes("fp4"), 0.5)
        self.assertEqual(collective.dtype_bytes(3),
                         3.0)  # bytes/element directly
        with self.assertRaises(ValueError):
            collective.dtype_bytes("float8_e4m3fnuz")

    def test_elems_dtype_equals_precomputed_bytes(self):
        n = 4096 * 8192
        a = collective.reduce_scatter(self.m, elems=n, dtype="bf16").total
        b = collective.reduce_scatter(self.m, n * 2).total
        self.assertAlmostEqual(a, b, places=12)

    def test_reducing_collective_may_upcast_on_the_wire(self):
        """fp8 operands summed in bf16 put 2x the bytes on the wire."""
        n = 4096 * 8192
        narrow = collective.all_reduce(self.m, elems=n, dtype="fp8")
        upcast = collective.all_reduce(self.m,
                                       elems=n,
                                       dtype="fp8",
                                       wire_dtype="bf16")
        self.assertAlmostEqual(upcast.t_bandwidth / narrow.t_bandwidth, 2.0)
        self.assertEqual(upcast.dtype, "fp8")
        self.assertEqual(upcast.wire_dtype, "bf16")
        self.assertIn("fp8->wire bf16", collective.describe(upcast))

    def test_permutations_refuse_a_wire_dtype(self):
        """AG and a2a cannot change dtype, so the argument is an error not a no-op."""
        for fn, kw in (
            (collective.all_gather, {
                "elems_full": 1024
            }),
            (collective.all_to_all, {
                "elems": 1024
            }),
        ):
            with self.assertRaises(ValueError):
                fn(self.m, dtype="fp8", wire_dtype="bf16", **kw)

    def test_payload_must_be_given_exactly_once(self):
        with self.assertRaises(ValueError):  # neither
            collective.reduce_scatter(self.m)
        with self.assertRaises(ValueError):  # both
            collective.reduce_scatter(self.m, 1024, elems=1024, dtype="bf16")
        with self.assertRaises(ValueError):  # elems without dtype
            collective.reduce_scatter(self.m, elems=1024)
        with self.assertRaises(ValueError):  # dtype without elems
            collective.reduce_scatter(self.m, 1024, dtype="bf16")


class AllToAllTest(unittest.TestCase):
    """All-to-all is link-bound, not egress-bound. See collective.py's module doc."""

    @staticmethod
    def _ring(x, wraparound, bw=90e9):
        return collective.Machine(
            axes=(collective.Axis(size=x, wraparound=wraparound,
                                  link_bw=bw), ),
            ici=collective.Interconnect(link_bw_oneway=bw, hop_latency=0.0),
        )

    def test_closed_form_single_axis(self):
        """T_bw = D*X/(4*lanes*B), checked against the hand derivation."""
        d, bw = 1 << 30, 90e9
        for x, lanes in ((4, 2), (8, 2), (16, 2)):
            est = collective.all_to_all(self._ring(x, True), d)
            self.assertAlmostEqual(
                est.t_bandwidth,
                d * x / (4 * lanes * bw),
                places=12,
                msg=f"torus X={x}",
            )

    def test_line_is_exactly_twice_the_torus(self):
        """One cut edge instead of two."""
        d = 1 << 30
        for x in (4, 8, 16):
            t_torus = collective.all_to_all(self._ring(x, True), d).t_bandwidth
            t_line = collective.all_to_all(self._ring(x, False), d).t_bandwidth
            self.assertAlmostEqual(t_line / t_torus, 2.0, msg=f"X={x}")

    def test_x2_coincides_with_the_egress_bound(self):
        """At X=2 nothing travels more than one hop, so link == egress. Both lane counts."""
        d = 1 << 20
        for wrap in (True, False):
            m = self._ring(2, wrap)
            a2a = collective.all_to_all(m, d).t_bandwidth
            egress = collective.reduce_scatter_lower_bound(m, d).t_bandwidth
            self.assertAlmostEqual(a2a,
                                   egress,
                                   places=12,
                                   msg=f"wraparound={wrap}")

    def test_bisection_cross_check(self):
        """Independent derivation: (N/2 devices)x(D/2 each) over the cut's links."""
        d, x, bw = 1 << 30, 16, 90e9
        crossing = (x / 2) * (d / 2)  # bytes that must change halves, one way
        directed_links = 2  # torus cut in two places, one direction each
        self.assertAlmostEqual(
            collective.all_to_all(self._ring(x, True), d).t_bandwidth,
            crossing / (directed_links * bw),
            places=12,
        )

    def test_ratio_against_an_rs_proxy(self):
        """The regression guard: X^2/(4(X-1)), i.e. 4.27x at X=16.

    An RS proxy is what a caller reaches for when the primitive is missing, and
    it is the error this function exists to remove.
    """
        d, x = 1 << 30, 16
        m = self._ring(x, True)
        ratio = (collective.all_to_all(m, d).t_bandwidth /
                 collective.reduce_scatter(m, d).t_bandwidth)
        self.assertAlmostEqual(ratio, x**2 / (4 * (x - 1)), places=9)
        self.assertAlmostEqual(ratio, 4.266666, places=5)

    def test_payload_is_preserved_across_phases(self):
        """A permutation moves everything every phase -- RS's `s /= X` must not appear."""
        d = 1 << 30
        est = collective.all_to_all(collective.tpu("v7x", (2, 2, 2), d2d=True),
                                    d,
                                    algo="sequential")
        self.assertGreater(len(est.phases), 1)
        for p in est.phases:
            self.assertEqual(p.bytes_in, d)

    def test_cost_is_independent_of_axis_order(self):
        """Follows from payload preservation; false for RS."""
        m = collective.tpu("v7x", (2, 4, 8), d2d=True)
        d = 1 << 28
        totals = {
            collective.all_to_all(m, d, algo="sequential", order=o).t_bandwidth
            for o in ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2))
        }
        self.assertEqual(len(totals), 1)

    def test_sequential_sums_and_concurrent_maxes(self):
        m = collective.tpu("v7x", (2, 2, 2), d2d=True)
        d = 1 << 30
        seq = collective.all_to_all(m, d, algo="sequential")
        con = collective.all_to_all(m, d, algo="concurrent")
        per_axis = [p.t_bandwidth for p in seq.phases]
        self.assertAlmostEqual(seq.t_bandwidth, sum(per_axis))
        self.assertAlmostEqual(con.t_bandwidth, max(per_axis))
        self.assertLess(con.total, seq.total)

    def test_concurrent_equals_the_lower_bound(self):
        """Concurrent is the bisection floor, which is why `auto` is refused."""
        m = collective.tpu("v7x", (2, 2, 2), d2d=True)
        d = 1 << 30
        con = collective.all_to_all(m, d, algo="concurrent")
        lb = collective.all_to_all_lower_bound(m, d)
        self.assertAlmostEqual(con.t_bandwidth, lb.t_bandwidth, places=12)

    def test_auto_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            collective.all_to_all(self._ring(8, True), 1024, algo="auto")
        self.assertIn("floor", str(cm.exception))

    def test_lower_bound_dominates_egress_beyond_x2(self):
        d = 1 << 30
        for x in (4, 8, 16):
            m = self._ring(x, True)
            self.assertGreater(
                collective.all_to_all_lower_bound(m, d).t_bandwidth,
                collective.reduce_scatter_lower_bound(m, d).t_bandwidth,
            )

    def test_single_device_group_is_free(self):
        m = self._ring(1, False)
        self.assertEqual(collective.all_to_all(m, 1 << 20).t_bandwidth, 0.0)
        self.assertEqual(collective.all_to_all_lower_bound(m, 1 << 20).t, 0.0)

    def test_monotone_in_axis_size(self):
        d = 1 << 30
        ts = [
            collective.all_to_all(self._ring(x, True), d).t_bandwidth
            for x in (2, 4, 8, 16)
        ]
        self.assertEqual(ts, sorted(ts))


class MoeExpertParallelTest(unittest.TestCase):
    """The comparison the EP variant table is for: a2a vs all-gather + reduce-scatter.

  Both price the same MoE layer on the same group; they differ in what the
  implementation issues. Hidden/token counts are shapes, not pins.
  """

    def setUp(self):
        self.m = collective.tpu("v7x", (2, 2, 2), d2d=True)  # 16 dies
        self.tokens, self.hidden = 4096, 6144

    def _a2a_pair(self, topk, dispatch="fp8", collect="bf16", field="total"):
        """Dispatch and collect are separate calls: the two legs differ in dtype."""
        routed = self.tokens * topk * self.hidden
        return sum(
            getattr(collective.all_to_all(self.m, elems=routed, dtype=dt),
                    field) for dt in (dispatch, collect))

    def _ag_rs(self, dtype="bf16", field="total"):
        """Gather every rank's tokens, run local experts, reduce-scatter back.

    Cost is independent of topk -- every token is gathered whatever its
    routing. That is the property that decides the crossover, and it is why
    topk is not a parameter here.
    """
        g = self.m.n_devices
        full = self.tokens * g * self.hidden
        return getattr(
            collective.all_gather(self.m, elems_full=full, dtype=dtype),
            field) + getattr(
                collective.reduce_scatter(self.m, elems=full, dtype=dtype),
                field)

    def test_a2a_volume_is_linear_in_topk(self):
        """On t_bandwidth, not total -- the per-launch latency does not scale."""
        base = self._a2a_pair(topk=1, field="t_bandwidth")
        for k in (2, 8, 32):
            self.assertAlmostEqual(
                self._a2a_pair(topk=k, field="t_bandwidth") / base,
                float(k),
                places=6,
                msg=f"topk={k}",
            )

    def test_the_two_variants_cross_over(self):
        """a2a wins at low topk, AG+RS at high. Neither dominates."""
        same = dict(dispatch="bf16", collect="bf16")
        self.assertLess(self._a2a_pair(topk=1, **same), self._ag_rs())
        self.assertGreater(self._a2a_pair(topk=32, **same), self._ag_rs())


if __name__ == "__main__":
    unittest.main()
