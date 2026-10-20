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
"""Unit tests for tpu_slots.py, against a parallel.json in a temp directory."""

from __future__ import annotations

import importlib.util
import io
import json
import multiprocessing
import pathlib
import sys
import tempfile
import unittest
import unittest.mock

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent

try:
    from . import tpu_slots  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "tpu_slots", _SCRIPTS_DIR / "tpu_slots.py")
    tpu_slots = importlib.util.module_from_spec(_spec)
    sys.modules["tpu_slots"] = tpu_slots
    _spec.loader.exec_module(tpu_slots)

IMAGE = "registry.example/vllm@sha256:abc"


def _gke(accelerator="tpu7x", topology="2x2x1", image=IMAGE, chips=4):
    return {
        "backend": "gke_job",
        "params": {
            "tpu_accelerator": accelerator,
            "tpu_topology": topology,
            "image": image,
            "tpu_chips": chips
        }
    }


def _seen(available, nodes, job_chips=4):
    return {
        "accelerator": "x",
        "topology": "y",
        "available_chips": available,
        "details": {
            "nodes": nodes,
            "job_chips": job_chips
        }
    }


PROFILES = {
    "v6e-a": {
        "backend": "gce_tpu_vm",
        "params": {}
    },
    "v6e-b": {
        "backend": "gce_tpu_vm",
        "params": {}
    },
    "gke": {
        "backend": "gke_job",
        "params": {}
    },
    "here": {
        "backend": "local",
        "params": {}
    },
}


def _claim_in_process(path: str, holder: str, out) -> None:
    """One process's claim, for the concurrency test: a slot name or ''."""
    try:
        out.put(tpu_slots.claim(pathlib.Path(path), holder)["resource"])
    except tpu_slots.Refused:
        out.put("")


class SlotCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "parallel.json"

    def state(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def discover(self, profiles=PROFILES, gke_slots=1, seen=None, **kw):
        return tpu_slots.discover(
            self.path,
            profiles,
            gke_slots,
            inspect=seen.get if isinstance(seen, dict) else
            (seen or (lambda _: None)),
            **kw)


class TestDiscover(SlotCase):

    def test_slots_follow_what_each_profile_can_serve(self):
        out = self.discover(gke_slots=2)
        # A VM is one slot, an unmeasured cluster --gke-slots, local none.
        self.assertEqual(sorted(self.state()["resources"]),
                         ["gke-1", "gke-2", "v6e-a", "v6e-b"])
        self.assertEqual(out["parallel"], 4)
        self.path.unlink()
        out = self.discover({"v6e": _gke("tpu-v6e-slice", "2x4", chips=8)},
                            seen=lambda _: _seen(20, ["n1", "n2", "n3"], 8))
        key = f"tpu-v6e-slice|2x4|8|{IMAGE}"
        self.assertEqual(sorted(self.state()["resources"]), ["v6e-1", "v6e-2"])
        self.assertEqual((out["hardware"][key]["free"], out["parallel"]),
                         (2, 2))
        # Free chips unreadable, or a pool with no nodes yet: --gke-slots.
        out = self.discover({
            "a": _gke(),
            "b": _gke(topology="2x2x2")
        },
                            gke_slots=3,
                            seen={
                                "a": None,
                                "b": _seen(64, [])
                            })
        self.assertEqual(
            (out["sized"]["a"]["slots"], out["sized"]["b"]["slots"]), (3, 3))

    def test_rediscovery_keeps_claims_and_the_parallel_cap(self):
        self.discover({"v6e-a": PROFILES["v6e-a"]})
        tpu_slots.claim(self.path, "sweep:a")
        state = self.state()
        state["parallel"] = 1
        self.path.write_text(json.dumps(state))
        out = self.discover(gke_slots=3)
        self.assertEqual(sorted(out["added"]),
                         ["gke-1", "gke-2", "gke-3", "v6e-b"])
        self.assertEqual(self.state()["resources"]["v6e-a"]["job"], "sweep:a")
        self.assertEqual(out["parallel"], 1)
        self.assertEqual(self.discover(parallel=4)["parallel"], 4)
        self.path.unlink()
        out = self.discover({"here": PROFILES["here"]})
        self.assertEqual(out["status"], "needs_input")

    def test_profiles_on_the_same_nodes_share_their_chips(self):
        # Two images on one pool with room for two servers: each profile
        # lists both, but only two can be held between them.
        out = self.discover(
            {
                "a": _gke(),
                "b": _gke(image="registry.example/other@sha256:d")
            },
            seen=lambda _: _seen(8, ["n1", "n2"]))
        self.assertEqual((out["sized"]["a"]["slots"],
                          out["sized"]["b"]["slots"], out["capacity"]),
                         (2, 2, 2))
        tpu_slots.claim(self.path, "h1", profile="a")
        tpu_slots.claim(self.path, "h2", profile="b")
        for kw in ({"profile": "a"}, {"resource": "b-2"}):
            with self.subTest(**kw), self.assertRaises(tpu_slots.Refused):
                tpu_slots.claim(self.path, "h3", **kw)

    def test_a_hardware_key_is_accelerator_topology_chips_and_image(self):
        no_chips = _gke()
        del no_chips["params"]["tpu_chips"]
        self.assertEqual(tpu_slots.hardware_key(no_chips),
                         f"tpu7x|2x2x1|4|{IMAGE}")
        self.assertEqual(tpu_slots.hardware_key(_gke(chips=4)),
                         tpu_slots.hardware_key(no_chips))
        self.assertNotEqual(tpu_slots.hardware_key(_gke(chips=2)),
                            tpu_slots.hardware_key(no_chips))
        self.assertIsNone(tpu_slots.hardware_key(_gke(image="")))
        vm = {"backend": "gce_tpu_vm"}
        self.assertIsNone(
            tpu_slots.hardware_key(vm, {
                "accelerator": "none",
                "topology": "none"
            }))
        key = tpu_slots.hardware_key(vm, {
            "accelerator": "v6e",
            "topology": "2x2",
            "chips": 4
        })
        self.assertEqual(key, "v6e|2x2|4|-")
        self.assertEqual(tpu_slots.Hardware.parse(key).chips, "4")
        self.assertIsNone(tpu_slots.Hardware.parse("a|b"))


class TestClaim(SlotCase):

    def test_claims_take_free_slots_and_refuse_what_is_held(self):
        self.discover()
        first = tpu_slots.claim(self.path, "sweep:a")
        self.assertEqual(tpu_slots.claim(self.path, "sweep:a"), first)
        self.assertNotEqual(
            tpu_slots.claim(self.path, "sweep:b")["resource"],
            first["resource"])
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.claim(self.path, "sweep:c", resource=first["resource"])
        with self.assertRaises(tpu_slots.NeedsInput):
            tpu_slots.claim(self.path, "sweep:c", resource="nope")
        # gke-1 and v6e-a are held: only v6e-b fits the profile asked for.
        self.assertEqual(
            tpu_slots.claim(self.path, "sweep:c", profile="v6e-b")["resource"],
            "v6e-b")
        state = self.state()
        state["parallel"] = 3
        self.path.write_text(json.dumps(state))
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.claim(self.path, "sweep:d")
        with self.assertRaises(tpu_slots.NeedsInput):
            tpu_slots.claim(pathlib.Path(self.tmp.name) / "none.json", "x")

    def test_release_and_set_job_belong_to_the_holder(self):
        self.discover()
        name = tpu_slots.claim(self.path, "sweep:a")["resource"]
        tpu_slots.set_job(self.path, name, "sweep:a", "job-1")
        self.assertEqual(self.state()["resources"][name]["job_id"], "job-1")
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.set_job(self.path, name, "sweep:b", "job-2")
        self.assertFalse(
            tpu_slots.release(self.path, name, "sweep:b")["released"])
        self.assertTrue(
            tpu_slots.release(self.path, name, "sweep:a")["released"])
        self.assertIsNone(self.state()["resources"][name]["job"])

    def test_most_available_takes_the_kind_with_room_for_the_most(self):
        profiles = {f"a{i}": _gke() for i in range(3)}
        profiles.update(big=_gke(), v6e=_gke("tpu-v6e-slice", "2x4", chips=8))
        seen = {f"a{i}": _seen(4, ["n1"]) for i in range(3)}
        seen.update(big=_seen(8, ["c1", "c2"]), v6e=_seen(16, ["m1", "m2"], 8))
        self.discover(profiles, seen=seen)
        # v7x: one pool of 1 listed thrice, one of 2 -> room for 3; v6e 2.
        got = tpu_slots.claim(self.path, "leader", most_available=True)
        self.assertEqual((got["hardware"], got["profile"]),
                         (f"tpu7x|2x2x1|4|{IMAGE}", "big"))
        self.assertEqual(got["free_like_this"], 2)
        follower = tpu_slots.claim(self.path, "f1", hardware=got["hardware"])
        self.assertEqual(follower["hardware"], got["hardware"])
        self.path.unlink()
        self.discover({"gke": {"backend": "gke_job"}}, gke_slots=2)
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.claim(self.path, "leader", most_available=True)

    def test_concurrent_claims_never_share_a_slot(self):
        self.discover(gke_slots=4)
        slots = len(self.state()["resources"])
        ctx = multiprocessing.get_context("spawn")
        out = ctx.Queue()
        workers = [
            ctx.Process(target=_claim_in_process,
                        args=(str(self.path), f"sweep:{i}", out))
            for i in range(slots + 3)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join(60)
        taken = [g for g in (out.get(timeout=10) for _ in workers) if g]
        self.assertEqual((len(taken), len(set(taken))), (slots, slots))
        held = {e["job"] for e in self.state()["resources"].values()}
        self.assertEqual(len(held), slots)


class TestCli(SlotCase):

    def test_state_path_and_exit_codes(self):
        with unittest.mock.patch.dict("os.environ",
                                      {"TPU_AGENT_SUITE_HOME": "/h"},
                                      clear=True):
            self.assertEqual(tpu_slots.state_path("r1"),
                             pathlib.Path("/h/runs/r1/parallel.json"))
            with self.assertRaises(tpu_slots.NeedsInput):
                tpu_slots.state_path(None)
        argv = ["claim", "--holder", "a", "--state", str(self.path)]
        with unittest.mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(tpu_slots.main(argv), tpu_slots.EXIT_NEEDS_INPUT)
        self.discover({"v6e-a": PROFILES["v6e-a"]})
        with unittest.mock.patch("sys.stdout", new=io.StringIO()) as out:
            self.assertEqual(tpu_slots.main(argv), tpu_slots.EXIT_OK)
        self.assertEqual(json.loads(out.getvalue())["resource"], "v6e-a")
        argv[2] = "b"
        with unittest.mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(tpu_slots.main(argv), tpu_slots.EXIT_REFUSED)


if __name__ == "__main__":
    unittest.main()
