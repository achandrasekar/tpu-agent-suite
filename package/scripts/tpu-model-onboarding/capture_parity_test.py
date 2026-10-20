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

import json
import os
import sys
import types
from unittest import mock

import capture_parity
import compare_parity
import torch
from absl.testing import absltest, parameterized
from torch import nn

VOCAB = 16
HIDDEN = 8


class TinyMixer(nn.Module):
    """Stands in for the module an unwiring playbook would replace."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self.scale = scale

    def forward(self, x):
        return self.linear(x) * self.scale


class TinyLayer(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.mixer = TinyMixer(scale)

    def forward(self, x):
        return self.mixer(x)


class TinyOutput:
    """Mimics a transformers output object carrying `.logits`."""

    def __init__(self, logits):
        self.logits = logits


class TinyModel(nn.Module):
    """A stand-in model: real nn.Module tree, no Hugging Face download."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([TinyLayer(scale), TinyLayer(scale)])
        self.head = nn.Linear(HIDDEN, VOCAB)
        self.config = mock.MagicMock()
        self.config._attn_implementation = "eager"

    def forward(self, input_ids=None):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return TinyOutput(self.head(hidden))


def make_model(scale=1.0, seed=0):
    torch.manual_seed(seed)
    return TinyModel(scale).eval()


MODULES = ["layers.0.mixer", "layers.1.mixer"]

IDS = [1, 2, 3, 4, 5]


def make_probe_dict(modules=None):
    return {
        "model_id":
        "tiny/model",
        "source": {
            "kind": "hf_pretrained",
            "path": "tiny/model"
        },
        "modules":
        MODULES if modules is None else modules,
        "probes": [{
            "id": "p0",
            "prompt": "hello world",
            "inputs": ["input_ids"],
            "n_tokens": len(IDS),
        }],
    }


def make_probe_tensors(ids=None):
    return {"p0/input_ids": torch.tensor([ids or IDS], dtype=torch.long)}


def write_probe_files(directory, probe=None, tensors=None, name="probe.json"):
    """Writes a probe pair and returns the JSON path."""
    path = os.path.join(directory, name)
    capture_parity.write_probe(
        path, probe if probe is not None else make_probe_dict(),
        make_probe_tensors() if tensors is None else tensors)
    return path


class ProbeTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path

    def test_digest_is_stable_across_key_order(self):
        a = {"model_id": "m", "modules": ["x"], "probes": [{"id": "p0"}]}
        b = {"probes": [{"id": "p0"}], "modules": ["x"], "model_id": "m"}
        self.assertEqual(capture_parity.probe_digest(a),
                         capture_parity.probe_digest(b))

    def test_digest_changes_with_module_list(self):
        self.assertNotEqual(
            capture_parity.probe_digest(make_probe_dict()),
            capture_parity.probe_digest(make_probe_dict(modules=["only.one"])))

    def test_digest_covers_the_tensor_file(self):
        # Identical JSON, different inputs. Hashing text alone would call
        # these the same probe and let two unrelated runs be compared.
        path = write_probe_files(self.tmp)
        first = capture_parity.probe_digest(
            make_probe_dict(), capture_parity.probe_tensors_path(path))
        capture_parity.write_probe(path, make_probe_dict(),
                                   make_probe_tensors([9, 9, 9, 9, 9]))
        second = capture_parity.probe_digest(
            make_probe_dict(), capture_parity.probe_tensors_path(path))
        self.assertNotEqual(first, second)

    def test_digest_is_stable_for_an_unchanged_probe(self):
        path = write_probe_files(self.tmp)
        tensors_path = capture_parity.probe_tensors_path(path)
        self.assertEqual(
            capture_parity.probe_digest(make_probe_dict(), tensors_path),
            capture_parity.probe_digest(make_probe_dict(), tensors_path))

    def test_missing_tensor_file_is_rejected_at_digest_time(self):
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "tensor file not found"):
            capture_parity.probe_digest(make_probe_dict(),
                                        os.path.join(self.tmp, "nope.st"))

    def test_load_probe_returns_json_and_tensors(self):
        probe, tensors = capture_parity.load_probe(write_probe_files(self.tmp))
        self.assertEqual(probe["model_id"], "tiny/model")
        self.assertTrue(
            torch.equal(tensors["p0/input_ids"],
                        torch.tensor([IDS], dtype=torch.long)))

    def test_tensor_path_is_derived_from_the_json_path(self):
        self.assertEqual(str(capture_parity.probe_tensors_path("/a/b.json")),
                         "/a/b.safetensors")

    def test_missing_probe_file_is_rejected(self):
        with self.assertRaisesRegex(capture_parity.CaptureError, "not found"):
            capture_parity.load_probe(os.path.join(self.tmp, "nope.json"))

    def test_missing_companion_tensor_file_is_rejected(self):
        path = os.path.join(self.tmp, "probe.json")
        with open(path, "w") as f:
            json.dump(make_probe_dict(), f)
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "tensor file not found"):
            capture_parity.load_probe(path)

    def test_declared_input_missing_from_tensor_file_is_rejected(self):
        probe = make_probe_dict()
        probe["probes"][0]["inputs"] = ["input_ids", "pixel_values"]
        path = write_probe_files(self.tmp, probe)
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "pixel_values"):
            capture_parity.load_probe(path)

    def test_probe_declaring_no_inputs_is_rejected(self):
        probe = make_probe_dict()
        probe["probes"][0]["inputs"] = []
        path = write_probe_files(self.tmp, probe)
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "declares no 'inputs'"):
            capture_parity.load_probe(path)

    def test_probe_missing_required_field_is_rejected(self):
        probe = make_probe_dict()
        del probe["modules"]
        path = write_probe_files(self.tmp, probe)
        with self.assertRaisesRegex(capture_parity.CaptureError, "modules"):
            capture_parity.load_probe(path)

    def test_empty_probe_list_is_rejected(self):
        probe = make_probe_dict()
        probe["probes"] = []
        path = write_probe_files(self.tmp, probe)
        with self.assertRaisesRegex(capture_parity.CaptureError, "no probes"):
            capture_parity.load_probe(path)

    def test_malformed_json_is_rejected(self):
        path = os.path.join(self.tmp, "bad.json")
        with open(path, "w") as f:
            f.write("{not json")
        with self.assertRaisesRegex(capture_parity.CaptureError, "valid JSON"):
            capture_parity.load_probe(path)


class CaptureTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path
        self.cpu = torch.device("cpu")

    def test_capture_produces_expected_keys(self):
        tensors = capture_parity.run_capture(make_model(), make_probe_dict(),
                                             make_probe_tensors(), self.cpu)
        self.assertCountEqual(
            tensors.keys(),
            [
                "p0/logits",
                "p0/module/layers.0.mixer",
                "p0/module/layers.1.mixer",
            ],
        )
        self.assertEqual(tensors["p0/logits"].shape, (1, 5, VOCAB))
        self.assertEqual(tensors["p0/module/layers.0.mixer"].shape,
                         (1, 5, HIDDEN))

    def test_hooks_are_removed_after_capture(self):
        model = make_model()
        capture_parity.run_capture(model, make_probe_dict(),
                                   make_probe_tensors(), self.cpu)
        for name in MODULES:
            self.assertEmpty(model.get_submodule(name)._forward_hooks)

    def test_hooks_are_removed_even_when_capture_fails(self):
        model = make_model()
        with self.assertRaises(Exception):
            # Out of vocab range: the embedding lookup raises mid-pass.
            capture_parity.run_capture(model, make_probe_dict(),
                                       make_probe_tensors([999]), self.cpu)
        for name in MODULES:
            self.assertEmpty(model.get_submodule(name)._forward_hooks)

    def test_unknown_module_is_rejected(self):
        probe = make_probe_dict(modules=["layers.0.does_not_exist"])
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "does not exist"):
            capture_parity.run_capture(make_model(), probe,
                                       make_probe_tensors(), self.cpu)

    def test_module_that_never_runs_is_rejected(self):
        # A hooked module that the forward pass skips yields no tensor. Left
        # unchecked it would silently vanish from the comparison.
        model = make_model()
        model.unused = TinyMixer()
        probe = make_probe_dict(modules=["unused"])
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "did not run"):
            capture_parity.run_capture(model, probe, make_probe_tensors(),
                                       self.cpu)

    def test_multiple_probes_are_captured_separately(self):
        probe = make_probe_dict()
        probe["probes"].append({
            "id": "p1",
            "prompt": "other",
            "inputs": ["input_ids"],
            "n_tokens": 2,
        })
        probe_tensors = make_probe_tensors()
        probe_tensors["p1/input_ids"] = torch.tensor([[6, 7]],
                                                     dtype=torch.long)
        tensors = capture_parity.run_capture(make_model(), probe,
                                             probe_tensors, self.cpu)
        self.assertEqual(tensors["p0/logits"].shape, (1, 5, VOCAB))
        self.assertEqual(tensors["p1/logits"].shape, (1, 2, VOCAB))

    def test_captured_tensors_are_detached_and_on_cpu(self):
        tensors = capture_parity.run_capture(make_model(), make_probe_dict(),
                                             make_probe_tensors(), self.cpu)
        for tensor in tensors.values():
            self.assertFalse(tensor.requires_grad)
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(tensor.is_contiguous())

    @parameterized.named_parameters(
        ("tensor", torch.zeros(2), (2, )),
        ("tuple", (torch.zeros(3), None), (3, )),
        ("nested", [[torch.zeros(4)]], (4, )),
        ("dict", {
            "a": None,
            "b": torch.zeros(5)
        }, (5, )),
    )
    def test_first_tensor_digs_out_the_payload(self, value, shape):
        self.assertEqual(capture_parity.first_tensor(value).shape, shape)

    def test_first_tensor_returns_none_when_absent(self):
        self.assertIsNone(capture_parity.first_tensor({"a": "no tensors"}))


class ManifestTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.cpu = torch.device("cpu")

    def test_reference_manifest_marks_independence(self):
        manifest = capture_parity.build_manifest(capture_parity.REFERENCE,
                                                 make_probe_dict(), self.cpu,
                                                 torch.float32, True,
                                                 make_model())
        self.assertEqual(manifest["role"], "reference")
        self.assertEqual(manifest["dtype"], "float32")
        self.assertTrue(manifest["independent_reference"])
        self.assertFalse(manifest["evidence"]["patches_applied"])
        self.assertEqual(manifest["modules"], MODULES)

    def test_patched_reference_records_playbooks(self):
        patches = mock.MagicMock()
        patches.APPLIED_PLAYBOOKS = ["triton-to-pytorch-ops.md"]
        manifest = capture_parity.build_manifest(capture_parity.REFERENCE,
                                                 make_probe_dict(),
                                                 self.cpu,
                                                 torch.float32,
                                                 False,
                                                 make_model(),
                                                 patch_module=patches)
        self.assertFalse(manifest["independent_reference"])
        self.assertTrue(manifest["evidence"]["patches_applied"])
        self.assertEqual(manifest["applied_playbooks"],
                         ["triton-to-pytorch-ops.md"])

    def test_manifest_digest_matches_probe(self):
        probe = make_probe_dict()
        manifest = capture_parity.build_manifest(capture_parity.CANDIDATE,
                                                 probe, self.cpu,
                                                 torch.bfloat16, False,
                                                 make_model())
        self.assertEqual(manifest["probe_sha256"],
                         capture_parity.probe_digest(probe))


class HostGuardTest(parameterized.TestCase):

    def test_reference_refuses_a_cuda_host(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(capture_parity.CaptureError, "CUDA"):
                capture_parity.check_reference_host()

    def test_reference_accepts_a_cpu_only_host(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            capture_parity.check_reference_host()  # does not raise


class PatchModuleTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path

    def write_module(self, body, name="server_patches.py"):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(body)
        return path

    def test_apply_is_called_on_import(self):
        path = self.write_module("CALLED = []\n"
                                 "PATCHED_MODULES = ['layers.0.mixer']\n"
                                 "APPLIED_PLAYBOOKS = ['flash.md']\n"
                                 "def apply():\n"
                                 "    CALLED.append(True)\n")
        module = capture_parity.load_patch_module(path)
        self.assertEqual(module.CALLED, [True])
        self.assertEqual(module.APPLIED_PLAYBOOKS, ["flash.md"])

    def test_module_without_apply_is_rejected(self):
        path = self.write_module("PATCHED_MODULES = []\n")
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "does not define apply"):
            capture_parity.load_patch_module(path)

    def test_missing_module_is_rejected(self):
        with self.assertRaisesRegex(capture_parity.CaptureError, "not found"):
            capture_parity.load_patch_module(os.path.join(self.tmp, "nope.py"))


class EndToEndTest(parameterized.TestCase):
    """Drives main() with a stub loader, then feeds the result to compare."""

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path
        self.probe_path = write_probe_files(self.tmp)

    def capture(self, role, prefix, model, extra=None):
        out = os.path.join(self.tmp, prefix)
        argv = [
            "--role", role, "--probe", self.probe_path, "--out", out,
            "--device", "cpu", "--dtype", "float32"
        ]
        adapter = capture_parity.get_adapter("hf_pretrained")
        with mock.patch.object(adapter, "load", return_value=model):
            code = capture_parity.main(argv + (extra or []))
        return code, out

    def test_matching_runs_compare_as_pass(self):
        model = make_model()
        ref_code, ref = self.capture("reference", "reference", model)
        cand_code, cand = self.capture("candidate", "candidate", model)
        self.assertEqual(ref_code, 0)
        self.assertEqual(cand_code, 0)

        ref_manifest, ref_tensors = compare_parity.load_capture(ref)
        cand_manifest, cand_tensors = compare_parity.load_capture(cand)
        report = compare_parity.compare(ref_manifest, ref_tensors,
                                        cand_manifest, cand_tensors)
        self.assertEqual(report["verdict"], compare_parity.PASS)
        self.assertEqual(report["probes"][0]["logits"]["max_abs"], 0.0)

    def test_a_broken_rewrite_compares_as_fail(self):
        # The candidate's mixer is scaled, standing in for a rewrite that got
        # the math wrong.
        _, ref = self.capture("reference", "reference", make_model(scale=1.0))
        _, cand = self.capture("candidate", "candidate", make_model(scale=1.5))

        ref_manifest, ref_tensors = compare_parity.load_capture(ref)
        cand_manifest, cand_tensors = compare_parity.load_capture(cand)
        report = compare_parity.compare(ref_manifest, ref_tensors,
                                        cand_manifest, cand_tensors)
        self.assertEqual(report["verdict"], compare_parity.FAIL)
        self.assertNotEmpty(report["failures"])

    def test_patched_reference_yields_unverified(self):
        patch_path = os.path.join(self.tmp, "patches.py")
        with open(patch_path, "w") as f:
            f.write("APPLIED_PLAYBOOKS = ['triton-to-pytorch-ops.md']\n"
                    "def apply():\n    pass\n")
        model = make_model()
        _, ref = self.capture("reference", "reference", model,
                              ["--patches", patch_path])
        _, cand = self.capture("candidate", "candidate", model,
                               ["--patches", patch_path])

        ref_manifest, ref_tensors = compare_parity.load_capture(ref)
        cand_manifest, cand_tensors = compare_parity.load_capture(cand)
        report = compare_parity.compare(ref_manifest, ref_tensors,
                                        cand_manifest, cand_tensors)
        self.assertEqual(report["verdict"], compare_parity.UNVERIFIED)

    def test_mismatched_probes_are_caught_downstream(self):
        model = make_model()
        _, ref = self.capture("reference", "reference", model)

        # Same JSON, different token ids: only the tensor file differs.
        self.probe_path = write_probe_files(self.tmp,
                                            tensors=make_probe_tensors(
                                                [5, 4, 3, 2, 1]),
                                            name="probe2.json")
        _, cand = self.capture("candidate", "candidate", model)

        ref_manifest, ref_tensors = compare_parity.load_capture(ref)
        cand_manifest, cand_tensors = compare_parity.load_capture(cand)
        with self.assertRaisesRegex(compare_parity.CaptureError,
                                    "different inputs"):
            compare_parity.compare(ref_manifest, ref_tensors, cand_manifest,
                                   cand_tensors)

    def test_main_reports_bad_request(self):
        self.assertEqual(capture_parity.main(["--role", "reference"]), 2)

    def test_make_probe_writes_both_files(self):
        out = os.path.join(self.tmp, "made.json")
        adapter = capture_parity.get_adapter("hf_pretrained")
        with mock.patch.object(adapter, "tokenize", return_value=[5, 6, 7]):
            code = capture_parity.main([
                "--make_probe", "--model", "tiny/model", "--out", out,
                "--prompt", "hello", "--modules", "layers.0.mixer"
            ])
        self.assertEqual(code, 0)

        probe, tensors = capture_parity.load_probe(out)
        self.assertEqual(probe["probes"][0]["inputs"], ["input_ids"])
        self.assertEqual(probe["source"]["kind"], "hf_pretrained")
        self.assertEqual(tensors["p0/input_ids"].tolist(), [[5, 6, 7]])
        # The JSON names its inputs; the values live only in the tensor
        # file.
        self.assertNotIn("[5, 6, 7]", json.dumps(probe))

    def test_make_probe_accepts_explicit_ids_without_a_tokenizer(self):
        # The path a model with no Hugging Face tokenizer takes.
        out = os.path.join(self.tmp, "made.json")
        code = capture_parity.main([
            "--make_probe", "--model", "some/path", "--out", out,
            "--input_ids", "4,5,6", "--modules", "layers.0.mixer"
        ])
        self.assertEqual(code, 0)
        _, tensors = capture_parity.load_probe(out)
        self.assertEqual(tensors["p0/input_ids"].tolist(), [[4, 5, 6]])

    def test_make_probe_records_custom_code_settings(self):
        out = os.path.join(self.tmp, "made.json")
        code = capture_parity.main([
            "--make_probe", "--model", "/local/model/dir", "--out", out,
            "--input_ids", "1,2", "--modules", "m", "--trust_remote_code",
            "--auto_class", "AutoModel"
        ])
        self.assertEqual(code, 0)
        source = capture_parity.load_probe(out)[0]["source"]
        self.assertTrue(source["trust_remote_code"])
        self.assertEqual(source["auto_class"], "AutoModel")
        self.assertEqual(source["path"], "/local/model/dir")


GCS_MODEL = "gs://bucket/models/tiny"


class GcsCheckpointTest(parameterized.TestCase):
    """A gs:// model is copied to the job's disk with fetch_weights.py."""

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path
        self.copies = []

        def fetch_gcs(url, small_only=False):
            self.copies.append((url, small_only))
            return f"/local/copy/{'tokenizer' if small_only else 'full'}"

        self.fetch_weights = types.ModuleType("fetch_weights")
        self.fetch_weights.fetch_gcs = fetch_gcs

    def test_the_probe_copies_only_the_tokenizer_and_names_the_gcs_folder(
            self):
        out = os.path.join(self.tmp, "made.json")
        adapter = capture_parity.get_adapter("hf_pretrained")
        with mock.patch.dict(sys.modules,
                             {"fetch_weights": self.fetch_weights}), \
                mock.patch.object(adapter, "tokenize",
                                  return_value=[5, 6]) as tokenize:
            code = capture_parity.main([
                "--make_probe", "--model", GCS_MODEL, "--out", out, "--prompt",
                "hi", "--modules", "m"
            ])
        self.assertEqual(code, 0)
        self.assertEqual(self.copies, [(GCS_MODEL, True)])
        # Tokenized from the local copy of the tokenizer...
        self.assertEqual(tokenize.call_args.args[0]["path"],
                         "/local/copy/tokenizer")
        # ...but the probe names the folder, so each capture copies it.
        probe = capture_parity.load_probe(out)[0]
        self.assertEqual(probe["model_id"], GCS_MODEL)
        self.assertEqual(probe["source"]["path"], GCS_MODEL)

    def test_a_capture_loads_its_own_full_copy(self):
        probe = make_probe_dict()
        probe["model_id"] = GCS_MODEL
        probe["source"]["path"] = GCS_MODEL
        probe_path = write_probe_files(self.tmp, probe=probe)
        out = os.path.join(self.tmp, "reference")
        adapter = capture_parity.get_adapter("hf_pretrained")
        with mock.patch.dict(sys.modules,
                             {"fetch_weights": self.fetch_weights}), \
                mock.patch.object(adapter, "load",
                                  return_value=make_model()) as load:
            code = capture_parity.main([
                "--role", "reference", "--probe", probe_path, "--out", out,
                "--device", "cpu", "--dtype", "float32"
            ])
        self.assertEqual(code, 0)
        self.assertEqual(self.copies, [(GCS_MODEL, False)])
        self.assertEqual(load.call_args.args[0]["path"], "/local/copy/full")
        # The manifest still names the gs:// folder, as the probe does.
        manifest, _ = compare_parity.load_capture(out)
        self.assertEqual(manifest["source"]["path"], GCS_MODEL)

    def test_without_fetch_weights_the_job_says_what_to_pass(self):
        out = os.path.join(self.tmp, "made.json")
        with mock.patch.dict(sys.modules, {"fetch_weights": None}):
            with self.assertRaisesRegex(capture_parity.CaptureError,
                                        "fetch_weights.py"):
                capture_parity.local_checkpoint(GCS_MODEL)
            self.assertEqual(
                capture_parity.main([
                    "--make_probe", "--model", GCS_MODEL, "--out", out,
                    "--prompt", "hi", "--modules", "m"
                ]), 2)

    def test_a_failed_copy_is_a_bad_request_not_a_traceback(self):

        def fetch_gcs(url, small_only=False):
            raise PermissionError(13, f"no permission to read {url}")

        self.fetch_weights.fetch_gcs = fetch_gcs
        with mock.patch.dict(sys.modules,
                             {"fetch_weights": self.fetch_weights}):
            with self.assertRaisesRegex(capture_parity.CaptureError,
                                        "no permission to read"):
                capture_parity.local_checkpoint(GCS_MODEL)

    def test_a_hub_id_or_local_dir_is_not_copied(self):
        out = os.path.join(self.tmp, "made.json")
        with mock.patch.dict(sys.modules,
                             {"fetch_weights": self.fetch_weights}):
            capture_parity.main([
                "--make_probe", "--model", "/local/model/dir", "--out", out,
                "--input_ids", "1,2", "--modules", "m"
            ])
        self.assertEqual(self.copies, [])


class AdapterSeamTest(parameterized.TestCase):
    """A source other than Hugging Face must work without touching the flow."""

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path
        self.original = dict(capture_parity._ADAPTERS)

    def tearDown(self):
        capture_parity._ADAPTERS.clear()
        capture_parity._ADAPTERS.update(self.original)
        super().tearDown()

    def test_unknown_source_kind_is_rejected(self):
        with self.assertRaisesRegex(capture_parity.CaptureError,
                                    "no adapter registered"):
            capture_parity.get_adapter("does_not_exist")

    def test_a_custom_adapter_drives_a_full_capture(self):

        class CheckpointAdapter(capture_parity.ModelAdapter):
            """Stands in for a source that is not Hugging Face at all."""

            kind = "local_checkpoint"

            def load(self, source, dtype, device):
                del source, dtype
                return make_model().to(device)

            def make_inputs(self, entry, tensors, device):
                return {
                    name: tensors[f"{entry['id']}/{name}"].to(device)
                    for name in entry["inputs"]
                }

            def extract_logits(self, outputs):
                return outputs.logits

        capture_parity.register_adapter(CheckpointAdapter())

        probe = make_probe_dict()
        probe["source"] = {
            "kind": "local_checkpoint",
            "path": "/models/weights.pt"
        }
        probe_path = write_probe_files(self.tmp, probe)

        out = os.path.join(self.tmp, "reference")
        code = capture_parity.main([
            "--role", "reference", "--probe", probe_path, "--out", out,
            "--device", "cpu", "--dtype", "float32"
        ])
        self.assertEqual(code, 0)

        manifest, tensors = compare_parity.load_capture(out)
        self.assertEqual(manifest["source"]["kind"], "local_checkpoint")
        self.assertIn("p0/logits", tensors)
        self.assertIn("p0/module/layers.0.mixer", tensors)

    def test_probe_without_source_or_model_id_is_rejected(self):
        probe = make_probe_dict()
        del probe["source"]
        del probe["model_id"]
        path = write_probe_files(self.tmp, probe)
        with self.assertRaisesRegex(capture_parity.CaptureError, "source"):
            capture_parity.load_probe(path)

    def test_legacy_probe_without_source_still_loads(self):
        probe = make_probe_dict()
        del probe["source"]
        self.assertEqual(capture_parity.resolve_source(probe), {
            "kind": "hf_pretrained",
            "path": "tiny/model"
        })


class NonTokenInputsTest(parameterized.TestCase):
    """Inputs that are not token ids travel the same single path."""

    def setUp(self):
        super().setUp()
        self.tmp = self.create_tempdir().full_path
        self.cpu = torch.device("cpu")
        self.adapter = capture_parity.get_adapter("hf_pretrained")

    def test_float_inputs_survive_byte_for_byte(self):
        # The reason every input goes in the tensor file: no decimal
        # round-trip sits between what was written and what is fed in.
        pixels = torch.rand((1, 3, 4, 4))
        probe = make_probe_dict()
        probe["probes"][0]["inputs"] = ["pixel_values"]
        path = write_probe_files(self.tmp, probe, {"p0/pixel_values": pixels})

        _, tensors = capture_parity.load_probe(path)
        self.assertTrue(torch.equal(tensors["p0/pixel_values"], pixels))

    def test_several_named_inputs_are_passed_through(self):
        entry = {"id": "p0", "inputs": ["input_ids", "attention_mask"]}
        tensors = {
            "p0/input_ids": torch.tensor([[1, 2]]),
            "p0/attention_mask": torch.tensor([[1, 0]]),
        }
        inputs = self.adapter.make_inputs(entry, tensors, self.cpu)
        self.assertCountEqual(inputs, ["input_ids", "attention_mask"])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 0]])

    def test_dtype_comes_from_the_tensor_file(self):
        entry = {"id": "p0", "inputs": ["pixel_values"]}
        tensors = {"p0/pixel_values": torch.zeros((1, 2), dtype=torch.float16)}
        inputs = self.adapter.make_inputs(entry, tensors, self.cpu)
        self.assertEqual(inputs["pixel_values"].dtype, torch.float16)


if __name__ == "__main__":
    absltest.main()
