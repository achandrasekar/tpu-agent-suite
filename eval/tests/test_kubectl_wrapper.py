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
"""Unit tests for eval/kubectl_wrapper.py per-trial resource isolation."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_WRAPPER = _REPO / "eval" / "kubectl_wrapper.py"
_TRIAL_A = "trial-alpha-111"
_TRIAL_B = "trial-beta-222"


class TestKubectlWrapper(unittest.TestCase):
    """Tests Rules 1-4 and cluster capacity redaction in kubectl_wrapper.py."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.trial_id_file = self.tmp / "trial_id"
        self.trial_id_file.write_text(_TRIAL_A + "\n", encoding="utf-8")

        self.kubeconfig_file = self.tmp / "kubeconfig"
        self.kubeconfig_file.write_text("apiVersion: v1\n", encoding="utf-8")

        self.adc_file = self.tmp / "adc.json"
        self.adc_file.write_text("{}", encoding="utf-8")

        self.state_file = self.tmp / "cluster_state.json"
        self.calls_file = self.tmp / "calls.jsonl"
        self.kubectl_real = self.tmp / "kubectl-real"
        self._write_fake_kubectl()

    def _write_fake_kubectl(self) -> None:
        """Write a fake kubectl-real script that records calls and simulates cluster state."""
        script = f"""#!{sys.executable}
import json, os, sys
from pathlib import Path

state_file = Path({str(self.state_file)!r})
calls_file = Path({str(self.calls_file)!r})

argv = sys.argv[1:]
stdin_text = sys.stdin.read() if not sys.stdin.isatty() else ""

with calls_file.open("a", encoding="utf-8") as f:
    f.write(json.dumps({{
        "argv": argv,
        "stdin": stdin_text,
        "kubeconfig": os.environ.get("KUBECONFIG"),
        "adc": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    }}) + "\\n")

state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {{}}
resources = state.get("resources", {{}})

def norm_kind(k):
    k = k.lower()
    return {{"pods": "pod", "po": "pod", "jobs": "job", "configmaps": "configmap", "cm": "configmap", "services": "service", "svc": "service"}}.get(k, k.rstrip("s"))

if "--dry-run=client" in argv and "-o" in argv and "json" in argv:
    # Simulate `kubectl create configmap <name> --dry-run=client -o json`
    kind = norm_kind(argv[1]).capitalize()
    name = argv[2]
    print(json.dumps({{"apiVersion": "v1", "kind": kind, "metadata": {{"name": name}}}}))
    sys.exit(0)

if argv[:3] == ["get", "pods", "--all-namespaces"] and "-o" in argv and "json" in argv:
    print(json.dumps({{"apiVersion": "v1", "kind": "PodList", "items": state.get("pod_list", [])}}))
    sys.exit(0)

if argv and argv[0] == "get" and any(a.startswith("jsonpath=") for a in argv):
    kind = norm_kind(argv[1])
    name = argv[2]
    key = f"{{kind}}/{{name}}"
    if key not in resources:
        sys.stderr.write(f'Error from server (NotFound): {{kind}}s "{{name}}" not found\\n')
        sys.exit(1)
    label = resources[key].get("eval-trial-id", "")
    sys.stdout.write(label)
    sys.exit(0)

print("OK " + " ".join(argv))
sys.exit(0)
"""
        self.kubectl_real.write_text(script, encoding="utf-8")
        self.kubectl_real.chmod(self.kubectl_real.stat().st_mode
                                | stat.S_IXUSR)

    def _set_state(
        self,
        resources: dict[str, dict[str, str]] | None = None,
        pod_list: list[dict] | None = None,
    ) -> None:
        self.state_file.write_text(
            json.dumps({
                "resources": resources or {},
                "pod_list": pod_list or [],
            }),
            encoding="utf-8",
        )

    def _calls(self) -> list[dict]:
        if not self.calls_file.exists():
            return []
        return [
            json.loads(line) for line in self.calls_file.read_text(
                encoding="utf-8").splitlines() if line.strip()
        ]

    def _run(
            self,
            *args: str,
            stdin_text: str | None = None) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "KUBE_GATE_TRIAL_ID_FILE": str(self.trial_id_file),
            "KUBE_GATE_KUBECONFIG": str(self.kubeconfig_file),
            "KUBE_GATE_ADC": str(self.adc_file),
            "KUBE_GATE_KUBECTL_REAL": str(self.kubectl_real),
        }
        return subprocess.run(
            [sys.executable, str(_WRAPPER), *args],
            input=stdin_text,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def _forwarded(self) -> list[list[str]]:
        """argv of every `kubectl-real` call except ownership lookups."""
        return [
            c["argv"] for c in self._calls()
            if not any(a.startswith("jsonpath=") for a in c["argv"])
        ]

    # ------------------------------------------------------------------
    # Rule 1: Read-Only Cluster Metadata
    # ------------------------------------------------------------------

    def test_rule1_passthrough_and_kubeconfig_override(self) -> None:
        for cmd in (
            ["version", "--client"],
            ["config", "current-context"],
            ["get", "nodes", "-L", "cloud.google.com/gke-tpu-topology"],
            ["get", "localqueues", "-n", "priority-dev"],
            ["describe", "node", "gke-tpu-node-1"],
        ):
            self.calls_file.unlink(missing_ok=True)
            res = self._run(*cmd)
            self.assertEqual(res.returncode, 0, res.stderr)
            calls = self._calls()
            self.assertEqual([c["argv"] for c in calls], [cmd])
            self.assertEqual(calls[0]["kubeconfig"], str(self.kubeconfig_file))
            self.assertEqual(calls[0]["adc"], str(self.adc_file))

        # A caller-supplied --kubeconfig is dropped.
        self.calls_file.unlink(missing_ok=True)
        self._run("--kubeconfig=/tmp/evil", "get", "nodes", "--kubeconfig",
                  "/tmp/evil2")
        self.assertEqual(self._calls()[0]["argv"], ["get", "nodes"])

    # ------------------------------------------------------------------
    # Rule 2: Create / Apply
    # ------------------------------------------------------------------

    def test_rule2_apply_labels_resources_and_pod_templates(self) -> None:
        # Empty `labels:` / `metadata:` load as None; a CronJob nests its
        # templates under `jobTemplate`.
        self._set_state(resources={})
        manifest = self.tmp / "m.yaml"
        manifest.write_text(
            "kind: Job\nmetadata:\n  name: j\n  labels:\nspec:\n  template:\n"
            "    metadata:\n      labels: {app: bench}\n"
            "---\n"
            "kind: Job\nmetadata: {name: j2}\nspec:\n  template:\n"
            "    metadata:\n"
            "---\n"
            "kind: CronJob\nmetadata: {name: c}\nspec:\n  jobTemplate:\n"
            "    spec:\n      template: {spec: {}}\n",
            encoding="utf-8")
        res = self._run("apply", "-f", str(manifest))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self._forwarded(), [["apply", "-f", "-"]])
        job, job2, cron = [
            json.loads(d) for d in self._calls()[-1]["stdin"].split("\n---\n")
        ]
        labelled = [
            job, job["spec"]["template"], job2, job2["spec"]["template"], cron,
            cron["spec"]["jobTemplate"],
            cron["spec"]["jobTemplate"]["spec"]["template"]
        ]
        for obj in labelled:
            self.assertEqual(obj["metadata"]["labels"]["eval-trial-id"],
                             _TRIAL_A)
        self.assertEqual(job["spec"]["template"]["metadata"]["labels"]["app"],
                         "bench")

    def test_rule2_apply_checks_existing_owner(self) -> None:
        self._set_state(
            resources={
                "job/mine": {
                    "eval-trial-id": _TRIAL_A
                },
                "job/theirs": {
                    "eval-trial-id": _TRIAL_B
                },
            })
        res = self._run("apply",
                        "-f",
                        "-",
                        stdin_text='{"kind": "Job", '
                        '"metadata": {"name": "theirs"}}')
        self.assertEqual(res.returncode, 1)
        self.assertIn("belongs to another trial", res.stderr)
        self.assertEqual(self._forwarded(), [])

        res = self._run("apply",
                        "-f",
                        "-",
                        stdin_text='{"kind": "Job", '
                        '"metadata": {"name": "mine"}}')
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self._forwarded(), [["apply", "-f", "-"]])

    def test_rule2_imperative_create_and_run(self) -> None:
        self._set_state(resources={})
        create = ("create", "configmap", "my-cm", "--from-file=p=/tmp/foo",
                  "-n", "default")
        # The plain create, and the dry run agents pipe into `kubectl apply`:
        # the dry run must stay a dry run.
        for extra, expected_tail in (
            ((), []),
            (("--dry-run=client", "-o", "yaml"),
             ["--dry-run=client", "-o", "yaml"]),
        ):
            self.calls_file.unlink(missing_ok=True)
            res = self._run(*create, *extra)
            self.assertEqual(res.returncode, 0, res.stderr)
            last = self._calls()[-1]
            self.assertEqual(last["argv"],
                             ["create", "-n", "default", "-f", "-"] +
                             expected_tail)
            self.assertEqual(
                json.loads(
                    last["stdin"])["metadata"]["labels"]["eval-trial-id"],
                _TRIAL_A)

        self.calls_file.unlink(missing_ok=True)
        self._run("run", "p", "--image=busybox", "--", "sleep", "10")
        self.assertEqual(self._calls()[-1]["argv"], [
            "run", "p", "--image=busybox", "-l", f"eval-trial-id={_TRIAL_A}",
            "--", "sleep", "10"
        ])

    # ------------------------------------------------------------------
    # Rule 3: List / Bulk Operations & `compute inspect` Redaction
    # ------------------------------------------------------------------

    def test_rule3_bulk_commands_get_trial_selector(self) -> None:
        sel = f"eval-trial-id={_TRIAL_A}"
        for cmd, expected in (
            (["get", "pods", "-n",
              "default"], ["get", "pods", "-n", "default", "-l", sel]),
            (["get", "pods", "-l",
              "job-name=x"], ["get", "pods", "-l", f"job-name=x,{sel}"]),
                # `-L` takes a value; it is not a pod name.
            (["get", "pods", "-L",
              "topo"], ["get", "pods", "-L", "topo", "-l", sel]),
                # For `logs`, `-f` is `--follow` and takes no value.
            (["logs", "-f", "-l",
              "app=x"], ["logs", "-f", "-l", f"app=x,{sel}"]),
                # `--all` and `-l` are mutually exclusive in kubectl.
            (["delete", "jobs", "--all"], ["delete", "jobs", "-l", sel]),
            (["wait", "--for=condition=Ready", "pod",
              "--all"], ["wait", "--for=condition=Ready", "pod", "-l", sel]),
        ):
            self.calls_file.unlink(missing_ok=True)
            res = self._run(*cmd)
            self.assertEqual(res.returncode, 0, f"{cmd}: {res.stderr}")
            self.assertEqual(self._calls()[0]["argv"], expected)

    def test_rule3_all_namespaces_pods_json_redacts_other_trials_pods(
            self) -> None:

        def pod(name: str, trial: str) -> dict:
            return {
                "metadata": {
                    "name": name,
                    "namespace": "default",
                    "labels": {
                        "eval-trial-id": trial
                    }
                },
                "spec": {
                    "nodeName":
                    "node-1",
                    "containers": [{
                        "name": "main",
                        "command": [f"{name}.py"],
                        "resources": {
                            "limits": {
                                "google.com/tpu": "4"
                            }
                        },
                    }],
                },
                "status": {
                    "phase": "Running"
                },
            }

        self._set_state(
            pod_list=[pod("own", _TRIAL_A),
                      pod("other", _TRIAL_B)])
        cmd = ["get", "pods", "--all-namespaces", "-o", "json"]
        res = self._run(*cmd)
        self.assertEqual(res.returncode, 0, res.stderr)
        # Run without `-l`, so `compute inspect` counts every pod's chips.
        self.assertEqual(self._calls()[0]["argv"], cmd)
        own, other = json.loads(res.stdout)["items"]
        self.assertEqual(own, pod("own", _TRIAL_A))
        # Another trial's pod keeps only its node, TPU limits and phase.
        self.assertEqual(
            other, {
                "metadata": {
                    "namespace": "default",
                    "name": "redacted"
                },
                "spec": {
                    "nodeName":
                    "node-1",
                    "containers": [{
                        "resources": {
                            "limits": {
                                "google.com/tpu": "4"
                            }
                        }
                    }],
                },
                "status": {
                    "phase": "Running"
                },
            })

    # ------------------------------------------------------------------
    # Rule 4: Named Resource Operations
    # ------------------------------------------------------------------

    def test_rule4_named_operations_allow_own_and_block_others(self) -> None:
        self._set_state(
            resources={
                f"{kind}/{owner}-{suffix}": {
                    "eval-trial-id": trial
                }
                for kind, suffix in (("pod", "pod"), ("job", "job"))
                for owner, trial in (("own", _TRIAL_A), ("other", _TRIAL_B))
            })
        for cmd in (
            ["get", "pod", "own-pod"],
            ["logs", "-f", "job/own-job"],
            ["exec", "own-pod", "--", "ls"],
            ["attach", "-i", "-c", "main", "own-pod"],
            ["cp", "default/own-pod:/tmp/out", "/tmp/local"],
            ["port-forward", "pod/own-pod", "8000:8000"],
            ["delete", "job", "own-job"],
        ):
            self.calls_file.unlink(missing_ok=True)
            res = self._run(*cmd)
            self.assertEqual(res.returncode, 0, f"{cmd}: {res.stderr}")
            self.assertEqual(self._calls()[-1]["argv"], cmd)

        for cmd in (
            ["describe", "job", "other-job"],
            ["logs", "-f", "job/other-job"],
            ["exec", "other-pod", "--", "ls"],
            ["attach", "-i", "-c", "main", "other-pod"],
            ["cp", "other-pod:/tmp/a", "/tmp/b"],
            ["delete", "job", "other-job"],
        ):
            self.calls_file.unlink(missing_ok=True)
            res = self._run(*cmd)
            self.assertEqual(res.returncode, 1, f"{cmd} should fail")
            self.assertIn("NotFound", res.stderr)
            self.assertEqual(self._forwarded(), [])

    def test_rule4_ignore_not_found_forwards_only_owned_targets(self) -> None:
        self._set_state(
            resources={
                "job/mine": {
                    "eval-trial-id": _TRIAL_A
                },
                "job/theirs": {
                    "eval-trial-id": _TRIAL_B
                },
            })
        # Nothing owned: exit 0 without forwarding.
        res = self._run("delete", "job", "theirs", "--ignore-not-found")
        self.assertEqual((res.returncode, res.stdout), (0, ""))
        self.assertEqual(self._forwarded(), [])

        # Mixed: only the owned target is forwarded.
        for sub in ("delete", "get"):
            self.calls_file.unlink(missing_ok=True)
            self._run(sub, "job", "mine", "theirs", "missing",
                      "--ignore-not-found", "-n", "default")
            self.assertEqual(
                self._forwarded(),
                [[sub, "job/mine", "--ignore-not-found", "-n", "default"]])

        # Same for targets read from a manifest.
        manifest = self.tmp / "jobs.yaml"
        manifest.write_text(
            "kind: Job\nmetadata: {name: mine}\n---\n"
            "kind: Job\nmetadata: {name: theirs}\n",
            encoding="utf-8")
        self.calls_file.unlink(missing_ok=True)
        self._run("delete", "-f", str(manifest), "--ignore-not-found")
        self.assertEqual(self._forwarded(),
                         [["delete", "job/mine", "--ignore-not-found"]])

    def test_rule4_name_check_reports_other_trials_name_as_taken(self) -> None:
        # The command `GkeJobDriver._name_unused` runs; empty stdout means
        # the name is free.
        name_check = ("get", "configmap,job", "bench", "--ignore-not-found",
                      "-o", "name")
        self._set_state(resources={"job/bench": {"eval-trial-id": _TRIAL_B}})
        res = self._run(*name_check)
        self.assertEqual((res.returncode, res.stdout), (0, "job/bench\n"))
        self.assertEqual(self._forwarded(), [])

        self._set_state(resources={})
        self.assertEqual(self._run(*name_check).stdout, "")

    def test_rule4_delete_from_stdin_forwards_manifest(self) -> None:
        # The wrapper reads stdin to find the targets, so it passes the same
        # text on; otherwise `kubectl-real` reads an empty stdin.
        self._set_state(resources={"job/mine": {"eval-trial-id": _TRIAL_A}})
        manifest = "kind: Job\nmetadata:\n  name: mine\n"
        self._run("delete", "-f", "-", stdin_text=manifest)
        last = self._calls()[-1]
        self.assertEqual((last["argv"], last["stdin"]),
                         (["delete", "-f", "-"], manifest))

    def test_disallowed_subcommand_is_rejected(self) -> None:
        for sub in ("edit", "replace", "proxy", "cordon", "drain", "taint"):
            res = self._run(sub, "node-1")
            self.assertEqual(res.returncode, 1)
            self.assertIn("Forbidden", res.stderr)


if __name__ == "__main__":
    unittest.main()
