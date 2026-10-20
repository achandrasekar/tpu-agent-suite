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
"""Contract enforcement through the hooks, against a real installed package.

The unit tests in `test_runtime.py` call the checkers directly, and the e2e
tests in `test_hooks_e2e.py` drive a real harness CLI and are skipped unless
`RUN_HARNESS_CLI_E2E=1`. Between those sits the layer that actually decides
whether a contract holds in production: `validate-delegate-request` and
`validate-subagent-result`, dispatched by the harness on every delegation.

This file covers that layer with no harness and no accelerator -- compile,
install into a temporary store, then hand `_builtin_validate_schema` the
canonical payloads the harness would give it.

It exists because a bug hid exactly here. `check_status_payload` was wired
into `validate_json_file` only, so a result claiming an outcome while carrying
none of its measurements passed the SubagentStop hook while the CLI validator
rejected it -- and `validate_json_file` is a call an agent chooses to make,
while the hook is the one that always fires. Every unit test passed throughout.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FRAMEWORK = REPO_ROOT / "framework"
PACKAGE = REPO_ROOT / "package"

RUN_ID = "20260923T101500Z-abcdef123456"
ARTIFACT = "artifacts/optimized.py"


class TestContractHooks(unittest.TestCase):
    """Delegation requests and subagent results, judged by the real hook."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls._tmp.name)
        cls.home = root / "home"
        workspace = root / "ws"
        workspace.mkdir(parents=True)
        dist = root / "dist"

        env = {**os.environ, "TPU_AGENT_SUITE_HOME": str(cls.home)}
        compile_res = subprocess.run(
            [
                sys.executable,
                str(FRAMEWORK / "compiler.py"), "translate",
                str(PACKAGE), "--harness", "claude-code", "--output",
                str(dist)
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        if compile_res.returncode != 0:
            raise unittest.SkipTest(
                f"compiler failed, cannot exercise hooks:\n{compile_res.stderr[-2000:]}"
            )

        install_res = subprocess.run(
            [
                sys.executable,
                str(FRAMEWORK / "installer.py"), "install",
                str(dist), "--destination",
                str(workspace), "--non-interactive", "--skip-harness-check",
                "--relocated-store"
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        if install_res.returncode != 0:
            raise unittest.SkipTest(
                f"installer failed, cannot exercise hooks:\n{install_res.stderr[-2000:]}"
            )

        runtimes = sorted(
            (cls.home / "packages").glob("*/framework/runtime.py"))
        if not runtimes:
            raise unittest.SkipTest("installer placed no runtime to dispatch")

        # Import the *installed* runtime, not the source tree's. The hook runs
        # from the installed copy in production, so a bug that survives the
        # build is only visible from there.
        spec = importlib.util.spec_from_file_location("installed_runtime",
                                                      runtimes[0])
        assert spec and spec.loader
        cls.runtime = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(runtimes[0].parent))
        spec.loader.exec_module(cls.runtime)

        cls.workspace = workspace
        cls._prev_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(cls.home)

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_prev_home", None) is None:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        else:
            os.environ["TPU_AGENT_SUITE_HOME"] = cls._prev_home
        cls._tmp.cleanup()

    def _request(self, agent: str, payload: dict | str) -> str:
        # A dict is encoded as the JSON payload a precise caller would send; a
        # str is passed through as the prose the kernel-worker actually dispatches.
        # Encoding the prose would wrap it in a JSON string literal, which the
        # hook reads as "no JSON object here" -- the opposite of the test.
        prompt = payload if isinstance(payload, str) else json.dumps(payload)
        decision = self.runtime._builtin_validate_schema(
            "tpu-agent-suite", self.workspace, {
                "event": "PreToolUse",
                "tool": "delegate",
                "subagent": agent,
                "args": {
                    "prompt": prompt
                },
                "context": {
                    "run_id": RUN_ID
                },
            })
        return decision.get("decision", "allow")

    def _result(self, agent: str, payload: dict) -> str:
        decision = self.runtime._builtin_validate_schema(
            "tpu-agent-suite", self.workspace, {
                "event": "SubagentStop",
                "subagent": agent,
                "output": "```json\n" + json.dumps(payload) + "\n```",
            })
        return decision.get("decision", "allow")

    # -- the tpu-agent -> tpu-kernel-development boundary -----------------------------

    def test_kernel_request_refuses_a_prompt_that_names_no_kernel(self):
        """The loop improves on an existing kernel, so a file is not optional.

        A bare prompt used to validate, and the loop then fell back to
        `<work>/base.py` -- a file the caller never named and may not have
        written. The failure surfaced stages later, as a planner with nothing
        to read, or worse as a real speedup measured against the wrong kernel.
        """
        self.assertEqual(
            self._request("tpu-kernel-development", {
                "prompt": "speed up fa",
                "run_id": RUN_ID
            }), "deny")
        self.assertEqual(
            self._request(
                "tpu-kernel-development", {
                    "prompt": "speed up fa",
                    "run_id": RUN_ID,
                    "base_kernel_path": "work/base.py"
                }), "allow")

    def test_kernel_request_refuses_an_empty_base_kernel_path(self):
        """`minLength: 1` -- an empty string names no file either."""
        self.assertEqual(
            self._request("tpu-kernel-development", {
                "prompt": "speed up fa",
                "run_id": RUN_ID,
                "base_kernel_path": ""
            }), "deny")

    def test_kernel_request_accepts_the_typed_fields(self):
        self.assertEqual(
            self._request(
                "tpu-kernel-development", {
                    "prompt": "speed up fa",
                    "run_id": RUN_ID,
                    "base_kernel_path": "work/base.py",
                    "target_op": "flash attention",
                    "max_iterations": 8,
                    "atol": 0.001,
                    "rtol": 0.001,
                }), "allow")

    def test_kernel_request_rejects_a_field_the_loop_cannot_honour(self):
        """A budget the loop has no stop condition for must not validate.

        The schema is closed precisely so a caller cannot promise the user an
        exit the agent will never take.
        """
        self.assertEqual(
            self._request("tpu-kernel-development", {
                "prompt": "x",
                "time_budget_minutes": 30
            }), "deny")

    def test_kernel_request_requires_a_prompt(self):
        self.assertEqual(
            self._request("tpu-kernel-development", {"max_iterations": 5}),
            "deny")

    def test_kernel_request_rejects_a_malformed_run_id(self):
        """`run_id` is optional -- standalone mode allocates its own -- but a
        value that is present and malformed resolves to the wrong artifact
        root, so it fails here rather than silently."""
        self.assertEqual(
            self._request("tpu-kernel-development", {
                "prompt": "x",
                "run_id": "not-a-run-id"
            }), "deny")

    def test_kernel_result_accepts_an_honest_failure(self):
        self.assertEqual(
            self._result(
                "tpu-kernel-development", {
                    "status":
                    "failed",
                    "run_id":
                    RUN_ID,
                    "failure_reason":
                    "no_device",
                    "error_message":
                    "TPU unreachable",
                    "iterations": [{
                        "iteration": 1,
                        "compile_ok": False,
                        "test_ok": False
                    }],
                }), "allow")

    def test_kernel_result_accepts_a_complete_success(self):
        self.assertEqual(
            self._result(
                "tpu-kernel-development", {
                    "status": "optimized",
                    "run_id": RUN_ID,
                    "artifacts": [ARTIFACT],
                    "best_kernel_artifact": ARTIFACT,
                    "baseline_latency_ms": 10.0,
                    "best_latency_ms": 4.0,
                    "speedup_factor": 2.5,
                    "iterations_run": 5,
                }), "allow")

    def test_kernel_result_rejects_the_retired_status_vocabulary(self):
        self.assertEqual(
            self._result("tpu-kernel-development", {
                "status": "success",
                "run_id": RUN_ID
            }), "deny")

    # -- the rule that used to reach only the CLI ----------------------------

    def test_claimed_outcome_without_measurements_is_denied_at_the_hook(self):
        """The regression this file exists for.

        `artifacts` is non-empty and the schema itself is satisfied -- only
        the conditional payload rule can catch this.
        """
        self.assertEqual(
            self._result(
                "tpu-kernel-development", {
                    "status": "optimized",
                    "run_id": RUN_ID,
                    "artifacts": [ARTIFACT],
                    "summary": "went well",
                }), "deny")

    def test_an_optimized_kernel_must_point_at_a_durable_artifact(self):
        """The rule that renaming the status could have switched off silently.

        `check_completed_has_artifact` -- 'a completed result must carry an
        artifact' -- fires only on the literal status `completed`. Every
        measurement below is present; only the artifacts are missing or empty.
        """
        measured = {
            "status": "optimized",
            "run_id": RUN_ID,
            "best_kernel_artifact": ARTIFACT,
            "best_latency_ms": 1.83,
            "speedup_factor": 2.31,
            "iterations_run": 3,
        }
        for label, artifacts in (("omitted", None), ("empty", [])):
            with self.subTest(artifacts=label):
                payload = dict(measured)
                if artifacts is not None:
                    payload["artifacts"] = artifacts
                self.assertEqual(
                    self._result("tpu-kernel-development", payload), "deny")

        self.assertEqual(
            self._result("tpu-kernel-development", {
                **measured, "artifacts": [ARTIFACT]
            }), "allow")

    def test_stage_subagents_are_held_to_the_same_rule(self):
        """A stage schema has no `artifacts` property at all, so the
        completed-artifact rule never applies to it and the payload rule is
        the only guard."""
        cases = [
            ("analyze-cuda-reference", {
                "status": "summarized",
                "run_id": RUN_ID
            }),
            ("kernel-worker", {
                "status": "optimized",
                "run_id": RUN_ID,
                "iteration": 1
            }),
        ]
        for agent, payload in cases:
            with self.subTest(agent=agent):
                self.assertEqual(self._result(agent, payload), "deny")

    def test_stage_subagents_may_report_failure_without_a_payload(self):
        """The whole point: each of these was unrepresentable before."""
        cases = [
            ("analyze-cuda-reference", {
                "status": "failed",
                "run_id": RUN_ID,
                "failure_reason": "timeout"
            }),
            ("kernel-worker", {
                "status": "failed",
                "run_id": RUN_ID,
                "iteration": 3,
                "failure_reason": "compilation_error"
            }),
        ]
        for agent, payload in cases:
            with self.subTest(agent=agent):
                self.assertEqual(self._result(agent, payload), "allow")

    # -- every handoff in the tpu-kernel-development workflow -----------------------------

    # What each caller says. Every dispatch goes out as a JSON object carrying
    # `prompt` and `run_id` -- prose alone is denied, because a subagent never
    # sees SessionStart and the delegation is its only source for the run it
    # must write under. `DISPATCH_EXTRAS` adds whatever else a callee requires.
    WORKFLOW = [
        ("tpu-agent", "tpu-kernel-development",
         "Make the paged-attention Pallas kernel faster on v6e."),
        ("tpu-kernel-development", "kernel-worker",
         "Run one iteration of the loop."),
        ("tpu-kernel-development", "tpu-diagnostics",
         "Analyse the XProf trace at /w/iter1/trace; write /w/iter1/p.md."),
        ("kernel-worker", "analyze-cuda-reference",
         "Analyze the reference kernel at /w/flash.cu for /w (state file:"
         " /w/state.json)."),
    ]

    # Fields a callee requires beyond `prompt` and `run_id`.
    DISPATCH_EXTRAS = {
        "tpu-kernel-development": {
            "base_kernel_path": "work/base.py"
        },
        "kernel-worker": {
            "work_dir": f"/runs/{RUN_ID}/artifacts/tpu-kernel-development/k01"
        },
    }

    def _dispatch(self, callee: str, prompt: str) -> dict:
        payload = {"prompt": prompt, "run_id": RUN_ID}
        payload.update(self.DISPATCH_EXTRAS.get(callee, {}))
        return payload

    def test_every_workflow_handoff_is_accepted_as_dispatched(self):
        """Each dispatch, shaped as the instructions tell the caller to send it.

        These were denied for a long time. The delegation hook treats `prompt`
        as routing metadata and strips it unless the target schema declares it
        as a property -- so a schema without `prompt` was handed an empty
        object and failed on every required field it had.

        Declaring `prompt` fixed that; dropping `run_id` from `required` went
        too far the other way. Every stage now requires both, so the callers
        send JSON, and this asserts each edge is still reachable first time.
        """
        for caller, callee, prompt in self.WORKFLOW:
            with self.subTest(edge=f"{caller} -> {callee}"):
                self.assertEqual(
                    self._request(callee, self._dispatch(callee, prompt)),
                    "allow")

    def test_a_dispatch_without_run_id_is_denied(self):
        """The regression this pass fixes.

        Six result schemas require `run_id`. A subagent never sees
        SessionStart, so the delegation is its only source for the value:
        dispatched without one it cannot write an artifact under the right run
        and cannot return a valid result. The denial has to land on the caller,
        which holds the run_id and can resend, rather than on the subagent that
        can no longer recover it.
        """
        for caller, callee, prompt in self.WORKFLOW:
            payload = self._dispatch(callee, prompt)
            payload.pop("run_id")
            accepts = self.contract_of(callee).get("accepts") or ""
            if not accepts.endswith(".json"):
                continue  # no request schema -- nothing to enforce
            with self.subTest(edge=f"{caller} -> {callee}"):
                self.assertEqual(self._request(callee, payload), "deny")

    def test_every_request_schema_declares_prompt(self):
        """The property that makes a prose delegation survive the hook.

        Only for callees that declare one; an agent with no request schema has
        nothing for the hook to strip a `prompt` out of, and is reached either
        way, which `test_every_workflow_handoff_is_accepted_as_dispatched`
        covers.

        `prompt` must be declared and required everywhere. Beyond it, a stage
        requires nothing, so a caller with only prose still gets through --
        except `tpu-kernel-development`, which also requires `base_kernel_path` because the
        loop has no kernel to optimize without one. That is a deliberate
        exception, so it is named here rather than left to drift.
        """
        schemas_dir = (pathlib.Path(__file__).resolve().parents[2] /
                       "package" / "schemas")
        beyond = {
            "tpu-kernel-development": ["base_kernel_path"],
            "kernel-worker": ["work_dir"],
        }
        checked = 0
        for _, callee, _ in self.WORKFLOW:
            accepts = self.contract_of(callee).get("accepts") or ""
            if not accepts.endswith(".json"):
                continue
            checked += 1
            with self.subTest(callee=callee):
                schema = json.loads(
                    (schemas_dir /
                     accepts.split("/")[-1]).read_text(encoding="utf-8"))
                self.assertIn("prompt", schema["properties"])
                self.assertEqual(schema["required"],
                                 ["prompt", "run_id"] + beyond.get(callee, []))
        self.assertTrue(checked, "no callee declares a request schema")

    def contract_of(self, agent: str) -> dict:
        """The built contract for an agent, as the harness resolves it."""
        build = json.loads(
            next((self.home /
                  "packages").glob("*/tpu-agent-suite-build.json")).read_text(
                      encoding="utf-8"))
        return build["contracts"][agent]

    def test_a_richer_typed_request_still_validates(self):
        """Relaxing `required` must not stop a caller from being precise."""
        self.assertEqual(
            self._request(
                "kernel-worker", {
                    "prompt":
                    "Run one iteration of the loop.",
                    "run_id":
                    RUN_ID,
                    "iteration":
                    1,
                    "base_kernel_path":
                    "work/base.py",
                    "work_dir":
                    f"/runs/{RUN_ID}/artifacts/tpu-kernel-development/k01",
                }), "allow")

    def test_relaxed_requests_still_reject_bad_payloads(self):
        """`required: [prompt]` is not `anything goes`."""
        for label, payload in [
            ("unknown field", {
                "prompt": "x",
                "bogus_field": 1
            }),
            ("malformed run_id", {
                "prompt": "x",
                "run_id": "not-a-run-id"
            }),
            ("no prompt at all", {
                "iteration": 1
            }),
        ]:
            with self.subTest(case=label):
                self.assertEqual(self._request("kernel-worker", payload),
                                 "deny")

    # -- the shared tpu-diagnostics boundary ------------------------------------

    def test_profiling_request_from_each_caller_is_accepted(self):
        """Callers ask for different quantities; the open `fields` takes both."""
        for caller, fields in (
            ("tpu-sharding", ["duty_cycle_percent", "tc_idle_ms_average"]),
            ("tpu-kernel-development",
             ["alu_utilization_percent", "dma_sync_wait_to_compute_ratio"]),
        ):
            with self.subTest(caller=caller):
                self.assertEqual(
                    self._request(
                        "tpu-diagnostics", {
                            "prompt": "Analyse the trace.",
                            "run_id": RUN_ID,
                            "traces": [{
                                "path": "/w/iter1/trace"
                            }],
                            "report_path": "/w/iter1/p.md",
                            "fields": fields,
                            "top_ops": 30,
                        }), "allow")

    def test_profiling_request_for_a_comparison_is_accepted(self):
        """Two traces in one call -- the xprof-compare shape one path lost."""
        self.assertEqual(
            self._request(
                "tpu-diagnostics", {
                    "prompt":
                    "Which run has the shorter step?",
                    "run_id":
                    RUN_ID,
                    "traces": [{
                        "path": "/p/a",
                        "label": "A"
                    }, {
                        "path": "/p/b",
                        "label": "B"
                    }],
                    "workload":
                    "/w/variant.py",
                }), "allow")

    def test_profiling_refuses_prose_without_a_run_id(self):
        """The regression a contract brings: bare prose used to pass.

        With no `accepts` the hook allowed anything. Now prose is validated as
        `{prompt}` and denied for the missing `run_id`, which is why every
        caller's instructions send a JSON object instead.
        """
        self.assertEqual(
            self._request("tpu-diagnostics",
                          "Analyse the XProf trace at /w/iter1/trace."),
            "deny")

    def test_profiling_result_passes_xprof_output_through(self):
        """XProf's own strings and shapes are not the framework's to constrain."""
        self.assertEqual(
            self._result(
                "tpu-diagnostics", {
                    "status":
                    "reported",
                    "summary":
                    "read in 143s",
                    "run_id":
                    RUN_ID,
                    "profiling_report_artifact":
                    "/w/iter1/p.md",
                    "xprof_version":
                    "xprof 2.24.0",
                    "traces": [{
                        "path": "/w/iter1/trace",
                        "captured": True,
                        "metrics": {
                            "peak_hbm_gib": {
                                "value": "N/A",
                                "source_tool": "get_kpi_metrics"
                            },
                            "alu_utilization_percent": {
                                "value": None,
                                "source_tool": "get_llo_analysis",
                                "tool_status": "UNAVAILABLE",
                                "tool_reason": "LLO_DATA_ABSENT",
                            },
                        },
                    }],
                }), "allow")

    def test_profiling_success_without_a_report_is_denied(self):
        """`status_requires` has to hold at the hook, not only in the CLI."""
        self.assertEqual(
            self._result(
                "tpu-diagnostics", {
                    "status": "reported",
                    "summary": "done",
                    "run_id": RUN_ID,
                    "xprof_version": "xprof 2.24.0",
                    "traces": [{
                        "path": "/w/iter1/trace",
                        "captured": False
                    }],
                }), "deny")

    def test_profiling_success_without_traces_is_denied(self):
        """A report that read nothing is not a report; [] counts as absent."""
        self.assertEqual(
            self._result(
                "tpu-diagnostics", {
                    "status": "reported",
                    "summary": "done",
                    "run_id": RUN_ID,
                    "profiling_report_artifact": "/w/iter1/p.md",
                    "xprof_version": "xprof 2.24.0",
                    "traces": [],
                }), "deny")

    def test_profiling_may_report_each_unhappy_path(self):
        for payload in (
            {
                "status": "invalid_input",
                "summary": "no traces, no workload",
                "run_id": RUN_ID
            },
            {
                "status": "needs_input",
                "summary": "workload builds no inputs",
                "run_id": RUN_ID
            },
            {
                "status": "failed",
                "summary": "read did not finish",
                "run_id": RUN_ID,
                "failure_reason": "read_timeout",
                "error_message": "killed after 3600s"
            },
        ):
            with self.subTest(status=payload["status"]):
                self.assertEqual(self._result("tpu-diagnostics", payload),
                                 "allow")

    def test_profiling_failure_without_a_reason_is_denied(self):
        self.assertEqual(
            self._result("tpu-diagnostics", {
                "status": "failed",
                "summary": "x",
                "run_id": RUN_ID
            }), "deny")


if __name__ == "__main__":
    unittest.main()
