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
"""GKE XPK cluster compute driver for multi-node TPU workloads."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
from typing import Any

from . import base, jobs, probe, registry, shell

Path = pathlib.Path

# How xpk reports workload state, mapped to the framework's vocabulary.
# Grouped by lifecycle category for readability and maintainability.
_STATUS_GROUPS: dict[str, tuple[str, ...]] = {
    "COMPLETED": ("SUCCESSFUL", "COMPLETE", "COMPLETED"),
    "FAILED": ("FAILED", "ERROR", "CRASHED"),
    "RUNNING": ("RUNNING", "ACTIVE"),
    # Kueue / JobSet non-terminal states holding or awaiting cluster quota.
    "PENDING": (
        "QUEUED",
        "PROVISIONING",
        "PENDING",
        "WAITING",
        "ADMITTED",
        "SUSPENDED",
        "PREEMPTED",
        "EVICTED",
        "QUOTA_RESERVED",
    ),
}
_XPK_STATUS: dict[str, str] = {
    raw: target
    for target, raws in _STATUS_GROUPS.items()
    for raw in raws
}

# States in which a workload still holds (or is queued for) cluster capacity.
_OCCUPYING_STATES = frozenset({"RUNNING", "PENDING"})

_TERMINAL_EXIT_CODES = {"COMPLETED": 0, "FAILED": 1}


def _row_tokens(line: str) -> list[str]:
    """Split an `xpk workload list` row into non-empty columns."""
    return [t.strip() for t in re.split(r"\s+|\|", line) if t.strip()]


def _is_header_row(tokens: list[str]) -> bool:
    """Report whether a row labels columns rather than naming a workload."""
    upper = [t.upper() for t in tokens]
    return "STATUS" in upper and any(t in ("NAME", "JOBSET", "WORKLOAD")
                                     for t in upper)


def _iter_workload_rows(stdout: str) -> list[list[str]]:
    """Return non-header token rows from `xpk workload list` output."""
    rows = []
    for line in (stdout or "").splitlines():
        tokens = _row_tokens(line)
        if tokens and not _is_header_row(tokens):
            rows.append(tokens)
    return rows


def _row_status(tokens: list[str]) -> str | None:
    """Read a workload's state from its columns.

    Statuses are matched as whole columns, never as substrings of the line: a
    workload named `pending-sweep` is not a pending workload. Where more than
    one column looks like a status, the last wins, since xpk prints the state
    to the right of the identifier columns.

    In Kueue, 'Finished' is the parent condition for both completed and failed
    workloads; if 'Finished' is present, the message is checked for failure
    keywords rather than unconditionally assuming success.
    """
    upper = [t.upper() for t in tokens]
    line_upper = " ".join(upper)

    if "FINISHED" in upper:
        if any(w in line_upper for w in ("FAIL", "ERROR", "CRASH")):
            return "FAILED"
        return "COMPLETED"

    found = None
    for token in tokens:
        status = _XPK_STATUS.get(token.upper())
        if status:
            found = status
    return found


def _gcs_root(bucket: str) -> str:
    """`gs://<bucket>[/prefix]` from a configured value, `""` when unset.

    Accepts the URI with or without its scheme and with a trailing slash,
    so the one place that reads the setting is the one place it is spelled.
    """
    cleaned = (bucket or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.startswith("gs://"):
        return cleaned
    return f"gs://{cleaned}"


def _gcs_bucket_check(uri: str) -> base.PreflightCheck:
    """Whether the bucket of `uri` (`gs://bucket[/prefix]`) exists, via gcloud."""
    bucket = uri.removeprefix("gs://").split("/", 1)[0]
    proc = base.probe_run([
        "gcloud", "storage", "buckets", "describe", f"gs://{bucket}",
        "--format=value(name)"
    ],
                          timeout=30)
    if proc.returncode == 0:
        return base.PreflightCheck("bucket", "ok", f"gs://{bucket}")
    error = base.probe_error(proc)
    lowered = error.lower()
    status = ("warn" if "403" in error or "forbidden" in lowered
              or "permission" in lowered else "fail")
    return base.PreflightCheck(
        "bucket", status, f"gs://{bucket}: {error}",
        f"Create it (`gcloud storage buckets create gs://{bucket}`) or set "
        "the bucket to one this account can use")


@registry.register_driver("gke_xpk")
class GkeXpkDriver(base.ComputeDriver):
    """Driver executing cluster-wide TPU workloads via XPK on GKE."""

    required_tools = ("xpk", "gcloud", *base.GKE_TOOLS)
    runs_in_container = True
    CHECK_FIELDS = {
        "cluster": ("cluster_name", "project"),
        "bucket": ("gcs_bucket", )
    }
    SETUP_FIELDS = base.SetupFields(
        required=("project", "zone", "cluster_name"),
        on_demand={
            "gcs_bucket": "collecting workload logs and artifacts from GCS",
            "python_interpreter": "a non-default Python inside the workload",
        },
    )

    def __init__(
        self,
        cluster_name: str,
        project: str = "",
        zone: str = "",
        gcs_bucket: str = "",
        python_interpreter: str | None = None,
    ) -> None:
        self.cluster_name = cluster_name
        self.project = project
        self.zone = zone
        self.gcs_bucket = _gcs_root(gcs_bucket)
        self.python_interpreter = python_interpreter or "python3"

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> GkeXpkDriver:
        for field, flag in (("cluster_name", "--cluster-name"),
                            ("zone", "--zone"), ("project", "--project")):
            if not params.get(field):
                raise base.ComputeConfigError(
                    f"Backend 'gke_xpk' requires '{field}' in the compute "
                    f"profile (`config set`) or via {flag}.",
                    error_code="MISSING_FIELD",
                )
        return cls(
            cluster_name=params["cluster_name"],
            project=params["project"],
            zone=params["zone"],
            gcs_bucket=params.get("gcs_bucket", ""),
            python_interpreter=params.get("python_interpreter"),
        )

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        cls.add_cli_arg(parser,
                        "--cluster-name",
                        help="GKE cluster name (gke_xpk)")
        cls.add_cli_arg(parser, "--zone", help="GCP zone of compute resource")
        cls.add_cli_arg(parser, "--project", help="GCP project ID")
        cls.add_cli_arg(parser,
                        "--gcs-bucket",
                        help="GCS bucket URI for artifacts and logs")

    def _xpk(self, *args: str) -> list[str]:
        """Build an xpk command carrying this driver's cluster coordinates."""
        cmd = ["xpk", *args, f"--cluster={self.cluster_name}"]
        if self.project:
            cmd.append(f"--project={self.project}")
        if self.zone:
            cmd.append(f"--zone={self.zone}")
        return cmd

    def _workload_for(self, job_id: str) -> tuple[jobs.JobRecord | None, str]:
        """Resolve the cluster workload name tracked for a job id."""
        record = jobs.JobRecord.load(job_id)
        if record and record.workload_name:
            return record, record.workload_name
        return record, job_id

    def _cluster_env(self) -> dict[str, str]:
        """Environment a recipe needs to address this cluster."""
        env = dict(os.environ)
        env["CLUSTER_NAME"] = self.cluster_name
        if self.project:
            env["PROJECT_ID"] = self.project
        if self.zone:
            env["ZONE"] = self.zone
        if self.gcs_bucket:
            env["BASE_OUTPUT_DIR"] = self.gcs_bucket
        return env

    def _runner_for(self, path: Path, python_interpreter: str | None) -> str:
        # As in shell.ShellDriver._runner_for.
        python = python_interpreter or self.python_interpreter or "python3"
        if path.suffix == ".sh":
            return f"TPU_AGENT_SUITE_PYTHON={python} bash"
        return python

    def describe_target(self) -> str:
        return (f"GKE cluster '{self.cluster_name}' via xpk (zone "
                f"{self.zone}, project {self.project})")

    def _preflight_target(self, checks: list[base.PreflightCheck]) -> bool:
        """An account to act as, and a RUNNING cluster where zone says."""
        checks.append(base.gcloud_account_check())
        if checks[-1].status == "fail":
            return False
        proc = base.probe_run([
            "gcloud", "container", "clusters", "list",
            f"--project={self.project}", f"--filter=name={self.cluster_name}",
            "--format=value(location,status)"
        ],
                              timeout=30)
        if proc.returncode != 0:
            return base.fail(
                checks, "cluster",
                f"could not list clusters in project {self.project}: "
                f"{base.probe_error(proc)}",
                "Check the profile's project, and that this account "
                "may list its GKE clusters")
        location, status = (proc.stdout.split() + ["", ""])[:2]
        if status != "RUNNING":
            return base.fail(
                checks, "cluster",
                f"cluster '{self.cluster_name}' is {status}" if status else
                f"no cluster '{self.cluster_name}' in project {self.project}",
                "Check the profile's cluster_name and project against "
                f"`gcloud container clusters list --project={self.project}`")
        checks.append(
            base.PreflightCheck("cluster", "ok", f"RUNNING in {location}"))
        if location not in (self.zone, self.zone.rsplit("-", 1)[0]):
            checks.append(
                base.PreflightCheck(
                    "zone", "warn",
                    f"the cluster is in {location}, but zone is {self.zone}",
                    "Set the profile's zone to the cluster's location "
                    "(`config set zone=...`)"))
        if self.gcs_bucket:
            checks.append(_gcs_bucket_check(self.gcs_bucket))
        return True

    def inspect(self) -> base.HardwareProfile:
        """Inspect cluster TPU type, topology, and workload occupancy."""
        try:
            proc = subprocess.run(self._xpk("cluster", "describe"),
                                  capture_output=True,
                                  text=True,
                                  timeout=30)
            if proc.returncode != 0:
                return self._unknown_profile("degraded",
                                             (proc.stderr or "").strip()
                                             or "xpk cluster describe failed")
            profile = self._profile_from_describe(proc.stdout)
            if profile.topology == "unknown" or profile.chips == 0 or profile.accelerator == "tpu":
                profile = self._enrich_profile_from_kubectl(profile)
            return profile
        except Exception as exc:  # noqa: BLE001 - reported in the profile
            return self._unknown_profile(f"error: {exc}", str(exc))

    def _enrich_profile_from_kubectl(
            self, profile: base.HardwareProfile) -> base.HardwareProfile:
        """Enrich profile with authoritative Kubernetes node labels if available."""
        try:
            cmd = [
                "kubectl",
                "get",
                "nodes",
                "-l",
                "cloud.google.com/gke-tpu-accelerator",
                "-o",
                "json",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=self._cluster_env(),
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return profile

            data = json.loads(proc.stdout)
            items = data.get("items", [])
            if not items:
                return profile

            total_chips = 0
            accel_name = profile.accelerator
            topology = profile.topology
            for node in items:
                labels = node.get("metadata", {}).get("labels", {})
                if accel_name in ("tpu", "unknown"):
                    accel_name = labels.get(
                        "cloud.google.com/gke-tpu-accelerator", accel_name)
                if topology == "unknown":
                    topology = labels.get("cloud.google.com/gke-tpu-topology",
                                          topology)

                allocatable = node.get("status", {}).get("allocatable", {})
                tpu_chips = int(allocatable.get("google.com/tpu", 0))
                total_chips += tpu_chips

            generation = probe.generation_for_name(accel_name) or accel_name
            spec = probe.spec_for(generation)
            chips = total_chips if total_chips > 0 else profile.chips
            available = (chips if profile.status == "healthy" else
                         profile.available_chips)

            return dataclasses.replace(
                profile,
                accelerator=generation,
                topology=topology,
                chips=chips,
                hbm_gb=spec["hbm_gb"]
                if spec["hbm_gb"] > 0 else profile.hbm_gb,
                available_chips=available,
            )
        except Exception:  # noqa: BLE001 - non-fatal fallback
            return profile

    def _profile_from_describe(self, stdout: str) -> base.HardwareProfile:
        """Build a profile from `xpk cluster describe` output.

        xpk prints for humans, so these are best-effort scrapes; every field
        falls back to an explicitly unknown value rather than a guess.
        """
        accelerator = _search(r"device[-_]type[:=\s]+([a-zA-Z0-9x-]+)", stdout,
                              "")
        if not accelerator:
            for match in re.finditer(r"^\s*\S+\s+\d+\s+([a-zA-Z0-9_-]+)",
                                     stdout, re.MULTILINE):
                candidate = match.group(1)
                if not any(
                        candidate.startswith(p)
                        for p in ("e2-", "n2-", "n1-", "c2-", "t2-")):
                    accelerator = candidate
                    break
            if not accelerator:
                accelerator = "tpu"

        topology = _search(r"topology[:=\s]+([0-9x]+)", stdout, "unknown")
        chips_text = _search(r"(?:chips|nodes|tpus)[:=\s]+([0-9]+)", stdout,
                             "")
        if chips_text:
            chips = int(chips_text)
        else:
            vms_match = re.search(r"contains\s+([0-9]+)\s+TPUVMs", stdout)
            if vms_match:
                vms = int(vms_match.group(1))
                chips = vms * (8 if ("v6" in accelerator
                                     or "6e" in accelerator) else 4)
            else:
                chips = 0

        generation = probe.generation_for_name(accelerator)
        spec = probe.spec_for(generation)

        occupying = self._occupying_workloads()
        if occupying is None:
            # Occupancy is unknown, which is not the same as "free". Leaving
            # available_chips unset keeps a scheduler from reading this as
            # spare capacity.
            status, available, held_by = "unknown", None, [
                "could not list cluster workloads"
            ]
        elif occupying:
            # xpk does not report per-workload chip counts, so how much of the
            # cluster is left cannot be derived from a workload count.
            status, available = "partially_busy", None
            held_by = [f"workload {name}" for name in occupying]
        else:
            status, available, held_by = "healthy", chips, []

        return base.HardwareProfile(
            accelerator=generation or accelerator,
            topology=topology,
            chips=chips,
            hbm_gb=spec["hbm_gb"],
            status=status,
            available_chips=available,
            held_by=held_by,
            details={
                "cluster": self.cluster_name,
                "project": self.project,
                "zone": self.zone,
                "accelerator_type": accelerator,
                "occupying_workloads": occupying or [],
            },
        )

    def _occupying_workloads(self) -> list[str] | None:
        """Names of workloads holding cluster capacity, or None if unlistable."""
        try:
            proc = subprocess.run(self._xpk("workload", "list"),
                                  capture_output=True,
                                  text=True,
                                  timeout=30)
        except Exception:  # noqa: BLE001 - reported as unknown occupancy
            return None
        if proc.returncode != 0:
            return None

        return [
            tokens[0] for tokens in _iter_workload_rows(proc.stdout)
            if _row_status(tokens) in _OCCUPYING_STATES
        ]

    def _unknown_profile(self, status: str,
                         detail: str) -> base.HardwareProfile:
        """Profile for a cluster whose state could not be determined."""
        return base.HardwareProfile(
            accelerator="tpu",
            topology="unknown",
            chips=0,
            hbm_gb=0.0,
            status=status,
            available_chips=None,
            held_by=[],
            details={
                "cluster": self.cluster_name,
                "project": self.project,
                "zone": self.zone,
                "error": detail,
            },
        )

    def run(
        self,
        code_path: str,
        timeout: int | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        inputs: list[base.StagedFile] | None = None,
        outputs: list[base.StagedFile] | None = None,
        job_name: str | None = None,
    ) -> base.SynchronousExecutionResult:
        del code_path, timeout, args, python_interpreter, inputs, outputs
        del job_name
        raise base.ComputeConfigError(
            "Backend 'gke_xpk' does not support synchronous `run`: cluster "
            "workloads are dispatched with `compute submit` and followed with "
            "`job status`, `job logs`, and `job artifacts`.",
            error_code="UNSUPPORTED_OPERATION",
        )

    def submit(
        self,
        recipe_path: str,
        job_name: str | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        run_id: str | None = None,
        inputs: list[base.StagedFile] | None = None,
        outputs: list[base.StagedFile] | None = None,
        serve: bool = False,
    ) -> str:
        """Run a recipe that creates an XPK workload; returns the job id."""
        if inputs or outputs:
            # The recipe creates the workload itself, so there is no pod spec
            # here to mount staged files into. Refusing is the only honest
            # answer: staging silently nothing leaves the recipe to die on a
            # missing path at runtime, long after the flag was accepted.
            raise base.ComputeConfigError(
                "Backend 'gke_xpk' does not support --input/--output file "
                "staging: the recipe creates the workload, so the files have "
                "to be staged by the recipe itself.",
                error_code="UNSUPPORTED_OPERATION",
            )
        src_path = shell.require_file(recipe_path, "recipe_path")
        workload_name = self._workload_name_for(job_name, src_path)
        record = self._new_record(job_name=job_name or workload_name,
                                  recipe_path=str(src_path),
                                  workload_name=workload_name,
                                  run_id=run_id or "",
                                  backend="gke_xpk",
                                  serve=serve,
                                  extra={"cluster_name": self.cluster_name})

        log_path = record.directory / "output.log"
        exit_code_path = record.directory / "exit_code.txt"
        record.log_path = str(log_path)

        env = self._cluster_env()
        # The recipe creates the workload itself; it must adopt this name so
        # status, log, artifact, and cancel lookups address the same workload.
        env["WORKLOAD_NAME"] = workload_name

        runner = self._runner_for(src_path, python_interpreter)
        quoted_args = " ".join(shlex.quote(a) for a in (args or []))
        launch_cmd = (f"({runner} {shlex.quote(str(src_path))} {quoted_args} "
                      f"> {shlex.quote(str(log_path))} 2>&1; "
                      f"echo $? > {shlex.quote(str(exit_code_path))})")

        proc = subprocess.Popen(["sh", "-c", launch_cmd],
                                start_new_session=True,
                                cwd=str(record.directory),
                                env=env)
        record.pid = proc.pid
        record.pid_started = jobs.process_started(proc.pid)
        record.status = "RUNNING"
        record.save()
        return record.job_id

    def _workload_name_for(self, job_name: str | None, src_path: Path) -> str:
        """Choose the cluster workload name this job will be tracked under."""
        timestamp = datetime.datetime.now(
            datetime.timezone.utc).strftime("%m%d%H%M%S")
        user = os.environ.get("USER", "tpuagent")[:8]
        workload_name = re.sub(r"[^a-z0-9-]", "",
                               (job_name
                                or f"{user}-{timestamp}").lower())[:28]
        if job_name:
            return workload_name

        # An unnamed job may still pin its own name inside the recipe; adopt it
        # so tracking follows the workload the cluster actually creates.
        try:
            match = re.search(r"--workload[=\s]+([a-zA-Z0-9_-]+)",
                              src_path.read_text(encoding="utf-8"))
        except OSError:
            return workload_name
        if match:
            parsed = match.group(1).strip()
            if not parsed.startswith("$"):
                return re.sub(r"[^a-z0-9-]", "", parsed.lower())[:28]
        return workload_name

    def get_status(self, job_id: str) -> base.AsynchronousExecutionResult:
        """Report workload state, with the cluster as the source of truth.

        The local recipe only *submits* the workload, so its exit code can
        prove a failed submission but never a finished job -- treating it as
        completion would report success while the workload is still queued.
        """
        record, workload_name = self._workload_for(job_id)
        details = {
            "cluster": self.cluster_name,
            "workload_name": workload_name
        }

        try:
            proc = subprocess.run(self._xpk(
                "workload", "list", f"--filter-by-job={workload_name}"),
                                  capture_output=True,
                                  text=True,
                                  timeout=30)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            return base.AsynchronousExecutionResult(job_id=job_id,
                                                    status="UNKNOWN",
                                                    error=str(exc),
                                                    details=details)

        target = workload_name.lower()
        for tokens in _iter_workload_rows(proc.stdout):
            # xpk prints both a jobset and a workload column; which one carries
            # the name varies, so match the name against any identifier column.
            if target not in [t.lower() for t in tokens]:
                continue
            status = _row_status(tokens)
            if not status:
                # The workload exists but reports a state this build does not
                # know. Say so, rather than reporting it as missing: an
                # unrecognised vocabulary is a different problem from a job
                # that was never created.
                return base.AsynchronousExecutionResult(
                    job_id=job_id,
                    status="UNKNOWN",
                    error=
                    f"unrecognised workload state in row: {' '.join(tokens)}",
                    details=details,
                )
            exit_code = _TERMINAL_EXIT_CODES.get(status)
            if record:
                record.mark(status, exit_code)
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status=status,
                exit_code=exit_code,
                start_time=record.start_time if record else None,
                end_time=record.end_time if record else None,
                details=details,
            )

        # No workload row yet. The recipe that creates the workload runs in the
        # background and can take tens of seconds (image staging, GKE calls),
        # so while it is alive the job is pending, not missing.
        if record and record.pid and jobs.process_is_alive(
                record.pid, record.pid_started):
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="PENDING",
                start_time=record.start_time,
                details=details,
            )

        # The recipe has exited. A non-zero code explains the absent workload.
        submission_code = self._submission_exit_code(record)
        if submission_code:
            if record:
                record.mark("FAILED", submission_code)
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="FAILED",
                exit_code=submission_code,
                error="recipe failed before the workload was created",
                details=details,
            )

        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status="UNKNOWN",
            error=f"workload '{workload_name}' not found on cluster "
            f"'{self.cluster_name}'",
            details=details,
        )

    @staticmethod
    def _submission_exit_code(record: jobs.JobRecord | None) -> int | None:
        """Return the recipe's exit code when it failed, else None."""
        if record is None:
            return None
        exit_code_file = record.directory / "exit_code.txt"
        if not exit_code_file.is_file():
            return None
        try:
            code = int(exit_code_file.read_text().strip())
        except (OSError, ValueError):
            return 1
        return code or None

    def fetch_logs(self, job_id: str) -> str:
        """Fetch workload logs from GCS, falling back to the recipe log."""
        record, workload_name = self._workload_for(job_id)

        if self.gcs_bucket:
            logs = self._gcs_logs(workload_name)
            if logs:
                return logs

        if record:
            log_path = record.directory / "output.log"
            if log_path.is_file():
                return log_path.read_text(encoding="utf-8")
        return ""

    def _gcs_logs(self, workload_name: str) -> str:
        """Read the workload's log from GCS.

        The filename is recipe-specific -- MaxText writes `train-0.log`, other
        recipes use `main.log`, `rank-0.log`, or their own convention -- so the
        prefix is listed and the first log read, rather than assuming a name.
        """
        prefix = f"{self.gcs_bucket}/{workload_name}/logs/"
        try:
            listing = subprocess.run(["gcloud", "storage", "ls", prefix],
                                     capture_output=True,
                                     text=True,
                                     timeout=60)
        except Exception:  # noqa: BLE001 - absence is not an error here
            return ""
        if listing.returncode != 0:
            return ""

        objects = [
            line.strip() for line in (listing.stdout or "").splitlines()
            if line.strip() and not line.strip().endswith("/")
        ]
        # Prefer an obvious log file, else fall back to the first object.
        logs = [o for o in objects if o.endswith((".log", ".txt"))] or objects
        for uri in sorted(logs):
            content = _gcloud_storage_cat(uri)
            if content:
                return content
        return ""

    def fetch_artifacts(self, job_id: str, dest_dir: Path) -> list[Path]:
        """Download the workload's TensorBoard traces from GCS."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not self.gcs_bucket:
            raise self.not_configured("gcs_bucket")

        _record, workload_name = self._workload_for(job_id)
        before = set(dest_dir.glob("**/*"))
        gcs_uri = f"{self.gcs_bucket}/{workload_name}/tensorboard"
        cmd = ["gcloud", "storage", "cp", "-r", gcs_uri, str(dest_dir)]
        try:
            proc = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  timeout=120)
        except Exception as exc:
            raise base.ComputeConfigError(
                f"Failed to fetch artifacts for job '{job_id}': {exc}",
                error_code="FETCH_FAILED",
            ) from exc
        if proc.returncode != 0:
            stderr = proc.stderr or ""
            if any(term in stderr.lower() for term in [
                    "not found", "no urls matched", "matched no objects",
                    "does not exist", "404"
            ]):
                return []
            raise base.ComputeConfigError(
                f"Failed to fetch artifacts for job '{job_id}' from {gcs_uri}: {stderr.strip()}",
                error_code="FETCH_FAILED",
            )
        return sorted(set(dest_dir.glob("**/*")) - before)

    def cancel(self, job_id: str) -> None:
        """Delete the workload so the cluster releases its TPU capacity.

        A workload that has already finished keeps its recorded outcome; a
        live one is recorded CANCELLED only once the delete has succeeded.
        A recipe still submitting is stopped first, or it would create the
        workload after the delete.
        """
        record, workload_name = self._workload_for(job_id)
        if self.get_status(job_id).status in jobs.TERMINAL_STATUSES:
            return
        if record and record.pid and jobs.process_is_alive(
                record.pid, record.pid_started):
            try:
                os.killpg(os.getpgid(record.pid), signal.SIGTERM)
            except OSError:
                pass
        try:
            proc = subprocess.run(self._xpk("workload", "delete",
                                            f"--workload={workload_name}",
                                            "--force"),
                                  capture_output=True,
                                  text=True,
                                  timeout=60)
        except Exception as exc:  # noqa: BLE001 - the workload may be live
            raise base.ComputeConfigError(
                f"xpk workload delete --workload={workload_name} did not "
                f"run: {exc}",
                error_code="CANCEL_FAILED",
            ) from exc
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        absent = re.search(r"not ?found|does not exist|no workload", output,
                           re.IGNORECASE)
        if proc.returncode != 0 and not absent:
            raise base.ComputeConfigError(
                f"xpk workload delete --workload={workload_name} failed "
                f"(exit {proc.returncode}): {(proc.stderr or '').strip()}",
                error_code="CANCEL_FAILED",
            )
        if record:
            record.mark("CANCELLED")


def _search(pattern: str, text: str, default: str) -> str:
    """Return the first capture of `pattern` in `text`, or `default`."""
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else default


def _gcloud_storage_cat(uri: str) -> str:
    """Read an object from GCS, returning "" when it is unavailable."""
    try:
        proc = subprocess.run(["gcloud", "storage", "cat", uri],
                              capture_output=True,
                              text=True,
                              timeout=60)
    except Exception:  # noqa: BLE001 - absence is not an error here
        return ""
    return proc.stdout if proc.returncode == 0 else ""
