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
"""Dispatch a single-host TPU workload to a GKE cluster as a Kubernetes Job.

This backend *creates* the workload. It carries the script in a ConfigMap
mounted into a stock image, so nothing is built and nothing is installed and
the image under test reaches the chips exactly as shipped.

How it differs from its two neighbours, since all three talk to GKE:

- `gke_pod` execs into a pod that is **already running**, and cannot start
  one. That is the right backend for attaching to a long-lived workload
  somebody else launched; it is not usable on an idle cluster, which is the
  normal state of one.
- `gke_xpk` launches multi-host, multi-slice workloads through `xpk`, and so
  requires the `xpk` binary. That is heavier than a one-off single-node check
  needs, and unavailable in a container carrying only `kubectl`.

This driver is the gap between them: one node, one pod, one Job, `kubectl`
only. TPU chips are allocated whole and a single pod can only initialise a
single-host slice, so a multi-host topology belongs to `gke_xpk` -- pointing
this at one reports `cpu-fallback`, because libtpu wants every worker's
hostname before it will open a device.

Requires `kubectl` on PATH and a kubeconfig that already names the target
cluster. Credentials are not minted here: dispatch does not shell out to
gcloud, so a caller that needs a fresh token mints it before calling. Only
preflight uses gcloud: to check that the image exists, and who the nodes
pull it as.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, NamedTuple, Sequence

from .. import env_check
from . import auth_utils, base, gcs_staging, jobs, probe, registry, shell
from .setup_fields import VALID_PRIORITIES

Path = pathlib.Path

# Node labels GKE puts on a TPU node pool, and the extended resource the chips
# are requested through.
ACCELERATOR_LABEL = "cloud.google.com/gke-tpu-accelerator"
TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
HOSTNAME_LABEL = "kubernetes.io/hostname"
# GKE sets this on nodes of a pool with --workload-metadata=GKE_METADATA,
# the only nodes where a pod can get its own Workload Identity token.
WORKLOAD_IDENTITY_LABEL = "iam.gke.io/gke-metadata-server-enabled"
NODEPOOL_LABEL = "cloud.google.com/gke-nodepool"
TPU_RESOURCE = "google.com/tpu"

# Kueue reads admission off these. Applied only when a queue is configured --
# Kueue reads admission off these. Applied only when a queue is configured --
# a cluster that schedules directly has no Kueue to read them.
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
# Opt a workload in to GKE's managed ML diagnostics. The operator
# watches for this label and injects profiling into the pod; without it a
# workload is invisible to the Diagnostics view in Console no matter what is
# enabled on the cluster. Harmless where the addon is absent -- an unrecognised
# label is just a label.
MLDIAGNOSTICS_LABEL = "managed-mldiagnostics-gke"
PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"

# The torch_tpu images keep their own software under /workspace, so a script
# volume mounted there fails container creation. /app is free.
SCRIPT_MOUNT = "/app"
# The ConfigMap key of the i-th --input: a key cannot hold the `/` a remote
# path may, and numbering keeps it clear of the script's own.
INPUT_KEY = "tpu-agent-suite-input-{}"
# Why a Job's outputs went to its log when no bucket was set; otherwise the
# reason is the error code that kept their upload URLs from being signed.
NO_BUCKET = "NO_BUCKET"

DEFAULT_NAMESPACE = "default"
# Only a fallback for a topology that cannot be parsed. Prefer
# `chips_for_topology`: a single-host slice has as many chips as its topology
# multiplies out to, and getting this wrong does not fail loudly. GKE's TPU
# webhook turns the chip request into a `gke-accelerator-count` node selector,
# so asking for 4 on a 2x4 node selects a label no node carries and the pod
# sits Pending with "didn't match Pod's node affinity/selector" -- which reads
# as a full cluster rather than a wrong number.
DEFAULT_CHIPS = 4
DEFAULT_WAIT_SEC = 900
SUBMIT_URL_EXP_SEC = 604800
# A serving job is meant to outlive the session, so a single crash must not
# end it the way it ends a batch run
SERVE_BACKOFF_LIMIT = 6
POLL_INTERVAL_SEC = 5

# Container waiting reasons for an image that will never pull as written.
IMAGE_REF_FAILURES = ("InvalidImageName", "ErrImageNeverPull")
# Waiting reasons for a pull that failed and is being retried. A registry 429
# or a DNS blip clears within a back-off or two; a missing image or one the
# nodes may not read does not, and the kubelet retries it forever while the
# pod holds its TPU chips. So pulls that have kept failing this long, first
# failure to latest by the pod's events, are taken as final.
IMAGE_PULL_RETRIES = ("ErrImagePull", "ImagePullBackOff")
PULL_FAILURE_GRACE_SEC = 60

# Kueue suspends every Job it manages and tracks it through a Workload owned
# by the Job and labelled with its uid. Only the Workload says why a Job is
# not admitted: a refused Job otherwise looks like one queued behind a full
# cluster, and polling it burns the whole wait for nothing.
WORKLOAD_RESOURCE = "workloads.kueue.x-k8s.io"
JOB_UID_LABEL = "kueue.x-k8s.io/job-uid"
# Kueue makes the Workload within a reconcile; a Job with none this long after
# its creationTimestamp is one it could not build a Workload for (usually a
# missing WorkloadPriorityClass), and stays suspended forever.
KUEUE_WORKLOAD_GRACE_SEC = 60
# QuotaReserved=False reasons that no amount of waiting fixes: a missing or
# inactive queue (`Inadmissible`, or `Misconfigured` on newer Kueue), or a
# request no flavor or quota can hold. Not here: `Suspended` (a queue held by
# its stopPolicy), `Pending`, `WaitingForQuota` and other waits that end.
KUEUE_REFUSALS = frozenset({
    "Inadmissible",
    "Misconfigured",
    "NoMatchingFlavor",
    "ExceedsMaxQuota",
})
# The same queue problems in the message: the scheduler can report them under
# plain `Pending`. A held queue also reads "is inactive"; see `_queue_held`.
KUEUE_QUEUE_PROBLEM = re.compile(
    r"\b(LocalQueue|ClusterQueue) (\S+) (doesn't exist|is inactive|"
    r"is terminating or missing)")
# A shared cluster's time-limit controller deactivates a Workload whose pods
# ran past the namespace's limit, and says so in these annotations. Kueue's
# own condition then reads only "deactivated", which looks like a refusal.
# The same controller also deactivates finished Workloads to collect them,
# under the `finished-gc` reason; that is not a kill.
EVICTION_REASON_ANNOTATION = "quota-system/eviction-reason"
EVICTION_MESSAGE_ANNOTATION = "quota-system/eviction-message"
EVICTED_AT_ANNOTATION = "quota-system/evicted-at"
FINISHED_GC_REASON = "finished-gc"

# The accelerator labels GKE puts on TPU nodes, which preflight checks
# `tpu_accelerator` against so a typo is flagged before any Job sits Pending.
# Exact spellings, not a spec table: HBM and devices per chip come from
# tpu_info via `probe.spec_for`, the same table the on-host probe reads.
KNOWN_ACCELERATOR_LABELS = (
    "tpu-v4-podslice",
    "tpu-v5-lite-podslice",
    "tpu-v5p-slice",
    "tpu-v6e-slice",
    "tpu7x",
)

# The memory-backed volume mounted at /dev/shm, in GiB. Without it a pod gets
# the container runtime's 64 MB, which is too small for vLLM. Without
# hf_cache_pvc it also holds the model and its compile caches (see
# POD_CACHE_ENV): Gemma-4-31B needs 76 GiB of it. The size is a cap, not a
# reservation: the node's memory backs only what is written. What is written
# stays in memory until deleted, though, and the pod requests no memory, so
# under node memory pressure it is the first evicted. So the cap stays below
# the smallest 4-chip TPU host's memory (v5e, 192 GB), leaving the rest to
# the engine's own processes.
SHM_MOUNT = "/dev/shm"
DEFAULT_SHM_SIZE_GB = 128
# Its cap in bytes, for a script to size a download against: statvfs inside
# the pod can report the node's whole memory instead.
SHM_LIMIT_ENV = "TPU_AGENT_SUITE_SHM_LIMIT_BYTES"
# Set in a --cpu-requests Job: the profile's TPU generation, as probe.py names
# it (v6e, tpu7x, ...), which the Job holds no chip of. A check that only
# plans for that TPU (onboarding's preflight) answers torch_tpu from it.
CHIPLESS_TPU_ENV = "TPU_AGENT_SUITE_CHIPLESS_TPU"
# A --cpu-requests `run` that exits with this, or never starts, is run again
# once on a TPU node: the work needed a chip after all, or no CPU node took it.
NEEDS_TPU_EXIT_CODE = 75

# Without a pod_service_account, pods get a GCS token limited to the staging
# bucket, so they can also read files staged outside their Job's folder (such
# as checkpoints for fetch_weights.py). It's minted from the caller's
# credentials and mounted from the `<job>-gcs-token` Secret.
TOKEN_SECRET_SUFFIX = "-gcs-token"
TOKEN_MOUNT = "/var/run/gcs-token"
# Tokens last about an hour, so status reads renew them this close to expiry.
# The margin leaves time for kubelet to sync the new Secret into the pod.
TOKEN_RENEW_MARGIN_SEC = 20 * 60
# How long to wait before retrying a failed renewal.
TOKEN_RENEW_RETRY_SEC = 5 * 60

# The shared Hugging Face cache: an admin-made PersistentVolumeClaim mounted
# in every pod, so a model downloaded by one Job is on disk for the next
# instead of being fetched again into each pod's empty filesystem. Only the
# hub cache (HF_HUB_CACHE) is pointed at it, not HF_HOME, which would also
# put a pod's Hugging Face token on a volume other Jobs read.
HF_CACHE_MOUNT = "/hf-cache"
HF_CACHE_ENV = "HF_HUB_CACHE"
HF_CACHE_VOLUME = "hf-cache"
# Jobs on different nodes mount it at once (a server and its parity check),
# which only ReadWriteMany allows; with ReadWriteOnce the second stays
# Pending on a Multi-Attach error.
HF_CACHE_ACCESS_MODE = "ReadWriteMany"
# The shared cache also holds compiled programs, under one folder per TPU
# shape and image (see `compile_cache_key`): vLLM's torch.compile and AOT
# artifacts in VLLM_CACHE_ROOT, and torch_tpu's compiled XLA programs in its
# tier-3 root (torch_tpu's library reads the INTERNAL spelling; the public
# spelling is the one its docs name and newer vllm_torchtpu copies onto it).
#
# Defaults only, as for POD_CACHE_ENV: an image may point these at a compile
# cache baked into it, and pod-spec env would override that. torch_tpu's two
# spellings move together, so an image that sets either keeps both as it has
# them. (Names, subfolder.)
COMPILE_CACHE_SUBDIR = "compile-cache"
COMPILE_CACHE_ENV = (
    (("VLLM_CACHE_ROOT", ), "vllm"),
    (("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
      "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"), "torch_tpu"),
)
# Every new image, unpinned KV cache size or patched build adds a folder, on
# the volume the weights need. A pod touches the stamp in its image folder
# (reads don't move a folder's mtime) and removes the others idle this long.
COMPILE_CACHE_STAMP = ".last_used"
COMPILE_CACHE_MAX_AGE_DAYS = 14
# Above this, a pod keeps its compile caches in /dev/shm, leaving the rest of
# the volume to weight downloads.
COMPILE_CACHE_MAX_USED_PCT = 90
# Without the shared cache, the downloads and compile caches go on /dev/shm,
# not the container's own disk: a TPU node's overlay can hold less than one
# model (41 GB, ~23 GB free, under Gemma-4-31B's 62.55 GB of weights), and
# filling it gets the pod evicted with exit 137. They share shm_size_gb's cap,
# as does everything else a pod caches or writes to a temporary file.
# TMPDIR is /dev/shm itself, which exists; the tools create the others.
#
# Defaults only: each is set in the container, at start, when neither it nor
# a variable that already moves it is set. Pod-spec env would override the
# image's ENV too, and an image may point these at weights or a compile cache
# baked into it. Name -> (default, the variables that move it).
POD_CACHE_DIR = f"{SHM_MOUNT}/cache"
POD_CACHE_ENV = {
    HF_CACHE_ENV:
    (f"{POD_CACHE_DIR}/huggingface/hub", ("HF_HOME", "HUGGINGFACE_HUB_CACHE")),
    "VLLM_CACHE_ROOT": (f"{POD_CACHE_DIR}/vllm", ("XDG_CACHE_HOME", )),
    "XDG_CACHE_HOME": (POD_CACHE_DIR, ()),
    "TORCHINDUCTOR_CACHE_DIR": (f"{POD_CACHE_DIR}/torchinductor", ()),
    "TMPDIR": (SHM_MOUNT, ()),
}
# Names the hub cache above as this pod's own, so a script may prune it: an
# HF_HUB_CACHE without it is a shared volume or the user's, and is kept.
POD_HF_CACHE_ENV = "TPU_AGENT_SUITE_POD_HF_CACHE"


def _pod_cache_command(command: list[str]) -> list[str]:
    """`command`, run after setting the POD_CACHE_ENV defaults it lacks.

    A shell sets them, not the pod spec, so the image's own ENV still wins.
    The hub cache is named in POD_HF_CACHE_ENV only when this set it.
    """
    lines = []
    for name, (default, movers) in POD_CACHE_ENV.items():
        unset = "".join(f"${{{var}+x}}" for var in (name, *movers))
        export = f"export {name}={shlex.quote(default)}"
        if name == HF_CACHE_ENV:
            export += f" {POD_HF_CACHE_ENV}={shlex.quote(default)}"
        lines.append(f'if [ -z "{unset}" ]; then {export}; fi')
    return ["sh", "-c", "\n".join(lines + ['exec "$@"']), "sh", *command]


def _compile_cache_exports(root: str) -> list[str]:
    """Shell lines pointing the COMPILE_CACHE_ENV that is unset at `root`."""
    lines = []
    for names, subdir in COMPILE_CACHE_ENV:
        unset = "".join(f"${{{name}+x}}" for name in names)
        value = shlex.quote(f"{root}/{subdir}")
        exports = " ".join(f"{name}={value}" for name in names)
        lines.append(f'if [ -z "{unset}" ]; then export {exports}; fi')
    return lines


def _compile_cache_command(command: list[str],
                           root: str,
                           max_age_days: int = COMPILE_CACHE_MAX_AGE_DAYS,
                           mount: str = HF_CACHE_MOUNT) -> list[str]:
    """`command`, run after pointing the COMPILE_CACHE_ENV it lacks at `root`.

    A shell sets them, not the pod spec, so the image's own ENV still wins.
    It stamps `root` as used, and, unless `max_age_days` is 0, removes in the
    background the other image folders nothing touched in that many days
    (the stamp, or any file in them). With `mount` nearly full, the caches
    stay in the pod instead.
    """
    q = shlex.quote
    lines = [
        f"used=$(df -P {q(mount)} 2>/dev/null | awk 'NR==2 {{print $5+0}}')",
        f'if [ "${{used:-0}}" -lt {COMPILE_CACHE_MAX_USED_PCT} ]; then',
        *_compile_cache_exports(root),
        f"mkdir -p {q(root)} 2>/dev/null && "
        f"touch {q(f'{root}/{COMPILE_CACHE_STAMP}')} 2>/dev/null",
        "else",
        f'echo "gke_job: {mount} is ${{used}}% full; compile caches stay '
        'in the pod" >&2',
        *_compile_cache_exports(POD_CACHE_DIR),
        "fi",
    ]
    if max_age_days > 0:
        # A folder find cannot read is kept, not taken for stale.
        lines.append(
            f"(for d in {q(mount)}/{COMPILE_CACHE_SUBDIR}/*/*/; do\n"
            f'd="${{d%/}}"; [ -d "$d" ] && [ "$d" != {q(root)} ] || continue\n'
            f'new=$(find "$d" -mtime -{max_age_days} 2>/dev/null) || continue\n'
            '[ -z "$new" ] || continue\n'
            f'echo "gke_job: removing compile cache $d, unused for '
            f'{max_age_days}+ days" >&2\n'
            'rm -rf "$d" || echo "gke_job: could not remove $d" >&2\n'
            "done) </dev/null &")
    return ["sh", "-c", "\n".join(lines + ['exec "$@"']), "sh", *command]


# What a PersistentVolumeClaim's name may be: a DNS-1123 subdomain.
PVC_NAME = re.compile(r"[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?")

# A Job's name is copied into a pod label, and label values max out at 63.
MAX_JOB_NAME_LEN = 63
# Leaves room for the job's own name and the run suffix.
MAX_JOB_NAME_PREFIX_LEN = 30


def _k8s_name(raw: str) -> str:
    """`raw` lowercased, with runs of other than [a-z0-9-] made '-'."""
    return re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower()).strip("-")


def compile_cache_key(image: str) -> str:
    """The compile-cache folder for `image`.

    A digest names the image's content, so it is used as is. A tag can move
    and two registries can share a name, so a tag is spelled readably and
    followed by a hash of the whole reference.
    """
    ref = image.strip()
    _, _, digest = ref.partition("@")
    if digest:
        algorithm, _, value = digest.partition(":")
        return _k8s_name(f"{algorithm}-{value[:32]}")
    name = _k8s_name(ref.rsplit("/", 1)[-1])[:48].strip("-")
    return f"{name}-{hashlib.sha256(ref.encode()).hexdigest()[:12]}"


def _chip_spec(accelerator: str) -> tuple[dict[str, Any] | None, str]:
    """HBM and devices per chip for a node label, or why they are unknown.

    `inspect` runs on the controller, before any pod exists to probe, so the
    label is all there is to go on. tpu_info's chip table is static and needs
    no TPU, but it is a TPU-side dependency a controller may lack; either way
    an unknown spec is reported as unknown rather than guessed, since a wrong
    capacity or device count sizes a model wrong.
    """
    try:
        spec = probe.spec_for(probe.generation_for_name(accelerator))
    except probe.MissingPackageError as exc:
        return None, ("hbm_gb and devices not reported: " +
                      env_check.ask_to_install([exc.package]))
    if not spec["hbm_gb"]:
        return None, (f"unknown accelerator '{accelerator}'; hbm_gb and "
                      "devices not reported")
    return spec, ""


def _positive_int(field: str, value: Any, minimum: int = 1) -> int:
    """`value` as an int >= `minimum`, or an INVALID_FIELD error naming it."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise base.ComputeConfigError(
            f"Backend 'gke_job' needs an integer '{field}', got {value!r}.",
            error_code="INVALID_FIELD",
        ) from exc
    if number < minimum:
        raise base.ComputeConfigError(
            f"Backend 'gke_job' needs '{field}' >= {minimum}, got {number}.",
            error_code="INVALID_FIELD",
        )
    return number


def _parse_requests(value: Any,
                    field: str = "colocate_requests") -> dict[str, str]:
    """`NAME=QTY,...` (or a dict) as pod resource requests; blank is {}."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    pairs = [
        p.partition("=") for p in str(value or "").split(",") if p.strip()
    ]
    if any(not name.strip() or not qty.strip() for name, _, qty in pairs):
        raise base.ComputeConfigError(
            f"Backend 'gke_job' needs '{field}' as "
            f"NAME=QTY,..., got {value!r}.",
            error_code="INVALID_FIELD",
        )
    return {name.strip(): qty.strip() for name, _, qty in pairs}


# How long one kubectl read may take. Without a bound, an unreachable API
# server or a stalled auth plugin hangs `job status` and `job list` with it;
# the shell and xpk drivers bound their status reads the same way.
KUBECTL_READ_TIMEOUT_SEC = 30

# How long a kubectl delete may take. It waits for the pod to terminate,
# which is its grace period (30s by default) plus the API round trips. It
# was the one unbounded call on `run`'s exit path; a delete that overruns is
# reported, the record stays live, and the run-end reaper retries.
KUBECTL_DELETE_TIMEOUT_SEC = 90

# A Job read right after it is created can come back NotFound. A Job whose
# record is younger than FRESH_JOB_SEC is read again after each delay.
FRESH_JOB_SEC = 60
NOT_FOUND_RETRY_DELAYS_SEC = (1, 2, 4)

# Suffixed names tried when a job outside this run holds the requested one.
NAME_SUFFIX_ATTEMPTS = 3

# The last resort for a Job's log once its pod is gone and no copy was kept:
# GKE ships container output to Cloud Logging. Bounded in lines and time, so
# a long-lived server cannot turn one `job logs` into a minutes-long read.
CLOUD_LOGGING_MAX_LINES = 2000
CLOUD_LOGGING_TIMEOUT_SEC = 60
# Margin around the record's start and end: the record is written around
# submit and when compute noticed the end, not when the container ran. Used
# only with the Job's UID in the query: a name can belong to an earlier or
# later Job (cancel and resubmit with the same --name), and the margin would
# then pull in that Job's output.
CLOUD_LOGGING_SLACK_SEC = 600
# The pod label carrying its Job's UID, as Cloud Logging spells it: the
# legacy key and the batch.kubernetes.io one, whose dots it turns into
# underscores. Both are set on Kubernetes 1.27+; either matches.
CLOUD_LOGGING_UID_LABELS = ("k8s-pod/controller-uid",
                            "k8s-pod/batch_kubernetes_io/controller-uid")


class KubectlError(base.ComputeConfigError):
    """A kubectl invocation failed.

    A subclass rather than a bare CalledProcessError so the CLI's existing
    ComputeConfigError handler renders it as a structured payload, with the
    command's own stderr in the message -- kubectl's diagnostics are the
    useful part and would otherwise be swallowed.
    """

    def __init__(self, argv: list[str], stderr: str) -> None:
        super().__init__(
            f"kubectl {' '.join(argv)} failed: {stderr.strip()}",
            error_code="KUBECTL_FAILED",
        )


def chips_for_topology(topology: str, default: int = DEFAULT_CHIPS) -> int:
    """Chips on one node of `topology`, e.g. `2x4` -> 8, `2x2x1` -> 4.

    This driver runs one pod on one node, so the whole topology is the node and
    its dimensions multiply out to the chip count. Returns `default` for
    anything unparseable rather than raising: a bad topology is already going
    to fail at scheduling with a better message than this could give.
    """
    try:
        dims = [int(d) for d in topology.lower().split("x")]
    except (AttributeError, ValueError):
        return default
    if not dims:
        return default
    product = 1
    for d in dims:
        product *= d
    return product or default


def _hf_cache_pvc_check(namespaced: list[str], kubectl: list[str], claim: str,
                        namespace: str) -> base.PreflightCheck:
    """Whether the shared cache claim exists, is ReadWriteMany, and binds."""
    name = "hf_cache_pvc"
    create = (f"Ask a cluster admin to create a {HF_CACHE_ACCESS_MODE} "
              f"PersistentVolumeClaim '{claim}' in namespace '{namespace}' "
              "(e.g. from a Filestore storage class), or unset hf_cache_pvc")
    proc = base.probe_run(namespaced + ["get", "pvc", claim, "-o", "json"])
    if proc.returncode != 0:
        error = base.probe_error(proc)
        if "forbidden" in error.lower():
            return base.PreflightCheck(
                name, "warn", f"cannot read PVC '{claim}': {error}",
                "Ask a cluster admin for `get` on persistentvolumeclaims, or "
                "check the claim yourself")
        # A pod naming a claim that does not exist never starts: every Job
        # would sit Pending.
        return base.PreflightCheck(
            name, "fail", f"PVC '{claim}' in namespace '{namespace}': "
            f"{error}", create)
    try:
        pvc = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return base.PreflightCheck(name, "warn",
                                   f"could not parse PVC '{claim}'")
    spec, status = pvc.get("spec") or {}, pvc.get("status") or {}
    modes = status.get("accessModes") or spec.get("accessModes") or []
    if HF_CACHE_ACCESS_MODE not in modes:
        return base.PreflightCheck(
            name, "fail",
            f"PVC '{claim}' is {', '.join(modes) or 'without an access mode'}"
            f", not {HF_CACHE_ACCESS_MODE}: Jobs on different nodes cannot "
            "all mount it, so a second one stays Pending", create)
    phase = status.get("phase") or "Pending"
    capacity = (status.get("capacity") or {}).get("storage")
    if phase == "Bound":
        return base.PreflightCheck(
            name, "ok", f"PVC '{claim}' is Bound, {HF_CACHE_ACCESS_MODE}" +
            (f", {capacity}" if capacity else ""))
    storage_class = spec.get("storageClassName")
    if phase == "Pending" and storage_class:
        # A WaitForFirstConsumer class binds only once a pod uses the claim,
        # so Pending is what a healthy, never-used claim looks like.
        sc = base.probe_run(kubectl + [
            "get", "storageclass", storage_class, "-o",
            "jsonpath={.volumeBindingMode}"
        ])
        if sc.returncode != 0:
            return base.PreflightCheck(
                name, "warn",
                f"PVC '{claim}' is Pending, and its storage class "
                f"'{storage_class}' could not be read: {base.probe_error(sc)}",
                "Fine if the class binds on first use; otherwise ask a "
                "cluster admin why the claim is not Bound")
        if sc.stdout.strip() == "WaitForFirstConsumer":
            return base.PreflightCheck(
                name, "warn", f"PVC '{claim}' is Pending until the first Job "
                f"mounts it (storage class '{storage_class}' binds on first "
                "use)")
    return base.PreflightCheck(
        name, "fail", f"PVC '{claim}' is {phase}, not Bound: a Job that "
        "mounts it stays Pending",
        "Ask a cluster admin why the claim is not Bound "
        f"(kubectl describe pvc {claim} -n {namespace}), or unset "
        "hf_cache_pvc")


def _kube_get_check(kubectl: list[str], name: str, what: str, argv: list[str],
                    fix: str) -> base.PreflightCheck:
    """Whether `kubectl get <argv>` finds `what`."""
    proc = base.probe_run(kubectl + ["get"] + argv + ["-o", "name"])
    if proc.returncode == 0:
        return base.PreflightCheck(name, "ok", f"{what} exists")
    error = base.probe_error(proc)
    status = "warn" if "forbidden" in error.lower() else "fail"
    return base.PreflightCheck(name, status, f"{what}: {error}", fix)


# Registries `gcloud artifacts docker images describe` can answer for: Artifact
# Registry, and Container Registry hosts that Artifact Registry now serves.
_GCLOUD_REGISTRY_SUFFIXES = ("-docker.pkg.dev", "gcr.io")


def _image_check(image: str) -> base.PreflightCheck:
    """Whether `image` resolves in its registry, as this machine's account.

    The nodes pull with their own service account, so an image this account
    can read may still fail to pull there; `_node_pull_check` warns of that
    where it can, and `compute setup verify --smoke-test` settles it. What this
    catches in a second is a tag, digest or repository that does not exist,
    which otherwise shows up as a Job stuck pulling.
    """
    registry = image.split("/", 1)[0]
    if not registry.endswith(_GCLOUD_REGISTRY_SUFFIXES):
        return base.PreflightCheck(
            "image", "warn", f"{image}: not in Artifact Registry, so it "
            "cannot be checked without pulling it")
    if not shutil.which("gcloud"):
        return base.PreflightCheck(
            "image", "warn", f"{image}: gcloud is not on PATH to check it")
    proc = base.probe_run([
        "gcloud", "artifacts", "docker", "images", "describe", image,
        "--format=value(image_summary.digest)"
    ],
                          timeout=30)
    if proc.returncode == 0:
        return base.PreflightCheck("image", "ok",
                                   f"{image} ({proc.stdout.strip()})")
    error = base.probe_error(proc)
    # The whole output, not the one line: which line says so varies.
    if re.search(r"not found|NOT_FOUND|manifest unknown",
                 f"{proc.stderr}\n{proc.stdout}", re.IGNORECASE):
        return base.PreflightCheck(
            "image", "fail", f"{image}: {error}",
            "Check the image's tag or digest against `gcloud artifacts docker "
            f"images list {image.split('@')[0].rsplit(':', 1)[0]} "
            "--include-tags`")
    # Not allowed to read it, or gcloud could not ask: the nodes may still
    # pull it, so only the smoke test can say.
    return base.PreflightCheck("image", "warn",
                               f"{image}: could not check it: {error}")


_AR_READER = "roles/artifactregistry.reader"
# <location>-docker.pkg.dev/<project>/<repository>/<image>: where the policy
# to read is, and what the fix names.
_AR_IMAGE = re.compile(r"([a-z0-9-]+)-docker\.pkg\.dev/([^/]+)/([^/]+)/.+")


def _gcloud_json(argv: list[str]) -> Any:
    """`gcloud <argv> --format=json`, parsed; None when it fails."""
    proc = base.probe_run(["gcloud"] + argv + ["--format=json"], timeout=30)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout or "null")
    except ValueError:
        return None


def _node_service_accounts(target: dict[str, str], pools: set[str],
                           topology: str | None) -> list[str] | None:
    """The service accounts the cluster's nodes run as, limited to `pools`
    when any are named, else to the pools of `topology` (a TPU pool scaled
    to zero has no nodes to name it); None when gcloud cannot say or no
    pool matches."""
    listed = _gcloud_json([
        "container", "node-pools", "list", "--cluster", target["cluster"],
        "--location", target["location"], "--project", target["project"]
    ])
    if not isinstance(listed, list):
        return None
    chosen = ([pool for pool in listed if pool.get("name") in pools] or [
        pool for pool in listed if topology and
        (pool.get("placementPolicy") or {}).get("tpuTopology") == topology
    ])
    if not chosen:
        return None
    accounts = {(pool.get("config") or {}).get("serviceAccount") or "default"
                for pool in chosen}
    if "default" in accounts:
        # GKE's "default" is the project's Compute Engine default account.
        described = _gcloud_json(["projects", "describe", target["project"]
                                  ]) or {}
        number = described.get("projectNumber")
        if not number:
            return None
        accounts.discard("default")
        accounts.add(f"{number}-compute@developer.gserviceaccount.com")
    return sorted(accounts)


def _list_node_pools(kubectl: list[str],
                     selector: str) -> subprocess.CompletedProcess:
    """`kubectl get nodes -l selector`: one line per node, naming its pool."""
    return base.probe_run(kubectl + [
        "get", "nodes", "-l", selector, "-o",
        "jsonpath={range .items[*]}{.metadata.labels." +
        NODEPOOL_LABEL.replace(".", r"\.") + '}{"\\n"}{end}'
    ])


def _node_pull_check(image: str, context: str, pools: set[str],
                     topology: str | None) -> base.PreflightCheck | None:
    """Whether the nodes can likely pull `image`, which `_image_check` saw
    only as this machine's account."""
    match = _AR_IMAGE.fullmatch(image)
    target = base.gke_target(context, prefer_context=True)
    if not match or any(v.startswith("<") for v in target.values()):
        return None
    location, project, repository = match.groups()
    if project == target["project"]:
        return None
    accounts = _node_service_accounts(target, pools, topology)
    who = (", ".join(accounts)
           if accounts else "the node pools' service account")
    grant = (f"gcloud artifacts repositories add-iam-policy-binding "
             f"{repository} --location {location} --project {project} "
             f"--role {_AR_READER} --member "
             f"serviceAccount:{accounts[0] if accounts else '<node SA>'}")
    policy = _gcloud_json([
        "artifacts", "repositories", "get-iam-policy", repository,
        "--location", location, "--project", project
    ])
    if isinstance(policy, dict):
        readers = {
            member
            for binding in policy.get("bindings") or []
            if binding.get("role") == _AR_READER
            for member in binding.get("members") or []
        }
        nodes = {f"serviceAccount:{account}" for account in accounts or []}
        if (nodes and nodes
                <= readers) or readers & {"allUsers", "allAuthenticatedUsers"}:
            return base.PreflightCheck(
                "image:node_pull", "ok",
                f"repository {project}/{repository} grants the nodes read")
        seen = "its policy does not name them"
    else:
        seen = "this account cannot read its policy to check"
    return base.PreflightCheck(
        "image:node_pull", "warn",
        f"the nodes pull {image.split('@')[0]} as {who}, from project "
        f"{project} rather than the cluster's ({target['project']}), and "
        f"{seen}; without read access the Job fails with ErrImagePull 403",
        f"Have an admin of {project} grant it {_AR_READER} "
        f"({grant}), or copy the image into a registry in {target['project']}")


def _mark_terminal(record: jobs.JobRecord,
                   status: base.AsynchronousExecutionResult) -> None:
    """Write a terminal status to the record, with why it ended.

    The reason has to outlive the Job: a Job whose image never pulled, or
    that Kueue refused, is deleted as soon as that is clear, and after that
    only the record can answer `job status` or explain why `job logs` has
    nothing.
    """
    extra = ({
        "never_started": True
    } if status.details.get("never_started") else {})
    record.mark(status.status,
                exit_code=status.exit_code,
                error=status.error,
                **extra)


class PodEnd(NamedTuple):
    """How a Job's pod ended, as `_pod_end` read it from the pods.

    The container's exit code and reason, and the pod's own reason and
    message (set when the kubelet or the node ended it); each None when the
    pod does not say. Field names match `_termination_error`'s arguments.
    """
    exit_code: int | None = None
    reason: str | None = None
    pod_reason: str | None = None
    pod_message: str | None = None


# Exit 137 is SIGKILL; with no reason anywhere, these are what send it.
SIGKILL_HINT = "SIGKILL: out of memory, an eviction, or the node going away"


def _termination_error(exit_code: int | None,
                       reason: str | None,
                       pod_reason: str | None = None,
                       pod_message: str | None = None) -> str | None:
    """Why a finished container failed, in words a person can act on.

    Kubernetes gives a container that exits non-zero the reason `Error`,
    which alone reads as the whole diagnosis. The exit code is what says
    something; a specific reason (OOMKilled, DeadlineExceeded) is kept.

    `pod_reason` and `pod_message` are the pod's own status, set when the
    kubelet or the node ended it rather than the workload: an eviction for
    a full disk kills the container with 137 and no traceback, which alone
    reads as a crash. The eviction is then the headline and the exit code
    the detail.
    """
    if exit_code is None:
        container = reason or None
    else:
        specific = reason if reason not in (None, "", "Error",
                                            "Completed") else None
        container = f"container exited with code {exit_code}"
        if specific:
            container += f" ({specific})"
        elif exit_code == 137 and not (pod_reason or pod_message):
            container += f" ({SIGKILL_HINT})"
    if _is_eviction(pod_reason, pod_message):
        evicted = (f"pod was evicted: {pod_message}"
                   if pod_message else "pod was evicted")
        return f"{evicted} ({container})" if container else evicted
    pod = ": ".join(part for part in (pod_reason, pod_message) if part)
    if not pod:
        return container
    return f"{container}; pod {pod}" if container else f"pod {pod}"


def _is_eviction(pod_reason: str | None, pod_message: str | None) -> bool:
    """Whether the kubelet evicted the pod, by its reason or its words.

    `Evicted` is the pod's reason when the kubelet evicts it; a pod whose
    reason was not kept still has the kubelet's message on its
    DisruptionTarget condition.
    """
    return pod_reason == "Evicted" or (pod_message or "").startswith(
        ("The node was low on resource", "Pod ephemeral local storage usage"))


def _pod_status_reason(pod: dict[str, Any]) -> tuple[str | None, str | None]:
    """The pod's own reason and message for ending, or (None, None).

    `status.reason` / `status.message` (`Evicted`, `Terminated`, `Shutdown`)
    first; else the DisruptionTarget condition, where newer kubelets and the
    taint manager say why they killed it.
    """
    status = pod.get("status", {})
    if status.get("reason") or status.get("message"):
        return status.get("reason"), status.get("message")
    for cond in status.get("conditions", []):
        if cond.get("type") == "DisruptionTarget" and cond.get(
                "status") == "True":
            return cond.get("reason"), cond.get("message")
    return None, None


def _terminated(pod: dict[str, Any]) -> dict[str, Any]:
    """The first container's `state.terminated` in `pod`, or {}."""
    return next((cs["state"]["terminated"]
                 for cs in pod.get("status", {}).get("containerStatuses", [])
                 if (cs.get("state") or {}).get("terminated")), {})


def _pod_end(pods: list[dict[str, Any]]) -> PodEnd:
    """How the Job's pod ended; all None when no pod says.

    The pod's own reason is read as well as the container's: a pod the
    kubelet evicted for a full disk shows only exit 137 on its container,
    and `Evicted` with the kubelet's words on the pod. The first pod that
    says anything is the answer.
    """
    for pod in pods:
        terminated = _terminated(pod)
        pod_reason, pod_message = _pod_status_reason(pod)
        if terminated or pod_reason or pod_message:
            return PodEnd(terminated.get("exitCode"), terminated.get("reason"),
                          pod_reason, pod_message)
    return PodEnd()


# The container reasons the kubelet gives when it lost the container's state:
# the node rebooted or went away under it (seen as exit 255 after a VM reset).
NODE_LOST_REASONS = ("Unknown", "ContainerStatusUnknown")
# How long after a lost node is seen Cloud Logging is asked for the lines the
# node's copy lost. It ingests a line within seconds as a rule, but can lag a
# minute; asked sooner, it can be missing the very lines that matter.
CLOUD_LOGGING_SETTLE_SEC = 90


def _node_lost(pods: list[dict[str, Any]]) -> bool:
    """Whether the Job's pod ended because its node was lost.

    Its log on the node's disk then misses whatever the kubelet had not yet
    flushed, often the last half minute, while Cloud Logging, shipped as the
    lines were printed, kept them.
    """
    return _pod_end(pods).reason in NODE_LOST_REASONS


def _with_cloud_tail(log: str, cloud: list[str]) -> tuple[str, int]:
    """`log` with the lines Cloud Logging kept after its end, and how many.

    The pod's copy is kept whole (Cloud Logging's is capped and may be cut
    at the front) and only the tail is taken from Cloud Logging, found by
    matching the copy's last complete lines. Unchanged when they cannot be
    found. A copy with no complete line takes all of Cloud Logging's.
    """
    lines = log.splitlines()
    if log and not log.endswith("\n"):
        lines = lines[:-1]  # Cut mid-line when the node went.
    tail = [line for line in lines if line.strip()][-3:]
    cloud = [line for line in cloud if line.strip()]
    if not tail:
        extra, lines = cloud, []
    else:
        for end in range(len(cloud), len(tail) - 1, -1):
            if cloud[end - len(tail):end] == tail:
                extra = cloud[end:]
                break
        else:
            return log, 0
    if not extra:
        return log, 0
    head = "".join(line + "\n" for line in lines)
    return (head + f"[compute] The node was lost, and its copy of this log "
            f"ends here; the {len(extra)} line(s) below are what Cloud "
            "Logging kept after it.\n" + "\n".join(extra) + "\n", len(extra))


def _write_atomically(path: Path, text: str) -> None:
    """Write `text` to `path` through a `.part` file beside it, so a reader
    never finds half of it. Raises OSError as `write_text` does."""
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    part.replace(path)


def _recorded_status(job_id: str) -> base.AsynchronousExecutionResult | None:
    """The status a settled record holds, for a Job no longer on the cluster.

    None when there is no record or it is not terminal: then the Job's absence
    is unexplained and the caller should say so.
    """
    record = jobs.JobRecord.load(job_id)
    if record is None or record.status not in jobs.TERMINAL_STATUSES:
        return None
    details: dict[str, Any] = {"from_record": True}
    if record.extra.get("never_started"):
        details["never_started"] = True
    return base.AsynchronousExecutionResult(job_id=job_id,
                                            status=record.status,
                                            start_time=record.start_time,
                                            end_time=record.end_time,
                                            exit_code=record.exit_code,
                                            error=record.extra.get("error"),
                                            details=details)


def _last_seen_path(job_id: str) -> Path:
    return jobs.jobs_dir() / job_id / "last_seen.json"


def _note_last_seen(job_id: str,
                    status: base.AsynchronousExecutionResult) -> None:
    """Keep what this read saw of the Job, for when it is later gone.

    A Job can vanish with nothing on the cluster to say why (a person or
    another run deleted it, or its namespace went), and kubectl's NotFound
    is then all `job status` would have. Kept beside the record rather than
    in it: a poll rewriting job.json could undo a terminal status another
    process just wrote. Only for a job with a record, and never fatal.
    """
    path = _last_seen_path(job_id)
    if not (path.parent / "job.json").is_file():
        return
    seen = {
        "status": status.status,
        "at": jobs.utc_now(),
        "node": status.details.get("node_name"),
        "pod": status.details.get("pod_name"),
        "reason": status.error,
    }
    try:
        _write_atomically(path, json.dumps(seen))
    except OSError:
        pass


def _last_seen_note(job_id: str) -> str:
    """`; last seen RUNNING on node N at T (pod P), reason: R`, or ''."""
    try:
        seen = json.loads(_last_seen_path(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(seen, dict) or not seen.get("status"):
        return ""
    note = f"; last seen {seen['status']}"
    if seen.get("node"):
        note += f" on node {seen['node']}"
    if seen.get("at"):
        note += f" at {seen['at']}"
    if seen.get("pod"):
        note += f" (pod {seen['pod']})"
    if seen.get("reason"):
        note += f", reason: {seen['reason']}"
    return note


def _is_not_found(exc: Exception) -> bool:
    """Whether kubectl said the object is not there. It spells it both ways."""
    return bool(re.search(r"not ?found", str(exc), re.IGNORECASE))


def _just_submitted(record: jobs.JobRecord | None) -> bool:
    """Whether a live record is young enough that NotFound may be a lag."""
    if record is None or record.status in jobs.TERMINAL_STATUSES:
        return False
    started = _parse_time(record.start_time)
    return started is not None and time.time() - started < FRESH_JOB_SEC


def _parse_time(stamp: str | None) -> float | None:
    """A Kubernetes RFC 3339 timestamp as epoch seconds, or None."""
    if not stamp:
        return None
    try:
        return datetime.datetime.fromisoformat(stamp.replace(
            "Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _rfc3339(epoch: float) -> str:
    """Epoch seconds as the UTC timestamp a Cloud Logging query compares."""
    return datetime.datetime.fromtimestamp(
        epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cloud_logging_query(namespace: str,
                         job_id: str,
                         cluster: str | None,
                         start: float | None,
                         end: float | None,
                         uid: str | None = None) -> str:
    """The Cloud Logging filter for a Job's container output.

    `start` and `end` (epoch seconds, from the job record) bound it;
    `cluster` narrows it when known. With the Job's `uid` the query matches
    only this Job's pods, so the bounds are widened by CLOUD_LOGGING_SLACK_SEC.
    Without one (a record from before the UID was kept) the name is all it
    has, and a widened window would take in the output of an earlier or later
    Job of the same name, so the bounds are exact. Pod labels reach Cloud
    Logging as `labels."k8s-pod/<key>"`.
    """
    filters = [
        'resource.type="k8s_container"',
        f'resource.labels.namespace_name="{namespace}"',
        f'labels."k8s-pod/job-name"="{job_id}"',
    ]
    if uid:
        filters.append("(" +
                       " OR ".join(f'labels."{label}"="{uid}"'
                                   for label in CLOUD_LOGGING_UID_LABELS) +
                       ")")
    if cluster:
        filters.append(f'resource.labels.cluster_name="{cluster}"')
    slack = CLOUD_LOGGING_SLACK_SEC if uid else 0
    if start is not None:
        filters.append(f'timestamp>="{_rfc3339(start - slack)}"')
    if end is not None:
        filters.append(f'timestamp<="{_rfc3339(end + slack)}"')
    return " AND ".join(filters)


def _log_entry_text(entry: dict[str, Any]) -> str | None:
    """The line a container printed, from one Cloud Logging entry.

    Plain output is `textPayload`; a line that parsed as JSON is
    `jsonPayload`, whose `message` is the text when it has one.
    """
    if isinstance(entry.get("textPayload"), str):
        return entry["textPayload"].rstrip("\n")
    payload = entry.get("jsonPayload")
    if isinstance(payload, dict):
        message = payload.get("message")
        return message.rstrip("\n") if isinstance(
            message, str) else json.dumps(payload, sort_keys=True)
    return None


def _log_entry_pod(entry: dict[str, Any]) -> str | None:
    return ((entry.get("resource") or {}).get("labels") or {}).get("pod_name")


def _time_limit_stop(workload: dict[str, Any]) -> dict[str, str] | None:
    """What the cluster's time-limit controller said when it stopped this
    Workload after it ran, or None when it did not.

    Only a deactivated Workload carrying the controller's eviction reason
    counts; `finished-gc` is the controller tidying up a finished one.
    """
    annotations = workload.get("metadata", {}).get("annotations") or {}
    reason = annotations.get(EVICTION_REASON_ANNOTATION)
    if (workload.get("spec", {}).get("active") is not False or not reason
            or reason == FINISHED_GC_REASON):
        return None
    message = (annotations.get(EVICTION_MESSAGE_ANNOTATION)
               or f"evicted ({reason})")
    stop = {"eviction_reason": reason, "eviction_message": message}
    if annotations.get(EVICTED_AT_ANNOTATION):
        stop["evicted_at"] = annotations[EVICTED_AT_ANNOTATION]
    return stop


def _workload_verdict(
        workload: dict[str, Any]) -> tuple[bool, str, str] | None:
    """(refused, reason, message) for a Workload Kueue has not admitted.

    Refused means Kueue will never admit it as written. Not refused is a
    Workload that is only waiting -- for quota, or requeued after Kueue
    preempted it (reason `Preempted`, the message naming the preemptor).
    None is a Workload Kueue has said nothing about: admitted, or not yet
    looked at, neither of which is a refusal.
    """
    conditions = {
        cond.get("type"): cond
        for cond in workload.get("status", {}).get("conditions", [])
    }
    finished = conditions.get("Finished") or {}
    if finished.get("status") == "True":
        # Finished while its Job is still suspended: it was never admitted,
        # and Kueue is done with it.
        return (True, finished.get("reason")
                or "Finished", finished.get("message")
                or "Kueue finished the Workload without admitting it.")
    if workload.get("spec", {}).get("active") is False:
        # Deactivated -- an admission check rejected it, or it was requeued
        # too often -- and Kueue does not requeue it until someone flips it
        # back. The condition that explains it depends on the cause.
        for kind in ("DeactivationTarget", "Evicted", "Requeued"):
            cond = conditions.get(kind) or {}
            if cond.get("message"):
                return (True, cond.get("reason")
                        or "Deactivated", cond["message"])
        return True, "Deactivated", "Kueue deactivated the Workload."
    # Older Kueue records preemption only as the eviction's reason.
    for kind in ("Preempted", "Evicted"):
        cond = conditions.get(kind) or {}
        if cond.get("status") == "True" and "Preempted" in (
                kind, cond.get("reason")):
            return False, "Preempted", cond.get("message") or "Preempted."
    reserved = conditions.get("QuotaReserved") or {}
    if reserved.get("status") != "False":
        return None
    reason = reserved.get("reason") or ""
    message = reserved.get("message") or ""
    if reason in KUEUE_REFUSALS or KUEUE_QUEUE_PROBLEM.search(message):
        return True, reason or "Inadmissible", message or reason
    return (False, reason or "Pending", message) if message else None


def _event_time(event: dict[str, Any]) -> float:
    """When an event last happened, for ordering; 0 when it does not say."""
    return _parse_time(
        event.get("lastTimestamp") or
        (event.get("series") or {}).get("lastObservedTime")
        or event.get("eventTime") or event.get("firstTimestamp")) or 0.0


def _kube_can_i_check(kubectl: list[str], verb: str,
                      resource: str) -> base.PreflightCheck:
    """Whether this account may `verb` `resource` where `kubectl` points."""
    proc = base.probe_run(kubectl + ["auth", "can-i", verb, resource])
    answer = proc.stdout.strip()
    name = f"rbac:{verb} {resource}"
    if answer == "yes":
        return base.PreflightCheck(name, "ok", "allowed")
    if answer.startswith("no"):
        return base.PreflightCheck(
            name, "fail", f"this account may not {verb} {resource}",
            f"Ask a cluster admin for a Role that grants {verb} on "
            f"{resource} in this namespace")
    return base.PreflightCheck(
        name, "warn", f"could not ask the cluster: {base.probe_error(proc)}")


@dataclasses.dataclass
class _Staging:
    """How a Job's files travel, as `GkeJobDriver._stage` arranged it."""

    configmap_files: list[str] = dataclasses.field(default_factory=list)
    mounted_inputs: list[tuple[str,
                               str]] = dataclasses.field(default_factory=list)
    in_urls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    out_urls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    logged_outputs: list[str] = dataclasses.field(default_factory=list)
    # Why outputs go to the log: NO_BUCKET, or the error code that kept
    # their upload URLs from being signed.
    logged_because: str = ""
    # Set before the first upload, so a failure after it knows to clean up.
    uploaded: bool = False

    def moves_files(self) -> bool:
        return bool(self.mounted_inputs or self.in_urls or self.out_urls
                    or self.logged_outputs)


@registry.register_driver("gke_job")
class GkeJobDriver(base.ComputeDriver):
    """Creates a Kubernetes Job on a TPU node and follows it to completion.

    Adapter: wraps kubectl behind the ComputeDriver interface. Not built on
    `shell.RemoteShellDriver` like its siblings, because that abstracts a shell
    on a host that already exists -- here there is no host until this driver
    makes one, and the lifecycle is the Job's rather than a process's.
    """

    required_tools = base.GKE_TOOLS
    # Set once the cluster refused the gcloud account but let ADC in.
    using_adc = False
    scheduled_run = True
    runs_in_container = True
    CHECK_FIELDS = {
        "cluster": ("kubeconfig", "kube_context"),
        "kube_context": ("kube_context", ),
        "image": ("image", ),
        "namespace": ("namespace", ),
        "kueue:queue": ("queue", "namespace"),
        "kueue:priority": ("priority", ),
        "pod_service_account": ("pod_service_account", ),
        "hf_cache_pvc": ("hf_cache_pvc", ),
        "staging:bucket": ("staging_bucket", ),
        "staging:signing":
        gcs_staging.POD_UPLOAD_FIELDS,
        "staging:credentials":
        ("staging_service_account", "staging_service_account_key"),
    }
    SETUP_FIELDS = base.SetupFields(
        required=("image", "tpu_accelerator", "tpu_topology"),
        optional=("kube_context", "namespace", "queue", "hf_cache_pvc"),
        conditional={
            "priority": "queue",
        },
        on_demand={
            "staging_bucket":
            "staging files too large to send inline",
            "staging_service_account":
            "signing staging URLs as a service account you can impersonate",
            "staging_service_account_key":
            "signing staging URLs with a key file when impersonation is "
            "unavailable",
            "pod_service_account":
            "Workload Identity GCS access without signing",
            "declared_duration_minutes":
            "Kueue suspension enforcement",
            "kubeconfig":
            "a non-default kubeconfig path",
            "python_interpreter":
            "a non-default Python inside the container",
            "tpu_chips":
            "overriding the chip count derived from tpu_topology",
            "shm_size_gb":
            f"a /dev/shm larger or smaller than {DEFAULT_SHM_SIZE_GB} GiB",
            "compile_cache_max_age_days":
            "keeping unused compile caches on hf_cache_pvc other than "
            f"{COMPILE_CACHE_MAX_AGE_DAYS} days (0 never removes them)",
            "job_name_prefix":
            "a recognisable prefix on generated Job names",
            "ml_diagnostics_cluster":
            "uploading profiles to an ML Diagnostics dashboard",
            "ml_diagnostics_gcs":
            "telling ML Diagnostics which GCS folder to read profiles from",
        },
    )

    def __init__(
        self,
        namespace: str | None = None,
        image: str | None = None,
        tpu_accelerator: str | None = None,
        tpu_topology: str | None = None,
        tpu_chips: int = DEFAULT_CHIPS,
        queue: str | None = None,
        priority: str | None = None,
        declared_duration_minutes: int | None = None,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
        python_interpreter: str = "python3",
        mldiagnostics: bool = False,
        staging_bucket: str | None = None,
        shm_size_gb: int = DEFAULT_SHM_SIZE_GB,
        job_name_prefix: str | None = None,
        staging_service_account: str | None = None,
        staging_service_account_key: str | None = None,
        colocate_with: str | None = None,
        colocate_requests: dict[str, str] | None = None,
        pod_service_account: str | None = None,
        hf_cache_pvc: str | None = None,
        cpu_requests: dict[str, str] | None = None,
        compile_cache_max_age_days: int = COMPILE_CACHE_MAX_AGE_DAYS,
    ) -> None:
        self.queue = queue
        # Unset (absent or blank) means `default`, as for kubectl and gke_pod.
        # It is never derived from the queue: Kueue needs the LocalQueue in
        # the Job's namespace, not a namespace named after it, and XPK and
        # Kueue's own quickstart both put theirs in `default`.
        self.namespace = (namespace or "").strip() or DEFAULT_NAMESPACE
        self.image = image
        self.tpu_accelerator = tpu_accelerator
        self.tpu_topology = tpu_topology
        self.tpu_chips = tpu_chips
        self.mldiagnostics = mldiagnostics
        if priority is not None and priority not in VALID_PRIORITIES:
            raise base.ComputeConfigError(
                f"Backend 'gke_job' priority must be one of "
                f"{sorted(VALID_PRIORITIES)}, got {priority!r}.",
                error_code="INVALID_FIELD",
            )
        self.priority = priority
        self.declared_duration_minutes = declared_duration_minutes
        self.kubeconfig = kubeconfig
        # Every kubectl call names this context, so a later
        # `kubectl config use-context` can't move the profile's Jobs.
        self.kube_context = kube_context
        self.python_interpreter = python_interpreter
        self.staging_bucket = staging_bucket
        self.shm_size_gb = shm_size_gb
        self.job_name_prefix = job_name_prefix
        if staging_service_account and staging_service_account_key:
            gcs_staging.staging_credentials(staging_service_account,
                                            staging_service_account_key)
        self.staging_service_account = staging_service_account
        self.staging_service_account_key = staging_service_account_key
        # The Kubernetes service account the pod runs as. Set, the pod moves
        # staged files with its own Workload Identity token and nothing is
        # signed, so a person's own login can stage files of any size.
        self.pod_service_account = (pod_service_account or "").strip() or None
        self._staging_client: Any = None
        # `_wait`'s latest (status, seconds since submit, ever RUNNING), for a
        # `run` that is killed before it can return.
        self._last_poll: tuple[base.AsynchronousExecutionResult, int,
                               bool] | None = None
        self.colocate_with = colocate_with
        self.colocate_requests = colocate_requests or {}
        self.hf_cache_pvc = hf_cache_pvc
        self.cpu_requests = cpu_requests or {}
        self.compile_cache_max_age_days = compile_cache_max_age_days

    # -- construction ------------------------------------------------------

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> GkeJobDriver:
        """Build a driver, validating only what every path needs.

        `image`, `tpu_accelerator` and `tpu_topology` are checked at dispatch
        rather than here: an exec-into-pod caller supplies none of them, and
        rejecting that configuration up front would make `inspect` unusable on
        a cluster nobody has picked a node pool for yet.
        """
        # Absent means "derive from the topology"; 0 means someone said 0 and
        # is wrong, and must still be rejected below. `or` would conflate them.
        chips = params.get("tpu_chips")
        if chips is None:
            chips = chips_for_topology(params.get("tpu_topology", ""))
        chips = _positive_int("tpu_chips", chips)

        declared_duration = params.get("declared_duration_minutes")
        if declared_duration is not None and declared_duration != "":
            try:
                declared_duration = int(declared_duration)
            except (TypeError, ValueError) as exc:
                raise base.ComputeConfigError(
                    f"Backend 'gke_job' needs an integer "
                    f"'declared_duration_minutes', got {declared_duration!r}.",
                    error_code="INVALID_FIELD",
                ) from exc
            if declared_duration == 0:
                declared_duration = None
            elif declared_duration < 0:
                raise base.ComputeConfigError(
                    f"Backend 'gke_job' needs 'declared_duration_minutes' >= 1, "
                    f"got {declared_duration}.",
                    error_code="INVALID_FIELD",
                )
        else:
            declared_duration = None

        priority = params.get("priority") or None
        if priority is not None and priority not in VALID_PRIORITIES:
            raise base.ComputeConfigError(
                f"Backend 'gke_job' priority must be one of "
                f"{sorted(VALID_PRIORITIES)}, got {priority!r}.",
                error_code="INVALID_FIELD",
            )

        queue = params.get("queue") or None

        # Absent or blank takes the default; 0 is rejected rather than read as
        # "none", since a pod without the volume is the failure this prevents.
        shm_size_gb = params.get("shm_size_gb")
        if shm_size_gb is None or shm_size_gb == "":
            shm_size_gb = DEFAULT_SHM_SIZE_GB
        shm_size_gb = _positive_int("shm_size_gb", shm_size_gb)

        raw_prefix = (params.get("job_name_prefix") or "").strip()
        job_name_prefix = _k8s_name(raw_prefix) or None
        if raw_prefix and not (job_name_prefix and len(job_name_prefix)
                               <= MAX_JOB_NAME_PREFIX_LEN):
            raise base.ComputeConfigError(
                f"Backend 'gke_job' 'job_name_prefix' needs 1 to "
                f"{MAX_JOB_NAME_PREFIX_LEN} letters, digits or '-', got "
                f"{raw_prefix!r}.",
                error_code="INVALID_FIELD",
            )

        colocate_with = params.get("colocate_with") or None
        colocate_requests = _parse_requests(params.get("colocate_requests"))
        # Kueue admits nothing that requests nothing, and what a client needs
        # depends on the client, so there is no default.
        if colocate_with and not colocate_requests:
            raise base.ComputeConfigError(
                "Backend 'gke_job' needs 'colocate_requests' "
                "(--colocate-requests NAME=QTY,...) with 'colocate_with'.",
                error_code="MISSING_FIELD",
            )

        max_age = params.get("compile_cache_max_age_days")
        if max_age is None or max_age == "":
            max_age = COMPILE_CACHE_MAX_AGE_DAYS
        max_age = _positive_int("compile_cache_max_age_days", max_age, 0)

        cpu_requests = _parse_requests(params.get("cpu_requests"),
                                       "cpu_requests")
        if cpu_requests and colocate_with:
            # A colocated Job already runs without chips, on a node another
            # Job holds; a CPU Job is placed by the cluster, on no TPU node.
            raise base.ComputeConfigError(
                "Backend 'gke_job' takes 'cpu_requests' or 'colocate_with', "
                "not both.",
                error_code="INVALID_FIELD",
            )

        # Unset (absent or blank) means no shared cache: every pod downloads
        # into its own filesystem, as before.
        hf_cache_pvc = (params.get("hf_cache_pvc") or "").strip() or None
        if hf_cache_pvc and not PVC_NAME.fullmatch(hf_cache_pvc):
            raise base.ComputeConfigError(
                f"Backend 'gke_job' 'hf_cache_pvc' must be the name of a "
                f"PersistentVolumeClaim (lowercase letters, digits, '-' or "
                f"'.'), got {hf_cache_pvc!r}.",
                error_code="INVALID_FIELD",
            )

        return cls(
            namespace=params.get("namespace"),
            image=params.get("image") or None,
            tpu_accelerator=params.get("tpu_accelerator") or None,
            tpu_topology=params.get("tpu_topology") or None,
            tpu_chips=chips,
            queue=queue,
            priority=priority,
            declared_duration_minutes=declared_duration,
            kubeconfig=params.get("kubeconfig") or None,
            kube_context=(params.get("kube_context") or "").strip() or None,
            python_interpreter=params.get("python_interpreter") or "python3",
            mldiagnostics=bool(params.get("mldiagnostics")),
            staging_bucket=params.get("staging_bucket") or None,
            shm_size_gb=shm_size_gb,
            job_name_prefix=job_name_prefix,
            staging_service_account=params.get(
                gcs_staging.SERVICE_ACCOUNT_PARAM) or None,
            staging_service_account_key=params.get(gcs_staging.KEY_FILE_PARAM)
            or None,
            colocate_with=colocate_with,
            colocate_requests=colocate_requests,
            pod_service_account=params.get(
                gcs_staging.POD_SERVICE_ACCOUNT_PARAM) or None,
            hf_cache_pvc=hf_cache_pvc,
            cpu_requests=cpu_requests,
            compile_cache_max_age_days=max_age,
        )

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        cls.add_cli_arg(
            parser,
            "--namespace",
            help="Kubernetes namespace to dispatch into (default: default)",
        )
        cls.add_cli_arg(
            parser,
            "--mldiagnostics",
            action="store_true",
            help="Label the Job for GKE managed ML diagnostics, so "
            "the workload appears in the Console Diagnostics view",
        )
        cls.add_cli_arg(
            parser,
            "--image",
            help="Container image the dispatched Job runs, unmodified",
        )
        cls.add_cli_arg(
            parser,
            "--tpu-accelerator",
            help=("Value of the cloud.google.com/gke-tpu-accelerator node "
                  "label, e.g. tpu7x or tpu-v6e-slice"),
        )
        cls.add_cli_arg(
            parser,
            "--tpu-topology",
            help=("Value of the cloud.google.com/gke-tpu-topology node label, "
                  "e.g. 2x2x1 or 2x2"),
        )
        cls.add_cli_arg(
            parser,
            "--tpu-chips",
            type=int,
            help=f"Chips to request per pod (default: {DEFAULT_CHIPS})",
        )
        cls.add_cli_arg(
            parser,
            "--queue",
            help="Kueue LocalQueue to admit through; omit on a direct cluster",
        )
        cls.add_cli_arg(
            parser,
            "--priority",
            help="Kueue priority class, one of " +
            ", ".join(sorted(VALID_PRIORITIES)) +
            "; omit to leave the Job unlabelled and take the cluster's default",
        )
        cls.add_cli_arg(
            parser,
            "--declared-duration-minutes",
            type=int,
            help="Declared runtime in minutes for Kueue suspension enforcement",
        )
        cls.add_cli_arg(
            parser,
            "--kubeconfig",
            help="Kubeconfig to use instead of the ambient one",
        )
        cls.add_cli_arg(
            parser,
            "--kube-context",
            help=base.KUBE_CONTEXT_HELP,
        )
        cls.add_cli_arg(
            parser,
            "--staging-bucket",
            help="GCS bucket URI for file staging via signed URLs",
        )
        cls.add_cli_arg(
            parser,
            "--staging-service-account",
            help="Service account that staging acts as, using your own "
            "login (no key file); kubectl keeps your identity",
        )
        cls.add_cli_arg(
            parser,
            "--staging-service-account-key",
            help="Key file of the service account staging uses, when you "
            "cannot use --staging-service-account; kubectl keeps your "
            "identity, unlike $GOOGLE_APPLICATION_CREDENTIALS",
        )
        cls.add_cli_arg(
            parser,
            "--pod-service-account",
            help="Kubernetes service account the Job's pod runs as. On a "
            "cluster with Workload Identity, the pod then moves staged files "
            "with its own token, so nothing is signed",
        )
        cls.add_cli_arg(
            parser,
            "--shm-size-gb",
            type=int,
            help=f"Size in GiB of the memory-backed /dev/shm mounted in the "
            f"pod (default: {DEFAULT_SHM_SIZE_GB})",
        )
        cls.add_cli_arg(
            parser,
            "--job-name-prefix",
            help="Prefix added to the name of every Job this backend "
            "creates, e.g. 'agent'",
        )
        cls.add_cli_arg(
            parser,
            "--hf-cache-pvc",
            help="ReadWriteMany PersistentVolumeClaim, made by a cluster "
            f"admin, mounted at {HF_CACHE_MOUNT} in every pod as "
            f"{HF_CACHE_ENV} (with vLLM's and torch_tpu's compile caches "
            f"under {HF_CACHE_MOUNT}/{COMPILE_CACHE_SUBDIR}/), so model "
            "weights download once and compiled programs are kept across Jobs",
        )
        cls.add_cli_arg(
            parser,
            "--compile-cache-max-age-days",
            type=int,
            help="With --hf-cache-pvc, each pod removes compile-cache image "
            "folders no pod has used in this many days (default: "
            f"{COMPILE_CACHE_MAX_AGE_DAYS}; 0 keeps them)",
        )
        cls.add_cli_arg(
            parser,
            "--colocate-with",
            metavar="JOB_ID",
            help="Run on the node of running Job JOB_ID, without chips (it "
            "holds them). For a client of a server Job, e.g. a benchmark: it "
            "reaches the server at http://JOB_ID:PORT without exec or a "
            "port-forward",
        )
        cls.add_cli_arg(
            parser,
            "--cpu-requests",
            metavar="NAME=QTY,...",
            help="Run on a CPU node, holding no TPU: the Job requests this "
            "CPU and memory instead of chips, e.g. cpu=2,memory=8Gi, and has "
            "no TPU node selector or toleration, so it books no TPU node. "
            "For work that needs the task image but not a chip. The Job gets "
            "the profile's TPU generation in TPU_AGENT_SUITE_CHIPLESS_TPU; a "
            "`run` that exits 75 or never starts is run again on a TPU node",
        )
        cls.add_cli_arg(
            parser,
            "--colocate-requests",
            metavar="NAME=QTY,...",
            help="Required with --colocate-with: the CPU and memory it "
            "requests instead of chips, e.g. cpu=2,memory=8Gi, sized to the "
            "client. Kueue admits no pod that requests nothing",
        )

    # -- kubectl plumbing --------------------------------------------------

    def needed_tools(self) -> tuple[str, ...]:
        return base.gke_tools(self._kubectl_cmd([]))

    def _kubectl(
        self,
        argv: list[str],
        stdin: str | None = None,
        check: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one kubectl command with the driver's namespace and kubeconfig."""
        try:
            proc = auth_utils.run_kubectl(self,
                                          self._kubectl_cmd(argv),
                                          kubeconfig=self.kubeconfig,
                                          raise_refused=check,
                                          input=stdin,
                                          timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise KubectlError(argv, f"no answer within {timeout}s") from exc
        if check and proc.returncode != 0:
            raise KubectlError(argv, proc.stderr)
        return proc

    def _kubectl_cmd(self, argv: list[str]) -> list[str]:
        """The kubectl command line for `argv`, with the driver's kubeconfig
        and context."""
        return base.kubectl_prefix(self.kubeconfig, self.kube_context) + argv

    def _kubectl_json(self, argv: list[str]) -> dict[str, Any]:
        proc = self._kubectl(argv + ["-o", "json"],
                             timeout=KUBECTL_READ_TIMEOUT_SEC)
        try:
            return json.loads(proc.stdout)
        except ValueError as exc:
            raise base.ComputeConfigError(
                f"kubectl {' '.join(argv)} returned unparseable JSON.",
                error_code="KUBECTL_BAD_OUTPUT",
            ) from exc

    def _ns(self) -> list[str]:
        return ["-n", self.namespace]

    # -- staging credentials -----------------------------------------------

    def _staging(self, best_effort: bool = False) -> dict[str, Any]:
        """`client=` for gcs_staging calls, when staging has its own credential.

        Empty when none is configured, and the calls fall back on ADC as
        before. Configured, one client carrying it serves every call, so the
        signed URLs, uploads, downloads and cleanup all act as the same
        account -- and none of it goes through the process environment that
        kubectl inherits. `best_effort` callers (cleanup) get ADC rather than
        an error when the credential cannot be built.
        """
        if not (self.staging_service_account
                or self.staging_service_account_key):
            return {}
        if self._staging_client is None:
            try:
                self._staging_client = gcs_staging.get_storage_client(
                    credentials=gcs_staging.staging_credentials(
                        self.staging_service_account,
                        self.staging_service_account_key))
            except base.ComputeConfigError:
                if best_effort:
                    return {}
                raise
        return {"client": self._staging_client}

    # -- inspect -----------------------------------------------------------

    def inspect(self) -> base.HardwareProfile:
        """Report the TPU node pool this driver would dispatch onto.

        Reads nodes rather than a cluster description, because the node labels
        are what scheduling actually keys on -- a cluster named after a chip
        generation proves nothing about the pool underneath it.
        """
        selector = self._node_selector()
        nodes = self._kubectl_json(["get", "nodes", "-l", selector])
        items = nodes.get("items", [])
        if not items:
            details: dict[str, Any] = {"selector": selector}
            if self.tpu_accelerator:
                # An autoscaled pool with no node yet, or a profile that names
                # a pool the cluster lacks: a Job still lands on what the
                # profile names, once a node comes up. What it will get is
                # known from the profile, so work that needs no chip (a CPU
                # Job planning for it) need not wait for a node to say so.
                details["tpu_accelerator"] = self.tpu_accelerator
                details["nodes_note"] = (
                    "no node matches the selector: the pool may be scaled to "
                    "zero, and a Job waits for one; job_chips, job_devices "
                    "and hbm_gb_per_device come from the profile")
                self._job_sizing(self.tpu_accelerator, details)
            return base.HardwareProfile(
                accelerator="none",
                topology="none",
                chips=0,
                hbm_gb=0.0,
                status="no_accelerator",
                available_chips=0,
                details=details,
            )

        labels = items[0].get("metadata", {}).get("labels", {})
        accelerator = labels.get(ACCELERATOR_LABEL, "unknown")
        topology = labels.get(TOPOLOGY_LABEL, "unknown")
        chips = 0
        node_names = []
        for node in items:
            node_names.append(node.get("metadata", {}).get("name", ""))
            allocatable = node.get("status", {}).get("allocatable", {})
            chips += int(allocatable.get(TPU_RESOURCE, 0) or 0)

        held_by, used = self._chips_in_use()
        # `None`, not `chips`, when the pod list was unreadable: a namespace
        # this identity cannot list pods in is a measurement failure, and
        # reporting the pool as fully free would send a scheduler at chips
        # somebody else is holding.
        available = None if used is None else max(chips - used, 0)

        details: dict[str, Any] = {
            "nodes": node_names,
            "namespace": self.namespace,
            "selector": selector,
            "tpu_accelerator": accelerator,
        }
        hbm_gb = self._job_sizing(accelerator, details)
        if self.queue:
            details["queue"] = self.queue
        if self.priority:
            details["priority"] = self.priority
        if self.declared_duration_minutes:
            details[
                "declared_duration_minutes"] = self.declared_duration_minutes

        status = "healthy"
        if available == 0:
            status = "busy"
        elif available is None:
            status = "unknown_contention"

        return base.HardwareProfile(
            accelerator=accelerator,
            topology=topology,
            chips=chips,
            hbm_gb=hbm_gb,
            status=status,
            available_chips=available,
            held_by=held_by,
            details=details,
        )

    def _job_sizing(self, accelerator: str, details: dict[str, Any]) -> float:
        """Add what one Job gets on `accelerator` to `details`; return the
        HBM per chip, 0.0 when unknown."""
        # `chips` counts the whole pool, but one Job gets `tpu_chips` of them,
        # and a Job's device count is what tensor parallelism is sized to.
        details["job_chips"] = self.tpu_chips
        spec, note = _chip_spec(accelerator)
        if not spec:
            details["hbm_gb_note"] = note
            return 0.0
        devices_per_chip = spec["devices_per_chip"]
        details["devices_per_chip"] = devices_per_chip
        details["job_devices"] = self.tpu_chips * devices_per_chip
        details["hbm_gb_per_device"] = spec["hbm_gb"] / devices_per_chip
        return spec["hbm_gb"]

    def describe_target(self) -> str:
        where = (f"kubeconfig {self.kubeconfig}"
                 if self.kubeconfig else "the current kubectl context")
        if self.kube_context:
            where = f"context '{self.kube_context}'" + (
                f" of kubeconfig {self.kubeconfig}" if self.kubeconfig else "")
        queue = f", Kueue queue '{self.queue}'" if self.queue else ""
        return (f"GKE cluster of {where}, namespace '{self.namespace}'"
                f"{queue}")

    def _node_selector(self) -> str:
        """The label selector for the nodes a Job of this driver lands on."""
        selector = f"{ACCELERATOR_LABEL}"
        if self.tpu_accelerator:
            selector = f"{ACCELERATOR_LABEL}={self.tpu_accelerator}"
        if self.tpu_topology:
            selector += f",{TOPOLOGY_LABEL}={self.tpu_topology}"
        if self.pod_service_account:
            selector += f",{WORKLOAD_IDENTITY_LABEL}=true"
        return selector

    def _preflight_target(self, checks: list[base.PreflightCheck]) -> bool:
        """The cluster answers, a Job can be built and created, and the
        queue and staging bucket it would use are there."""
        kubectl = self._kubectl_cmd([])
        # Without a cluster to probe, the profile's own settings are still
        # checked below.
        reachable = auth_utils.cluster_check(self, kubectl, checks,
                                             self.kubeconfig, self.namespace)
        try:
            self._require_dispatch_config()
        except base.ComputeConfigError as exc:
            base.fail(checks, "dispatch_settings", str(exc))
        image = _image_check(self.image) if self.image else None
        if image:
            checks.append(image)
        if (self.tpu_accelerator
                and self.tpu_accelerator not in KNOWN_ACCELERATOR_LABELS):
            # Only a warning: a generation newer than this table is right.
            checks.append(
                base.PreflightCheck(
                    "tpu_accelerator", "warn",
                    f"'{self.tpu_accelerator}' is not a known "
                    f"{ACCELERATOR_LABEL} value; a Job that asks for a "
                    "label no node has stays Pending", "Use one of " +
                    ", ".join(sorted(KNOWN_ACCELERATOR_LABELS)) +
                    ", or ignore this if it is a newer generation"))
        if reachable:
            pools = self._preflight_cluster(kubectl, checks)
            if image and image.status == "ok":
                pull = _node_pull_check(self.image,
                                        base.current_context(kubectl), pools,
                                        self.tpu_topology)
                if pull:
                    checks.append(pull)
        if self.staging_bucket:
            checks.extend(self._staging_preflight())
        return reachable

    def _staging_preflight(self) -> list[base.PreflightCheck]:
        """The staging checks, made with the credential staging will use."""
        bucket = gcs_staging.clean_bucket_name(self.staging_bucket or "")
        try:
            credentials = gcs_staging.staging_credentials(
                self.staging_service_account, self.staging_service_account_key)
        except base.ComputeConfigError as exc:
            return [
                base.PreflightCheck(
                    "staging:credentials", "fail", str(exc),
                    f"Fix or unset '{gcs_staging.SERVICE_ACCOUNT_PARAM}' / "
                    f"'{gcs_staging.KEY_FILE_PARAM}'")
            ]
        return gcs_staging.preflight_checks(
            bucket,
            credentials=credentials,
            pod_service_account=self.pod_service_account)

    def _preflight_cluster(self, kubectl: list[str],
                           checks: list[base.PreflightCheck]) -> set[str]:
        """The namespace, the right to create Jobs, and Kueue's objects.

        Returns the node pools of the TPU nodes a Job would land on.
        """
        namespaced = kubectl + ["-n", self.namespace]
        if self.namespace != DEFAULT_NAMESPACE:
            checks.append(
                _kube_get_check(
                    kubectl, "namespace", f"namespace '{self.namespace}'",
                    ["namespace", self.namespace],
                    f"kubectl create namespace {self.namespace}, or set "
                    "namespace to an existing one"))
        pools: set[str] = set()
        if self.tpu_accelerator and self.tpu_topology:
            nodes, pools = self._preflight_nodes(kubectl)
            checks.append(nodes)
        # A run creates the ConfigMap carrying the script, then the Job.
        for resource in ("configmaps", "jobs"):
            checks.append(_kube_can_i_check(namespaced, "create", resource))
        if self.staging_bucket and not self.pod_service_account:
            # The pod token is stored in a Secret. Only warn: without it the
            # Job still runs, it just can't reach GCS.
            secrets = _kube_can_i_check(namespaced, "create", "secrets")
            if secrets.status == "fail":
                secrets = dataclasses.replace(
                    secrets,
                    status="warn",
                    detail=f"{secrets.detail}, so pods won't get a token for "
                    "the staging bucket")
            checks.append(secrets)
        if self.pod_service_account:
            # A Job naming a missing account is accepted, and its pod is
            # never created: the Job just sits there.
            checks.append(
                _kube_get_check(
                    namespaced, "pod_service_account",
                    f"service account '{self.pod_service_account}' in "
                    f"namespace '{self.namespace}'",
                    ["serviceaccount", self.pod_service_account],
                    f"kubectl create serviceaccount "
                    f"{self.pod_service_account} -n {self.namespace}, then "
                    "grant roles/storage.objectAdmin on the staging bucket "
                    "either to its Workload Identity principal directly, or "
                    "to a Google service account it names in an "
                    "iam.gke.io/gcp-service-account annotation (setup step "
                    "4 in docs/install.md)"))
        if self.hf_cache_pvc:
            checks.append(
                _hf_cache_pvc_check(namespaced, kubectl, self.hf_cache_pvc,
                                    self.namespace))
        if not self.queue:
            return pools
        # One call answers both "is Kueue installed" (the resource type
        # exists) and "is the queue there", without the cluster-wide read
        # that listing CRDs needs.
        checks.append(
            _kube_get_check(
                namespaced, "kueue:queue",
                f"LocalQueue '{self.queue}' in namespace '{self.namespace}'",
                ["localqueue", self.queue],
                "Install Kueue on the cluster and create the LocalQueue "
                "(https://kueue.sigs.k8s.io/docs/installation/), or unset "
                "queue to schedule Jobs directly"))
        # The priority label is applied only on a queued Job.
        if self.priority:
            checks.append(
                _kube_get_check(
                    kubectl, "kueue:priority",
                    f"WorkloadPriorityClass '{self.priority}'",
                    ["workloadpriorityclass", self.priority],
                    "Ask a cluster admin to create the "
                    "WorkloadPriorityClass, or unset priority"))
        return pools

    def _preflight_nodes(
            self, kubectl: list[str]) -> tuple[base.PreflightCheck, set[str]]:
        """Whether any node carries the labels a Job will select on, and
        the node pools of those that do.

        Never a failure: a node pool that scales from zero, or one node
        auto-provisioning creates on demand, has no nodes until a Job asks,
        and that looks the same here as a label nothing will ever match.
        TPU nodes that only lack the Workload Identity label get a warning
        of their own, with the fix: auto-provisioning can still add a pool
        that has it.
        """
        selector = self._node_selector()
        proc = _list_node_pools(kubectl, selector)
        if proc.returncode != 0:
            return base.PreflightCheck(
                "tpu_nodes", "warn",
                f"could not list nodes: {base.probe_error(proc)}"), set()
        nodes = proc.stdout.splitlines()
        if nodes:
            return base.PreflightCheck(
                "tpu_nodes", "ok",
                f"{len(nodes)} node(s) match {selector}"), set(nodes) - {""}
        if self.pod_service_account:
            tpu_selector = selector.removesuffix(
                f",{WORKLOAD_IDENTITY_LABEL}=true")
            tpu = _list_node_pools(kubectl, tpu_selector)
            pools = sorted(set(tpu.stdout.splitlines()) - {""})
            if tpu.returncode == 0 and pools:
                return base.PreflightCheck(
                    "tpu_nodes", "warn",
                    f"TPU node pool(s) {', '.join(pools)} match "
                    f"{tpu_selector} but don't run the GKE metadata server, "
                    f"which {gcs_staging.POD_SERVICE_ACCOUNT_PARAM} needs: "
                    "Jobs won't schedule on them",
                    "gcloud container node-pools update <pool> "
                    "--workload-metadata=GKE_METADATA, or unset "
                    f"{gcs_staging.POD_SERVICE_ACCOUNT_PARAM}. Fine if "
                    "auto-provisioning will add a pool with Workload "
                    "Identity"), set()
        return base.PreflightCheck(
            "tpu_nodes", "warn", f"no node matches {selector} right now",
            "Fine for a node pool that scales from zero or is auto-"
            "provisioned; otherwise check tpu_accelerator and tpu_topology "
            "against the TPU node pool's labels"), set()

    def _chips_in_use(self) -> tuple[list[str], int | None]:
        """Pods holding TPU chips right now, and how many chips that is.

        Returns `(names, None)` when the pods cannot be listed, so the caller
        can tell "nothing is holding chips" from "nobody told me".
        """
        proc = self._kubectl(
            ["get", "pods", "--all-namespaces", "-o", "json"],
            check=False,
        )
        if proc.returncode != 0:
            return [], None
        try:
            payload = json.loads(proc.stdout)
        except ValueError:
            return [], None

        held: list[str] = []
        total = 0
        for pod in payload.get("items", []):
            phase = pod.get("status", {}).get("phase", "")
            if phase in ("Succeeded", "Failed"):
                continue
            # Only a scheduled pod holds chips. A Pending one with no node
            # is waiting for them, and counting it read a full cluster as
            # fuller, and an empty one as busy.
            if not pod.get("spec", {}).get("nodeName"):
                continue
            requested = 0
            for container in pod.get("spec", {}).get("containers", []):
                limits = container.get("resources", {}).get("limits", {})
                requested += int(limits.get(TPU_RESOURCE, 0) or 0)
            if requested:
                meta = pod.get("metadata", {})
                held.append(f"{meta.get('namespace')}/{meta.get('name')}")
                total += requested
        return held, total

    # -- dispatch ----------------------------------------------------------

    def _clean_bucket(self, needed: bool) -> str:
        """The staging bucket, demanded only when files actually need one."""
        clean = (gcs_staging.clean_bucket_name(self.staging_bucket)
                 if self.staging_bucket else "")
        if needed and not clean:
            raise self.not_configured(
                "staging_bucket", "These --input files are over "
                f"{gcs_staging.CONFIGMAP_MAX_BYTES // 1024} KiB in total "
                "(script included), too large for the Job's ConfigMap.")
        return clean

    def _stage(
        self,
        staging: _Staging,
        name: str,
        script: Path,
        inputs: Sequence[base.StagedFile],
        outputs: Sequence[base.StagedFile],
        url_exp: int,
    ) -> None:
        """Move the inputs and arrange the outputs' way back, into `staging`.

        Inputs that fit ride in the ConfigMap, and outputs come back in the
        pod's log unless the staging bucket can take them: neither needs a
        bucket or credentials that can sign, which a person's own ADC cannot.
        """
        for staged in inputs:
            shell.require_file(staged.local, f"--input {staged.local}")
        size = sum(
            Path(path).stat().st_size
            for path in [script, *(staged.local for staged in inputs)])
        mount = size <= gcs_staging.CONFIGMAP_MAX_BYTES
        bucket = self._clean_bucket(needed=bool(inputs) and not mount)
        for index, staged in enumerate(inputs):
            if mount:
                key = INPUT_KEY.format(index)
                staging.configmap_files.append(
                    f"--from-file={key}={staged.local}")
                staging.mounted_inputs.append((staged.remote, key))
                continue
            blob_name = f"{name}/in/{staged.remote}"
            staging.uploaded = True
            gcs_staging.upload_file_to_gcs(staged.local, bucket, blob_name,
                                           **self._staging())
            staging.in_urls.append(
                (staged.remote, self._pod_url(bucket, blob_name, "GET",
                                              url_exp)))
        if outputs and bucket:
            try:
                staging.out_urls = [
                    (staged.remote,
                     self._pod_url(bucket, f"{name}/out/{staged.remote}",
                                   "PUT", url_exp)) for staged in outputs
                ]
            except base.ComputeConfigError as exc:
                staging.logged_because = exc.error_code
                print(
                    "gke_job: returning --output files through the pod's "
                    f"log -- {exc.error_code}: could not sign upload URLs; "
                    "outputs over "
                    f"{gcs_staging.LOGGED_OUTPUT_MAX_BYTES // 1024} KiB in all "
                    "will not come back (see `compute preflight`)",
                    file=sys.stderr)
        if outputs and not staging.out_urls:
            staging.logged_outputs = [staged.remote for staged in outputs]
            staging.logged_because = staging.logged_because or NO_BUCKET

    def _pod_url(self, bucket: str, blob_name: str, method: str,
                 expiration: int) -> str:
        """The URL the pod moves `blob_name` through.

        Signed, unless the pod runs as `pod_service_account`: it then sends
        its own Workload Identity token to the plain object URL, and nothing
        here needs credentials that can sign.
        """
        if self.pod_service_account:
            return gcs_staging.object_url(bucket, blob_name)
        return gcs_staging.generate_v4_signed_url(
            bucket,
            blob_name,
            method=method,
            expiration_seconds=expiration,
            **self._staging())

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
        """Execute a script and wait for its output.

        With `cpu_requests`, a run that never starts (no CPU node took it) or
        exits NEEDS_TPU_EXIT_CODE (it found it needs a chip) is run again
        once on a TPU node, and `details.cpu_fallback` says why.
        """
        kwargs = dict(code_path=code_path,
                      timeout=timeout,
                      args=args,
                      python_interpreter=python_interpreter,
                      inputs=inputs,
                      outputs=outputs,
                      job_name=job_name)
        result = self._run_once(**kwargs)
        if not self.cpu_requests:
            return result
        if result.details.get("never_started"):
            why = "no CPU node took it"
        elif result.exit_code == NEEDS_TPU_EXIT_CODE:
            why = (f"it exited {NEEDS_TPU_EXIT_CODE}: it needs a chip, "
                   "or could not do without one")
        else:
            return result
        print(
            f"gke_job: the CPU run {result.details.get('job_id')} did not "
            f"do the work ({why}); running it again on a TPU node.",
            file=sys.stderr,
            flush=True)
        cpu_requests, self.cpu_requests = self.cpu_requests, {}
        try:
            result = self._run_once(**kwargs)
        finally:
            self.cpu_requests = cpu_requests
        result.details["cpu_fallback"] = why
        return result

    def _run_once(
        self,
        code_path: str,
        timeout: int | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        inputs: list[base.StagedFile] | None = None,
        outputs: list[base.StagedFile] | None = None,
        job_name: str | None = None,
    ) -> base.SynchronousExecutionResult:
        """One `run`: execute a script and wait for its output.

        Dispatches a Job and blocks on it, so the result is the pod's own logs
        and exit code. It cleans up after itself -- reading the logs first,
        since deleting a Job deletes its pod and takes the logs with it, and a
        Job left holding chips keeps the next run pending.
        """
        outputs = outputs or []
        clean_bucket = self._clean_bucket(needed=False)

        # A requested name gets a random suffix so parallel runs don't collide.
        job_name = self._job_name(job_name, unique=True)
        # A synchronous run is over when the wait is: the URLs need to outlive
        # the wait and nothing more. `_wait` budgets the queue and the run
        # separately, so that is up to twice the timeout.
        job_id = self.submit(
            code_path,
            job_name=job_name,
            args=args,
            python_interpreter=python_interpreter or self.python_interpreter,
            # Tagged like a submitted job, so the run-end reaper finds this
            # one if the run is killed before its own cleanup runs.
            run_id=self.run_id,
            inputs=inputs,
            outputs=outputs,
            url_exp_sec=min(2 * (timeout or DEFAULT_WAIT_SEC),
                            SUBMIT_URL_EXP_SEC),
        )
        # `is None`, not falsy: `--timeout 0` means do not wait, and folding it
        # into the default turns a poll into a fifteen-minute block.
        wait_for = DEFAULT_WAIT_SEC if timeout is None else timeout
        status: base.AsynchronousExecutionResult | None = None
        never_ran = False
        self._last_poll = None
        # The caller's shell may SIGTERM a long queue wait. Say what the Job
        # was doing, then let the delete in `finally` run.
        with shell.sigterm_raises():
            try:
                status = self._wait(job_id, wait_for)
                self._settle_record(job_id, status)
                # Image never pulled, Kueue refused, or timed out in the
                # queue: no container ran, so there are no logs or outputs
                # to ask for, and every read skipped is one less step before
                # the delete.
                never_ran = bool(status.details.get("never_started"))
                logs, carried = (("", {}) if never_ran else
                                 gcs_staging.split_logged_outputs(
                                     self._read_logs(job_id), job_id))
                stderr = status.error or ""
                if never_ran:
                    # `job_status` below is the status at the timeout; the
                    # Job itself is deleted on the way out.
                    stderr = f"{stderr} The Job has been deleted.".strip()
                result = base.SynchronousExecutionResult(
                    exit_code=status.exit_code if status.exit_code is not None
                    else (0 if status.status == "COMPLETED" else 1),
                    stdout=logs,
                    stderr=stderr,
                    details={
                        "job_id": job_id,
                        "job_status": status.status,
                        **status.details,
                    },
                )
                if never_ran:
                    for staged in outputs:
                        result = shell._missing_output(
                            result, staged, "the Job never ran, so it was "
                            "never written")
                    return result
                job = jobs.JobRecord.load(job_id)
                missing, fields = self._collect_outputs(
                    job_id, base.StagedFile.as_records(outputs), carried,
                    job.extra.get("outputs_in_log") if job else None,
                    lambda record: Path(record["local"]))
                for staged in outputs:
                    if staged.remote in missing:
                        result = shell._missing_output(result, staged,
                                                       missing[staged.remote])
                if fields:
                    # The CLI turns these into the `setup plan` command to run.
                    result.details["setup_fields"] = fields
                return result
            except SystemExit:
                # Killed from outside, usually the caller's shell timeout.
                # Without a line here the caller sees only its own timeout
                # and cannot tell a full cluster from a hang.
                print(self._killed_message(job_id), file=sys.stderr)
                raise
            finally:
                # Always delete the Job: a finished one is just leftover
                # objects, a running one holds a TPU node, and a queued one
                # would start later with nobody watching. If the delete
                # fails, print and move on so the result above still comes
                # back; the run-end reaper will try again. The delete reads
                # the Job's status again first, so one that finished since the
                # last poll is recorded COMPLETED; only a Job that never
                # started skips that, since nothing has changed to read.
                try:
                    self._delete(job_id, status if never_ran else None)
                except base.ComputeConfigError as exc:
                    print(f"gke_job: could not delete Job {job_id} -- {exc}",
                          file=sys.stderr)
                if clean_bucket:
                    gcs_staging.delete_staged_prefix(clean_bucket,
                                                     f"{job_name}/",
                                                     **self._staging(True))

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
        url_exp_sec: int = SUBMIT_URL_EXP_SEC,
    ) -> str:
        """Dispatch the script as a Job and return its name as the job id.

        The script travels in a ConfigMap mounted into the stock image, so no
        image is built and the runtime under test is the one that shipped. The
        ConfigMap takes the Job's name so a run can be found and deleted as a
        unit.
        """
        self._require_dispatch_config()
        # Before `_stage` sizes it: a missing script is then the not-found
        # error callers handle, not an OSError from the stat.
        shell.require_file(recipe_path, "recipe_path")
        recipe = Path(recipe_path)
        name = self._job_name(job_name, unique=False)
        # By extension, as every other driver does: a shell recipe handed to
        # the configured Python dies on its first line with a SyntaxError. It
        # gets that Python as TPU_AGENT_SUITE_PYTHON instead, to find tools beside.
        python = python_interpreter or self.python_interpreter
        shell_recipe = recipe.suffix == ".sh"
        interpreter = "bash" if shell_recipe else python

        inputs, outputs = inputs or [], outputs or []
        clean_bucket = self._clean_bucket(needed=False)
        staging = _Staging()
        configmap = False
        token_expires: float | None = None
        # Before anything is staged: the cleanup below deletes the ConfigMap
        # and staged files under `name`, which a name in use makes another's.
        name = self._unused_job_name(name, job_name, run_id)
        placement = None
        if self.colocate_with:
            placement = self._placement_of(self.colocate_with)
            # The server's address, before the client starts asking for it.
            # A batch server gets it here, so it needs no flag and stays
            # reapable; a --serve one already has it.
            self._expose(self.colocate_with)

        try:
            # Inside the try, not before it: an input uploaded before a later
            # one fails to upload or sign, or before a ConfigMap that fails to
            # create (name taken, admission webhook, API timeout), would
            # otherwise stay in the bucket with no JobRecord naming the prefix
            # to clean.
            self._stage(staging, name, recipe, inputs, outputs, url_exp_sec)
            log_url = self._log_url(clean_bucket, name)
            configmap = True
            try:
                self._kubectl(["create", "configmap", name] + self._ns() +
                              [f"--from-file={recipe.name}={recipe_path}"] +
                              staging.configmap_files)
            except KubectlError as exc:
                # Taken since the check: what's under `name` is another job's.
                if re.search(r"already ?exists", exc.message, re.IGNORECASE):
                    configmap = staging.uploaded = False
                raise
            token_expires = self._put_token_secret(name)
            manifest = self._job_manifest(
                name=name,
                script_name=recipe.name,
                interpreter=interpreter,
                args=args or [],
                staging=staging,
                log_url=log_url,
                serve=serve,
                placement=placement,
                shell_python=python if shell_recipe else None,
                token_secret=token_expires is not None,
            )
            # JSON on stdin rather than a templated heredoc: kubectl accepts
            # it as YAML, and a value carrying a colon or a leading dash
            # cannot break the document the way it can in hand-rolled text.
            # The UID tells this Job from an earlier or later one of the same
            # name, which Cloud Logging cannot do by name alone.
            uid = self._kubectl(
                ["apply", "-f", "-", "-o", "jsonpath={.metadata.uid}"],
                stdin=json.dumps(manifest)).stdout.strip()
            extra: dict[str, Any] = {}
            if token_expires is not None:
                # Cancel and renewal check this, so Jobs without a token
                # never touch Secrets.
                extra["token_secret"] = True
                extra["token_expires"] = token_expires
            if uid:
                extra["uid"] = uid
                if token_expires is not None:
                    self._own_token_secret(name, uid)
            if staging.in_urls or staging.out_urls:
                extra["staged_prefix"] = f"{name}/"
            if outputs:
                extra["outputs"] = base.StagedFile.as_records(outputs)
            if staging.logged_outputs:
                extra["outputs_in_log"] = staging.logged_because
            self._new_record(
                job_id=name,
                job_name=job_name or name,
                recipe_path=str(recipe),
                run_id=run_id or "",
                backend="gke_job",
                status="PENDING",
                serve=serve,
                extra=extra,
            )
        except Exception:
            # An orphaned ConfigMap is invisible and accumulates, and a
            # caller retrying under the same name would fail on the create
            # instead of the apply. Blobs uploaded for a Job that never
            # started are just as orphaned, and no record will point at them.
            if configmap:
                self._kubectl(["delete", "configmap", name] + self._ns() +
                              ["--ignore-not-found"],
                              check=False)
            if token_expires is not None:
                self._kubectl(
                    ["delete", "secret", name + TOKEN_SECRET_SUFFIX] +
                    self._ns() + ["--ignore-not-found"],
                    check=False)
            if staging.uploaded:
                gcs_staging.delete_staged_prefix(clean_bucket, f"{name}/",
                                                 **self._staging(True))
            raise
        # After the record and outside the cleanup above: the Job is already
        # running, so a Service failure must not delete its ConfigMap or lose
        # its job id.
        if serve:
            self._expose(name, uid)
        return name

    def _unused_job_name(self, name: str, requested: str | None,
                         run_id: str | None) -> str:
        """`name`, or `name` plus a random suffix if a job outside this run
        holds it. Raises NAME_TAKEN if this run's live job holds it, since
        the caller may mean that job."""
        if self._name_unused(name):
            return name
        # Cleanup and trimming can map two --name values to one Job name.
        asked = (f" (from --name {requested!r})"
                 if requested not in (None, name) else "")
        mine = jobs.JobRecord.load(name)
        if (mine is not None and run_id and mine.run_id == run_id
                and mine.status not in jobs.TERMINAL_STATUSES):
            raise base.ComputeConfigError(
                f"Job name {name!r}{asked} is already this run's job, still "
                f"{mine.status}. Wait for it with `job wait --job-id {name}`, "
                f"cancel it first with `job cancel --job-id {name}`, or "
                "choose a different --name.",
                error_code="NAME_TAKEN",
            )
        for _ in range(NAME_SUFFIX_ATTEMPTS):
            candidate = self._job_name(name, unique=True)
            if self._name_unused(candidate):
                print(
                    f"gke_job: Job name {name!r}{asked} is used by a job "
                    f"outside this run in namespace {self.namespace!r}; "
                    f"submitting as {candidate!r}.",
                    file=sys.stderr)
                return candidate
        raise base.ComputeConfigError(
            f"Job name {name!r}{asked} and {NAME_SUFFIX_ATTEMPTS} suffixed "
            f"variants are all in use in namespace {self.namespace!r}. "
            "Choose a different --name. Don't cancel or delete jobs this run "
            "didn't submit.",
            error_code="NAME_TAKEN",
        )

    def _name_unused(self, name: str) -> bool:
        """Whether no Job or ConfigMap called `name` exists. A failed lookup
        answers True, leaving the ConfigMap create as the check."""
        try:
            proc = self._kubectl(["get", "configmap,job", name] + self._ns() +
                                 ["--ignore-not-found", "-o", "name"],
                                 check=False,
                                 timeout=KUBECTL_READ_TIMEOUT_SEC)
        except KubectlError:
            return True
        return proc.returncode != 0 or not proc.stdout.strip()

    def _placement_of(self, job_id: str) -> dict[str, Any]:
        """The server pod's node, nodeSelector and tolerations.

        Copying them lands the client on the server's node and makes Kueue
        admit it under the server's TPU flavor, not a CPU pool's.
        """
        spec = (self._live_pod(self._pods(job_id)) or {}).get("spec", {})
        if not spec.get("nodeName"):
            raise base.ComputeConfigError(
                f"--colocate-with {job_id!r}: no scheduled pod. Wait for it "
                f"with `job wait --job-id {job_id} --until running`.",
                error_code="INVALID_FIELD",
            )
        return {
            "nodeSelector": {
                **spec.get("nodeSelector", {}),
                HOSTNAME_LABEL: spec["nodeName"],
            },
            "tolerations": spec.get("tolerations", []),
        }

    def _expose(self, name: str, uid: str = "") -> None:
        """Give a Job the address `<name>.<namespace>.svc`: a --serve Job when
        it is submitted, any other when a --colocate-with client is.

        A headless Service owned by the Job, so deleting the Job deletes it.
        Applying it again is harmless. Best effort: a client can still reach
        the pod IP without it. The address alone does not exempt the Job from
        reaping; only --serve does.
        """
        try:
            uid = uid or self._kubectl(["get", "job", name] + self._ns() +
                                       ["-o", "jsonpath={.metadata.uid}"],
                                       timeout=KUBECTL_READ_TIMEOUT_SEC).stdout
            self._kubectl(["apply", "-f", "-"],
                          stdin=json.dumps({
                              "apiVersion": "v1",
                              "kind": "Service",
                              "metadata": {
                                  "name":
                                  name,
                                  "namespace":
                                  self.namespace,
                                  "ownerReferences": [{
                                      "apiVersion": "batch/v1",
                                      "kind": "Job",
                                      "name": name,
                                      "uid": uid.strip(),
                                  }],
                              },
                              "spec": {
                                  "clusterIP": "None",
                                  "selector": {
                                      "job-name": name
                                  }
                              },
                          }),
                          timeout=KUBECTL_READ_TIMEOUT_SEC)
        # ComputeConfigError, not just KubectlError: a 403 on creating a
        # Service (RBAC) goes through the ADC retry and surfaces as
        # ClusterAccessDenied.
        except base.ComputeConfigError as exc:
            print(f"warning: no Service for {name}: {exc}", file=sys.stderr)

    def _job_name(self, requested: str | None, unique: bool) -> str:
        """`requested` (or a fresh job id) as a valid, prefixed Job name.

        With `unique`, a requested name gets a random suffix. The result is
        trimmed to MAX_JOB_NAME_LEN, and naming it again returns it unchanged,
        so `run` can pass the name it picked to `submit`.
        """
        name = _k8s_name(requested or "")
        if requested and not name:
            raise base.ComputeConfigError(
                f"Job name needs at least one letter or digit, got "
                f"{requested!r}.",
                error_code="INVALID_FIELD",
            )
        suffix = f"-{uuid.uuid4().hex[:6]}" if name and unique else ""
        name = name or jobs.new_job_id()
        prefix = self.job_name_prefix
        if prefix and name != prefix and not name.startswith(f"{prefix}-"):
            name = f"{prefix}-{name}"
        return name[:MAX_JOB_NAME_LEN - len(suffix)].rstrip("-") + suffix

    def _require_dispatch_config(self) -> None:
        missing = [
            field for field, value in (
                ("image", self.image),
                ("tpu_accelerator", self.tpu_accelerator),
                ("tpu_topology", self.tpu_topology),
            ) if not value
        ]
        if missing:
            raise base.ComputeConfigError(
                "Backend 'gke_job' needs " + ", ".join(f"'{f}'"
                                                       for f in missing) +
                " to dispatch a Job. Set them in the compute profile"
                " (`config set`) or pass " +
                " ".join(f"--{f.replace('_', '-')}" for f in missing) + ".",
                error_code="MISSING_FIELD",
            )

    def _token_bucket(self) -> str | None:
        """The bucket the Job's pod token is limited to.

        None if the pods don't get a token: no staging bucket, or they have
        their own pod_service_account.
        """
        if not self.staging_bucket or self.pod_service_account:
            return None
        return gcs_staging.locate(self.staging_bucket, "")[0]

    def _put_token_secret(self, name: str) -> float | None:
        """Mint the Job's pod token and store it in its Secret.

        Returns the token's expiry, or None if no Secret was written. Failures
        only warn, since plenty of Jobs never use GCS.
        """
        bucket = self._token_bucket()
        if bucket is None:
            return None
        try:
            token, expires = gcs_staging.mint_bucket_token(bucket)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(
                f"warning: couldn't create a GCS token for {name}, so its "
                f"pods can't use gs://{bucket}: {exc}",
                file=sys.stderr)
            return None
        # Server-side: a client-side apply would keep a copy of the token in
        # the Secret's last-applied-configuration annotation.
        try:
            proc = self._kubectl(
                ["apply", "--server-side", "--force-conflicts", "-f", "-"] +
                self._ns(),
                stdin=json.dumps({
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {
                        "name": name + TOKEN_SECRET_SUFFIX,
                        "namespace": self.namespace
                    },
                    "type": "Opaque",
                    "stringData": {
                        "token": token
                    },
                }),
                check=False,
                timeout=KUBECTL_READ_TIMEOUT_SEC)
            # In case kubectl echoes the object back in its error.
            error = (proc.stderr or "").replace(token, "<redacted>").strip()
        except base.ComputeConfigError as exc:
            # A timeout. The message is ours and never includes the token.
            proc, error = None, str(exc)
        if proc is None or proc.returncode != 0:
            print(
                f"warning: couldn't save the GCS token for {name}, so its "
                f"pods can't use gs://{bucket}: {error}",
                file=sys.stderr)
            return None
        return expires

    def _own_token_secret(self, name: str, uid: str) -> None:
        """Make the Job the owner of its token Secret, so the Secret is
        deleted along with the Job."""
        try:
            self._kubectl(["patch", "secret", name + TOKEN_SECRET_SUFFIX] +
                          self._ns() + [
                              "--type=merge", "-p",
                              json.dumps({
                                  "metadata": {
                                      "ownerReferences": [{
                                          "apiVersion": "batch/v1",
                                          "kind": "Job",
                                          "name": name,
                                          "uid": uid
                                      }]
                                  }
                              })
                          ],
                          timeout=KUBECTL_READ_TIMEOUT_SEC)
        # The Job is already running, so this must not fail the submit.
        except base.ComputeConfigError as exc:
            print(
                f"warning: the GCS token for {name} won't be deleted with "
                f"the Job; `compute job cancel` deletes it: {exc}",
                file=sys.stderr)

    def _renew_token_secret(self, job_id: str) -> None:
        """Re-mint a running Job's pod token if it's close to expiring.

        Called on status reads (`job wait`, `job status`, `run`). The pod
        reads the token from a mounted file, and kubelet updates the file
        when the Secret changes, so no restart is needed. If nobody checks
        on a Job for over an hour its token lapses; long unattended Jobs
        should use pod_service_account instead.
        """
        record = jobs.JobRecord.load(job_id)
        if record is None or not record.extra.get("token_secret"):
            return
        now = time.time()
        expires = record.extra.get("token_expires") or 0
        if (expires - now > TOKEN_RENEW_MARGIN_SEC
                or now < (record.extra.get("token_retry_at") or 0)):
            return
        renewed = self._put_token_secret(job_id)
        if renewed is None:
            # Back off so `job wait` doesn't retry (and warn) on every poll.
            record.extra["token_retry_at"] = now + TOKEN_RENEW_RETRY_SEC
        else:
            record.extra["token_expires"] = renewed
            record.extra.pop("token_retry_at", None)
        record.save()

    def _job_manifest(
        self,
        name: str,
        script_name: str,
        interpreter: str,
        args: list[str],
        staging: _Staging | None = None,
        log_url: str | None = None,
        serve: bool = False,
        placement: dict[str, Any] | None = None,
        shell_python: str | None = None,
        token_secret: bool = False,
    ) -> dict[str, Any]:
        """Build the Job. Every field below is load-bearing -- see the notes."""
        metadata: dict[str, Any] = {"name": name, "namespace": self.namespace}
        template_metadata: dict[str, Any] = {}
        if self.queue:
            # A Kueue cluster admits work through a queue rather than
            # scheduling it, and suspends a Job that names none. Some clusters
            # inject a default via webhook, so an unlabelled Job may still run;
            # naming it is what makes that predictable.
            metadata_labels: dict[str, str] = {QUEUE_LABEL: self.queue}
            # Define workload labels in the pod template (crucial for Kueue /
            # shared capacity admission and quota tracking).
            pod_labels: dict[str, str] = {"namespace": self.namespace}
            # Only when configured: Kueue refuses a Job whose priority class
            # does not exist, and a cluster that defines none would refuse
            # every Job labelled with a default nobody chose. Unlabelled, the
            # Job takes the cluster's default priority.
            if self.priority:
                metadata_labels[PRIORITY_LABEL] = self.priority
                pod_labels["priority"] = self.priority
            metadata["labels"] = metadata_labels
            if self.declared_duration_minutes:
                pod_labels["declared-duration-minutes"] = str(
                    self.declared_duration_minutes)
            template_metadata["labels"] = pod_labels

        # Set regardless of Kueue: profiling and admission are unrelated, and
        # the block above runs only on a queued cluster.
        if self.mldiagnostics:
            template_metadata.setdefault("labels",
                                         {})[MLDIAGNOSTICS_LABEL] = "true"
            metadata.setdefault("labels", {})[MLDIAGNOSTICS_LABEL] = "true"

        staging = staging or _Staging()
        command = (gcs_staging.build_pod_bootstrap_command(
            script_mount=SCRIPT_MOUNT,
            script_name=script_name,
            interpreter=interpreter,
            args=args,
            input_urls=staging.in_urls,
            output_urls=staging.out_urls,
            log_url=log_url,
            mounted_inputs=staging.mounted_inputs,
            logged_outputs=staging.logged_outputs,
            job=name,
            pod_token=bool(self.pod_service_account),
            service_account=self.pod_service_account or "",
            namespace=self.namespace,
            bucket=gcs_staging.clean_bucket_name(self.staging_bucket or ""),
        ) if staging.moves_files() or log_url else [
            interpreter,
            f"{SCRIPT_MOUNT}/{script_name}",
        ] + args)

        env = [{
            "name": SHM_LIMIT_ENV,
            "value": str(self.shm_size_gb * 2**30),
        }]
        # Where the pod goes and what it holds, decided once.
        scheduling: dict[str, Any]
        if placement:
            # A colocated Job: on the other Job's node, without chips, which
            # that Job already holds. It requests `colocate_requests`
            # instead, as Kueue gives no flavor to a pod that requests nothing.
            scheduling = dict(placement)
            resources: dict[str, Any] = {"requests": self.colocate_requests}
        elif self.cpu_requests:
            # A CPU Job: no chips, and nothing that sends it to a TPU node.
            # TPU nodes carry a NoSchedule taint this pod does not tolerate,
            # so it lands on a CPU node; on a Kueue cluster the flavor that
            # covers CPU and memory adds its own selector and toleration.
            scheduling = {}
            resources = {"requests": self.cpu_requests}
            generation = probe.generation_for_name(self.tpu_accelerator or "")
            if generation:
                env.append({"name": CHIPLESS_TPU_ENV, "value": generation})
        else:
            # TPU nodes carry a NoSchedule taint precisely so that ordinary
            # workloads do not land on them, so the selector and the
            # toleration go together.
            scheduling = {
                "nodeSelector": {
                    ACCELERATOR_LABEL: self.tpu_accelerator,
                    TOPOLOGY_LABEL: self.tpu_topology,
                },
                "tolerations": [{
                    "key": TPU_RESOURCE,
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }],
            }
            # Chips are allocated whole, not time-shared: a pod that does
            # not request them is given no device to open and silently
            # lands on the CPU.
            resources = {"limits": {TPU_RESOURCE: self.tpu_chips}}

        template: dict[str, Any] = {
            "spec": {
                "restartPolicy":
                "OnFailure" if serve else "Never",
                # By default, Kubernetes adds env vars for every Service in the
                # namespace. On a shared cluster, those belong to other teams,
                # and some clash with vars our workloads read (e.g. VLLM_PORT).
                "enableServiceLinks":
                False,
                **scheduling,
                "containers": [{
                    "name":
                    "workload",
                    "image":
                    self.image,
                    "command":
                    command,
                    "volumeMounts": [{
                        "name": "script",
                        "mountPath": SCRIPT_MOUNT,
                    }, {
                        "name": "dshm",
                        "mountPath": SHM_MOUNT,
                    }],
                    "env":
                    env,
                    "resources":
                    resources,
                }],
                "volumes": [
                    {
                        "name": "script",
                        "configMap": {
                            "name": name
                        },
                    },
                    {
                        # See DEFAULT_SHM_SIZE_GB: the runtime's 64 MB default
                        # fails vLLM and torch_tpu without naming /dev/shm.
                        "name": "dshm",
                        "emptyDir": {
                            "medium": "Memory",
                            "sizeLimit": f"{self.shm_size_gb}Gi",
                        },
                    }
                ],
            }
        }
        if self.pod_service_account:
            spec = template["spec"]
            spec["serviceAccountName"] = self.pod_service_account
            # A node without the GKE metadata server hands the pod the node's
            # own account instead, and every transfer fails with a 403 that
            # names neither: Pending, which preflight explains, is clearer.
            # A colocated Job keeps the server's selector as it is.
            if not placement:
                spec.setdefault("nodeSelector",
                                {})[WORKLOAD_IDENTITY_LABEL] = "true"
        if token_secret:
            # gcloud and the scripts' storage clients read the token here.
            # Mount the whole volume, not a subPath: kubelet doesn't update
            # subPath mounts, so renewed tokens would never show up.
            container = template["spec"]["containers"][0]
            container["volumeMounts"].append({
                "name": "gcs-token",
                "mountPath": TOKEN_MOUNT,
                "readOnly": True,
            })
            env = container.setdefault("env", [])
            for var in ("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE", "GCS_TOKEN_FILE"):
                env.append({"name": var, "value": f"{TOKEN_MOUNT}/token"})
            template["spec"]["volumes"].append({
                "name": "gcs-token",
                "secret": {
                    "secretName": name + TOKEN_SECRET_SUFFIX
                },
            })
        if self.hf_cache_pvc:
            container = template["spec"]["containers"][0]
            container["volumeMounts"].append({
                "name": HF_CACHE_VOLUME,
                "mountPath": HF_CACHE_MOUNT,
            })
            container.setdefault("env", []).append({
                "name": HF_CACHE_ENV,
                "value": HF_CACHE_MOUNT
            })
            if self.image:
                # Keyed on the image this container runs (`image` above).
                # See COMPILE_CACHE_ENV. A recipe that exports its own still
                # wins.
                container["command"] = _compile_cache_command(
                    container["command"],
                    f"{HF_CACHE_MOUNT}/{COMPILE_CACHE_SUBDIR}/"
                    f"{self.tpu_accelerator}-{self.tpu_topology}/"
                    f"{compile_cache_key(self.image)}",
                    self.compile_cache_max_age_days)
            template["spec"]["volumes"].append({
                "name": HF_CACHE_VOLUME,
                "persistentVolumeClaim": {
                    "claimName": self.hf_cache_pvc
                },
            })
        else:
            # See POD_CACHE_ENV. A recipe that exports its own still wins.
            container = template["spec"]["containers"][0]
            container["command"] = _pod_cache_command(container["command"])
        if shell_python:
            # Not expanded by Kubernetes: a `~` path reaches the script as
            # is, and the scripts expand it themselves.
            template["spec"]["containers"][0].setdefault("env", []).append({
                "name":
                "TPU_AGENT_SUITE_PYTHON",
                "value":
                shell_python
            })
        if template_metadata:
            template["metadata"] = template_metadata
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": metadata,
            "spec": {
                "backoffLimit": SERVE_BACKOFF_LIMIT if serve else 0,
                "template": template,
            },
        }

    # -- job lifecycle -----------------------------------------------------

    def get_status(self, job_id: str) -> base.AsynchronousExecutionResult:
        """Read the Job, distinguishing a queued one from a stuck one.

        Kueue suspension is reported as PENDING with `suspended` set rather
        than folded into the generic pending case. It does not look like
        anything is wrong -- the Job exists, reports no error, and never
        starts a pod -- so a caller that cannot tell the two apart reads a
        full queue as unreachable hardware. A suspended Job Kueue has
        refused outright (its Workload is inadmissible, or Kueue never made
        one) is not queued at all, and is FAILED with Kueue's reason.
        """
        try:
            status = self._live_status_retrying_new_job(job_id)
        except base.ComputeConfigError as exc:
            # The Job may be gone because it finished and we deleted it (a
            # pull failure is deleted as soon as it is clear). Its settled
            # record is then the answer, not the cluster's NotFound. Any
            # other error (auth, an unreachable cluster) is still raised.
            if not _is_not_found(exc):
                raise
            recorded = _recorded_status(job_id)
            if recorded is not None:
                return recorded
            record = jobs.JobRecord.load(job_id)
            if record is None:
                raise
            # Something other than compute removed the Job. Keep the record:
            # the profile may now point at another cluster, where it runs.
            raise base.ComputeConfigError(
                f"Job {job_id} not found in namespace {self.namespace!r}"
                f"{_last_seen_note(job_id)}. compute didn't delete it (its "
                f"record says {record.status}). Another run or a person "
                "removed it, or the compute profile now points at a "
                f"different cluster. `job logs --job-id {job_id}` returns any "
                "saved output, or what Cloud Logging kept. To carry on, "
                "submit it again; don't look for it with kubectl. "
                f"({exc})",
                error_code="JOB_GONE",
            ) from exc
        _note_last_seen(job_id, status)
        if status.status in ("PENDING", "RUNNING"):
            self._renew_token_secret(job_id)
        if status.details.get("never_started"):
            # A Job that failed by itself has finished its pod; this one's
            # pod is live, pulling forever and holding a TPU node -- or, one
            # Kueue refused, is still queued to run should the cluster change
            # -- and a settled record is one the run-end reaper skips. So
            # delete it here, and record this status once it is gone: a
            # second read could find the Job already vanishing and lose why
            # it failed.
            try:
                self._delete(job_id, status)
            except base.ComputeConfigError as exc:
                # Left unsettled, so the reaper tries again.
                print(f"gke_job: could not delete Job {job_id} -- {exc}",
                      file=sys.stderr)
                return status
        self._settle_record(job_id, status)
        return status

    def _live_status_retrying_new_job(
            self, job_id: str) -> base.AsynchronousExecutionResult:
        """`_live_status`, retrying NotFound for a Job submitted moments ago,
        whose first read can race its creation."""
        delays = list(NOT_FOUND_RETRY_DELAYS_SEC)
        while True:
            try:
                return self._live_status(job_id)
            except base.ComputeConfigError as exc:
                if not (delays and _is_not_found(exc)
                        and _just_submitted(jobs.JobRecord.load(job_id))):
                    raise
                time.sleep(delays.pop(0))

    def _live_status(self, job_id: str) -> base.AsynchronousExecutionResult:
        """`get_status` without the record write: the cluster's answer only."""
        job = self._kubectl_json(["get", "job", job_id] + self._ns())
        spec = job.get("spec", {})
        job_status = job.get("status", {})
        details: dict[str, Any] = {"namespace": self.namespace}
        queued = bool(
            job.get("metadata", {}).get("labels", {}).get(QUEUE_LABEL))

        pods = self._pods(job_id)
        live_pod = self._live_pod(pods)
        if live_pod:
            pod_name = live_pod.get("metadata", {}).get("name")
            if pod_name:
                details["pod_name"] = pod_name
            node_name = live_pod.get("spec", {}).get("nodeName")
            if node_name:
                details["node_name"] = node_name

        start_time = job_status.get("startTime")
        end_time = job_status.get("completionTime")

        if job_status.get("succeeded", 0):
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="COMPLETED",
                start_time=start_time,
                end_time=end_time,
                exit_code=0,
                details=details,
            )

        if job_status.get("failed", 0):
            end = _pod_end(pods)
            if end.pod_reason:
                details["pod_reason"] = end.pod_reason
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="FAILED",
                start_time=start_time,
                end_time=end_time,
                exit_code=end.exit_code,
                error=_termination_error(**end._asdict()),
                details=details,
            )

        if spec.get("suspend"):
            refused, kueue = self._kueue_admission(job_id, job)
            details.update(kueue)
            if kueue.get("eviction_reason"):
                # It ran, and the cluster's time limit stopped it. Not
                # never_started: the Job and its record stay, for the logs.
                return base.AsynchronousExecutionResult(
                    job_id=job_id,
                    status="FAILED",
                    start_time=start_time,
                    error=("The cluster stopped the Job after it ran: "
                           f"{kueue['eviction_message']}. Kueue will not "
                           "requeue it; split the work into shorter jobs or "
                           "resume from a checkpoint."),
                    details=details,
                )
            if refused:
                # Waiting cannot change Kueue's mind. Left queued, the Job
                # would run unwatched once the cluster is fixed, so
                # `get_status` deletes it, as it does a pod that never pulls.
                details["never_started"] = True
                return base.AsynchronousExecutionResult(
                    job_id=job_id,
                    status="FAILED",
                    start_time=start_time,
                    error=(f"Kueue will not admit the Job "
                           f"({kueue['kueue_reason']}): "
                           f"{kueue['kueue_message']}"),
                    details=details,
                )
            details["suspended"] = True
            details["pending_reason"] = base.PENDING_QUEUED
            error = ("Kueue preempted the Job to make room for another "
                     "workload and requeued it; it runs again when quota "
                     "frees up." if kueue.get("kueue_reason") == "Preempted"
                     else "Job is suspended: the cluster admits work through "
                     "a queue and has not admitted this one. Free chips, not "
                     "configuration, is the usual cause.")
            if kueue.get("kueue_message"):
                error += f" Kueue says: {kueue['kueue_message']}"
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="PENDING",
                start_time=start_time,
                error=error,
                details=details,
            )

        if job_status.get("active", 0):
            # `active` counts pending pods as well as running ones, so it
            # cannot be read as "running" on its own: a pod that will never be
            # scheduled -- no free chips, or a node selector matching nothing
            # -- keeps the Job active indefinitely and would otherwise be
            # reported as healthy progress until the caller's timeout expired.
            # The pod's own phase is what separates the two.
            pull_error, final = self._image_pull_problem(pods)
            if final:
                # Waiting on it only burns the caller's timeout while the Job
                # holds a TPU node. The pod is still live, unlike a Job that
                # failed on its own, so `get_status` deletes it.
                details["pod_reason"] = pull_error
                details["never_started"] = True
                return base.AsynchronousExecutionResult(
                    job_id=job_id,
                    status="FAILED",
                    start_time=start_time,
                    error=f"The image could not be pulled: {pull_error}",
                    details=details,
                )
            phase, reason = self._pod_phase(job_id, pods)
            # Scheduled but not yet given up on: the pull is the reason.
            reason = reason or pull_error
            if phase == "Pending":
                details["pod_phase"] = phase
                if reason:
                    details["pod_reason"] = reason
                # Bound to a node, it is pulling or starting. Unbound, it is
                # waiting for one: with a queue, the quota is held and the
                # pool is full; without, nothing holds its place at all.
                details["pending_reason"] = (
                    base.PENDING_STARTING if "node_name" in details else
                    base.PENDING_ADMITTED_NO_NODE
                    if queued else base.PENDING_UNSCHEDULED)
                if not reason:
                    reason = (
                        f"Pod is on node {details['node_name']} but not yet "
                        "running: the image is still pulling or the "
                        "container is starting." if "node_name" in details else
                        "Pod is not scheduled yet: no node matches the "
                        "selector, or the chips are taken.")
                return base.AsynchronousExecutionResult(
                    job_id=job_id,
                    status="PENDING",
                    start_time=start_time,
                    error=reason,
                    details=details,
                )
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="RUNNING",
                start_time=start_time,
                details=details,
            )

        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status="PENDING",
            start_time=start_time,
            details=details,
        )

    def _kueue_admission(self, job_id: str,
                         job: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Whether Kueue has refused a suspended Job for good, and its words.

        Returns (refused, details), the details under `kueue_*` keys. Refused
        only on an answer the cluster gave: a Workload Kueue will not admit,
        or none at all after `KUEUE_WORKLOAD_GRACE_SEC`. A failed read (no
        CRD, no rights, no answer) says nothing, and the Job stays PENDING.
        A Workload only waiting -- for quota, behind a held queue, or
        requeued after a preemption -- is not refused, but Kueue's message
        is passed on: it says why the Job is not running.
        """
        metadata = job.get("metadata", {})
        uid = metadata.get("uid")
        # A Job a JobSet (or LWS, or Pathways) controls has no Workload of
        # its own: Kueue admits the parent, whose Workload owns the verdict.
        if not uid or any(
                ref.get("controller")
                for ref in metadata.get("ownerReferences", [])):
            return False, {}
        workloads = self._workloads(uid)
        if workloads == []:
            created = _parse_time(metadata.get("creationTimestamp"))
            if (created is None
                    or time.time() - created < KUEUE_WORKLOAD_GRACE_SEC):
                return False, {}
            # A Kueue too old to label its Workloads with the Job's uid would
            # look like one that made none; its owner reference still says.
            workloads = self._workloads(uid, labelled=False)
            if workloads == []:
                return True, self._no_workload(job_id, metadata)
        waiting: dict[str, Any] = {}
        for workload in workloads or []:
            stop = _time_limit_stop(workload)
            if stop:
                return True, stop
            verdict = _workload_verdict(workload)
            if verdict is None:
                continue
            refused, reason, message = verdict
            if refused and self._queue_held(message):
                refused, reason = False, "Stopped"
            name = workload.get("metadata", {}).get("name")
            details = {"kueue_reason": reason, "kueue_message": message}
            if name:
                details["kueue_workload"] = name
            if not refused:
                waiting = waiting or details
                continue
            events = self._warning_events("Workload", name) if name else []
            if events:
                details["kueue_events"] = events
            return True, details
        return False, waiting

    def _queue_held(self, message: str) -> bool:
        """Whether the queue `message` calls inactive may only be on hold.

        A LocalQueue or ClusterQueue its `stopPolicy` holds reads "is
        inactive" in the Workload's conditions, as one missing its flavor
        does; but a hold is lifted by hand, and the Job then runs. The
        queue's own Active condition tells them apart (reason `Stopped`).
        A queue that cannot be read is given the benefit of the doubt.
        """
        match = KUEUE_QUEUE_PROBLEM.search(message)
        if not match or match.group(3) != "is inactive":
            return False
        kind, name = match.group(1), match.group(2)
        queue = self._read(["get", f"{kind.lower()}s.kueue.x-k8s.io", name] +
                           (self._ns() if kind == "LocalQueue" else []))
        return queue is None or any(
            cond.get("type") == "Active" and cond.get("reason") == "Stopped"
            for cond in queue.get("status", {}).get("conditions", []))

    def _no_workload(self, job_id: str, metadata: dict[str,
                                                       Any]) -> dict[str, Any]:
        """Why a Job Kueue made no Workload for will never be admitted.

        Kueue explains that only in its controller's log, and in newer
        releases in a Warning event on the Job (`WorkloadPriorityClassNotFound`),
        which is used when there is one. Otherwise, a priority class the
        cluster does not define is by far the usual cause, and is blamed
        only once the cluster confirms it has no such class.
        """
        details: dict[str, Any] = {"kueue_reason": "NoWorkload"}
        events = self._warning_events("Job", metadata.get("name") or job_id)
        if events:
            details.update(kueue_message=events[0], kueue_events=events)
            return details
        priority = metadata.get("labels", {}).get(PRIORITY_LABEL)
        cause = ("Kueue could not build one for it, and says why only in the "
                 "kueue-controller-manager log (namespace kueue-system).")
        if priority:
            try:
                proc = self._kubectl([
                    "get", "workloadpriorityclasses.kueue.x-k8s.io", priority
                ],
                                     check=False,
                                     timeout=KUBECTL_READ_TIMEOUT_SEC)
            except KubectlError:
                proc = None
            if proc is not None and "NotFound" in proc.stderr:
                cause = (f"The Job names the WorkloadPriorityClass "
                         f"'{priority}' ({PRIORITY_LABEL}), which the cluster "
                         "does not define -- `kubectl get "
                         "workloadpriorityclasses` lists those it does.")
        details["kueue_message"] = (
            f"no Kueue Workload was created for the Job within "
            f"{KUEUE_WORKLOAD_GRACE_SEC}s, so it will never be admitted. "
            f"{cause}")
        return details

    def _read(self, argv: list[str]) -> dict[str, Any] | None:
        """`kubectl <argv> -o json`, or None if it could not be read.

        For Kueue's objects, where a failed read -- no CRD, no right to
        read, no answer -- says nothing, and must not fail the Job.
        """
        try:
            proc = self._kubectl(argv + ["-o", "json"],
                                 check=False,
                                 timeout=KUBECTL_READ_TIMEOUT_SEC)
            payload = json.loads(proc.stdout) if proc.returncode == 0 else None
        except (KubectlError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _workloads(self,
                   uid: str,
                   labelled: bool = True) -> list[dict[str, Any]] | None:
        """The Kueue Workloads of the Job with `uid`, or None if unreadable.

        None is not an empty list: it is a cluster without Kueue's CRD, an
        account that may not list Workloads, or an API server that did not
        answer, and none of those says the Job has no Workload. `labelled`
        asks for Kueue's job-uid label, one cheap query; without it every
        Workload in the namespace is read and matched by owner reference.
        """
        argv = ["get", WORKLOAD_RESOURCE] + self._ns()
        if labelled:
            argv += ["-l", f"{JOB_UID_LABEL}={uid}"]
        items = (self._read(argv) or {}).get("items")
        if not isinstance(items, list):
            return None
        if labelled:
            return items
        return [
            workload for workload in items if any(
                ref.get("uid") == uid
                for ref in workload.get("metadata", {}).get(
                    "ownerReferences", []))
        ]

    def _warning_events(self, kind: str, name: str) -> list[str]:
        """An object's Warning events as `Reason: message`, newest first.

        Kueue records why it will not admit a Workload as events on it (and
        on the Job) as well as in its conditions, sometimes in more detail.
        Events that cannot be read are simply not reported.
        """
        events = (self._read(["get", "events"] + self._ns() + [
            "--field-selector",
            f"involvedObject.kind={kind},involvedObject.name={name}"
        ]) or {}).get("items") or []
        seen: list[str] = []
        for event in sorted(events, key=_event_time, reverse=True):
            if event.get("type") != "Warning" or not event.get("message"):
                continue
            line = f"{event.get('reason') or 'Warning'}: {event['message']}"
            if line not in seen:
                seen.append(line)
        return seen[:5]

    def _pods(self, job_id: str) -> list[dict[str, Any]]:
        """The Job's pods, or an empty list if they cannot be read."""
        proc = self._kubectl(
            ["get", "pods"] + self._ns() +
            ["-l", f"job-name={job_id}", "-o", "json"],
            check=False,
            timeout=KUBECTL_READ_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            return []
        try:
            return json.loads(proc.stdout).get("items", [])
        except ValueError:
            return []

    @staticmethod
    def _live_pod(pods: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The pod worth reading: Running, else Pending, else the first there is."""
        ranked = {"Running": 0, "Pending": 1}
        best: dict[str, Any] | None = None
        best_rank = len(ranked)
        for pod in pods:
            rank = ranked.get(pod.get("status", {}).get("phase"), len(ranked))
            if best is None or rank < best_rank:
                best, best_rank = pod, rank
        return best

    def _pod_phase(
        self,
        job_id: str,
        pods: list[dict[str, Any]] | None = None,
    ) -> tuple[str | None, str | None]:
        """The pod's phase and, when it is stuck, why.

        The message comes from the `PodScheduled` condition, which is where
        the scheduler explains itself -- "0/2 nodes are available: 2
        Insufficient google.com/tpu" tells the caller the cluster is full
        rather than the manifest being wrong, and the two need different
        responses.

        `pods` is for a caller that has already read them: every status poll
        goes through here, and the list costs a `kubectl get pods` round trip.
        """
        pod = self._live_pod(self._pods(job_id) if pods is None else pods)
        if pod is None:
            return None, None
        status = pod.get("status", {})
        reason = None
        for cond in status.get("conditions", []):
            if cond.get("type") == "PodScheduled" and cond.get(
                    "status") == "False":
                reason = cond.get("message") or cond.get("reason")
                break
        return status.get("phase"), reason

    def _image_pull_problem(
            self, pods: list[dict[str, Any]]) -> tuple[str | None, bool]:
        """Why a container's image is not pulling, and whether that is final.

        Final for a reference that can never pull, or for pulls that have
        kept failing for `PULL_FAILURE_GRACE_SEC`. Not measured to now: the
        waiting reason stays `ImagePullBackOff` while a retry pulls, and a
        multi-gigabyte retry after one blip can take minutes to succeed.
        """
        for pod in pods:
            status = pod.get("status", {})
            for cs in (status.get("initContainerStatuses", []) +
                       status.get("containerStatuses", [])):
                waiting = cs.get("state", {}).get("waiting") or {}
                reason = waiting.get("reason")
                if reason not in IMAGE_REF_FAILURES + IMAGE_PULL_RETRIES:
                    continue
                detail = waiting.get("message") or cs.get("image")
                message = f"{reason}: {detail}" if detail else reason
                if reason in IMAGE_REF_FAILURES:
                    return message, True
                span = self._pull_failure_span(
                    pod.get("metadata", {}).get("name"))
                return message, (span is not None
                                 and span >= PULL_FAILURE_GRACE_SEC)
        return None, False

    def _pull_failure_span(self, pod_name: str | None) -> float | None:
        """Seconds from the pod's first failed pull to its latest, or None.

        The waiting state records no time; the kubelet's `Failed to pull
        image` events do, one per attempt, and a series of them keeps its
        first timestamp and moves its last, so this works the same for
        `run` and for a `job status` in a later process. The `BackOff` and
        `Error: ImagePullBackOff` events are left out: they repeat while the
        kubelet waits, not per attempt. Events that cannot be read (not
        allowed, or aged out) give None, and the pull is never taken as
        final: the wait runs its course, as it did before.
        """
        if not pod_name:
            return None
        proc = self._kubectl(
            ["get", "events"] + self._ns() + [
                "--field-selector",
                f"involvedObject.kind=Pod,involvedObject.name={pod_name}",
                "-o", "json"
            ],
            check=False,
        )
        if proc.returncode != 0:
            return None
        try:
            events = json.loads(proc.stdout).get("items", [])
        except ValueError:
            return None
        failed = [
            event for event in events if event.get("reason") == "Failed" and (
                event.get("message") or "").startswith("Failed to pull image")
        ]
        firsts = [
            _parse_time(event.get("firstTimestamp") or event.get("eventTime"))
            for event in failed
        ]
        lasts = [
            _parse_time(
                event.get("lastTimestamp")
                or (event.get("series") or {}).get("lastObservedTime")
                or event.get("eventTime")) for event in failed
        ]
        firsts = [stamp for stamp in firsts if stamp is not None]
        lasts = [stamp for stamp in lasts if stamp is not None]
        return max(lasts) - min(firsts) if firsts and lasts else None

    def _wait(self, job_id: str,
              timeout: int) -> base.AsynchronousExecutionResult:
        """Poll until the Job settles, or return its last state at timeout.

        Polling rather than `kubectl wait`, which takes one condition at a
        time: waiting on `complete` does not return early on failure, it burns
        the whole timeout, and a run that failed in seconds should say so in
        seconds.

        `timeout` is timed from when the pod starts running, not from submit:
        time spent queued (waiting for Kueue or a node) is unrelated to how
        long the script needs. The wait to start has a separate budget,
        also `timeout`, so a Job that is never admitted still gives up.

        Each poll is kept in `self._last_poll` as (status, seconds since
        submit, whether RUNNING was ever seen), so a `run` killed mid-wait
        can still say what it saw.
        """
        submitted = time.monotonic()
        started: float | None = None
        status = self.get_status(job_id)
        while status.status in ("PENDING", "RUNNING"):
            now = time.monotonic()
            if started is None and status.status == "RUNNING":
                started = now
            self._last_poll = (status, round(now - submitted), started
                               is not None)
            # After starting, the run clock keeps going even if Kueue
            # preempts the Job back to PENDING.
            elapsed = now - (submitted if started is None else started)
            if elapsed >= timeout:
                queued = (now if started is None else started) - submitted
                return self._timed_out(job_id, status, timeout, queued, started
                                       is not None)
            time.sleep(POLL_INTERVAL_SEC)
            status = self.get_status(job_id)
        return status

    @staticmethod
    def _timed_out(job_id: str, status: base.AsynchronousExecutionResult,
                   timeout: int, queued: float,
                   started: bool) -> base.AsynchronousExecutionResult:
        """Mark `status` as timed out, saying whether the Job ever ran and
        how long it was queued, so a slow Job and a busy cluster are not
        confused.
        """
        status.details["timed_out_after_sec"] = timeout
        status.details["queued_sec"] = round(queued)
        if started:
            message = (f"Job {job_id} was still {status.status} after "
                       f"running for {timeout}s. It had also waited "
                       f"{round(queued)}s in the queue, which does not count "
                       "toward --timeout.")
        else:
            # No container ran: `run` and `_read_logs` skip the pod.
            status.details["never_started"] = True
            message = (f"Job {job_id} was still {status.status} after "
                       f"waiting {timeout}s in the queue. It never ran, so "
                       "no work was lost.")
        status.error = (f"{message} {status.error}"
                        if status.error else message)
        return status

    def _killed_message(self, job_id: str) -> str:
        """One line for a `run` killed from outside, from its last poll."""
        if self._last_poll is None:
            return (f"gke_job: stopped before Job {job_id}'s first status "
                    "read; deleting it.")
        status, since, ever_ran = self._last_poll
        why = f" {status.error.rstrip('.')}." if status.error else ""
        tail = ("Logs so far are saved for `job logs` if the pod can still be "
                "read." if ever_ran else
                "Nothing ran. A resubmit would wait the same way; check "
                "`compute inspect` first.")
        return (f"gke_job: stopped while Job {job_id} was {status.status}, "
                f"{since}s after submission; deleting it.{why} {tail}")

    def fetch_logs(self, job_id: str) -> str:
        """Return the Job's pod output, without any outputs framed in it."""
        return gcs_staging.split_logged_outputs(self._read_logs(job_id),
                                                job_id)[0]

    def _read_logs(self, job_id: str) -> str:
        """Return the Job's pod output, empty if it never produced any.

        A finished Job is read from the copy `_save_logs` took, because its
        pod does not last: pod GC collects it once the autoscaler removes the
        TPU node. Without one, the pod is asked, then the copy the pod
        uploaded to GCS. A Job none of them can answer for raises unless it
        has yet to start -- kubectl's complaint returned as the log reads as
        a job that printed one line and succeeded.
        """
        saved = self._saved_log(job_id)
        if not saved.is_file():
            record = jobs.JobRecord.load(job_id)
            if record is not None and record.extra.get("never_started"):
                # Its image never pulled or Kueue never admitted it, so no
                # container ever ran: there is nothing to find, and pointing
                # at Cloud Logging sends the reader looking for output that
                # cannot exist.
                raise base.ComputeConfigError(
                    f"Job {job_id} never started, so it has no logs: "
                    f"{record.extra.get('error') or 'its image never pulled'}",
                    error_code="JOB_NEVER_STARTED",
                )
            proc = self._pod_logs(job_id)
            if proc.returncode == 0:
                if not _node_lost(self._pods(job_id)):
                    return proc.stdout
                # Its container is gone with the node, so its log is final:
                # kept as a finished Job's is, to be settled below.
                self._keep_lost_node_log(job_id, proc.stdout)
            else:
                self._download_log(job_id, saved)
        if saved.is_file():
            if self._lost_node_marker(job_id).is_file():
                return self._settle_lost_node_log(job_id)
            return saved.read_text(encoding="utf-8")
        try:
            live = self._live_status(job_id).status
        except base.ComputeConfigError:
            live = None  # The Job itself is gone.
        if live == "PENDING":
            return ""
        record = jobs.JobRecord.load(job_id)
        why = ("It was cancelled before it finished, and cancelling deletes "
               "its pod"
               if record is not None and record.status == "CANCELLED" else
               "Its pod could not be read (reaped after its node scaled away, "
               "evicted, expired, or deleted outside compute)")
        logs, why_not = self._cloud_logging_logs(job_id, record)
        if logs is not None:
            return logs
        raise base.ComputeConfigError(
            f"Logs for Job {job_id} are unavailable: "
            f"{proc.stderr.strip()}. {why}, and no copy was "
            "saved locally or in the staging bucket. Cloud Logging: "
            f"{why_not}",
            error_code="LOGS_UNAVAILABLE",
        )

    def _cloud_logging_logs(
            self, job_id: str,
            record: jobs.JobRecord | None) -> tuple[str | None, str]:
        """The Job's container output as Cloud Logging kept it, or (None,
        why not).

        Read only once the pod and every copy are gone: GKE ships what the
        container printed there, so a pod reaped with its node, or evicted
        before `_save_logs` saw it finish, still has its last lines. The
        text opens with a line saying where it came from, since it may be
        cut short or missing what was not yet ingested. Never raises: the
        caller says why there are no logs.
        """
        lines, where = self._cloud_logging_lines(job_id, record)
        if lines is None:
            return None, where
        header = (f"[compute] Job {job_id}'s pod is gone and no copy of its "
                  f"log was saved: the {len(lines)} line(s) below are what "
                  f"Cloud Logging kept ({where})")
        return header + ".\n" + "\n".join(lines) + "\n", ""

    def _cloud_logging_lines(
            self, job_id: str,
            record: jobs.JobRecord | None) -> tuple[list[str] | None, str]:
        """The Job's lines in Cloud Logging, oldest first, and where they
        came from; or (None, why not). Never raises."""
        target = base.gke_target(base.current_context(self._kubectl_cmd([])),
                                 prefer_context=True)
        known = {
            part: value
            for part, value in target.items() if not value.startswith("<")
        }
        start = _parse_time(record.start_time) if record else None
        end = _parse_time(record.end_time) if record else None
        query = _cloud_logging_query(
            self.namespace,
            job_id,
            known.get("cluster"),
            start,
            end,
            uid=record.extra.get("uid") if record else None)
        if not shutil.which("gcloud"):
            return None, ("gcloud is not on PATH to read it. Install the "
                          "Google Cloud CLI, or run this query in the Logs "
                          f"Explorer: {query}")
        argv = [
            "gcloud", "logging", "read", query,
            f"--limit={CLOUD_LOGGING_MAX_LINES}", "--order=desc",
            "--format=json"
        ]
        if start is None:
            # gcloud reads only the last day unless the query bounds it.
            argv.append("--freshness=30d")
        if "project" in known:
            argv.append(f"--project={known['project']}")
        proc = base.probe_run(argv, timeout=CLOUD_LOGGING_TIMEOUT_SEC)
        if proc.returncode != 0:
            return None, (f"`gcloud logging read` failed: "
                          f"{base.probe_error(proc)}. Query: {query}")
        try:
            entries = json.loads(proc.stdout or "[]")
        except ValueError:
            entries = None
        if not isinstance(entries, list):
            return None, ("`gcloud logging read` returned output that is not "
                          f"a list of entries. Query: {query}")
        # Asked newest first, so the limit keeps the end, where a crash is;
        # read oldest first.
        entries = [
            entry for entry in reversed(entries) if isinstance(entry, dict)
        ]
        pods = {_log_entry_pod(entry) for entry in entries} - {None}
        lines = []
        for entry in entries:
            text = _log_entry_text(entry)
            if text is None:
                continue
            pod = _log_entry_pod(entry)
            # A Job that retried ran several pods; say whose line it is.
            lines.append(f"[{pod}] {text}" if len(pods) > 1 and pod else text)
        if not lines:
            return None, ("it holds no output for the Job: container logging "
                          "may be off for the cluster, the entries expired, "
                          f"or they are not ingested yet. Query: {query}")
        where = (f"project {known['project']}"
                 if "project" in known else "gcloud's default project")
        if len(entries) >= CLOUD_LOGGING_MAX_LINES:
            where += f", the last {CLOUD_LOGGING_MAX_LINES} only"
        return lines, where

    def _keep_lost_node_log(self, job_id: str, log: str) -> None:
        """Save a lost node's copy of the log, marked for `_read_logs` to add
        what Cloud Logging kept after it. Asks Cloud Logging nothing, so the
        status reads that call this never wait on it."""
        # A file, not a record field: callers save a record they loaded
        # before this, which would drop it. Before the copy, which reads as
        # final without it.
        self._lost_node_marker(job_id).touch()
        _write_atomically(self._saved_log(job_id), log)

    def _settle_lost_node_log(self, job_id: str) -> str:
        """A lost node's saved log, with Cloud Logging's tail added once it
        has had CLOUD_LOGGING_SETTLE_SEC to ingest it, waiting out what is
        left of that; then saved, and never asked again. The copy alone if
        Cloud Logging cannot be read, asking again on the next read."""
        saved, marker = self._saved_log(job_id), self._lost_node_marker(job_id)
        text = saved.read_text(encoding="utf-8")
        wait = marker.stat().st_mtime + CLOUD_LOGGING_SETTLE_SEC - time.time()
        if wait > 0:
            print(
                f"gke_job: {job_id}'s node was lost, so its log on the node "
                f"misses its last lines; waiting {wait:.0f}s for Cloud "
                "Logging to have them.",
                file=sys.stderr,
                flush=True)
            time.sleep(wait)
        lines, why_not = self._cloud_logging_lines(job_id,
                                                   jobs.JobRecord.load(job_id))
        if lines is None:
            print(
                f"gke_job: {job_id}'s log is the lost node's copy, which may "
                f"miss its last lines. Cloud Logging: {why_not}",
                file=sys.stderr)
            return text
        text = _with_cloud_tail(text, lines)[0]
        _write_atomically(saved, text)
        marker.unlink(missing_ok=True)
        return text

    def _save_logs(self, job_id: str) -> None:
        """Copy a finished Job's logs beside its record, once."""
        saved = self._saved_log(job_id)
        if saved.is_file():
            return
        # Pods first: `kubectl logs` on a Job without one blocks for its
        # 20s pod-running timeout, on every status read that retries this.
        pods = self._pods(job_id)
        if pods:
            proc = self._pod_logs(job_id)
            if proc.returncode == 0:
                if _node_lost(pods):
                    self._keep_lost_node_log(job_id, proc.stdout)
                else:
                    _write_atomically(saved, proc.stdout)
                return
        self._download_log(job_id, saved)

    def _pod_logs(self, job_id: str) -> subprocess.CompletedProcess[str]:
        return self._kubectl(["logs", f"job/{job_id}"] + self._ns(),
                             check=False,
                             timeout=KUBECTL_READ_TIMEOUT_SEC)

    @staticmethod
    def _saved_log(job_id: str) -> Path:
        return jobs.jobs_dir() / job_id / "output.log"

    @staticmethod
    def _lost_node_marker(job_id: str) -> Path:
        """Beside a saved log that a lost node may have cut short, from when
        the loss was seen until Cloud Logging's tail is added to it."""
        return jobs.jobs_dir() / job_id / "output.log.node-lost"

    @staticmethod
    def _log_blob(job_id: str) -> str:
        return f"{job_id}/out/{gcs_staging.POD_LOG_FILE}"

    def _log_url(self, clean_bucket: str, job_id: str) -> str | None:
        """A PUT URL for the pod's log copy, or None if there cannot be one.

        Best-effort, like the upload in the pod: the copy only backs up the
        one `_save_logs` takes, and credentials that cannot sign (a person's
        ADC) must not fail a Job that stages no files of its own.
        """
        if not clean_bucket:
            return None
        try:
            return self._pod_url(clean_bucket, self._log_blob(job_id), "PUT",
                                 SUBMIT_URL_EXP_SEC)
        except base.ComputeConfigError as exc:
            print(
                f"gke_job: not keeping a GCS copy of {job_id}'s log -- "
                f"{exc.error_code}: could not sign an upload URL",
                file=sys.stderr)
            return None

    def _download_log(self, job_id: str, dest: Path) -> None:
        """Fetch the log the pod uploaded, if a staging bucket holds one.

        Never raises: a status read or cancel that fails here over a copy
        leaves the Job's state unread or its TPUs held. The download lands
        beside `dest` first, since a partial `dest` passes for the whole log.
        """
        if not self.staging_bucket:
            return
        part = dest.with_name(dest.name + ".part")
        try:
            if gcs_staging.download_file_from_gcs(
                    gcs_staging.clean_bucket_name(self.staging_bucket),
                    self._log_blob(job_id), part, **self._staging()):
                part.replace(dest)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            print(
                f"gke_job: could not fetch the log of {job_id} from GCS "
                f"-- {exc}",
                file=sys.stderr)
        finally:
            part.unlink(missing_ok=True)

    def fetch_artifacts(self, job_id: str, dest_dir: Path) -> list[Path]:
        """Write the Job's logs to `dest_dir` and return the file.

        Logs are the whole artifact: the workload writes to a container
        filesystem that goes away with the pod, and this driver mounts no
        shared volume to survive it. A driver that returned an empty list here
        would read as "the job produced nothing".
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / f"{job_id}.log"
        logs, carried = gcs_staging.split_logged_outputs(
            self._read_logs(job_id), job_id)
        target.write_text(logs, encoding="utf-8")
        retrieved = [target]
        record = jobs.JobRecord.load(job_id)
        declared = record.extra.get("outputs") if record else None
        if not declared:
            return retrieved
        in_log = record.extra.get("outputs_in_log")
        if not in_log and not self.staging_bucket:
            # The job declared outputs, so a bucket existed at submit time and
            # the environment has changed under it. Returning the log alone
            # here would read as "the job produced nothing else".
            raise self.not_configured(
                "staging_bucket",
                f"Job {job_id} staged its --output files to GCS: set the same "
                "bucket it was submitted with to retrieve them.")
        missing, fields = self._collect_outputs(
            job_id, declared, carried, in_log,
            lambda record: base.StagedFile.artifact_path(dest, record))
        for out_spec in declared:
            if out_spec["remote"] in missing:
                print(
                    f"gke_job: output {out_spec['remote']!r} of {job_id} not "
                    f"retrieved -- {missing[out_spec['remote']]}",
                    file=sys.stderr)
            else:
                retrieved.append(base.StagedFile.artifact_path(dest, out_spec))
        # The staged copies have served their purpose once they are on disk
        # here; what was never fetched stays, because it is the caller's data
        # and nothing else holds it.
        if not in_log and not missing:
            gcs_staging.delete_staged_prefix(
                gcs_staging.clean_bucket_name(self.staging_bucket),
                f"{job_id}/out/", **self._staging(True))
        if fields:
            # The pod is gone with them: only a rerun, set up so the pod can
            # upload, produces them.
            raise base.NotConfigured(
                f"Job {job_id} produced outputs too large for the pod's log, "
                "and the pod could not upload them, so they are lost; its "
                f"log is in {target}. Set up an upload path and rerun.",
                "STAGING_NOT_CONFIGURED", fields)
        return retrieved

    def _collect_outputs(
        self,
        job_id: str,
        declared: list[dict[str, str]],
        carried: dict[int, bytes | str],
        in_log: str | bool | None,
        dest_of: Callable[[dict[str, str]], Path],
    ) -> tuple[dict[str, str], list[str]]:
        """Land each declared output at `dest_of(it)`, from its log or GCS.

        `carried` is what `split_logged_outputs` found in the Job's log;
        `in_log` is the record's `outputs_in_log`: why the Job sent every
        output there (so none is in GCS), or falsy. Returns why each output
        that did not land is missing, by its path on the target, and, if one
        was produced but lost in the log, the settings that let the pod
        upload it instead.
        """
        bucket = gcs_staging.clean_bucket_name(self.staging_bucket or "")
        missing: dict[str, str] = {}
        lost = False
        for index, out_spec in enumerate(declared):
            dest, remote = dest_of(out_spec), out_spec["remote"]
            # The log first: a pod whose upload the bucket refused frames
            # the output there instead.
            got = carried.get(index)
            if isinstance(got, bytes):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(got)
                continue
            if got and in_log:
                lost = True
                missing[remote] = f"{got}, and {_why_not_uploaded(in_log)}"
                continue
            if got:
                # Framed only because its upload failed; the log says why.
                missing[remote] = (f"{got}, after the pod failed to upload it "
                                   f"to gs://{bucket} (see the job's log)")
                continue
            blob_name = f"{job_id}/out/{remote}"
            if in_log or not gcs_staging.download_file_from_gcs(
                    bucket, blob_name, dest, **self._staging()):
                missing[remote] = (
                    "was not created" if in_log else
                    f"gs://{bucket}/{blob_name} was not created")
        if not lost:
            return missing, []
        return missing, (["staging_bucket"] if in_log == NO_BUCKET else list(
            gcs_staging.POD_UPLOAD_FIELDS))

    def _port_forward_command(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None,
    ) -> tuple[list[str], str]:
        """Tunnel to the Job's running pod, falling back to the Job itself.

        `job/<name>` is what kubectl accepts when no pod is up yet; naming the
        running pod directly avoids kubectl picking a terminal one of a Job
        that retried.
        """
        if not job_id:
            raise base.ComputeConfigError(
                "Backend 'gke_job' requires '--job-id' for port-forwarding.",
                error_code="MISSING_FIELD",
            )
        auth_utils.resolve_credentials(
            self,
            self._kubectl_cmd(["get", "job", job_id, "-o", "name"] +
                              self._ns()), self.kubeconfig)
        target = f"job/{job_id}"
        live_pod = self._live_pod(self._pods(job_id))
        if live_pod and live_pod.get("status", {}).get("phase") == "Running":
            pod_name = live_pod.get("metadata", {}).get("name")
            if pod_name:
                target = f"pod/{pod_name}"
        cmd = self._kubectl_cmd(
            self._ns() +
            ["port-forward", target, f"{local_port}:{remote_port}"])
        return cmd, f"{job_id}-{local_port}"

    def cancel(self, job_id: str) -> None:
        """Delete the Job and its ConfigMap, and record what that meant."""
        self._delete(job_id)

    def _delete(
            self,
            job_id: str,
            status: base.AsynchronousExecutionResult | None = None) -> None:
        """`cancel`, given the Job's status when the caller has just read it."""
        record = jobs.JobRecord.load(job_id)
        settled = record is not None and record.status in jobs.TERMINAL_STATUSES
        if not settled and status is None:
            try:
                status = self._live_status(job_id)
            except base.ComputeConfigError:
                # No Job to read: never created, or already gone.
                status = None
        finished = settled or (status is not None
                               and status.status in jobs.TERMINAL_STATUSES)
        running = status is not None and status.status == "RUNNING"
        never_ran = bool(record and record.extra.get("never_started")) or (
            status is not None and status.details.get("never_started"))
        if record and (finished or running) and not never_ran:
            # Deleting the Job deletes its pod, and the logs with it. A pod
            # whose image never pulled has none, and asking blocks on it. A
            # running one's output so far is what `job logs` returns after a
            # cancel; without the copy, the next read fails on NotFound.
            self._save_logs(job_id)
        objects = [("job", job_id), ("configmap", job_id)]
        if record is not None and record.extra.get("token_secret"):
            # Only for Jobs with a token, so cancel doesn't need Secret access.
            objects.append(("secret", job_id + TOKEN_SECRET_SUFFIX))
        for kind, obj in objects:
            proc = self._kubectl(["delete", kind, obj] + self._ns() +
                                 ["--ignore-not-found"],
                                 check=False,
                                 timeout=KUBECTL_DELETE_TIMEOUT_SEC)
            # `--ignore-not-found` covers the object; this covers a namespace
            # that is itself gone, which is absence all the same. kubectl
            # spells it both `(NotFound)` and `not found`.
            if proc.returncode != 0 and not re.search(
                    r"not ?found", proc.stderr or "", re.IGNORECASE):
                raise KubectlError(["delete", kind, obj], proc.stderr)
        if record is not None and self.staging_bucket:
            # Inputs are of no use to a Job that no longer exists. Outputs are
            # left to `fetch_artifacts`, which deletes them once they are on
            # disk: reaping a run must not destroy results nobody collected.
            prefix = record.extra.get("staged_prefix")
            if prefix:
                gcs_staging.delete_staged_prefix(
                    gcs_staging.clean_bucket_name(self.staging_bucket),
                    f"{prefix}in/", **self._staging(True))
        if record is None or settled:
            return
        if status is not None and status.status in jobs.TERMINAL_STATUSES:
            _mark_terminal(record, status)
        else:
            # Once the Job is deleted, only the record can say why. The
            # status error is its state at the cancel, not a failure.
            error = status.error if status is not None else None
            record.mark("CANCELLED",
                        error=f"Cancelled while: {error}" if error else None,
                        **({
                            "never_started": True
                        } if never_ran else {}))

    def _settle_record(self, job_id: str,
                       status: base.AsynchronousExecutionResult) -> None:
        """Write a terminal cluster status through to the job record.

        The record is what `job list` and the reaper read, and nothing else
        updates it once the Job is dispatched, so every status read that
        finds the Job finished is the moment to say so.
        """
        if status.status not in jobs.TERMINAL_STATUSES:
            return
        record = jobs.JobRecord.load(job_id)
        if record is None:
            return
        # Any sight of the Job finished may be the last of its pod, so the
        # logs are copied now rather than when asked for; `_save_logs` keeps
        # the first copy it gets. A Job whose image never pulled has none.
        if not status.details.get("never_started"):
            self._save_logs(job_id)
        if record.status not in jobs.TERMINAL_STATUSES:
            _mark_terminal(record, status)


def _why_not_uploaded(in_log: str | bool) -> str:
    """Why the pod sent its outputs to its log rather than to GCS."""
    if in_log == NO_BUCKET:
        return "no 'staging_bucket' was set for the pod to upload it to"
    if isinstance(in_log, str):
        return f"the pod could not upload it: signing its URL failed ({in_log})"
    # A record from before the reason was kept.
    return "the pod had no way to upload it"
