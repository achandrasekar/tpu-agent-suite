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
"""TorchTPU VLLM Pre-flight Guardrail Check."""

import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys

# Weights may take this fraction of one device's HBM; the rest is left for
# the KV cache and runtime overhead.
WEIGHT_HBM_FRACTION = 0.8

# Bytes per parameter for the dtypes safetensors metadata names.
_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def weights_bytes(info, index_total=None):
    """Size of a hub model's weights in bytes, or None when unknown.

    `info.safetensors.total` is a parameter count, not bytes, so parameters
    per dtype are multiplied out first. Failing that, the checkpoint index's
    `total_size` is used. Summing the `.safetensors` files is the last resort:
    some repos ship the weights twice (Mistral-7B-Instruct-v0.3 has a
    consolidated file beside the split ones), which that sum double-counts.
    """
    params = getattr(getattr(info, "safetensors", None), "parameters", None)
    if params and all(dtype in _DTYPE_BYTES for dtype in params):
        return sum(n * _DTYPE_BYTES[dtype] for dtype, n in params.items())
    if index_total:
        return index_total
    files = sum(s.size or 0 for s in info.siblings or []
                if s.rfilename.endswith(".safetensors"))
    return files or None


def index_total_size(path):
    """`metadata.total_size` from a safetensors index file, or None."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("metadata", {}).get("total_size") or None
    except (OSError, ValueError):
        return None


def local_weights_bytes(model_dir):
    """Size of a local checkpoint directory's weights, or None when unknown.

    The index's `total_size` is preferred for the same reason as on the hub:
    a directory holding both a consolidated and a split copy would otherwise
    count twice.
    """
    for index in sorted(
            glob.glob(os.path.join(model_dir, "*.safetensors.index.json"))):
        total = index_total_size(index)
        if total:
            return total
    files = sum(
        os.path.getsize(f)
        for f in glob.glob(os.path.join(model_dir, "*.safetensors")))
    return files or None


GCS_SCHEME = "gs://"


def load_fetch_weights():
    """fetch_weights.py, which reads a gs:// checkpoint folder.

    In a job it is shipped beside this script (`--input
    <fetch_weights.py>:fetch_weights.py`); in the package it is in the
    tpu-model-onboarding scripts.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here,
                   os.path.join(os.path.dirname(here),
                                "tpu-model-onboarding")):
        if (os.path.isfile(os.path.join(folder, "fetch_weights.py"))
                and folder not in sys.path):
            sys.path.insert(0, folder)
    try:
        import fetch_weights  # pylint: disable=g-import-not-at-top
    except ImportError as e:
        raise ImportError(
            "a gs:// model needs fetch_weights.py beside this script: pass "
            "--input <fetch_weights.py>:fetch_weights.py to compute run"
        ) from e
    return fetch_weights


def gcs_model(url):
    """(local copy of the small files, weight bytes or None) for a gs://
    checkpoint folder.

    vLLM reads the config and tokenizer from the copy, so the dry run needs
    no weights and never sees the URL. The size is the index's `total_size`
    when the folder has one, otherwise the weights `fetch_weights` would copy,
    from the folder's listing.
    """
    fetch_weights = load_fetch_weights()
    local = fetch_weights.fetch_gcs(url, small_only=True)
    size = local_weights_bytes(local) or fetch_weights.gcs_weight_bytes(
        url, local)
    return local, size


def head_problem(tp,
                 num_heads,
                 num_kv_heads,
                 use_mla=False,
                 linear_heads=None):
    """Why vLLM cannot shard this model's attention `tp` ways, or None.

    Query heads must split evenly; vLLM checks this when it builds the
    config. KV heads either split evenly or, when there are fewer of them
    than shards, are replicated, which needs `tp` to be a multiple of the KV
    head count. vLLM enforces that only when it builds the attention layers
    (`QKVParallelLinear`), after the weights download, so it is mirrored
    here. MLA models decode as MQA with one KV head, which always replicates.
    `linear_heads` is a hybrid model's Gated-DeltaNet (key, value) head
    counts; vLLM splits both evenly across shards, again only when it builds
    the layers, so both must divide `tp`.
    """
    if num_heads % tp:
        return f"TP {tp} does not divide {num_heads} attention heads"
    for count, kind in zip(linear_heads or (), ("key", "value")):
        if count % tp:
            return (f"TP {tp} does not divide {count} linear-attention "
                    f"{kind} heads")
    if use_mla or num_kv_heads % tp == 0 or tp % num_kv_heads == 0:
        return None
    return (f"{num_kv_heads} KV heads can neither be split {tp} ways nor "
            f"replicated evenly across {tp} shards")


def fits(tp, weights_gib, hbm_gib_per_device):
    """Whether one shard of the weights leaves room for the KV cache.

    An unknown size passes here so that no TP is ruled out, and the plan
    says the fit went unchecked.
    """
    if weights_gib is None:
        return True
    return weights_gib / tp <= WEIGHT_HBM_FRACTION * hbm_gib_per_device


def replica_cost(replicas):
    """What `replicas` engines cost that the weight-fit check leaves out."""
    return (f"each of the {replicas} replicas compiles its own graphs at "
            "startup and holds its own KV cache and per-sequence state, which "
            "the weight-fit check does not count")


def plan_parallelism(devices,
                     hbm_gib_per_device,
                     num_heads,
                     num_kv_heads,
                     weights_gib,
                     user_tp=None,
                     use_mla=False,
                     rejected=None,
                     linear_heads=None):
    """Choose TP and DP so that TP x DP uses every device.

    A TP the user states always wins; the plan only records what it costs.
    Otherwise the largest TP that divides the device count, suits the head
    counts, and fits the weights is chosen, so one engine serves the model.
    Data-parallel replicas are added only when the head counts, the device
    layouts, or vLLM rule out a TP that spans every device. A smaller TP
    with more replicas is not the default: the weight-fit check leaves out
    the KV cache and per-sequence state (hybrid models' linear-attention
    slots alone can exhaust one device), and every replica compiles its own
    graphs at startup, which on one host can outlast the engine-ready
    timeout. `smallest_fitting_tp` names the smallest TP the weights fit, so
    a report can say when that layout might serve faster.
    `linear_heads` is a hybrid model's Gated-DeltaNet (key, value) head
    counts, or None for a model without linear attention.
    `rejected` maps TPs that cannot run to why, said in full ("vLLM rejected
    TP 1: ...", "torch_tpu has no device layout for TP 1 ..."); they are not
    candidates.

    Returns the plan dict; its `ok` is False when no deployable TP exists.
    """
    rejected = rejected or {}
    plan = {
        "devices": devices,
        "hbm_gib_per_device": hbm_gib_per_device,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "use_mla": use_mla,
        "linear_attention_heads": list(linear_heads) if linear_heads else None,
        "weights_gib": None if weights_gib is None else round(weights_gib, 2),
        "recommended_engine": "custom_fastapi",
    }
    head_ok = [
        t for t in range(1, devices + 1) if devices % t == 0 and head_problem(
            t, num_heads, num_kv_heads, use_mla, linear_heads) is None
    ]
    candidates = [t for t in head_ok if t not in rejected]
    fitting = [
        t for t in candidates if fits(t, weights_gib, hbm_gib_per_device)
    ]
    # With the size unknown every TP "fits", so no smaller TP is known to.
    plan["smallest_fitting_tp"] = (min(fitting) if fitting
                                   and weights_gib is not None else None)

    if user_tp is not None:
        plan["source"] = "user"
        plan["tensor_parallel_size"] = user_tp
        plan["data_parallel_size"] = max(devices // user_tp, 1)
        notes = ["TP stated by the user"]
        problem = head_problem(user_tp, num_heads, num_kv_heads, use_mla,
                               linear_heads)
        if problem:
            plan["ok"] = False
            plan["reason"] = f"{problem}; vLLM will refuse it"
            return plan
        if user_tp > devices:
            plan["ok"] = False
            plan["reason"] = (f"TP {user_tp} exceeds the {devices} devices "
                              "available")
            return plan
        if user_tp in rejected:
            plan["ok"] = False
            plan["reason"] = rejected[user_tp]
            return plan
        unused = devices - user_tp * plan["data_parallel_size"]
        if unused:
            notes.append(f"{unused} of {devices} devices unused")
        if plan["data_parallel_size"] > 1:
            notes.append(replica_cost(plan["data_parallel_size"]))
        if weights_gib is None:
            notes.append("weight size unknown, so the memory fit was not "
                         "checked")
        elif not fits(user_tp, weights_gib, hbm_gib_per_device):
            notes.append(f"weights exceed {WEIGHT_HBM_FRACTION:.0%} of HBM "
                         "per device at this TP; expect OOM")
        plan["ok"] = True
        plan["recommended_engine"] = "vllm"
        plan["reason"] = "; ".join(notes)
        return plan

    plan["source"] = "recommended"
    skipped = "".join(f"; {why}" for _, why in sorted(rejected.items()))
    if not fitting:
        plan["ok"] = False
        plan["recommended_engine"] = "custom_fastapi"
        plan["tensor_parallel_size"] = None
        plan["data_parallel_size"] = None
        if candidates:
            plan["reason"] = (
                f"weights do not fit on {devices} x {hbm_gib_per_device:g} "
                f"GiB devices at any head-compatible TP {candidates}")
        elif head_ok:
            plan["reason"] = (
                f"every head-compatible TP {head_ok} was rejected")
        else:
            plan["reason"] = (
                f"no TP dividing {devices} devices suits {num_heads} "
                f"attention heads ({num_kv_heads} KV heads")
            if linear_heads:
                plan["reason"] += (f"; {linear_heads[0]} key and "
                                   f"{linear_heads[1]} value linear-attention "
                                   "heads")
            plan["reason"] += ")"
        plan["reason"] += skipped
        return plan

    tp = max(fitting)
    plan["ok"] = True
    plan["recommended_engine"] = "vllm"
    plan["tensor_parallel_size"] = tp
    plan["data_parallel_size"] = devices // tp
    if weights_gib is None:
        plan["reason"] = (
            f"weight size unknown; the largest head-compatible TP ({tp}) "
            "gives each device the smallest shard")
    else:
        plan["reason"] = (
            f"largest TP that divides {num_heads} heads and whose weight "
            f"shard ({weights_gib / tp:.1f} GiB) fits "
            f"{WEIGHT_HBM_FRACTION:.0%} of {hbm_gib_per_device:g} GiB, "
            "leaving the most HBM for the KV cache and compiling one engine")
    if devices // tp > 1:
        plan["reason"] += (f"; {devices // tp} replicas use all {devices} "
                           f"devices; {replica_cost(devices // tp)}")
    smallest = plan["smallest_fitting_tp"]
    if smallest is not None and smallest < tp:
        plan["reason"] += (
            f"; TP {smallest} also fits and, with more replicas, may "
            "serve faster: pass --tp to measure it, at the cost of startup "
            "time and memory headroom")
    plan["reason"] += skipped
    return plan


def vllm_error(exc):
    """The useful line of a vLLM config error, without pydantic's framing."""
    text = str(exc)
    match = re.search(r"Value error, (.*?)(?: \[type=|$)", text, re.MULTILINE)
    if match:
        return match.group(1)
    return text.splitlines()[0] if text else repr(exc)


# How torch_tpu says it has no layout for a device count. Only this, or no
# layout returned at all, rules a TP out; any other failure of the lookup says
# nothing about the TP.
_NO_LAYOUT = "No TPU topology found"


class TopologyLookupError(Exception):
    """The topology lookup failed for a reason other than refusing a TP."""


def topology_rejections(devices, get_topology):
    """TPs dividing `devices` that the installed torch_tpu cannot lay out.

    Each engine asks torch_tpu for the device layout of its own TP group (a
    data-parallel replica is a separate engine), and torch_tpu knows layouts
    only for some device counts, which differ by TPU generation and by
    torch_tpu version. `get_topology(n)` is that same lookup, so asking it
    here keeps the plan to what the installed stack will start, without a
    list of counts kept in this script. Maps each refused TP to why.

    Raises TopologyLookupError when the lookup fails any other way (a changed
    signature, a busy runtime, no TPU visible): that is no evidence against
    any TP, so the caller rules none out rather than all of them.
    """
    refused = {}
    for tp in range(1, devices + 1):
        if devices % tp:
            continue
        try:
            layout = get_topology(tp)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # The whole message decides; only its first line is reported.
            why = str(e).splitlines()[0] if str(e) else repr(e)
            if _NO_LAYOUT not in str(e):
                raise TopologyLookupError(f"{type(e).__name__}: {why}") from e
        else:
            if layout is not None:
                continue
            why = "no layout returned"
        refused[tp] = f"torch_tpu has no device layout for TP {tp} ({why})"
    return refused


def find_topology_lookup():
    """The plugin's per-engine topology lookup, or None if it has none.

    vllm_torchtpu's workers call `TpuPlatform._get_tpu_topology(world_size)`
    for their engine's TP group. Plugins from before it existed set up any
    device count, so its absence means nothing is ruled out.
    """
    try:
        # pylint: disable=g-import-not-at-top
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
    except ImportError:
        return None
    return getattr(TpuPlatform, "_get_tpu_topology", None)


def apply_vllm_patch(patch):
    """Applies `patch` to the installed vllm_torchtpu before it is imported.

    The applier, `apply_vllm_patch.sh`, travels beside the patch. Returns
    None on success, else the reason the patch could not be applied.
    """
    applier = os.path.join(os.path.dirname(os.path.abspath(patch)),
                           "apply_vllm_patch.sh")
    if not os.path.isfile(patch) or not os.path.isfile(applier):
        return (f"{patch} and apply_vllm_patch.sh must both be passed to the"
                " target as inputs, in the same directory")
    # Located, not imported: importing the plugin starts the TPU runtime,
    # and the patch must land before anything loads it.
    spec = importlib.util.find_spec("vllm_torchtpu")
    if spec is None or not spec.submodule_search_locations:
        return f"vllm_torchtpu is not importable with {sys.executable}"
    root = os.path.dirname(
        os.path.abspath(list(spec.submodule_search_locations)[0]))
    env = dict(os.environ, TPU_AGENT_SUITE_PYTHON=sys.executable)
    done = subprocess.run(["bash", applier, patch, root],
                          capture_output=True,
                          text=True,
                          env=env,
                          check=False)
    print(done.stdout, end="")
    if done.returncode:
        return (done.stderr.strip()
                or f"apply_vllm_patch.sh exited {done.returncode}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="TorchTPU VLLM Pre-flight Guardrail Check")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Hugging Face model id, local directory, or "
                        "gs:// checkpoint folder")
    parser.add_argument("--devices",
                        type=int,
                        required=True,
                        help="TPU devices one deployment gets")
    parser.add_argument("--hbm-gb-per-device",
                        type=float,
                        required=True,
                        help="HBM per device, as compute inspect reports it")
    parser.add_argument("--tp",
                        type=int,
                        default=None,
                        help="Tensor parallel size the user stated, if any")
    parser.add_argument("--plan-json",
                        type=str,
                        default="tp_plan.json",
                        help="Where to write the TP/DP plan")
    parser.add_argument("--vllm-patch",
                        type=str,
                        default=None,
                        help="vllm_torchtpu patch the model needs, applied"
                        " before vLLM is imported; apply_vllm_patch.sh must"
                        " sit beside it")
    args = parser.parse_args()

    if args.devices < 1 or args.hbm_gb_per_device <= 0:
        print("CRITICAL ERROR: --devices and --hbm-gb-per-device must be "
              "positive. compute inspect did not report the hardware; "
              "check details.hbm_gb_note.")
        sys.exit(1)
    if args.tp is not None and args.tp < 1:
        print("CRITICAL ERROR: --tp must be at least 1.")
        sys.exit(1)

    if args.vllm_patch:
        problem = apply_vllm_patch(args.vllm_patch)
        if problem:
            print("CRITICAL ERROR: the vllm_torchtpu patch this model needs"
                  f" did not apply.\n{problem}")
            sys.exit(1)

    # A gs:// folder: vLLM gets a copy of its config and tokenizer, never the
    # URL, and the weight size comes from the folder itself.
    engine_model, gcs_size = args.model, None
    if args.model.startswith(GCS_SCHEME):
        try:
            engine_model, gcs_size = gcs_model(args.model)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            print(f"CRITICAL ERROR: could not read {args.model}: {e}")
            sys.exit(1)

    try:
        # pylint: disable=g-import-not-at-top
        # pylint: disable=g-importing-member
        # Ensure vllm_torchtpu is patched into standard vLLM registry
        import vllm_torchtpu  # noqa: F401  # pylint: disable=unused-import
        from vllm.engine.arg_utils import EngineArgs
    except ImportError as e:
        print("CRITICAL ERROR: Failed to import vLLM or TorchTPU backend. Is"
              f" the environment set up? \n{e}")
        sys.exit(1)

    print("==================================================")
    print(f" PRE-FLIGHT CHECK: {args.model}")
    print(f" DEVICES: {args.devices} x {args.hbm_gb_per_device:g} GiB")
    print("==================================================")

    # We enforce VLLM_TARGET_DEVICE="tpu" implicitly via the environment or setup.
    # vllm_torchtpu overrides standard vLLM mechanisms to hook into the TPU.
    def engine_config(tp, dp=1):
        return EngineArgs(
            model=engine_model,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            # We don't want to actually allocate TPU memory or download heavy
            # weights during pre-flight. We only want to force the config
            # parsers to execute their static analysis.
            enforce_eager=True,
        ).create_engine_config()

    print("\n[1/3] Introspecting Architecture and Pallas Kernels...")
    try:
        # This function fetches the config.json and runs it through vLLM's
        # initialization logic. The TorchTPU patches will intercept it. If it sees
        # an unsupported feature (like AWQ, missing Sliding Window Attention
        # masking, or ALiBi), it crashes here. TP 1 keeps a TP problem from
        # reading as an architecture one; TP is checked in step 2.
        config = engine_config(1)
        print(
            "  ✅ Architecture is officially supported by the TorchTPU backend."
        )
        print(f"  ✅ Quantization Config: {config.model_config.quantization}")
        print("  ✅ Max Model Length Allowed: "
              f"{config.model_config.max_model_len}")

    except NotImplementedError as e:
        print(
            "\n❌ PRE-FLIGHT FAILED: TorchTPU Pallas Kernel Compatibility Error"
        )
        print(f"     Reason: {str(e)}")
        print(
            "     Action: This model requires an operation that has not been ported"
            " to TPU yet.")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print("\n❌ PRE-FLIGHT FAILED: Standard Hub/Config Verification Error")
        print(f"     Reason: {str(e)}")
        if "401" in str(e) or "gated" in str(e).lower():
            print(
                "     Action: This is a gated model. Please export HF_TOKEN via the"
                " environment.")
        sys.exit(1)

    print("\n[2/3] Planning Tensor / Data Parallelism...")
    # Read heads the way vLLM does: its getters resolve architecture-specific
    # names (Falcon's multi_query, n_head_kv, ...) that the raw HF config
    # does not.
    model_config = config.model_config
    num_heads = model_config.model_arch_config.total_num_attention_heads
    num_kv_heads = model_config.get_total_num_kv_heads()
    use_mla = bool(model_config.use_mla)
    # Gated-DeltaNet layers in hybrid models (Qwen3-Next, Qwen3.5) shard
    # their own heads, which the getters above do not report.
    text_config = model_config.hf_text_config
    linear_heads = tuple(
        getattr(text_config, name, None)
        for name in ("linear_num_key_heads", "linear_num_value_heads"))
    if not all(linear_heads):
        linear_heads = None
    weights_gib = None
    try:
        if args.model.startswith(GCS_SCHEME):
            size = gcs_size
        elif os.path.isdir(args.model):
            size = local_weights_bytes(args.model)
        else:
            # pylint: disable=g-import-not-at-top
            from huggingface_hub import hf_hub_download, model_info

            info = model_info(args.model, files_metadata=True)
            index_total = None
            index = "model.safetensors.index.json"
            if any(s.rfilename == index for s in info.siblings or []):
                index_total = index_total_size(
                    hf_hub_download(args.model, index))
            size = weights_bytes(info, index_total)
        if size:
            weights_gib = size / (1024**3)
            print(f"  ✅ Base Model Size on Disk: ~{weights_gib:.1f} GiB")
        else:
            print("  ⚠️ WARNING: Could not read the weight size; the plan "
                  "takes the largest head-compatible TP.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"  ⚠️ WARNING: Could not read the weight size ({e}); the plan "
              "takes the largest head-compatible TP.")

    # vLLM has the last word on the (TP, DP) pairs the plan names: the one to
    # deploy, and the single replica without data parallelism. A TP it
    # refuses is dropped and the plan is redone, so the next fitting TP is
    # tried. vLLM does not check TP x DP against the devices; the plan does.
    rejected, accepted = {}, set()
    lookup = find_topology_lookup()
    if lookup is None:
        print("  ⚠️ WARNING: this vllm_torchtpu has no topology lookup, so "
              "no TP is ruled out for lack of a device layout.")
    else:
        try:
            rejected.update(topology_rejections(args.devices, lookup))
        except TopologyLookupError as e:
            print(f"  ⚠️ WARNING: the topology lookup failed ({e}), so no TP "
                  "is ruled out for lack of a device layout.")
    while True:
        plan = plan_parallelism(args.devices, args.hbm_gb_per_device,
                                num_heads, num_kv_heads, weights_gib, args.tp,
                                use_mla, rejected, linear_heads)
        unchecked = [] if not plan["ok"] else [
            # (TP, 1) is what an engine that cannot run replicas starts.
            pair for pair in dict.fromkeys(((plan["tensor_parallel_size"],
                                             plan["data_parallel_size"]),
                                            (plan["tensor_parallel_size"], 1)))
            if pair not in accepted
        ]
        if not unchecked:
            break
        for tp, dp in unchecked:
            try:
                engine_config(tp, dp)
                accepted.add((tp, dp))
            except Exception as e:  # pylint: disable=broad-exception-caught
                rejected[tp] = f"vLLM rejected TP {tp}: {vllm_error(e)}"
    with open(args.plan_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"  ℹ️  Heads: {num_heads} attention, {num_kv_heads} KV"
          f"{' (MLA: KV replicates at any TP)' if use_mla else ''}" +
          (f", {linear_heads[0]} linear-attention key, "
           f"{linear_heads[1]} value" if linear_heads else ""))
    print(f"  ℹ️  TP {plan['tensor_parallel_size']} x DP "
          f"{plan['data_parallel_size']} ({plan['source']})")
    print(f"  ℹ️  Reason: {plan['reason']}")
    print(f"  ℹ️  Plan written to {args.plan_json}")

    print("\n[3/3] Final Verdict")
    if not plan["ok"]:
        print(f"  ❌ NOT DEPLOYABLE AS PLANNED: {plan['reason']}")
        print("==================================================")
        sys.exit(1)
    print(
        "  ✅ CLEAR FOR DEPLOYMENT. You may proceed to `torch_inference_coder`."
    )
    print("==================================================")
    sys.exit(0)


if __name__ == "__main__":
    main()
