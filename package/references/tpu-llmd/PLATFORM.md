# The platform: what a cluster needs before a release can land

`{{runtime}} deploy preflight` is the only place the study inspects the
cluster, and `{{runtime}} deploy probe` the only place it inspects the
serving image. Both read the `llmd` section of the package environment. This
note explains each check and what `--fix` may touch, so a blocker can be
read as the action it implies.

## The environment's `llmd` section

Collected by the environment questions when the compute backend is a GKE
one (`{{schema:environment.json}}`):

| key | meaning | default |
| :--- | :--- | :--- |
| `namespace` | where study releases live; everything there is disposable | `llmd-study` |
| `production_namespace` | where a promoted release would live; `teardown` refuses to touch it without `--include-production` | unset |
| `gateway_mode` | `standalone` (llm-d router with its own Envoy, no Gateway API class needed) or `gke_gateway` | `standalone` |
| `hf_token_secret` | Secret holding `HF_TOKEN` for weight download | `hf-token` |
| `kube_context` | kubeconfig context, when not the current one | unset |
| `prometheus_project` / `prometheus_url` | where `collect` reads vLLM metrics (Managed Prometheus by project, or a PromQL endpoint) | the compute project |
| `inference_perf_image` | the load generator image | pinned |
| `max_chips_default` | chip budget when the workload spec names none | unset |

The cluster, project, location and TPU accelerator come from the compute
section the GKE backend already filled in; `llmd` adds only what serving
needs beyond job submission.

## The checks, in order

| check | fails when | `--fix` | otherwise |
| :--- | :--- | :--- | :--- |
| `tool.kubectl`, `tool.helm` | not on `PATH` | no | install; helm is how the router chart is applied |
| `tool.gcloud` | not on `PATH` (warn) | no | `collect` needs it for a Managed Prometheus token; the study runs without a profile until then |
| `cluster.reachable` | the API server does not answer for the configured context | no | fix credentials/context; nothing else is checked |
| `namespace.<ns>` | the study namespace is missing | **creates it** | -- |
| `crd.gateway` | Gateway API CRDs absent (`gke_gateway` mode only) | no | enable the GKE Gateway controller on the cluster |
| `crd.inferencepool` | `inferencepools.inference.networking.k8s.io` absent | **applies the pinned GAIE CRD manifest** | the router's EPP registers an InferencePool even in standalone mode |
| `crd.inferenceobjective` | `inferenceobjectives.llm-d.ai` absent | **applies the pinned llm-d-router CRD manifest** (the release's `manifests.yaml`) | the chart renders one `InferenceObjective` (`llm-d.ai/v1alpha2`) per tenant; GAIE's bundle does not carry that group, so a GKE cluster with the Inference Gateway CRDs still fails the install without it |
| `secret.hf_token` | the token secret is not in the namespace | no | create it; a model-server pod without it fails on weight download, after the chips were scheduled |
| `tpu.nodes` | no nodes carry the recipe's accelerator label, or fewer free chips than one replica needs | no | add or resize the TPU node pool; `chips_free` is reported for the orchestrator's budget |
| `crd.podmonitoring` | Managed Prometheus CRD absent (warn) | no | enable managed collection on the cluster; without it `collect` has nothing to read |
| `helm.router_chart` | the pinned standalone router chart cannot be pulled | no | registry access; the chart coordinates are pinned in the runtime, never typed by an agent |

`--fix` is limited to the two actions that are idempotent, namespace-scoped
or additive, and reversible (the namespace by `teardown`, the CRDs by
removing the pinned manifest). It never
creates node pools, secrets or IAM bindings: each of those is a billing or a
security decision a person makes.

Why standalone by default: llm-d's router ships a standalone chart that
bundles its own Envoy and EPP and needs no Gateway API class or GKE Inference
Gateway enablement on the cluster
(https://github.com/llm-d/llm-d-router/blob/main/config/charts/README.md).
That removes a cluster prerequisite from the critical path of a study; the
`gke_gateway` mode exists for a production cluster that already standardises
on the GKE Inference Gateway
(https://cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway).

## The probe

`deploy probe --image <recipe image> --out <dir>` runs one short TPU Job on
the image; `deploy probe --selector <labels> --out <dir>` runs the same
check inside a pod that is already serving (an adopted fleet, where no
chip is free for a Job). Both import the connector modules llm-d's TPU
guides configure for KV events, P/D and KV offload with JAX and torch
pinned to CPU (`JAX_PLATFORMS=cpu`), so a running server is never
disturbed, and write `engine-capabilities.json`
(`{{schema:engine-capabilities.json}}`): each capability true or false, the
module that satisfied it, and the vLLM / tpu-inference / vllm-torchtpu
versions.

The modules looked for, per the guides at the pinned release
(https://github.com/llm-d/llm-d/tree/v0.10.0/guides): for `tpu_inference`
images `tpu_inference.distributed.tpu_connector_hma` (P/D) and
`tpu_inference.offload.tpu_offload_connector` (host tier); for
vllm-torchtpu images `vllm_torchtpu.distributed.kv_transfer.tpu_connector`
and `vllm_torchtpu.offload.raiden_connector`; and vLLM's own
`vllm.distributed.kv_events` for precise prefix routing. The recipe's
`engine_flavor` says which family the renderer emits.

The probe **gates** arms and never prunes them: the seeder and planner
write an arm whose capability is `false` as `blocked` with
`blocked_on_engine_capability`, so a missing connector is visible on round
one rather than discovered when the arm is finally opened. With no probe
every connector arm stays open, on the assumption that the vllm-torchtpu
image carries what the guides name. A connector reported absent is a
`blockers[]` line tagged informational and a sentence in the study
summary, not a platform failure.

## What the platform specialist does not do

It does not render or apply a release, size a node pool, change quotas, or
install anything beyond the two `--fix` actions. A platform problem the
checks cannot repair is a blocked study, reported with the action required,
and that is the correct outcome: an arm applied to an unready cluster fails
after the chips were scheduled, which is the most expensive way to learn
the same thing.
