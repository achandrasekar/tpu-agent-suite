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
the image that imports the connector modules llm-d's TPU guides configure
for P/D and KV offload (the `tpu_connector` for prefill/decode transfer and
the offloading connector for a host-memory tier, per
https://github.com/llm-d/llm-d/tree/main/guides) and writes
`engine-capabilities.json` (`{{schema:engine-capabilities.json}}`): each
connector `present` or not, and the vLLM / tpu-inference versions the image
reports.

The study **assumes** the vllm-torchtpu image carries those connectors, as
the guides describe. The probe exists so that an image that does not is
reported on round one rather than discovered when a P/D arm is finally
opened; it changes nothing about which arms are seeded or open. A connector
reported absent is a `blockers[]` line tagged informational and a sentence
in the study summary.

## What the platform specialist does not do

It does not render or apply a release, size a node pool, change quotas, or
install anything beyond the two `--fix` actions. A platform problem the
checks cannot repair is a blocked study, reported with the action required,
and that is the correct outcome: an arm applied to an unready cluster fails
after the chips were scheduled, which is the most expensive way to learn
the same thing.
