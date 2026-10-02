# HANDOFF: what a promoted llm-d release leaves behind

The deliverable of a study is not the best arm; it is a running, observable
release and the bundle to recreate it. `llmd-productionizer` writes the
bundle under `<run>/artifacts/llmd/production/`, and `promote_release.py
bundle` is the only thing that writes it, so two promotions of the same
incumbent produce the same files.

## Layout

```
production/
├── production-config.json     # incumbent + router.epp_replicas/epp_ha_mode; what `render --production` consumed
├── llmd-config.json           # THE result: the config the release runs, with provenance.promoted_from
├── serving-recipe.json        # the single-replica recipe the study tuned around
├── search-ledger.md           # every arm: kind, status, verdict, metric, delta, execution; blocked arms as findings
├── router.values.yaml         # llm-d-router-standalone values as applied (EPP replicas, flags, plugin config)
├── epp-config.yaml            # EndpointPickerConfig: the tuned plugin set and weights
├── inference-objectives.yaml  # one per tenant with priority (rendered by the chart; listed for review)
├── exposure.yaml              # the front-door Service (ClusterIP, or internal passthrough LB)
├── workload/                  # workload-spec.json, resolved profile, the inference-perf configs used
├── observability/             # podmonitoring.yaml as applied; README on what is and is not there
├── autoscaling/README.md      # UNVERIFIED, with the reason and the recipe for doing it
├── acceptance-report.md       # every check with its measured value; every drill; everything UNVERIFIED
├── RUNBOOK.md                 # health, re-tune, roll back, scale, rotate image, re-run study, drill, remove, migrate
├── release.json               # digests and pins: config, incumbent, chart, EPP image, llm-d ref, inference-perf ref
├── llmd-promote-result.json   # the envelope returned to tpu-llmd
├── acceptance/                # `deploy bench` output of the acceptance run
├── drills/                    # one JSON row per drill
└── release/                   # the rendered release directory the runtime applied (manifests, bench Jobs)
```

## What "accepted" means

`promote_release.py bundle` judges three things, each as its own row in
`acceptance-report.md`:

1. **Gates.** Every gate in the acceptance bench report passed, in the same
   execution: each tenant's SLO percentiles, each tenant's error rate, the
   chip budget, and the load generator was not the bottleneck
   (`load.client_limited`). The report must be `admissible`.
2. **Same as measured.** When the request names the incumbent's own bench
   report (`acceptance.baseline_report_path`), the primary metric in
   production is within `acceptance.noise_floor_pct` (default 5%) of it, or
   better. The production release is the incumbent re-applied; a worse
   number says the render differs from what the study measured.
3. **Drills.** Every drill in the request recovered within its timeout and
   the front door answered afterwards.

Any row failing makes the result `acceptance_failed`. The release is left
up for the user to inspect; nothing is torn down.

## What is always UNVERIFIED here

- **Autoscaling.** No ScaledObject or HPA is rendered. An adopted fleet's
  Deployment is not the release's to scale; a managed fleet's replica count
  is a cold path. No KEDA soak was run. `autoscaling/README.md` carries the
  recipe: scale on the EPP's inflight-token metric against the measured
  knee, min = the capacity floor, max = the chip budget, and re-run
  acceptance under the sweep ladder.
- **Zonal failover.** Single cluster, single zone.
- **Canary rollout.** The release is applied whole. A weighted split needs
  a second router release behind the front door, or HTTPRoute weights in
  Gateway mode.
- **Model-server drills** on an adopted fleet: the pods are the owner's.
- **EPP HA, exposure and the EPP drill** when the incumbent is passthrough:
  no router was installed, so there is nothing to make highly available.

## Production settings the bundle adds

Only two, both on the router, both recorded in
`llmd-config.json.provenance.promoted_from.production_changes`:

| setting | default | why |
| :-- | :-- | :-- |
| `router.epp_replicas` | 2 | one EPP is a single point of failure for every request through the router |
| `router.epp_ha_mode` | `active-passive` | with `epp.replicas > 1` the chart enables leader election by itself; the standby is deliberately not Ready so the Service follows the leader (https://github.com/llm-d/llm-d-router/tree/v0.11.0/config/charts). `active-active` is for plugin sets that only read model-server-pulled state. |

The front door is a Service named `<release>-ingress` selecting the EPP
pods (Envoy is their sidecar). `cluster-ip` is the default; `internal-lb`
adds GKE's `networking.gke.io/load-balancer-type: Internal` annotation and
asks for an internal passthrough load balancer
(https://cloud.google.com/kubernetes-engine/docs/concepts/service-load-balancer).

## Guards around production

- `deploy teardown` refuses a production release without
  `--include-production`.
- The `llmd-production-guard` hook denies that flag to agents unless the
  user has placed a `.teardown-approved` marker file in the bundle
  directory (next to `release.json`), and denies raw mutations that name
  the production label or namespace.
- The reaper (`SessionEnd`) tears down study releases and leaves
  `serve=true` records alone.
- `llmd-chip-budget-guard` subtracts production holdings from the study's
  chip budget before any later arm is dispatched.
