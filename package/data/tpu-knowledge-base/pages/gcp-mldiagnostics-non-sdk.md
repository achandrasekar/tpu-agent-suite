<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: gcp-mldiagnostics-non-sdk
keyword: non-sdk-xprof
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-09-13
source_updated: not applicable
generator: authored
---

# Getting a hosted XProf link without the ML Diagnostics SDK

Google Cloud ML Diagnostics can serve a hosted XProf dashboard for a run whose
traces you produced and uploaded yourself. Nothing needs to import an ML
Diagnostics SDK or wrap the training loop: the workload writes standard XProf
traces, you copy them to a Cloud Storage prefix, and you register the run
against the Hypercompute Cluster API. Three decoupled steps, each one a thing
you can already do.

This matters when the SDK is not available in the image, when the workload is
not Python, or when profiling has to be a post-step on artifacts that already
exist.

## 1. Profile the workload with the stock JAX profiler

```python
import jax

profile_dir = "/tmp/profile"

jax.profiler.start_trace(profile_dir)
# ... run the steps to be profiled ...
out.block_until_ready()          # dispatch is async; without this the trace is empty
jax.profiler.stop_trace()
```

This writes the standard XProf layout, and the shape of it is what the rest of
the workflow depends on:

```
/tmp/profile/plugins/profile/<session_id>/<hostname>.xplane.pb
/tmp/profile/plugins/profile/<session_id>/<hostname>.trace.json.gz
```

## 2. Upload the artifacts

```bash
gcloud storage cp -r /tmp/profile/plugins/profile/* \
  gs://<BUCKET>/profiles/<RUN_ID>/plugins/profile/
```

**The destination layout is load-bearing, not a convention.** The backend looks
for `.xplane.pb` files under `<gcs_path>/<run_id>/plugins/profile/`, so the
`plugins/profile/` segment must survive the copy and `<RUN_ID>` must match the
run you register in step 3. Upload to a flattened prefix and the files are
there, the API returns success, and no session is ever created.

## 3. Register the run

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "<RUN_ID>",
    "runGroup": "<GROUP>",
    "artifacts": {
      "gcsPath": "gs://<BUCKET>/profiles"
    },
    "tools": [{ "xprof": {} }]
  }' \
  "https://hypercomputecluster.googleapis.com/v1alpha/projects/<PROJECT>/locations/<REGION>/machineLearningRuns?machine_learning_run_id=<RUN_ID>"
```

`artifacts.gcsPath` is the prefix **above** the run id, not the directory the
traces are in: the backend appends `<run_id>/plugins/profile/` itself. `tools`
must include `xprof` or no dashboard is proxied.

## 4. Ask for the session, which is what triggers the scan

```bash
curl -s -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  "https://hypercomputecluster.googleapis.com/v1alpha/projects/<PROJECT>/locations/<REGION>/machineLearningRuns/<RUN_ID>/profilerSessions"
```

Listing sessions is not a passive read. Uploaded traces that were never
registered by an SDK have no session until something looks for one; this call
is what makes the backend scan the prefix, create a session for what it finds,
and return the dashboard URI. Upload and then check the console without calling
this, and it will look like the upload failed.

## What you get

A profiler session whose kind marks it as discovered-by-scan rather than
SDK-registered, in a succeeded state, carrying the dashboard URI for the hosted
XProf UI.

## Constraints

- `v1alpha`. The path carries the version, and an alpha surface can be renamed
  or withdrawn; a call that 404s here is more likely a moved surface than a
  missing run.
- The caller needs ADC that can mint a token for the project, and write access
  to the bucket. On GKE, bucket writes from inside the workload also need the
  node pool's OAuth scope to permit storage writes — a property fixed at node
  pool creation that IAM cannot grant after the fact.
- `<REGION>` must be one the API serves; not every region that runs TPUs is
  one.

> [!inferred]
> The scan appears to be idempotent — re-listing after uploading more traces
> picks them up — but this was not tested deliberately, and no public page
> states the behaviour. Do not rely on it to refresh a session in place.

## Evidence

This entry has no source URL, which is why it is authored rather than
generated. It records a workflow that was run and observed, not a page that was
read, and the two are not the same kind of claim.

What is publicly documented, and can be checked:

- [Workload monitoring with ML Diagnostics](https://docs.cloud.google.com/tpu/docs/ml-diagnostics/workload-monitoring)
  — the product, its GKE integration and what it collects.
- [JAX profiling](https://docs.jax.dev/en/latest/201/profiling.html) —
  `jax.profiler.start_trace` and `stop_trace`. Note this is the maintained
  page; the unnumbered `/profiling.html` is a retained older version.

What is **not** publicly documented, and is reported here as observed
behaviour rather than cited: the `v1alpha` request shape above, that
`artifacts.gcsPath` sits above the run id, and that listing profiler sessions
is what triggers the scan. Treat those as true of the day in `retrieved` and
verifiable only by running them.
