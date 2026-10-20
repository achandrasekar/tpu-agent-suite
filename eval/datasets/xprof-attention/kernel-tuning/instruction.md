The attention layer in `/workspace/attention.py` is the hot spot in my model and
it is slower than I think it should be. Make it faster on the TPU, and show me
the measurements that justify what you changed.

`attention()` is the function that matters; the rest of that file is the harness
that profiles it. Keep the shapes and the causal masking as they are — I need
the same computation, faster.

`/workspace/run_on_tpu.sh <label>` runs it on a TPU node and brings the trace
back to `/logs/artifacts/xprof/<label>/`. Use a different label per variant so
they do not overwrite each other. You can dispatch the work yourself if you
prefer, but the trace has to end up in that directory either way.

Profile the code as it stands first, then **at most 5 further attempts** -- six
profiles in all, and the sixth is the last one read. Anything you measure after
that is invisible to me, so the number you report must come from one of the six.

Write every trace under `/logs/artifacts/xprof/`. That directory is collected
off the machine; anything written elsewhere is discarded when the run ends.

The chips are on a Kubernetes cluster your environment is already configured
for. `$TASK_DOCKER` names an image with an accelerator runtime, but **not one
that can run this workload** — its JAX is too old for the profiling flags the
harness sets, and the run dies with `Unknown command line flag`. Bring your own
image, as `/workspace/run_on_tpu.sh` does, or install what you need into the
pod. If you do send work into `$TASK_DOCKER`, put it under `/app` — `/workspace`
is taken inside it. `$GKE_TPU_ACCELERATOR` and `$GKE_TPU_TOPOLOGY` are the two
node labels that
pick a TPU node; use them as given, since a wrong pair leaves the work pending
rather than failing. If that cluster admits work through Kueue, create the Job
in namespace `$GKE_NAMESPACE` and label it
`kueue.x-k8s.io/queue-name: $GKE_QUEUE` — a Job that names no queue is accepted
and then suspended forever, which looks like a slow pod rather than an error.

`$GKE_PRIORITY` is present for clusters that use priority classes. This one
does not admit arbitrary values: setting `priorityClassName` from it on a Job
here is rejected outright with `[FORBIDDEN] Invalid priority class specified`.
Leave it alone unless you have checked what the cluster accepts.

Make it faster **on one chip**. The node your job lands on has several, and
the workload uses one; sharding the same arithmetic across the rest is not what
is being measured here and a result that spans more than one device is
discarded. Everything about the kernel itself is in scope.

YOU MUST use the TPU resources named in your environment, the cluster in
`$GKE_CLUSTER` and no other. Do not provision new ones.

Work out up front what tooling is installed for you — agents, skills,
command-line profilers — and use it. Do that before you start rather than when
you hit something you cannot answer: what is available changes how you should
approach this. If nothing is, do the work yourself.

End your reply with a single line, exactly this shape:

    FINAL: <device_us> us, <what you changed>, from <baseline_device_us> us

Both numbers are per-device time for the attention computation, taken from a
profile rather than from wall clock.
