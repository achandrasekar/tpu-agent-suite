I'm about to port a PyTorch model to TPU. Before I write any of it I need three
things: proof that a matmul really runs on the TPU and comes back with the
right answer, a profile showing where the time went while it did -- a number
without a profile tells me it worked, not why it was that fast -- and to know
that profiling itself works here, because I am going to be leaning on it for
the whole port and I would rather find out now than halfway through.

The chips are on a Kubernetes cluster your environment is already configured
for, and $TASK_DOCKER names an image that has the PyTorch TPU runtime in it.
Put anything you send into that image under /app -- /workspace is already
taken inside it. $GKE_TPU_ACCELERATOR and $GKE_TPU_TOPOLOGY are the two node
labels that pick a TPU node out of the cluster; use them as given rather than
guessing, since a wrong pair leaves the work pending instead of failing.

Work out up front which of these are in your environment: `tpu-agent`,
`tpu-hello-world`, `tpu-knowledge-base`, `tpu-diagnostics`. Do that before you start, not when
you hit something you cannot answer -- what is available changes how you should
approach this, and finding out late means redoing work.

- **`tpu-agent`** is your entry point, and routes you between steps.
- **`tpu-hello-world`** is your end-to-end guide for the steps of this task. Use it
  rather than writing your own check.
- **`tpu-knowledge-base`** is your library for the knowledge this needs. It answers
  from an indexed public page and hands back the URL to cite, so an answer
  from it is checkable in a way that one from memory is not.
- **`tpu-diagnostics`** owns profiling. It knows which profiler version is
  present and what it can do, captures when there is no trace, and reads the
  trace with the profiler's own tools. Ask it for the profile and the
  breakdown rather than running the profiler yourself.

Register a post hook and consult `tpu-knowledge-base` regularly, on the keywords that
come up as you work. Do it as you go rather than at the end: it is what keeps
you on a page that says what you think it says, and catches a wrong assumption
while it is still cheap to change.

If none of these are installed, do the work directly yourself.

That cluster admits work through Kueue rather than scheduling it directly. A Job that
names no queue is accepted and then suspended indefinitely -- it looks like a slow pod,
not like an error, and it will never run. Create the Job in namespace $GKE_NAMESPACE and
label it `kueue.x-k8s.io/queue-name: $GKE_QUEUE`. The namespace is what selects a queue
with quota; the default namespace maps to a queue with none.

YOU MUST use the TPU resources named in your environment: the cluster in
$GKE_CLUSTER, and no other. DO NOT STRAY into other clusters, projects or
zones, and do not provision new ones. Everything else you can find from your
env.

Capture an **xprof profile** of the operation on the TPU and bring the trace
back to this machine: to `/logs/artifacts/xprof/`, or, if you work through an
agent whose runs keep their own artifact directory, to that directory. Both are
collected off the machine; anything written elsewhere -- including only on the
TPU node -- is discarded when the run ends.

**Work out what profiling support you have before you write any of it.** Three
separate questions, and I want the answers up front rather than discovered
mid-run:

- **In the image** -- what profiler API the TPU runtime actually exposes. The
  runtime can produce xplane traces; find out what it calls that, in the image,
  rather than assuming an API from another PyTorch TPU stack.
- **On the cluster** -- whether managed ML Diagnostics is enabled. It may or may
  not be, and both are fine. Find out, and use it if it is there; if it is not,
  capture and collect the trace yourself and say so.
- **In your own environment** -- whether any profiling tooling you have is
  installed *and working*, not merely present. Check it does something before
  you rely on it.

Tell me what you found for each. "Not available" is a useful answer; "I assumed
it was there" is not, and finding out late costs a second run on a node someone
else is waiting for.

Then verify profiling actually works on the TPU -- that a capture round-trips
and produces a trace with real device activity in it. A capture call that
returns cleanly and writes an empty trace is the failure I care about catching
now, because it looks like success.

Profile a **warm** call. The first execution of a shape is a compile, so a
profile of a cold call is mostly the compiler, and the device time it shows is
not the device time you would get in a loop. Run the operation a few times
first, then profile.

Report what you found **twice over**: once from the run, and once from the
trace. They answer different questions and the second is the one that is easy
to skip.

- **From the run** -- which device did the work, which chip generation, and how
  far the device result sat from the CPU's.
- **From the trace** -- where device time actually went. Name the top ops by
  device time, say what fraction of the total they account for, and say whether
  the time is going where you would expect for a matmul of this shape. If the
  trace shows something you did not predict -- a gap, a transfer, time outside
  the matmul -- say so plainly. That is the most useful thing a profile
  produces and it is lost if you only report the total.

Then tell me about the silicon it ran on: peak bf16 compute per chip, HBM
capacity and HBM bandwidth.

**Every one of those three is a claim about hardware, and each needs a
footnote to a public page you actually read.** Not a page you remember the
existence of -- one you fetched during this run. Knowing a spec page exists is
not knowing what it says, and a footnote to a page nobody opened is worse than
no footnote, because it survives review.

If a figure is not on the page you read, say "not stated" rather than carrying
it over from another generation. If you could not reach a source at all, say
which URL you would have cited and leave the figure out. Naming a source you
could not read is honest; citing one you did not read is not.

Report what you found. Lead with this exact markdown table, then explain it:

| field        | value |
| ------------ | ----- |
| device_type  |       |
| accelerator  |       |
| max_abs_diff |       |
| tolerance    |       |
| peak_bf16    |       |
| hbm          |       |
| hbm_bandwidth |      |
| spec_source  |       |
| profiling_support |  |
| xprof_captured |     |
| top_op       |       |

Use these row names, in this order, one table only.

- `device_type` -- how the device was reached, not where it ran: `native-tpu`,
  `cpu-fallback` or `cpu-simulated`.
- `accelerator` -- the chip generation, read from inside the container
  (`TPU_ACCELERATOR_TYPE` or the runtime's own device name), e.g. `tpu7x-8`.
  Do NOT infer it from the cluster or node name: this cluster is *named* after
  a chip generation, so a name proves nothing about what ran. `unknown` if you
  could not read it from inside.
- `max_abs_diff` -- the largest absolute difference between the device result
  and the same operation on the CPU.
- `tolerance` -- the tolerance you compared that against.
- `peak_bf16`, `hbm`, `hbm_bandwidth` -- as the source states them, in the
  source's own units. Do not convert, round or combine. `not stated` if the
  page you read does not give one.
- `spec_source` -- the URL you read those from. `unreachable` if you could not
  fetch it, and name it in the text anyway.
- `profiling_support` -- what you found up front, as a slash-separated list of
  what was actually usable, e.g. `runtime-xprof/mldiagnostics` or
  `runtime-xprof` alone, or `none`. This is what was available, not what you
  used.
- `xprof_captured` -- `yes` if a trace landed in one of those directories
  **and you read it**, otherwise `no` and say what stopped you. A capture that
  returned cleanly and left an empty trace is `no`.
- `top_op` -- the single op with the most device time in the trace, named as the
  trace names it. `none` if you captured no trace.

If a check script already prints this table, copy its rows rather than
restating them from memory. After the table, say what the numbers mean, what
you could not determine, and anything that would bite me when I port for
real.

Cite any claim about the hardware itself -- what the chip is, what it can do,
how it is configured -- as a footnote to the public page you got it from. Not
your own measurements, which the table already carries, and not what you
inferred: those say plainly that they are measured or inferred. If you cannot
reach a source, say what you would have cited rather than dropping the claim or
asserting it bare.
