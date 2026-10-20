I'm about to port a PyTorch model to TPU, and before I write any of it I need
to know that PyTorch can really use the hardware -- not that a node answers,
but that a matmul actually runs on the TPU and comes back with the right
answer. I also need to know what the chip I landed on actually is, from a
source I can check, because the number I plan against comes from that spec and
not from the run.

The chips are on a Kubernetes cluster your environment is already configured
for, and $TASK_DOCKER names an image that has the PyTorch TPU runtime in it.
Put anything you send into that image under /app -- /workspace is already
taken inside it. $GKE_TPU_ACCELERATOR and $GKE_TPU_TOPOLOGY are the two node
labels that pick a TPU node out of the cluster; use them as given rather than
guessing, since a wrong pair leaves the work pending instead of failing.

Work out up front which of these are in your environment: `tpu-agent`,
`tpu-hello-world`, `tpu-knowledge-base`. Do that before you start, not when you hit something
you cannot answer -- what is available changes how you should approach this,
and finding out late means redoing work.

- **`tpu-agent`** is your entry point, and routes you between steps.
- **`tpu-hello-world`** is your end-to-end guide for the steps of this task. Use it
  rather than writing your own check.
- **`tpu-knowledge-base`** is your library for the knowledge this needs. It answers
  from an indexed public page and hands back the URL to cite, so an answer
  from it is checkable in a way that one from memory is not.

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
