# XProf Attention Kernel Tuning Evaluation (`xprof`)

An evaluation suite that hands an agent a plain-JAX causal attention layer, asks for it faster on one TPU chip, and decides from the profiles it produced rather than the table it wrote.

Where a coding benchmark checks the diff, this one checks the traces:
1. **Speedup**: per-device time for the attention computation, baseline vs. the variant the agent kept.
2. **Kernel authorship**: whether a `tpu_custom_call` appeared at all.
3. **Instruction following**: six profiles in all — one baseline plus five attempts — counted from disk and from a capture ledger.
4. **Result validity**: the kept variant still computes the original, at the original precision, on a single device.
5. **Search behaviour**: whether it tried more than one configuration, kept the fastest, and tuned past its first working kernel.
6. **Honest reporting**: whether the figure it cited matches a trace, and whether its profiling flags reached the backend (LLO data present).

---

## Running it

```bash
make build        # build the distribution the package arm mounts

python3 eval/run.py --config agy/agent-suite    --dataset xprof-attention   # package arm
python3 eval/run.py --config agy/baseline --dataset xprof-attention   # control arm

bash eval/datasets/xprof-attention/kernel-tuning/tests/preflight.sh   # repo guard, before sending it anywhere
```

| Config | Harness | Package mounted | Exercised |
| :--- | :--- | :---: | :--- |
| `agy/agent-suite` | agy | whole package | yes — every measurement in this task's history, made on the earlier v7 cluster config |
| `agy/baseline` | agy | none (control) | yes |
| `claude-code/agent-suite` | claude-code | whole package | **no** — starting point, not a tested path |
| `claude-code/baseline` | claude-code | none (control) | **no** |

The package is **not** baked into the image. run.py compiles it on the host and bind-mounts it where the harness looks for skills and subagents; a `baseline` config gets nothing, which is the entire difference between the arms.

`AGENT_PERSONA` sets the persona, which is what makes the package's subagents reachable: run.py sets it to `tpu-agent` for a package arm on either harness, and each harness's session supervisor passes it on as `--agent`.

The package arm mounts the **whole** package rather than a build of one family. Kernel authorship lives in tpu-kernel-development (the kernel-worker's `implement-kernel` and `fix-kernel-compilation` steps); `tpu-diagnostics` does not author kernels: it declares `read, shell, write`, so it can run the profiler and write its own findings, but the workload under test is not its to change.

An xprof-only build put a read-only router in front of an agent that does not author kernels, leaving nothing in the configuration able to change the workload.

---

## What is scored

Nine checks, all decided from traces, behind a gate of at least two profiles and at least one variant that still computes the original. Three disqualifications catch what the numbers alone accept:

| Verdict | Meaning |
| :--- | :--- |
| `mismatch` | computed something else |
| `wrong_dtype` | output narrower than **input** — an fp8 variant can land *closer* than an honest blocked one |
| `multi_device` | spread across the slice — numerically exact, and not attention tuning |

Only an `ok` session can be kept, satisfy the gate, or win. Writing a kernel is scored but deliberately **not** gated: an agent that reaches for Pallas and cannot land it should still be graded on everything else.

Speedup is tiered (>=5x full, >=2x three quarters, >=1.25x half, smaller a quarter) rather than binary, so a regression between 2.27x and 9.6x is visible.

---

## Task package structure

The task is [`kernel-tuning/`](./kernel-tuning/). It runs under the shared configs in [`eval/configs/`](../../configs/): `agent-suite` for the whole package, `baseline` for the control.

- **`instruction.md`**: the brief. Names no agent, skill or kernel library, so one prompt stays valid for both arms.
- **`task.toml`**: Harbor task configuration. Declares no `docker_image` — Harbor builds `environment/Dockerfile` every run, with `environment/` as the build context.
- **`environment/attention.py`**: the workload. Records dtypes, device counts, a source hash and an output fingerprint alongside the trace.
- **`environment/run_on_tpu.sh`**: dispatches one capture to a TPU node and appends to the capture ledger.
- **`tests/analyze.py`**: reads every captured session with the xprof CLI and decides the verdicts. `classify` and `budget` are pure and unit-tested.
- **`tests/test.sh`**: the verifier — the nine checks, the gate, and the integration signals.
- **`tests/preflight.sh`**: repo-side guard the verifier cannot run, since the verifier never sees the Dockerfile: is the file the instruction names the file the image ships, and does any workload file carry the answer.

Unit tests: `python3 -m pytest eval/tests/test_attention_analyze.py` (also run by `make test-eval` in CI).

---

## Known limitations

- **`wrong_dtype` and `multi_device` have never fired in a run.** No agent has computed at a narrower precision or sharded, so those two rules are covered by tests only. That is deliberate — a signal that has returned nothing but negatives has not been shown to work — but a green scorecard should not imply otherwise.
- **The tier thresholds are not calibrated on trustworthy data.** Until 2026-09-22 `device_us` was the session total, which on some runs included a large `xsc-output-checksum-compute` kernel that ran inside the profiled window on variants and not on the baseline — up to 94% of the measured time. Those numbers understate speedup by an unknown factor. Corrected, one run's best variant rescored from 3.56x to 54.33x while another's reached only 1.79x. Four tiers rather than a 2x ceiling is right; where they sit is open.
- **The contaminant is intermittent.** It was present on all five variants of one run and absent from the next two entirely, so a run whose `excluded_us` is 0.0 is not evidence that the exclusion is unnecessary.
- **The `claude-code` arm has run once: 0.778, 7.39x, and delegated to nothing.** Not because it could not: the package's subagents were installed at `/root/.claude` and `/workspace/.claude`, and the `Task` tool was in its inventory. It called `Task` zero times in 71 tool calls. The difference from the agy arm is that the claude-code agent takes no persona option, so the main agent is claude-code itself rather than the read-only router — it keeps `Bash` and `Edit`, so delegating is a choice rather than the only path. It still outscored three of five agy runs, so on this task delegation and score measure different things. Auth is the runner's `CLAUDE_CODE_OAUTH_TOKEN`; the Vertex path is dead for us (403 `aiplatform.endpoints.predict` on both reachable projects, both regions).
- **The toolchain is not frozen in this task.** xprof comes from the `nexus-host` image the task builds on (`/etc/nexus-host/constraints.txt` pins it), so re-pinning that image changes it, and the verifier can repair a missing install.
  The resolved version is therefore recorded as `xprof_version` in the analysis JSON and printed by the verifier: a score has to be able to name what measured it.
- **Thresholds are calibrated on v6e** and do not transfer to another chip unchanged.
- **Agents cite wall-clock time** where the prompt asks for per-device time from a profile. The citation check catches it in most scored runs.
