---
title: "GraphTrainer autoresearch: agent-driven graph-pass optimization loop"
type: doc-concept
provenance: doc
source: torchtitan/experiments/graph_trainer/autoresearch/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# GraphTrainer autoresearch: agent-driven graph-pass optimization loop

## Definition
The autoresearch README describes "an autonomous experiment loop for graph-pass / kernel
optimization" in `graph_trainer`. An LLM agent edits exactly one file,
`torchtitan/experiments/graph_trainer/passes.py`, and benchmarks each change. It keeps a change
only if training-step time improves "**and** numerics stay bitwise-identical to the eager
reference". The loop runs until someone stops it. The human and the agent communicate only
through shared files: `autoresearch.md` (the operating manual with `[SETUP]` sections),
`ideas.md`, `experiment_log.md`, `learnings.md`, and a machine-readable `results.tsv`.

## In torchtitan (grounded)
- **What the agent edits.** The search space is the pass list that
  [`GraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/trainer.py#L142 "torchtitan/experiments/graph_trainer/trainer#GraphTrainer")
  applies to its traced step. `autoresearch.md` tells the agent to register new passes in
  [`construct_default_graph_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L420 "torchtitan/experiments/graph_trainer/passes#construct_default_graph_passes")
  or [`compile_time_passes`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/passes.py#L150 "torchtitan/experiments/graph_trainer/passes#compile_time_passes").
  The tracer, trainer and model code are read-only for the agent.
- **Benchmark.** `autoresearch/scripts/run_benchmark.sh` runs `graph_trainer.llama3` /
  [`graph_trainer_llama3_8b_c4_test`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/llama3/config_registry.py#L129 "torchtitan/experiments/graph_trainer/llama3/config_registry#graph_trainer_llama3_8b_c4_test")
  with `--compile.mode aot_fx_trace`, FSDP=4, TP=2, the `c4_test` dataset, 8192 tokens per DP
  rank (`--training.num_tokens_per_microbatch_per_dp_rank 8192`), seed 42 and 20 steps. Its
  header says "on 8xH100". Edit it for your own model and parallelism.
- **Numerics gate.** The manual's example check is the bitwise-vs-eager test
  [`TestLlama3BitwiseDeterministic`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py#L387 "torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic#TestLlama3BitwiseDeterministic")
  (`-k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager"`). The agent is told not to
  read that test's source.
- **Keep or revert.** A `keep` commits `passes.py` together with the tracking files. A `discard`
  or `crash` reverts only `passes.py` (`git checkout -- .../passes.py`) and still commits the
  tracking files. The best result so far is therefore always HEAD, and failed attempts stay in
  the log.

## Why it matters / when it applies
The README lists four design properties. The loop is asynchronous (humans and agent share only
files), auditable (the commit tree keeps every code state), safe (numerics is a hard gate) and
compounding (each keep builds on earlier ones). The overrides README makes the same point:
making implementations swappable by config is "a building block for automated hill-climbing".
The manual also forbids the agent from reading git history, so that it rediscovers known
optimizations on its own instead of copying earlier commit messages.

**On TPU.** As shipped, the harness does not apply to TPU. It edits the CUDA default pipeline
in `passes.py`, and its gate assumes Inductor/CUDA bitwise parity with eager.
[`TpuGraphTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer.py#L90 "torchtitan/experiments/tpu/graph_trainer/tpu_graph_trainer#TpuGraphTrainer")
builds its passes from a different file,
[`construct_tpu_pass_pipeline`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass.py#L647 "torchtitan/experiments/tpu/graph_trainer/tpu_backend_pass#construct_tpu_pass_pipeline")
in `experiments/tpu/graph_trainer/tpu_backend_pass.py`, so changes to `passes.py` never reach
the XLA-lowered TPU graph. Porting the loop to TPU means three changes: point the editable
file at the TPU pass module, swap the benchmark for a TPU recipe, and replace the gate with a
TPU-vs-reference numerics check.

## Connections
- Code concepts: [torch-compile-integration](../concepts/torch-compile-integration.md),
  [profiling-and-tooling](../concepts/profiling-and-tooling.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md),
  [training-loop](../concepts/training-loop.md).
- Module catalogs: [module map](../catalog/index.md) (`experiments/graph_trainer/passes`).
- Related doc-concepts: [graph-trainer](graph-trainer.md), [kernel-overrides](kernel-overrides.md).

## Source
Extracted from `torchtitan/experiments/graph_trainer/autoresearch/README.md` (kept in place).
