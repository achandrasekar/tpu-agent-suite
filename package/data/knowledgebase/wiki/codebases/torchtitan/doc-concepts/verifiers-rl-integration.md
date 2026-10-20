---
title: "Verifiers integration: running TitanRL rollouts in an external Verifiers environment"
type: doc-concept
provenance: doc
source: torchtitan/experiments/rl/examples/verifiers/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Verifiers integration: running TitanRL rollouts in an external Verifiers environment

## Definition
An optional bridge between TitanRL ([rl-post-training](rl-post-training.md)) and the Verifiers
library (PrimeIntellect). No other TitanRL recipe imports or requires Verifiers. The README draws an
explicit ownership boundary:
- **TitanRL** loads samples, schedules rollout groups, routes generation, tracks policy versions,
  computes advantages and trains the policy.
- **Verifiers** runs the environment, the agent or tool loop, the runtime, and the task reward.
- **The adapter** converts samples and traces between the two.

Reusable bridge code lives in `examples/verifiers/components/`; each experiment is its own package
with its own dependency list. The first one, `verifiers/dapo_math`, reruns the
[DAPO Math](dapo-math-rl-recipe.md) workload unchanged except for the rollout path. The integration
pins Verifiers 0.3.1 and its `verifiers.v1` API.

## In torchtitan (grounded)
- **Rollouter.** [`VerifiersRollouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/rollouter.py#L69 "torchtitan/experiments/rl/examples/verifiers/components/rollouter#VerifiersRollouter")
  subclasses `Rollouter` but bypasses TitanRL's `RolloutWorkerActor`, `RolloutWorker`, `MessageEnv`
  and `TokenEnv`: its `worker` field is hidden and unused, and `worker_pool_size` /
  `num_threads_per_worker` are suppressed because the Verifiers server owns its own worker pool.
  [`run_group_rollouts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/rollouter.py#L228 "torchtitan/experiments/rl/examples/verifiers/components/rollouter#VerifiersRollouter.run_group_rollouts")
  sends each sample to Verifiers, and
  [`trace_to_rollout_turns`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/rollouter.py#L339 "torchtitan/experiments/rl/examples/verifiers/components/rollouter#VerifiersRollouter.trace_to_rollout_turns")
  flattens the returned trace graph into TitanRL
  [`RolloutTurn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/types.py#L83 "torchtitan/experiments/rl/rollout/types#RolloutTurn") objects: each run of trainable
  (sampled) tokens becomes one turn whose prompt is everything before it, and shared sampled graph
  nodes are emitted once. TitanRL's rubric and advantage estimator then run normally; the reward
  Verifiers computed reaches the rubric through
  [`RewardFromVerifiers`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/rollouter.py#L54 "torchtitan/experiments/rl/examples/verifiers/components/rollouter#RewardFromVerifiers").
- **Single source of the taskset.** The training dataset's taskset is the only place the environment
  taskset is configured.
  [`VerifiersRollouter.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/rollouter.py#L126 "torchtitan/experiments/rl/examples/verifiers/components/rollouter#VerifiersRollouter.Config.__post_init__")
  copies `train_dataset.verifiers_taskset` into the environment-server config and raises if
  `verifiers_env_server.environment.taskset` was set to anything different.
- **Data (step 1 of the flow).** [`VerifiersTaskDataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/data.py#L44 "torchtitan/experiments/rl/examples/verifiers/components/data#VerifiersTaskDataset")
  loads the Verifiers taskset inside the TitanRL controller, where TitanRL samples each task. It
  ships the task's typed data as a plain dict
  ([`VerifiersTaskSample`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/data.py#L38 "torchtitan/experiments/rl/examples/verifiers/components/data#VerifiersTaskSample")),
  because the Verifiers env client sends it as msgpack over ZMQ to the separate server process,
  which rebuilds the typed task.
- **Environment server (step 2).** [`VerifiersEnvServer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/env_server.py#L68 "torchtitan/experiments/rl/examples/verifiers/components/env_server#VerifiersEnvServer")
  spawns and owns the Verifiers server as a separate local process on the controller host
  ([`start`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/env_server.py#L108 "torchtitan/experiments/rl/examples/verifiers/components/env_server#VerifiersEnvServer.start")).
  The agent runtime inside it is configured independently: `SubprocessConfig`, `DockerConfig` or
  `PrimeConfig`.
- **Generation server (step 3).** [`GenerationServer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/generation_server.py#L45 "torchtitan/experiments/rl/examples/verifiers/components/generation_server#GenerationServer")
  exposes TitanRL's controller-provided
  [`GenerateFn`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/types.py#L29 "torchtitan/experiments/rl/rollout/types#GenerateFn") through the HTTP
  token-generation protocol Verifiers expects (a local aiohttp app, so token-in/token-out is
  preserved). It keeps what Verifiers traces lack:
  [`VerifiersGenerationMetadata`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/components/generation_server.py#L27 "torchtitan/experiments/rl/examples/verifiers/components/generation_server#VerifiersGenerationMetadata")
  accumulates a conservative min/max policy-version span and generator metrics per rollout, which
  are stamped on every emitted turn (metrics only once, to avoid double counting).
- **DAPO Math example.** [`VerifiersMathRollouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/dapo_math/rollouter.py#L37 "torchtitan/experiments/rl/examples/verifiers/dapo_math/rollouter#VerifiersMathRollouter")
  uses Verifiers' null harness, a local `SubprocessConfig` runtime, `max_turns=1` and a static
  one-worker pool. Its taskset
  ([`VerifiersMathTaskset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/dapo_math/data.py#L45 "torchtitan/experiments/rl/examples/verifiers/dapo_math/data#VerifiersMathTaskset"))
  reuses the in-tree `DapoMathDataset` / `AIME2025Dataset` and scores with the same
  `score_math_response` as the native recipe. Recipes:
  [`rl_dapo_qwen3_4b_verifiers_8k`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/dapo_math/config_registry.py#L136 "torchtitan/experiments/rl/examples/verifiers/dapo_math/config_registry#rl_dapo_qwen3_4b_verifiers_8k")
  and
  [`rl_dapo_qwen3_4b_verifiers_32k`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/verifiers/dapo_math/config_registry.py#L145 "torchtitan/experiments/rl/examples/verifiers/dapo_math/config_registry#rl_dapo_qwen3_4b_verifiers_32k"),
  run with `--module verifiers.dapo_math` (registered in
  [`_supported_experiments`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/__init__.py#L7 "torchtitan/experiments/__init__#_supported_experiments")).

## Adding an experiment
One package per task under `examples/verifiers/` (`__init__.py`, `config_registry.py`, `data.py`,
`requirements.txt`, `rollouter.py`):
1. `data.py`: define the Verifiers `Taskset` and `TasksetConfig`.
2. `rollouter.py`: put that config in `VerifiersTaskDataset.Config.verifiers_taskset`; configure the
   agent, harness, runtime and worker pool, but leave `verifiers_env_server.environment.taskset`
   unset (derived from the training dataset).
3. `config_registry.py`: select the rollouter from `Controller.Config`.
4. Register the module in `torchtitan/experiments/__init__.py`, then run
   `python -m torchtitan.experiments.rl.train --module verifiers.my_task --config <name>`.

## Why it matters / when it applies
Use it to reuse Verifiers environments, agent harnesses and sandboxed runtimes without
reimplementing them as TitanRL `MessageEnv`s, while keeping TitanRL's scheduling, policy-version
tracking and training path. For CPU-heavy environments the README says to raise the Verifiers
worker-pool size first (the DAPO example's `StaticPoolConfig(num_workers=1)`). Placing the
environment server in a Monarch actor is a possible future mode; it would need routable ZMQ and
generation-server addresses and actor-owned startup/shutdown, which the current integration does
not provide. The DAPO example has no tools or sandbox, so it must not be used for untrusted code
execution.

**TPU applicability.** The bridge itself is CPU-side Python (a subprocess server, ZMQ and a local
HTTP server), so nothing in it is GPU-specific; it inherits the GPU-only trainer and vLLM generator
of the in-tree RL stack, and the TPU RL port ([tpu-rl-with-ray](tpu-rl-with-ray.md)) does not use it
at this pin.

## Connections
- Code concepts: [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md),
  [job-configuration](../concepts/job-configuration.md),
  [loss-and-validation](../concepts/loss-and-validation.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/rl/examples/verifiers/components/*`, `experiments/rl/examples/verifiers/dapo_math/*`).
- Related doc-concepts: [rl-post-training](rl-post-training.md),
  [dapo-math-rl-recipe](dapo-math-rl-recipe.md),
  [search-r1-tool-use-rl](search-r1-tool-use-rl.md) (the native multi-turn tool path).

## Source
Extracted from `torchtitan/experiments/rl/examples/verifiers/README.md`, with the example section
from `torchtitan/experiments/rl/examples/verifiers/dapo_math/README.md` (both kept in place).
