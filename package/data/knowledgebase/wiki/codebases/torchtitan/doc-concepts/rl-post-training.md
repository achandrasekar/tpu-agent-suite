---
title: "RL post-training (TitanRL): TorchTitan trainer + vLLM generator with a shared model definition"
type: doc-concept
provenance: doc
source: torchtitan/experiments/rl/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# RL post-training (TitanRL): TorchTitan trainer + vLLM generator with a shared model definition

## Definition
The RL experiment, branded **TitanRL** in its README, runs GRPO/DAPO-style RL with TorchTitan as
the trainer and vLLM as the rollout generator. The README names three advantages:
1. **Unified model definition.** (Mostly) the same TorchTitan model runs inside the trainer and
   inside vLLM. Only the attention modules differ, and they call the same kernels. New models and
   kernels are written once, which removes a major source of training/inference mismatch bugs.
2. **Batch invariance.** In batch-invariant mode, supported configurations produce bitwise-identical
   log-probs across different data-parallel layouts. See [rl-batch-invariance](rl-batch-invariance.md).
3. **One stack from pretraining to post-training.** The same model definitions, kernels and training
   components serve both phases.

The README warns that unified-model performance varies: it can trail native vLLM on inference-only
workloads but beat it end to end in some RL configurations. Batch invariance trades throughput for
exact numerics.

The pipeline has three layers:
- **Experiment logic.** A `Rollouter` composes training/validation data, a `MessageEnv`, a rubric
  and a rollout function.
- **Controller and dataflow.** Independent loops load data, produce rollouts, pack batches and update
  the policy. A `RolloutGroupWorkBuffer` connects them and bounds policy lag.
- **Distributed execution.** A router sends requests to one or more vLLM generator replicas;
  `PolicyTrainer` runs on its own TorchTitan mesh; **TorchStore** publishes new weights back to the
  generators (including direct GPU-to-GPU RDMA); **Monarch** is the actor framework that places
  trainers and generators on separate GPU meshes with async calls. Training and generation scale
  independently.

The README's reference experiment is now DAPO Math on Qwen3-4B-Base
([dapo-math-rl-recipe](dapo-math-rl-recipe.md)), replacing the old `alphabet_sort` smoke test.

## In torchtitan (grounded)
- **Launcher.** `python -m torchtitan.experiments.rl.train` runs
  [`main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/train.py#L273 "torchtitan/experiments/rl/train#main"). It parses a
  [`Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L281 "torchtitan/experiments/rl/controller#Controller.Config"), spawns the trainer and generator
  process meshes ([`spawn_proc_mesh`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/train.py#L188 "torchtitan/experiments/rl/train#spawn_proc_mesh")), then runs
  [`setup_async`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L533 "torchtitan/experiments/rl/controller#Controller.setup_async") and
  [`run`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L762 "torchtitan/experiments/rl/controller#Controller.run"). `--module` / `--config` select a
  config-registry function; [`ConfigManager._load_config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/manager.py#L58 "torchtitan/config/manager#ConfigManager._load_config")
  resolves a short name listed in
  [`_supported_experiments`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/__init__.py#L7 "torchtitan/experiments/__init__#_supported_experiments") (`dapo_math`,
  `search_r1`, `verifiers.dapo_math`, ...) under `torchtitan.experiments.rl.examples`, and imports a
  fully qualified module path such as the README's `my_project.my_experiment` directly.
- **Controller loop.** [`Controller`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L260 "torchtitan/experiments/rl/controller#Controller") pipelines four
  asyncio loops: data input -> rollout workers -> batcher -> trainer. These are
  [`_data_input_loop`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L920 "torchtitan/experiments/rl/controller#Controller._data_input_loop"),
  [`_rollout_loop`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L949 "torchtitan/experiments/rl/controller#Controller._rollout_loop"),
  [`_batcher_loop`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L999 "torchtitan/experiments/rl/controller#Controller._batcher_loop") and
  [`_trainer_loop`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L1043 "torchtitan/experiments/rl/controller#Controller._trainer_loop"), which share a
  [`RolloutGroupWorkBuffer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/components/work_buffer.py#L54 "torchtitan/experiments/rl/components/work_buffer#RolloutGroupWorkBuffer").
  [`AsyncLoopConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L155 "torchtitan/experiments/rl/controller#AsyncLoopConfig") sets
  [`num_samples_per_prompt`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L163 "torchtitan/experiments/rl/controller#AsyncLoopConfig.num_samples_per_prompt")
  (GRPO siblings per prompt, default 8),
  [`num_prompts_per_train_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L159 "torchtitan/experiments/rl/controller#AsyncLoopConfig.num_prompts_per_train_step")
  (default 8) and
  [`target_offpolicy_steps`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L166 "torchtitan/experiments/rl/controller#AsyncLoopConfig.target_offpolicy_steps")
  (default 3), which sizes the active buffer to `(S + 1) * P` prompt groups.
  [`window_fraction`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L175 "torchtitan/experiments/rl/controller#AsyncLoopConfig.window_fraction") (default
  0.3, `None` = strict FIFO) lets the batcher skip an unfinished head group, so observed lag can
  exceed the target up to the derived
  [`max_offpolicy_steps`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L246 "torchtitan/experiments/rl/controller#AsyncLoopConfig.max_offpolicy_steps")
  property, `(B + W - 2) // P`.
- **Trainer actor.** [`PolicyTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L48 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer") is a
  Monarch actor built from TorchTitan components. It exposes
  [`forward_backward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L348 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer.forward_backward"),
  [`optim_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L431 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer.optim_step") and
  [`push_model_state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L491 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer.push_model_state_dict")
  (TorchStore). Losses: [`GRPOLoss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/losses/grpo.py#L17 "torchtitan/experiments/rl/losses/grpo#GRPOLoss") is a
  [`DAPOLoss`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/losses/dapo.py#L23 "torchtitan/experiments/rl/losses/dapo#DAPOLoss") with a symmetric clip (`clip_eps=0.2`);
  the DAPO Math recipe uses `DAPOLoss` directly with clip-higher bounds.
- **Generator actor.** [`VLLMGenerator`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/generator.py#L653 "torchtitan/experiments/rl/actors/generator#VLLMGenerator")
  drives a vLLM engine with continuous batching. The model is registered with vLLM by
  [`register_to_vllm`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/models/vllm_registry.py#L260 "torchtitan/experiments/rl/models/vllm_registry#register_to_vllm") and wrapped by
  [`VLLMModelWrapper`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/models/vllm_wrapper.py#L268 "torchtitan/experiments/rl/models/vllm_wrapper#VLLMModelWrapper"), which uses
  TorchTitan model args directly, swaps in vLLM paged attention, converts DTensors, and loads HF
  weights (`supports_pp = False`). Several replicas are configured with
  [`num_generators`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L327 "torchtitan/experiments/rl/controller#Controller.Config.num_generators") and fronted
  by [`InterGeneratorRouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/routing/inter_generator_router.py#L58 "torchtitan/experiments/rl/routing/inter_generator_router#InterGeneratorRouter")
  ([`generator_router`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L335 "torchtitan/experiments/rl/controller#Controller.Config.generator_router")).
- **Weight sync.** [`WeightSyncManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/components/weight_sync.py#L30 "torchtitan/experiments/rl/components/weight_sync#WeightSyncManager")
  overlaps the trainer push and generator pull with the next training step, and releases
  buffer slots only after the pull so that "no sample will be born stale".
- **Rollouts and the "write an experiment" contract.** The README asks for four pieces: datasets, a
  `MessageEnv`, a rubric, and a `Rollouter` that composes them. In code,
  [`Rollouter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L43 "torchtitan/experiments/rl/rollout/rollouter#Rollouter") owns the datasets and a pool of
  CPU worker processes
  ([`worker_pool_size`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L100 "torchtitan/experiments/rl/rollout/rollouter#Rollouter.Config.worker_pool_size"),
  default 4, each hosting a
  [`RolloutWorkerActor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/rollout_worker.py#L22 "torchtitan/experiments/rl/actors/rollout_worker#RolloutWorkerActor")). The env
  and rubric live on
  [`RolloutWorker.Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L221 "torchtitan/experiments/rl/rollout/rollouter#RolloutWorker.Config")
  (`rubric`, `message_env`, `token_env`, `advantage`), which builds
  [`TokenEnv`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/environment/token.py#L52 "torchtitan/experiments/rl/environment/token#TokenEnv") /
  [`MessageEnv`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/environment/message.py#L51 "torchtitan/experiments/rl/environment/message#MessageEnv") instances per group.
  Coarse customization overrides
  [`Rollouter.run_group_rollouts`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L178 "torchtitan/experiments/rl/rollout/rollouter#Rollouter.run_group_rollouts");
  fine customization subclasses the worker
  ([`make_env_group`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L255 "torchtitan/experiments/rl/rollout/rollouter#RolloutWorker.make_env_group"),
  [`score_group`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L278 "torchtitan/experiments/rl/rollout/rollouter#RolloutWorker.score_group"),
  [`_run_single_rollout`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout/rollouter.py#L370 "torchtitan/experiments/rl/rollout/rollouter#RolloutWorker._run_single_rollout")).
  Examples: [dapo-math-rl-recipe](dapo-math-rl-recipe.md) (single-turn),
  [search-r1-tool-use-rl](search-r1-tool-use-rl.md) (multi-turn tools),
  [verifiers-rl-integration](verifiers-rl-integration.md) (external rollout runtime), and the
  older `alphabet_sort` recipes such as
  [`rl_grpo_qwen3_0_6b_varlen`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py#L86 "torchtitan/experiments/rl/examples/alphabet_sort/config_registry#rl_grpo_qwen3_0_6b_varlen").
- **Chat rendering.** [`Controller.Config.renderer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L306 "torchtitan/experiments/rl/controller#Controller.Config.renderer")
  takes a [`RendererConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/renderer.py#L21 "torchtitan/experiments/rl/renderer#RendererConfig"). The README's
  example uses [`RenderersLibraryConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/renderer.py#L33 "torchtitan/experiments/rl/renderer#RenderersLibraryConfig"),
  which wraps a typed config from the `renderers` library (for example
  `Qwen3RendererConfig(enable_thinking=False)`) and builds it on TorchTitan's own tokenizer through
  [`RendererTokenizerWrapper`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/renderer.py#L78 "torchtitan/experiments/rl/renderer#RendererTokenizerWrapper") (no second
  HF tokenizer is loaded). It rejects the library's `auto` and `default` renderers. In-tree renderers
  such as [`MuseGlimmerRendererConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/models/muse_glimmer/renderer.py#L72 "torchtitan/experiments/rl/models/muse_glimmer/renderer#MuseGlimmerRendererConfig")
  subclass the same base.
- **Flash Attention selection.** The README's install step says TorchTitan picks FA4 on Blackwell,
  FA3 on Hopper and the bundled FA2 on older GPUs. The choice comes from
  [`get_cuda_flash_attention_impl`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/tools/utils.py#L38 "torchtitan/tools/utils#get_cuda_flash_attention_impl").

## Observability
The README lists four complementary views of a run:
- **System timelines** from the structured logger's per-rank JSONL and its Gantt generator
  ([structured-trace-logging](structured-trace-logging.md)).
- **Training curves**: typed metrics reduced once per step ([rl-metrics](rl-metrics.md)).
- **Inference engine health**: vLLM engine metrics over OpenTelemetry, to JSONL or OTLP
  ([rl-metrics](rl-metrics.md#vllm-engine-metrics-opentelemetry)).
- **Rollout inspection**: [`RolloutSampleRecorder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/rollout_recorder.py#L65 "torchtitan/experiments/rl/rollout_recorder#RolloutSampleRecorder")
  writes selected train/validation rollouts to `rollout_samples.jsonl`
  ([`Controller.Config.rollout_recorder`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L310 "torchtitan/experiments/rl/controller#Controller.Config.rollout_recorder")).

The README carries a TODO to add `Profiler.Config` support to `PolicyTrainer`. At this pin
[`PolicyTrainer.Config.profiler`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L92 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer.Config.profiler")
(and the matching generator field) is declared, but nothing in `experiments/rl` reads it, so the
TODO still stands.

## Monarch: every actor endpoint is `@concurrent_endpoint`
The README makes this a rule: every endpoint on a TitanRL Monarch actor (`PolicyTrainer`,
`VLLMGenerator`, the rollout worker, the inter-generator router) is declared with
`@concurrent_endpoint`, not `@endpoint`. A plain `@endpoint` processes the actor's message queue
sequentially and blocks the next message. `@concurrent_endpoint` runs each message in its own asyncio
task. Mixing the two is a trap, because a plain `@endpoint` blocks all other endpoints, the
concurrent ones included. A future plain `@endpoint` must justify itself in its docstring. At this
pin no file under `experiments/rl` (outside tests) uses a plain `@endpoint`; for example
[`RolloutWorkerActor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/rollout_worker.py#L22 "torchtitan/experiments/rl/actors/rollout_worker#RolloutWorkerActor") declares
all three of its endpoints concurrent.

## Why it matters / when it applies
Sharing one model definition removes a whole class of RL bugs, where the sampler's log-probs
silently differ from the learner's because the two stacks implement the model differently. The
async buffer with bounded staleness (`target_offpolicy_steps`, `window_fraction`) trades
on-policyness for throughput.

**TPU applicability.** The in-tree RL stack is GPU-only. Its install steps are CUDA 13 wheels,
Flash Attention 3/4 and Triton batch-invariant kernels, and TorchStore RDMA and Monarch assume
GPUs. The TPU port lives in `torchtitan/experiments/tpu/rl_nc_ray`. It is a *non-colocated*
setup driven by Ray: [`RayController`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/grpo.py#L123 "torchtitan/experiments/tpu/rl_nc_ray/grpo#RayController") runs
[`RayTPUPolicyTrainer`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/trainer.py#L120 "torchtitan/experiments/tpu/rl_nc_ray/trainer#RayTPUPolicyTrainer") (a `PolicyTrainer`
subclass that trains through torch_tpu) and
[`RayVLLMGenerator`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/generator.py#L194 "torchtitan/experiments/tpu/rl_nc_ray/generator#RayVLLMGenerator") (vLLM on TPU) on
separate TPU slices, so that the training state and the KV cache do not compete for HBM. Weights
move through the Ray object store, not TorchStore. Its package `__init__` *mocks* `monarch` and
`torchstore` in `sys.modules`
([`DummyActor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/rl_nc_ray/__init__.py#L40 "torchtitan/experiments/tpu/rl_nc_ray/__init__#DummyActor")) so the upstream actor classes
can be subclassed without those dependencies. See [tpu-rl-with-ray](tpu-rl-with-ray.md) and
[torchtpu-backend](../concepts/torchtpu-backend.md).

> [!warning] Doc vs code
> - The README's architecture section says "Set `max_offpolicy_steps=0` for synchronous execution".
>   `max_offpolicy_steps` is a read-only derived property of `AsyncLoopConfig`; the settable field
>   is `target_offpolicy_steps`. With the default `window_fraction=0.3`, `target_offpolicy_steps=0`
>   still allows a bound of `(B + W - 2) // P` (1 for `P=8`), so strict lockstep also needs
>   `window_fraction=None` *(agent inference from the property formula)*.
> - The README's "Write an experiment" snippet puts `message_env` and `rubric` on `Rollouter.Config`
>   and overrides `_run_single_rollout` on the `Rollouter`. In code those fields live on
>   `RolloutWorker.Config`, `Rollouter.Config` requires a `worker`, and `_run_single_rollout` is a
>   `RolloutWorker` method (the shipped `DapoMathWorker` / `SearchR1Worker` follow the code).

## Connections
- Code concepts: [training-loop](../concepts/training-loop.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [model-definitions](../concepts/model-definitions.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [job-configuration](../concepts/job-configuration.md),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [fsdp-sharding](../concepts/fsdp-sharding.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [state-dict-adapters](../concepts/state-dict-adapters.md),
  [metrics-and-observability](../concepts/metrics-and-observability.md),
  [profiling-and-tooling](../concepts/profiling-and-tooling.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`experiments/rl/controller`, `rollout/rollouter`, `actors/trainer`, `actors/generator`,
  `models/vllm_wrapper`, `renderer`, `experiments/tpu/rl_nc_ray/*`).
- Related doc-concepts: [rl-batch-invariance](rl-batch-invariance.md),
  [rl-metrics](rl-metrics.md), [dapo-math-rl-recipe](dapo-math-rl-recipe.md),
  [verifiers-rl-integration](verifiers-rl-integration.md),
  [search-r1-tool-use-rl](search-r1-tool-use-rl.md),
  [structured-trace-logging](structured-trace-logging.md),
  [tpu-rl-with-ray](tpu-rl-with-ray.md) and [tpu-rl-kuberay-deployment](tpu-rl-kuberay-deployment.md)
  (the TPU port in depth), [checkpoint-conversion](checkpoint-conversion.md).

## Source
Extracted from `torchtitan/experiments/rl/README.md` (kept in place).
