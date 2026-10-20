---
title: "Batch-invariant mode and trainer/generator log-prob parity (RL)"
type: doc-concept
provenance: doc
source: torchtitan/experiments/rl/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Batch-invariant mode and trainer/generator log-prob parity (RL)

## Definition
Batch-invariant mode guarantees that a model's output for a given input is identical regardless
of what other inputs are in the batch. In RL this is what makes true on-policy training
checkable. The generator computes log-probs in one batch composition (say, 8 completions), and
the trainer recomputes them in another (say, 2 per DP shard). Without batch invariance,
floating-point accumulation order makes the two disagree. The TitanRL README now states the
guarantee as **bitwise-identical log-probs across different data-parallel layouts** for supported
configurations, and calls it a throughput-for-exact-numerics trade useful for debugging and
controlled on-policy studies. It points to the bitwise parity guide
(`torchtitan/experiments/rl/docs/bitwise_parity.md`) for configuration and limits: trainer and
generator must still use the same tensor-parallel degree, sequence parallelism is unsupported, and
models with randomness (dropout) also need a shared `seed`.

## In torchtitan (grounded)
- **Switch.** `debug.batch_invariant` on [`DebugConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/config/configs.py#L398 "torchtitan/config/configs#DebugConfig"), together
  with `deterministic=True`; [`Controller.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/controller.py#L345 "torchtitan/experiments/rl/controller#Controller.Config.__post_init__")
  rejects `batch_invariant` without `deterministic`, and rejects it on ROCm (the varlen path there
  cannot force `num_splits=1`). The alphabet_sort recipe
  [`rl_grpo_qwen3_0_6b_varlen_batch_invariant`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py#L886 "torchtitan/experiments/rl/examples/alphabet_sort/config_registry#rl_grpo_qwen3_0_6b_varlen_batch_invariant")
  uses `DebugConfig(batch_invariant=True, deterministic=True)`; sibling variants cover Flex
  attention, GPT-OSS, a Qwen3 MoE debug model and Qwen3.5.
- **Enable path.** [`set_batch_invariance`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/utils.py#L340 "torchtitan/distributed/utils#set_batch_invariance") is called by
  [`PolicyTrainer.__init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/trainer.py#L95 "torchtitan/experiments/rl/actors/trainer#PolicyTrainer.__init__"),
  [`VLLMGenerator.__init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/actors/generator.py#L804 "torchtitan/experiments/rl/actors/generator#VLLMGenerator.__init__") and the
  standalone [`generate`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/generate.py#L77 "torchtitan/experiments/rl/generate#generate") script.
  It enables the `batch_invariant_ops` Triton overrides for `mm`, `addmm`, `_log_softmax` and
  `mean.dim`, sets NCCL env vars (single channel, `Simple` protocol, `allreduce:tree`, one
  thread, no SHARP/NVLS), and turns off bf16/fp16 reduced-precision reductions and TF32.
  [`is_in_batch_invariant_mode`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/distributed/utils.py#L313 "torchtitan/distributed/utils#is_in_batch_invariant_mode") is the global flag that the model
  code reads.
- **Model-side hooks.** [`VarlenInnerAttention.forward`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/attention.py#L165 "torchtitan/models/common/attention#VarlenInnerAttention.forward")
  forces `num_splits=1` for FA3 varlen in this mode (it forces it for FA2 unconditionally, as a
  paged-KV NaN workaround).
  [`BatchInvariantFlexConverter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/batch_invariance.py#L25 "torchtitan/experiments/rl/batch_invariance#BatchInvariantFlexConverter") pins
  `BACKEND=TRITON`, `BLOCK_M=BLOCK_N=16` on every
  [`FlexInnerAttention`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/common/attention.py#L233 "torchtitan/models/common/attention#FlexInnerAttention") layer to match vLLM.
  [`force_logprobs_fn_for_batch_invariance`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/rl/batch_invariance.py#L57 "torchtitan/experiments/rl/batch_invariance#force_logprobs_fn_for_batch_invariance") replaces
  vLLM's fused Triton log-prob kernel with TorchTitan's `compute_logprobs`, so both sides
  compute log-probs with the same ops. The separate vLLM batch-invariant `bmm` patch for the MoE
  router (`patch_bmm_for_batch_invariance`) was removed at this pin.
- **Precision contract.** The trainer keeps fp32 master weights. FSDP mixed precision
  (`training.mixed_precision_param="bfloat16"`, the default) casts them to bf16 for the forward,
  which matches the generator's bf16 forward bit for bit. The trainer always wraps the model in
  FSDP, so this cast happens even at `data_parallel_shard_degree=1`.
- **Guard.** Pre-training refuses the flag:
  [`Trainer.Config.__post_init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/trainer.py#L132 "torchtitan/trainer#Trainer.Config.__post_init__") raises
  "Batch-invariant mode is not supported in pre-training".

## Why it matters / when it applies
Turn it on when debugging generator/trainer log-prob mismatches or before claiming on-policy
correctness. It costs speed (Triton fixed-tile GEMMs, single-channel NCCL), so it is a debugging
and verification mode, not a production default. The README's background reading is the
train/inference mismatch write-up for asynchronous RL and "Defeating Nondeterminism in LLM
Inference".

**TPU applicability.** Everything in the mechanism is CUDA-specific: Triton kernels, NCCL env
vars, and `torch.backends.cuda` flags.
Nothing in `experiments/tpu` (including `rl_nc_ray`) provides a TPU equivalent. On the torch_tpu
lane ([torchtpu-backend](../concepts/torchtpu-backend.md)), determinism depends on XLA compiling
the same program for the same shapes. There is no in-repo tool here for batch-composition
invariance on TPU.

## Connections
- Code concepts: [fsdp-sharding](../concepts/fsdp-sharding.md) (mixed-precision boundary),
  [loss-and-validation](../concepts/loss-and-validation.md),
  [model-definitions](../concepts/model-definitions.md),
  [job-configuration](../concepts/job-configuration.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`torchtitan/distributed/utils`, `experiments/rl/batch_invariance`).
- Related doc-concepts: [rl-post-training](rl-post-training.md),
  [testing-strategy](testing-strategy.md) (bitwise loss comparison tooling),
  [tpu-rl-with-ray](tpu-rl-with-ray.md).

## Source
Extracted from `torchtitan/experiments/rl/README.md` (kept in place). The README's
"Trainer/generator consistency" section now defers mechanism details to
`torchtitan/experiments/rl/docs/bitwise_parity.md`, which this page also draws on.
