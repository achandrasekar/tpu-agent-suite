---
title: "Checkpoint conversion HF <-> TorchTitan DCP, and how to verify it"
type: doc-concept
provenance: doc
source: scripts/checkpoint_conversion/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Checkpoint conversion HF <-> TorchTitan DCP, and how to verify it

## Definition
Converting a checkpoint between formats (HF safetensors <-> TorchTitan DCP) is correct only if
the converted weights give the same outputs when loaded into the target model definition. The
README gives two checks, in increasing strength:
1. **Greedy-decode sanity check.** Decode greedily from both checkpoints and compare. This is a
   fast "true negative" detector, not a proof. The README now points to the vLLM-based
   `torchtitan/experiments/rl/generate.py`, which is "largely model-agnostic" (pass
   `--temperature 0` for greedy decode; "tested on some but not all models"). It also needs the right HF `config.json`: "an
   incorrect `config.json` ... will result in incorrect generations despite a correct weight
   conversion".
2. **KL-divergence check.** Run forward on the DCP checkpoint in TorchTitan and on the safetensors
   in HF `AutoModelForCausalLM`, then compare the logit distributions. The Llama 3 example shows
   the check catching a subtle bug: omitting the RoPE Q/K permutation still decodes plausibly,
   but its KL rises from ~1e-13 to ~5e-6.

## In torchtitan (grounded)
- **Adapters do the mapping.** Each model's `ModelSpec.state_dict_adapter` (a
  [`StateDictAdapter`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/state_dict_adapter.py#L80 "torchtitan/protocols/state_dict_adapter#StateDictAdapter")) implements `to_hf` / `from_hf`.
  For Llama 3,
  [`Llama3StateDictAdapter.to_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/state_dict_adapter.py#L69 "torchtitan/models/llama3/state_dict_adapter#Llama3StateDictAdapter.to_hf")
  and [`from_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/state_dict_adapter.py#L113 "torchtitan/models/llama3/state_dict_adapter#Llama3StateDictAdapter.from_hf")
  rename keys and apply
  [`_permute`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/state_dict_adapter.py#L46 "torchtitan/models/llama3/state_dict_adapter#Llama3StateDictAdapter._permute") /
  [`_reverse_permute`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/models/llama3/state_dict_adapter.py#L58 "torchtitan/models/llama3/state_dict_adapter#Llama3StateDictAdapter._reverse_permute")
  to Q/K, the "permutation on several of the attention matrices" that the README describes.
  [`fqn_to_index_mapping`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/protocols/state_dict_adapter.py#L29 "torchtitan/protocols/state_dict_adapter#BaseStateDictAdapter.fqn_to_index_mapping") records
  which HF shard file each key belongs to.
- **Offline scripts** (`scripts/` is not in the symbol index, so they are described here in
  prose). Both import `torchtitan.models.<model_name>`, call its `model_registry(flavor)`, build
  the model *on CPU*, and wrap it in [`ModelWrapper`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/base.py#L93 "torchtitan/components/checkpointer/base#ModelWrapper") (now imported from the
  `torchtitan.components.checkpointer` package, which replaced the single `checkpoint.py`
  module) to get a correctly shaped state dict.
  - `convert_from_hf.py` (HF -> DCP): runs `to_hf` on the *empty* TorchTitan state dict to get
    HF-keyed buffers, `dcp.load`s safetensors into them through `HuggingFaceStorageReader`,
    runs `from_hf`, then `dcp.save`s a DCP checkpoint.
  - `convert_to_hf.py` (DCP -> HF): `dcp.load`s the DCP checkpoint, runs `to_hf`, optionally
    casts to `--export_dtype` (float16/bfloat16/float32), and saves through
    `HuggingFaceStorageWriter(save_distributed=True, fqn_to_index_mapping=...,
    enable_consolidation=True)`. It reads the index mapping from `--hf_assets_path`.
- **In-trainer equivalents.** The shared policy config
  (`BaseCheckpointManager.Config`) carries
  [`initial_load_in_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/base.py#L547 "torchtitan/components/checkpointer/base#BaseCheckpointManager.Config.initial_load_in_hf"),
  which loads HF weights at step 0 through the same adapter (it now requires
  `initial_load_model_only`), and
  [`last_save_in_hf`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/base.py#L556 "torchtitan/components/checkpointer/base#BaseCheckpointManager.Config.last_save_in_hf")
  (with [`export_dtype`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/base.py#L559 "torchtitan/components/checkpointer/base#BaseCheckpointManager.Config.export_dtype")),
  which exports at the last step. Two backends implement the export: the default DCP
  [`CheckpointManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/dcp.py#L89 "torchtitan/components/checkpointer/dcp#CheckpointManager")
  via [`dcp_save`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/dcp.py#L252 "torchtitan/components/checkpointer/dcp#CheckpointManager.dcp_save"),
  and the optional
  [`TorchCheckpointingManager`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/torch_checkpointing.py#L209 "torchtitan/components/checkpointer/torch_checkpointing#TorchCheckpointingManager")
  via [`_save_last_step`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/checkpointer/torch_checkpointing.py#L456 "torchtitan/components/checkpointer/torch_checkpointing#TorchCheckpointingManager._save_last_step"),
  which consolidates with the `torch_checkpointing` library's
  `consolidate_hf_safetensors_checkpoint` in a pre-finalize callback.
- **Numerical tests.** `numerical_tests_example.py` holds the KL test for Llama 3.
  `numerical_tests_qwen3_5.py` adds top-k agreement and image inputs for Qwen 3.5, and
  `numerical_tests_qwen3_5_shard.py` checks logits across no-parallel, FSDP, FSDP+EP and
  FSDP+EP+TP. New at this pin, `numerical_tests_kimi.py` (Kimi-VL) and
  `numerical_tests_kimi_k3.py` (Kimi K3) run the released HF model and TorchTitan in one process
  on the same text+image prompt and compare last-token logits, covering preprocessing, vision
  encoder, projector and decoder; the Kimi-VL script calls `--dtype float32` "the correctness
  gate".

## Why it matters / when it applies
Onboarding almost always starts from an HF release and ends with an HF export, so this is the
validation gate for any new model or adapter change. Run the KL check whenever an adapter
permutes, fuses or splits tensors: greedy decode alone can miss layout bugs like the RoPE
permutation.

**Last-step HF export vs the offline script (the NameError finding).** In this fork, the DCP
backend `components/checkpointer/dcp.py` (formerly `components/checkpoint.py`) still comments out
the import of `consolidate_safetensors_files_on_every_rank` (a comment says it is unavailable in
the fork's build environment). `dcp_save` still calls it whenever `to_hf` is set and the adapter
has an `fqn_to_index_mapping`, which is the case for any multi-file HF release. So
`last_save_in_hf=True` on the default DCP backend with an indexed `hf_assets_path` will raise a
`NameError` at the end of training (see [checkpointing](../concepts/checkpointing.md)). The
`torch_checkpointing` backend does not use that function; it consolidates through its own
library, which needs the `torch_checkpointing` package installed. The
offline `convert_to_hf.py` does not take that path: it never calls the missing function and
instead lets `HuggingFaceStorageWriter` consolidate (`enable_consolidation=True`). The safe
recipe on this fork is therefore to save DCP during training, then run `convert_to_hf.py`. One
caveat is unverified here: upstream PyTorch implements the writer's consolidation with the same
private `_consolidate_hf_safetensors` module that the fork's comment says is missing, so in that
environment the offline route may also fail. Confirm it on your torch build.

**TPU notes.** The scripts are device-agnostic and run on CPU, so no TPU is needed to convert.
They only accept models that have `torchtitan.models.<name>.model_registry` (at this pin:
llama3, qwen3, qwen3_5, qwen3_6, qwen3_8, deepseek_v3, deepseek_v4, gpt_oss, flux, kimi_k2_7,
kimi_k3, muse_glimmer). TPU-lane registries under `experiments/tpu/*` reuse the
upstream adapters (for example, `Qwen3StateDictAdapter`), so a DCP checkpoint written by a TPU
run of an upstream model converts the same way. TPU recipes usually import HF weights in-trainer
instead. [`sft_llama3_8b_metamath`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/config_registry.py#L376 "torchtitan/experiments/tpu/llama3/config_registry#sft_llama3_8b_metamath") sets
`initial_load_in_hf=True` with `load_only=True`, which also suppresses the last-step save and so
avoids the export path entirely. See [torchtpu-backend](../concepts/torchtpu-backend.md).

> [!warning] Doc vs code
> - The README does not mention the Qwen 3.5 or Kimi numerical tests in this directory.
> - (Fixed at this pin: the README now names `numerical_tests_example.py` correctly and points
>   the greedy-decode check at `experiments/rl/generate.py` instead of the missing
>   `generation/test_generate.py`.)

## Connections
- Code concepts: [state-dict-adapters](../concepts/state-dict-adapters.md),
  [checkpointing](../concepts/checkpointing.md),
  [train-spec-and-model-protocol](../concepts/train-spec-and-model-protocol.md),
  [model-definitions](../concepts/model-definitions.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`torchtitan/components/checkpointer/*`, `torchtitan/protocols/state_dict_adapter`,
  `torchtitan/models/llama3/state_dict_adapter`).
- Related doc-concepts: [transformers-modeling-backend](transformers-modeling-backend.md)
  (a near-identity adapter), [testing-strategy](testing-strategy.md),
  [tpu-training-recipes](tpu-training-recipes.md), [rl-post-training](rl-post-training.md)
  (home of the `generate.py` used for the greedy check).

## Source
Extracted from `scripts/checkpoint_conversion/README.md` (kept in place).
