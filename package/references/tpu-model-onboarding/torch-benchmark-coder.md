---
scripts:
  - scripts/tpu-model-onboarding/benchmark_tpu.py
  - scripts/tpu-model-onboarding/run_benchmarks.py
references:
  - references/tpu-model-onboarding/torch-benchmark-coder/causal_lm.md
  - references/tpu-model-onboarding/torch-benchmark-coder/vlm.md
  - references/tpu-model-onboarding/torch-benchmark-coder/clip.md
  - references/tpu-model-onboarding/torch-benchmark-coder/whisper.md
  - references/tpu-model-onboarding/torch-benchmark-coder/wav2vec2.md
  - references/tpu-model-onboarding/torch-benchmark-coder/text_to_audio.md
  - references/tpu-model-onboarding/torch-benchmark-coder/timm.md
  - references/tpu-model-onboarding/torch-benchmark-coder/image_classification.md
  - references/tpu-model-onboarding/torch-benchmark-coder/seq2seq.md
  - references/tpu-model-onboarding/torch-benchmark-coder/embeddings.md
  - references/tpu-model-onboarding/torch-benchmark-coder/grounding_dino.md
  - references/tpu-model-onboarding/torch-benchmark-coder/vision_segmentation.md
  - references/tpu-model-onboarding/torch-benchmark-coder/diffusers.md
  - references/tpu-model-onboarding/torch-benchmark-coder/speechbrain.md
  - references/tpu-model-onboarding/torch-benchmark-coder/clap.md
  - references/tpu-model-onboarding/torch-benchmark-coder/fallback.md
---
# Torch Benchmark Coder Skill (Benchmark Generator)

This skill specifies how to generate or refactor benchmark execution scripts
tailored for running deep learning models on Google Cloud TPU architectures
using `torch_tpu`.

> [!IMPORTANT]
> `torch_xla` (and `pytorch_xla`) is **DEPRECATED**. Never generate code that
> imports or relies on `torch_xla`. Always use native PyTorch TPU bindings (`import torch`,
> `device = torch.device("tpu")`).

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`.
Return `invalid_input` if `run_id` is missing. Write only to `<work>`, never to CWD or package files.

## Preconditions & Inputs

You receive candidate model metadata from prior discovery steps (e.g.
`model_id`, `library`, and `pipeline_tag`).

The skill contains a template script `{{script:tpu-model-onboarding/benchmark_tpu.py}}` which already implements
all standard patches, helpers, and utility functions (e.g., `get_cache_dir()`,
`import_dependency()`, `run_benchmark_pass()`).

## Task

1.  Read the template script `{{script:tpu-model-onboarding/benchmark_tpu.py}}`.
2.  Synthesize `<work>/benchmark_tpu.py` by **replacing the `benchmark_model`
    placeholder function** in the template with the bespoke loading and
    execution logic matching the candidate model's type or pipeline_tag.
3.  Write the completed script to `<work>/benchmark_tpu.py`.
4.  DO NOT modify or overwrite the shipped package files. Shipped `run_benchmarks.py`
    is executed directly from `{{script:tpu-model-onboarding/run_benchmarks.py}}`
    via compute runtime, and your synthesized `<work>/benchmark_tpu.py` is passed to it
    via `--input <work>/benchmark_tpu.py:benchmark_tpu.py`.

--------------------------------------------------------------------------------

## Bespoke Loader Recipes (To replace `benchmark_model` placeholder)

Choose the recipe that matches the candidate model's tag/library and overwrite
`benchmark_model`:

-   **Causal LM (Standard LLMs - Llama, Gemma, Qwen, GPT, Mistral)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/causal_lm.md}}`.
-   **Vision-Language Models (VLM - LLaVA, SmolVLM, Qwen2.5-VL, InternVL)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/vlm.md}}`.
-   **CLIP (Multimodal Image-Text Alignment & Zero-Shot Classification)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/clip.md}}`.
-   **Whisper (Automatic Speech Recognition & Speech Seq2Seq)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/whisper.md}}`.
-   **Wav2Vec2 (Audio Feature Representation & CTC Speech Recognition)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/wav2vec2.md}}`.
-   **Text-to-Audio / TTS (SpeechT5, Bark, AudioLDM)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/text_to_audio.md}}`.
-   **Timm (Torch Image Models & ConvNet Backbones)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/timm.md}}`.
-   **Image Classification / ViT (Vision Transformers & Torchvision)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/image_classification.md}}`.
-   **Seq2Seq / Translation (T5, FLAN-T5, BART)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/seq2seq.md}}`.
-   **Embeddings & Masked LMs (BERT, DistilBERT, ModernBERT, RoBERTa)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/embeddings.md}}`.
-   **Object Detection & Document Structuring (DETR, Table Transformer,
    Grounding DINO)**: See `{{reference:tpu-model-onboarding/torch-benchmark-coder/grounding_dino.md}}`
-   **Vision Segmentation (SAM, SegFormer, Mask2Former)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/vision_segmentation.md}}`.
-   **Diffusers (Stable Diffusion, SDXL, Flux Image Generation)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/diffusers.md}}`.
-   **SpeechBrain & CLAP (Audio Separation & CLAP Embeddings)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/speechbrain.md}}` and
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/clap.md}}`.
-   **Generic Fallback Loader (Unsupported / Custom Architecture Heads)**: See
    `{{reference:tpu-model-onboarding/torch-benchmark-coder/fallback.md}}`.

--------------------------------------------------------------------------------

## Unwired Models

Models flagged `[STATUS: ONBOARDING_REQUIRED]` are unwired by tpu-unwiring
before you get here. You do not write patches. Each unwired model comes with a
`patches_path`, an importable module exposing `PATCHED_MODULES`,
`APPLIED_PLAYBOOKS` and `apply()`, already proven to run a forward pass on the
accelerator.

-   It is shipped to the target as `patches/<model_slug>.py`. In
    `<work>/benchmark_tpu.py`, import that module and call `apply()` before the
    model is constructed, only for the model it belongs to.
-   Never inline the patches into `benchmark_tpu.py`: the numerical parity check
    imports the same module in its own process.
