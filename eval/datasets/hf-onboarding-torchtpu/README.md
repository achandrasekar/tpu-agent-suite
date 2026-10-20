# Hugging Face Model Onboarding Evaluation (`hf-onboarding`)

A benchmark and evaluation suite for onboarding Hugging Face models to Cloud TPU using `torch_tpu`, Google's native PyTorch TPU backend.

Where single-op benchmarks measure isolated kernels, this dataset evaluates an end-to-end model onboarding pipeline:
1. **Architecture & Memory Roofline**: Pre-execution capacity and bandwidth analysis.
2. **Continuous Batching Execution**: Serving throughput, TTFT, and TPOT on Cloud TPU.
3. **Compilation Persistence**: Verifying and persisting XLA computation cache.
4. **Hardware Profiling**: Capturing xprof / XPlane execution traces on the TPU.
5. **Greedy Output Correctness**: Verifying numerical correctness without degradation.

---

## Environment-Driven Task Architecture

Rather than duplicating task directories and verifier scripts across models, the evaluation uses a single, parameterized task package in [`./baseline/`](./baseline/) driven entirely by environment variables:

- **`HF_MODEL_URL`**: Hugging Face model URL or model ID to onboard (defaults to `https://huggingface.co/Qwen/Qwen3-0.6B` if unset).
- **`HF_TOKEN`**: Hugging Face access token for gated models (e.g. Gemma, Llama).
- **`TASK_DOCKER`**: Pinned TPU container image deployed on the target GKE cluster.

### Running the evaluation

```bash
# 1. Run with the default baseline model (Qwen/Qwen3-0.6B)
./eval/run.py --config claude-code/agent-suite-shared-capacity --dataset hf-onboarding-torchtpu

# 2. Run with any open / ungated model by passing HF_MODEL_URL
HF_MODEL_URL="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B" \
./eval/run.py --config claude-code/agent-suite-shared-capacity --dataset hf-onboarding-torchtpu

# 3. Run with a gated model (requires HF_TOKEN)
HF_TOKEN="hf_..." \
HF_MODEL_URL="https://huggingface.co/meta-llama/Llama-3.2-1B" \
./eval/run.py --config claude-code/agent-suite-shared-capacity --dataset hf-onboarding-torchtpu
```

---

## Complete Model Catalog & Validated URLs (36 Models $\le 30\text{B}$)

All 36 models below are verified for Cloud TPU onboarding and placement analysis. The task itself is
model-agnostic — pass any of these URLs as `HF_MODEL_URL` to run it. The slice and TP columns are
placement recommendations for reference; the brief has the agent derive placement from the
checkpoint's own dimensions rather than take them as given.

### 1. Dense Decoder-Only LLMs (19 Models)

| Model | Hugging Face URL | Params | Gating | Target Slice | Rec. TP | Context |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-0.6B** | [https://huggingface.co/Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 590M | Open | v6e-1 | 1 | 32k |
| **SmolLM2-135M** | [https://huggingface.co/HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 135M | Open | v6e-1 | 1 | 8k |
| **SmolLM2-360M** | [https://huggingface.co/HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 360M | Open | v6e-1 | 1 | 8k |
| **gemma-3-270m** | [https://huggingface.co/google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | 270M | Gated (HF_TOKEN) | v6e-1 | 1 | 8k |
| **Qwen3-1.7B** | [https://huggingface.co/Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | Open | v6e-1 | 1 | 32k |
| **SmolLM2-1.7B** | [https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 1.7B | Open | v6e-1 | 1 | 8k |
| **Qwen3-4B** | [https://huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B | Open | v6e-1 | 1 | 32k |
| **Phi-3.5-mini-instruct** | [https://huggingface.co/microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.82B | Open | v6e-1 | 1 | 128k |
| **Llama-3.2-1B** | [https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1.23B | Gated (HF_TOKEN) | v6e-1 | 1 | 128k |
| **gemma-3-1b-pt** | [https://huggingface.co/google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt) | 1B | Gated (HF_TOKEN) | v6e-1 | 1 | 8k |
| **gemma-2-2b** | [https://huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) | 2.61B | Gated (HF_TOKEN) | v6e-1 | 1 | 8k |
| **Llama-3.2-3B** | [https://huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | 3.21B | Gated (HF_TOKEN) | v6e-1 | 1 | 128k |
| **Qwen3-8B** | [https://huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.2B | Open | v6e-1 | 1 | 32k |
| **Mistral-7B-Instruct-v0.3** | [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7.25B | Open | v6e-1 | 1 | 32k |
| **Llama-3.1-8B** | [https://huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 8.03B | Gated (HF_TOKEN) | v6e-1 | 1 | 128k |
| **gemma-2-9b** | [https://huggingface.co/google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) | 9.24B | Gated (HF_TOKEN) | v6e-1 | 1 | 8k |
| **Ministral-3-14B-Instruct-2512** | [https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) | 14B | Open | v7x-1 | 1 | 32k |
| **Phi-4** | [https://huggingface.co/microsoft/Phi-4](https://huggingface.co/microsoft/Phi-4) | 14.7B | Open | v7x-1 | 1 | 16k |
| **gemma-2-27b** | [https://huggingface.co/google/gemma-2-27b](https://huggingface.co/google/gemma-2-27b) | 27.2B | Gated (HF_TOKEN) | v7x-1 | 2 | 8k |

### 2. Reasoning & Chain-of-Thought LLMs (2 Models)

| Model | Hugging Face URL | Params | Gating | Target Slice | Rec. TP | Context |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-R1-Distill-Qwen-1.5B** | [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.78B | Open | v6e-1 | 1 | 32k |
| **DeepSeek-R1-Distill-Qwen-7B** | [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.61B | Open | v6e-1 | 1 | 32k |

### 3. Sparse Mixture-of-Experts (MoE) (4 Models)

| Model | Hugging Face URL | Total / Active Params | Gating | Target Slice | Rec. TP | Architecture |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **DeepSeek-V2-Lite** | [https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | 15.7B / 2.4B | Open | v6e-1 | 1 | 16B MoE with Multi-head Latent Attention (MLA) and 64 routed experts (2.4B active). |
| **Qwen3-Coder-30B-A3B-Instruct** | [https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 30.7B / 3.3B | Open | v7x-1 | 2 | 30B MoE with 3B active parameters. Tested in torch_tpu/examples/huggingface_transformers. |
| **gpt-oss-20b** | [https://huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 20.2B / 4B | Open | v7x-1 | 1 | 32 local experts with sliding window attention; preconfigured in torch_tpu. |
| **gemma-4-26B-A4B** | [https://huggingface.co/google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | 26B / 4B | Gated (HF_TOKEN) | v7x-1 | 1 | Google Gemma 4 26B MoE model with 4B active parameters; config in torch_tpu. |

### 4. Vision-Language Models (VLMs) (5 Models)

| Model | Hugging Face URL | Params | Gating | Target Slice | Visual Architecture |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Qwen3-VL-2B-Instruct** | [https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2.2B | Open | v6e-1 | Multimodal vision-language model with dynamic resolution ViT patch merge. |
| **paligemma2-3b-pt-224** | [https://huggingface.co/google/paligemma2-3b-pt-224](https://huggingface.co/google/paligemma2-3b-pt-224) | 3B | Gated (HF_TOKEN) | v6e-1 | Google PaliGemma 2 3B vision-language model based on Gemma 2. |
| **Qwen3-VL-8B-Instruct** | [https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8.4B | Open | v6e-1 | 16.8M downloads open vision-language model. Supported in torch_tpu. |
| **paligemma2-10b-pt-224** | [https://huggingface.co/google/paligemma2-10b-pt-224](https://huggingface.co/google/paligemma2-10b-pt-224) | 10B | Gated (HF_TOKEN) | v6e-1 | Google PaliGemma 2 10B multimodal vision-language model. |
| **gemma-3-12b-pt** | [https://huggingface.co/google/gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt) | 12B | Gated (HF_TOKEN) | v6e-1 | Google Gemma 3 12B multimodal model. |

### 5. Multimodal Diffusion & Video Generation (2 Models)

| Model | Hugging Face URL | Params | Gating | Target Slice | Pipeline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **stable-diffusion-xl-base-1.0** | [https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 3.5B | Open | v6e-1 | Stable Diffusion XL text-to-image pipeline. Existing diffusers test in torch_tpu. |
| **Wan2.2-TI2V-5B-Diffusers** | [https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | 5.2B | Open | v7x-1 | Flow-matching video diffusion transformer with native test in torch_tpu. |

### 6. Audio, Speech & Representation Encoders (3 Models)

| Model | Hugging Face URL | Params | Gating | Target Slice | Modality |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **bge-m3** | [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 567M | Open | v6e-1 | Top open multilingual embedding encoder (37M+ downloads). Validates encoder-only pooling. |
| **detr-resnet-50** | [https://huggingface.co/facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) | 41M | Open | v6e-1 | End-to-end transformer object detector from benchmark_registry.pbtxt. |
| **whisper-large-v3** | [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 1.54B | Open | v6e-1 | Industry standard speech-to-text encoder-decoder; tested in torch_tpu model configs. |

### 7. State-Space Models (SSM) & Attention Hybrids (1 Models)

| Model | Hugging Face URL | Total / Active Params | Gating | Target Slice | Rec. TP | Architecture |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** | [https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 30.1B / 3B | Open | v7x-1 | 2 | Interleaved Mamba SSM + Attention + MoE layers. In torch_tpu model configs. |

### 8. Frontier Models (6 Models, $> 30\text{B}$ except where noted)

These sit outside the $\le 30\text{B}$ catalog above and are recorded here so the URLs are pinned.
Parameter counts are read from each repo's safetensors index, and expert counts from its
`config.json`. The minimum chip
count is the bf16 weight footprint over TPU7x HBM (192 GB/chip) — a floor that ignores KV cache,
activations and sharding overhead, so treat it as a lower bound rather than a validated placement.
At 1M context the KV cache alone dominates for the GLM and DeepSeek entries.

| Model | Hugging Face URL | Params | Gating | bf16 Weights | Min TPU7x Chips | Context | Architecture |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen3.8-27B** | [https://huggingface.co/Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27.8B | Open | 55.6 GB | 1 | 256k | Dense multimodal (`qwen3_5`), 64 layers, 4 KV heads. Only entry here that fits the $\le 30\text{B}$ scope. |
| **gemma-4-31B-it** | [https://huggingface.co/google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31.3B | Open | 62.5 GB | 1 | 256k | Dense multimodal (`gemma4`), 60 layers, 16 KV heads. |
| **Qwen3.8-Flash-Next** | [https://huggingface.co/Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 180B | Open | 360 GB | 2 | 256k | MoE multimodal (`qwen4_exp`), 512 experts, 10 active/token, 48 layers. |
| **GLM-5.3-Flash** | [https://huggingface.co/zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 321B | Open | 643 GB | 4 | 1M | MoE multimodal (`glm5_next`), 288 experts, 8 active/token, 45 layers. |
| **GLM-5.2** | [https://huggingface.co/zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 753B | Open | 1,507 GB | 8 | 1M | MoE text-only (`glm_moe_dsa`), 256 experts, 8 active/token, 78 layers. |
| **DeepSeek-V4.1-Flash** | [https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 763B | Open | 1,526 GB | 8 | 1M | MoE multimodal (`deepseek_v41`), 384 experts, 6 active/token, 40 layers, 1 KV head. |

`gpt-oss-20b` was also requested and is already catalogued under
[Sparse Mixture-of-Experts](#3-sparse-mixture-of-experts-moe-4-models) above.

Every model above is ungated, so each runs from the environment with `HF_MODEL_URL` alone — no
`HF_TOKEN` required:

```bash
RUN="./eval/run.py --config claude-code/agent-suite-shared-capacity --dataset hf-onboarding-torchtpu"

# Fits a single TPU7x chip
HF_MODEL_URL="https://huggingface.co/Qwen/Qwen3.8-27B"            $RUN
HF_MODEL_URL="https://huggingface.co/google/gemma-4-31B-it"       $RUN

# Multi-chip: the task.toml topology must supply the chip count in the table above
HF_MODEL_URL="https://huggingface.co/Qwen/Qwen3.8-Flash-Next"     $RUN
HF_MODEL_URL="https://huggingface.co/zai-org/GLM-5.3-Flash"       $RUN
HF_MODEL_URL="https://huggingface.co/zai-org/GLM-5.2"             $RUN
HF_MODEL_URL="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash" $RUN

# gpt-oss-20b, already catalogued above
HF_MODEL_URL="https://huggingface.co/openai/gpt-oss-20b"          $RUN
```

The four multi-chip entries will not fit the single-chip slice the baseline `task.toml` pins today
(`v7x-1`); they need a topology matching the **Min TPU7x Chips** column before the run can succeed.

---

## Hardware Placement: Cloud TPU v6e vs v7x

| TPU Generation | HBM Capacity | HBM Bandwidth | Peak BF16 Compute | Optimal Model Size Range |
| :--- | :--- | :--- | :--- | :--- |
| **Cloud TPU v6e** (Trillium) | 32 GB | 1,638 GB/s | 918 TFLOPs | Models < 10B parameters, MoEs $\le 4\text{B}$ active |
| **Cloud TPU v7x** (Ironwood, `tpu7x`) | 192 GB | 7,380 GB/s | 2,307 TFLOPs | Models 10B–30B parameters, 30B MoEs |

---

## Task Package Structure

The dataset contains the single canonical task under [`./baseline/`](./baseline/):
- **`instruction.md`**: Dynamic onboarding brief directing the agent to parse `$HF_MODEL_URL`, inspect Hugging Face `config.json`, roofline v6e vs v7x, benchmark continuous batching on Cloud TPU, profile steady-state decode with xprof, and verify output correctness.
- **`task.toml`**: Harbor task configuration with pinned runtime image, timeouts, and default environment mapping.
- **`tests/test.sh`**: Dynamic multi-aspect verifier that resolves keywords from `$HF_MODEL_URL` / agent output and scores script validity, native TPU device execution, and profiling traces.
