You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Synthesize a benchmark script `benchmark_tpu.py` for a Causal LM model using PyTorch/TPU using the `torch_benchmark_coder` reference guide and templates.

Route the request by delegating to `tpu-model-onboarding` to synthesize `benchmark_tpu.py` and update the `benchmark_model` placeholder using the Causal LM recipe.

Important execution instructions:
- Delegate to `tpu-model-onboarding` to synthesize the benchmark script.
- Forward the delegation request directly and concisely without adding unrequested tasks or expanding requirements.
- Synthesize `benchmark_tpu.py` for Causal LM with TPU device initialization. Do not execute or benchmark the script on hardware.
- Do not run shell `sleep` commands while waiting on subagents.
