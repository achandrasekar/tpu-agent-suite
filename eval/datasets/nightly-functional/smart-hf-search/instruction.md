You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Find the 5 most popular Hugging Face PyTorch models with under 8B parameters using the `smart_hf_search` reference guide and tool scripts.

Route the request by delegating to `tpu-model-onboarding` to search the Hugging Face hub, filter models with under 8 billion parameters, and report the top 5 model names with their parameter counts and download metrics.

Important execution instructions:
- Delegate to `tpu-model-onboarding` to execute the model search.
- Forward the delegation request directly and concisely without adding unrequested tasks or expanding requirements.
- Report the top 5 models with parameter counts and download metrics.
- Do not run shell `sleep` commands while waiting on subagents.
