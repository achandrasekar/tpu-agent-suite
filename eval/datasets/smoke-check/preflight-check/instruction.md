You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Run a local preflight compatibility check on the model `Qwen/Qwen2.5-7B-Instruct` to verify whether its architecture and tokenizer are compatible with Cloud TPU.

Route the request by delegating to `tpu-model-onboarding` to run the verification and output a concise compatibility summary.

Important execution instructions:
- Delegate to `tpu-model-onboarding` to execute the preflight check.
- Forward the delegation request directly and concisely without adding unrequested tasks or expanding requirements (do not deploy or serve the model).
- Execute the existing `preflight_check.py` script once and report its verdict and findings. Do not write custom scripts to manually inspect model configs or calculate sharding dimensions.
- Do not run shell `sleep` commands while waiting on subagents.
