You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Run a vLLM pre-flight compatibility check on the model `Qwen/Qwen2.5-7B-Instruct` at tensor parallelism 1 on the GKE TPU cluster configured for this package using the `vllm-engine-compatibility` reference guide and scripts.

Route the request by delegating to `tpu-model-onboarding` to run the verification and output a concise compatibility summary.

Important execution instructions:
- Delegate to `tpu-model-onboarding` to execute the engine compatibility check.
- Forward the delegation request directly and concisely without adding unrequested tasks or expanding requirements.
- Run the compatibility check against the remote GKE TPU cluster. Do not install dependencies or modify the environment to force the check to succeed.
- Report the check's actual outcome and findings (including if vLLM is unavailable on the cluster).
- Do not run shell `sleep` commands while waiting on subagents.
