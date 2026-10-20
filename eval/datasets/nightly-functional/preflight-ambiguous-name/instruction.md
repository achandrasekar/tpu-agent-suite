You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Run tpu-model-onboarding's preflight compatibility check for the model the user calls `llama 3 8b`, and report whether it is compatible with Cloud TPU.

Important execution instructions:
- This exercises tpu-model-onboarding's own resolution of model names. Delegate the request to `tpu-model-onboarding` with the name `llama 3 8b` exactly as given; do not resolve it, or ask about it, yourself first.
- Only the preflight check: do not deploy, serve or benchmark the model.
- Do not run shell `sleep` commands while waiting on subagents.
