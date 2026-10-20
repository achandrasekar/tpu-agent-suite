You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Onboard and verify inference for the model `HuggingFaceTB/SmolLM2-135M-Instruct` on TPU using the inference workflow (`torch-tpu-inference`) in verification mode (quick set up and tear down). Deploy the model with the `vllm` inference engine (not `custom_fastapi`). Proceed without checking in. Accept all suggested defaults and do not stop to confirm the plan.

Important execution instructions:
- Do not run shell `sleep` commands while waiting on subagents.
- Proceed without checking in; do not stop or wait for user confirmation before executing on TPU.
