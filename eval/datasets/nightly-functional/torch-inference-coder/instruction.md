You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Synthesize a serving script `inference.py` for a Causal LM model using the `torch_inference_coder` reference guide and recipes.

Route the request by delegating to `tpu-model-onboarding` to adapt the server script for Causal LM serving using the reference recipe.

Important execution instructions:
- Delegate to `tpu-model-onboarding` to synthesize the serving script.
- Forward the delegation request directly and concisely without adding unrequested tasks or expanding requirements.
- Synthesize `inference.py` for Causal LM with TPU device initialization. Do not create deployments.json and do not launch the server.
- Do not run shell `sleep` commands while waiting on subagents.
