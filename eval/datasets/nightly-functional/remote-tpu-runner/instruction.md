You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

Profile the TPU accelerator hardware of the GKE TPU cluster configured for this package using the `remote_tpu_runner` reference guide and tools.

Route the request by delegating to `tpu-model-onboarding` to inspect the cluster hardware and report the hardware profile it returns.

Important execution instructions:
- Delegate to `tpu-model-onboarding` to inspect the remote TPU cluster.
- Forward the delegation request directly and concisely without adding unrequested tasks or expanding requirements.
- Profile the TPU hardware on the remote GKE cluster rather than the local agent container.
- Report the hardware profile and cluster health returned by the cluster.
- Do not run shell `sleep` commands while waiting on subagents.
