Plan the torch_tpu inference benchmark for meta-llama/Meta-Llama-3-8B in bf16 on a TPU v6e-8 slice.

The plan should size the largest batch that fits at 4096 tokens of context from the chip's HBM capacity, and say where each hardware figure comes from.

Planning only. There is no TPU or cluster in this session, so nothing can be launched: stop once the plan is written and return it. Answer clearly, citing any references you can for grounding. Drop a JSON at `/logs/artifacts/answer.json` with keys `answer` and `reference`, for scoring.
