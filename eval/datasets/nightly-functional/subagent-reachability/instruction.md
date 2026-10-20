You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

This task checks that delegation works. Delegate exactly once, to `tpu-knowledge-base`, with this JSON request:

```json
{"query": "240-rule", "allow_crawl": false}
```

Wait for its result, then reply with what the 240-rule is in one sentence.

- Do not answer the question yourself, and do not delegate to any other agent.
- If the delegation fails, do not retry with a different agent and do not work around it: reply with the error message exactly as it was returned.
- Do not run shell `sleep` commands while waiting on the subagent.
