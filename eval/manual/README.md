# Manual E2E Testing

Optional: For a full reset run `rm -rf ~/.tpu-agent-suite` (WARNING: deletes installed TPU agent package, compute & env).

Create and activate a venv: `python3 -m venv ~/.venv && source ~/.venv/bin/activate`

Edit `e2e_env.sh` (repo path, harness, task, and GKE settings), then run:

```bash
./1_install_workspace.sh
./2_setup_compute.sh
```

Skip `./2_setup_compute.sh` to set up compute interactively inside the harness.

Launch your harness in `$WORKSPACE_DIR`:
- **Terminal (in `tmux`)**: `source ./e2e_env.sh && cd "$WORKSPACE_DIR"`, then start your harness (e.g., `agy --agent tpu-agent`).
- **Antigravity UI**: Click **New Project**, select `$WORKSPACE_DIR`, and start a **+ New Conversation**.

Prompt: `Follow the instructions in @PROMPT.md end-to-end.`

Note: Manual runs do not run Harbor's verifier or cheating judge.
