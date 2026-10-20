# Connectivity Verification

Before executing diagnostics, perform a connectivity check to ensure the
target compute (local, GCE TPU VM, or GKE) is operational.

## Instructions

Connectivity and accelerator health are checked via `{{runtime}} compute inspect`:

1. Run the inspection:

   ```bash
   {{runtime}} compute inspect --json
   ```

2. If the command fails or reports status not healthy, do not proceed. Report the
   connectivity blocker and exit.
