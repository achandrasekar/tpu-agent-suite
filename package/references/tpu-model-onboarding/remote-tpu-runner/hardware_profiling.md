# Accelerator Hardware Profiling

Verify availability of specialized accelerators (TPU, GPU, System RAM) on the
machine.

## Instructions

1. Execute accelerator inspection via `{{runtime}} compute inspect`:

   ```bash
   {{runtime}} compute inspect --json
   ```

   This reads the target cluster's node labels or the VM itself and returns
   chip allocations, accelerator topology, and memory capacity. Don't query
   the cluster or VM yourself.

2. Parse the JSON output payload:
   - **TPU**: Check `chips`, `accelerator`, and `status`. `hbm_gb` is
   memory per **chip**, not per device.
   - **Devices, for tensor parallelism**: TP is sized to devices, not chips,
   and some chips are more than one device -- a tpu7x chip is two devices of
   96 GB each, so a 2x2x1 slice is 4 chips but 8 devices. Use
   `details.job_devices` and `details.hbm_gb_per_device` on every backend: on
   `gke_job`, where `chips` counts the whole node pool, they are what one Job
   gets. If they are missing, `details.hbm_gb_note` says why; do not assume
   one device per chip. If it says a package is not installed, ask the user to
   run the install command it gives, then run `inspect` again.
   - **GPU**: Check device capacity if available.
   - **System RAM**: Check for system memory baseline.
