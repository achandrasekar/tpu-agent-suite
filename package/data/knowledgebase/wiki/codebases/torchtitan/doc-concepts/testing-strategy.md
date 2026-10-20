---
title: "Testing strategy: unit tests, Fake-PG and Real-PG integration matrix, numerics goldens, and TPU parity tests"
type: doc-concept
provenance: doc
source: tests/README.md
updated: 2026-10-15
generated: {by: wikify/0.3.4, at: 2026-09-25T01:07:30Z}
---
# Testing strategy: unit tests, Fake-PG and Real-PG integration matrix, numerics goldens, and TPU parity tests

## Definition
`tests/README.md` sets three test goals. **Unit tests** check one module at a time. They are split
by hardware need into `tests/unit_tests/cpu/` and `tests/unit_tests/gpu/`, and GPU tests that need
several devices carry the `multi_gpu` pytest marker. **Integration tests**
(`tests/integration_tests/`) check end-to-end composability. Each case is a real training launch of
a *full configuration* taken from `torchtitan_recipes/tests/` (one module per suite). The suites are
`features` (feature composability), `models` (width across architectures), `h100` and `b200`
(hardware-specific). **Numerics tests** give deterministic regression coverage: selected
integration cases compare their loss and grad-norm curves against golden files under
`tests/assets/losses/`. The CI design rests on Fake process groups (Fake PG): fast, broad coverage on
pull requests from a single physical GPU, with Real PG used where real collectives matter and for
the full post-merge suite.

## In torchtitan (grounded)
- **Test definition.** Each case is an [`OverrideDefinitions`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/__init__.py#L19 "tests/integration_tests/__init__#OverrideDefinitions"). New entries set
  [`configs`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/__init__.py#L43 "tests/integration_tests/__init__#OverrideDefinitions.configs"), a list of functions returning a `Trainer.Config` with one function per run (runs share a
  dump folder, which is how save-then-resume cases are written). The legacy
  [`override_args`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/__init__.py#L24 "tests/integration_tests/__init__#OverrideDefinitions.override_args") form (base config plus CLI overrides) is kept for the suites under `torchtitan/experiments`.
  The other fields are `test_name`, `ngpu` (default 4), `disabled`, `skip_rocm_test`, `timeout`,
  [`golden_numerics_path`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/__init__.py#L39 "tests/integration_tests/__init__#OverrideDefinitions.golden_numerics_path") and [`use_real_pg`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/__init__.py#L41 "tests/integration_tests/__init__#OverrideDefinitions.use_real_pg"). The suites come from [`build_features_test_list`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/features.py#L13 "tests/integration_tests/features#build_features_test_list"), [`build_model_tests_list`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/models.py#L13 "tests/integration_tests/models#build_model_tests_list"), [`build_h100_tests_list`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/h100.py#L13 "tests/integration_tests/h100#build_h100_tests_list") and [`build_b200_tests_list`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/b200.py#L12 "tests/integration_tests/b200#build_b200_tests_list"). A typical config is
  [`llama3_debugmodel_fsdp2_cp2`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan_recipes/tests/features.py#L408 "torchtitan_recipes/tests/features#llama3_debugmodel_fsdp2_cp2"), which sets FSDP 2 x CP 2 and applies the CP transform (see
  [model-config-transforms](model-config-transforms.md)).
- **Runner.** [`main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/run_tests.py#L472 "tests/integration_tests/run_tests#main") -> [`run_tests`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/run_tests.py#L373 "tests/integration_tests/run_tests#run_tests") -> [`run_single_test`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/run_tests.py#L189 "tests/integration_tests/run_tests#run_single_test"). `main` accepts a comma-separated `--test_suite` and rejects
  `--execution_mode fake_pg` for `h100`/`b200` and for `--test_scope real_pg_required`. [`_filter_tests`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/run_tests.py#L335 "tests/integration_tests/run_tests#_filter_tests") drops
  `use_real_pg` cases from a Fake-PG run, keeps only them under `real_pg_required`, and applies
  `--test_name`, `--exclude`, `disabled`, ROCm skips and the GPU-count check. For each run,
  `run_single_test` exports the config function's module and name as `MODULE`/`CONFIG` to
  `./run_train.sh`. Under Fake PG it sets `COMM_MODE=fake_backend` and first calls
  [`validate_fake_pg_compatibility`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/__init__.py#L65 "tests/integration_tests/__init__#validate_fake_pg_compatibility"), which raises unless `use_real_pg=True` when a config enables checkpointing (including
  seed-checkpoint creation), pipeline parallelism or a named known-bad combination.
  `run_train.sh` in fake-backend mode runs a single process with `--comm.mode=fake_backend
  --training.steps 1`. [`GPUPool`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/run_tests.py#L54 "tests/integration_tests/run_tests#GPUPool") packs tests largest-first onto `--ngpu` physical GPUs. A Fake-PG test
  takes one physical GPU while keeping its `ngpu` as the simulated world size. `--no-parallel` runs
  tests one at a time.
- **Numerics path.** A case with `golden_numerics_path` must define exactly one config and does not
  call `run_train.sh` directly. The runner calls `scripts/loss_compare.py` with the same config as
  baseline and test, `--import-result=<golden>` and `--assert-equal`. Under Fake PG it adds
  `--no-seed-checkpoint`. A `{execution_mode}` placeholder in the path selects
  `tests/assets/losses/fake_pg/` or `real_pg/`. [`_read_golden_spec`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/run_tests.py#L137 "tests/integration_tests/run_tests#_read_golden_spec") derives the step count and metric columns
  from the golden file. `--export-numerics` writes fresh results instead, and [`_add_parallelism_header`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/integration_tests/run_tests.py#L179 "tests/integration_tests/run_tests#_add_parallelism_header") stamps the
  parallelism plan (`FSDP=.., TP=.., ...`) into the header. Inside `loss_compare.py`,
  [`main`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/scripts/loss_compare.py#L1189 "scripts/loss_compare#main") drives [`create_seed_checkpoint`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/scripts/loss_compare.py#L535 "scripts/loss_compare#create_seed_checkpoint"), [`read_metrics_from_file`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/scripts/loss_compare.py#L610 "scripts/loss_compare#read_metrics_from_file") / [`export_metrics_to_file`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/scripts/loss_compare.py#L646 "scripts/loss_compare#export_metrics_to_file") and [`assert_losses_equal`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/scripts/loss_compare.py#L809 "scripts/loss_compare#assert_losses_equal").
- **Bitwise tools.** [`hash_model`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/utils.py#L64 "tests/utils#hash_model") and [`hash_gradient`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/tests/utils.py#L94 "tests/utils#hash_gradient") hash parameters and gradients for bitwise comparisons in unit
  tests.
- **Experiment suites.** Several experiments ship their own `tests/integration_tests.py` built on
  `OverrideDefinitions`, for example [`build_transformers_modeling_backend_test_list`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/transformers_modeling_backend/tests/integration_tests.py#L16 "torchtitan/experiments/transformers_modeling_backend/tests/integration_tests#build_transformers_modeling_backend_test_list") (also torchft, rl, graph_trainer).
  `tests/integration_tests/flux.py` is a standalone runner with its own `main` (it also takes
  `--execution_mode` and `--test_scope`). It is not a `--test_suite` choice; CI runs it inside the
  8-GPU model job.
- **CI images.** `.ci/docker/` (README: "Docker images for TorchTitan CI") is copied from PyTorch CI
  and keeps its layout. `build.sh` dispatches the builds, `common/` holds the build-stage scripts
  and `ubuntu/` the Dockerfile for CPU build and test jobs. Usage is
  `./build.sh <image-name> <docker build params>`, for example
  `./build.sh torchtitan-ubuntu-22.04-clang12 -t myimage:latest`; `build.sh` also knows a
  `torchtitan-rocm-ubuntu-22.04-clang12` image.
- **Micro test encoders.** `tests/assets/flux_test_encoders/clip-micro` and `t5-micro` are
  configs "significantly downsized from the original model (e.g., num_hidden_layers reduced to 1,
  hidden_size to 64) to fit within the memory constraints of the test infrastructure". Each folder
  holds only a `config.json`, so the encoders start from random weights. The TPU Flux distributed
  tests point `--encoder.t5_encoder` / `--encoder.clip_encoder` at them (see
  [flux-model](flux-model.md)).

## Why it matters / when it applies
Most TorchTitan bugs appear only when features are combined (FSDP x TP x CP x PP x EP x compile x
checkpoint resume), so the integration matrix, not the unit tests, is the main gate. The README
splits the work by PG mode: Fake-PG runs check that a feature combination "configure[s],
transform[s], and complete[s] training", and Real-PG runs also cover real collectives and
distributed state. On pull requests the Fake-PG lane and the `real_pg_required` lane partition the
enabled A10G tests with no overlap. Post-merge, six-hourly schedules and the `ciflow/8gpu` label run
the full suites with Real PG. The `ciflow/h100.8` and `ciflow/b200` labels opt into the hardware
lanes; B200 also runs on `main` pushes that touch Kimi K3.

Fake-PG goldens have limits that the README states plainly. They guard FakeProcessGroup's
"deterministic synthetic numerical contract". They do not validate remote-rank values or EP load
balance, and a golden "does not establish that the simulated gradients are numerically
representative". Some cases carry no golden and give end-to-end coverage only: Qwen3.5 MoE (A10G
Real-PG not bitwise deterministic), DeepSeek V4 FSDP 2 x TP 2, EP 2 (Real-PG only: under Fake PG
its sequence-parallel collectives return outputs aliasing their inputs and grad_norm explodes at
step 1), and Kimi K2.5 DistMuon FSDP+EP (the multimodal bicubic-upsampling CUDA backward is
nondeterministic). Adding a feature means adding a config in `torchtitan_recipes/tests/` plus an
`OverrideDefinitions` entry that names it, and marking it `use_real_pg=True` if it checkpoints, uses
PP or otherwise needs real collectives.

**TPU view.** Everything above is GPU-oriented: `NGPU`, `CUDA_VISIBLE_DEVICES`, A10G/H100/B200
lanes and `cuda` / `rocm` arch choices. None of it launches on TPU. TPU coverage lives elsewhere:
- `torchtitan/experiments/tpu/tests/` (training, model, checkpoint-verification, flags, MFU and
  profiler tests).
- The TPU-named unit tests at the root of `tests/unit_tests/` (`test_tpu_packed_qkv*.py`,
  `test_tpu_grouped_experts.py`), which exercise the TPU converters on CPU.
- Per-model CPU-vs-device parity suites such as [`Llama3Test`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/llama3/tests/test_llama3.py#L43 "torchtitan/experiments/tpu/llama3/tests/test_llama3#Llama3Test") on [`BaseAcceleratorDeviceTest`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/base_device_test.py#L18 "torchtitan/experiments/tpu/base_device_test#BaseAcceleratorDeviceTest"), and the distributed harness
  [`BaseDistributedDeviceTest`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/base_distributed_device_test.py#L699 "torchtitan/experiments/tpu/base_distributed_device_test#BaseDistributedDeviceTest"), which picks the device from a `--device` flag and, on TPU, counts chips through torch_tpu
  when `--num_devices` is unset. [`DistributedUnitTestRunner`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/experiments/tpu/base_distributed_device_test.py#L58 "torchtitan/experiments/tpu/base_distributed_device_test#DistributedUnitTestRunner") checks a parallelized device model against a single-device
  *CPU reference* for forward, backward and optimizer step.

When porting a feature to TPU, add a parity test on that harness rather than a `run_tests` entry.
See [torchtpu-backend](../concepts/torchtpu-backend.md). The Fake-PG idea carries over: a
fake-process-group test can check sharding and config at a world size you do not have.

> [!warning] Doc vs code
> - The README says a Fake-PG configuration fails validation if it uses "an explicit non-Fake
>   communication backend". `validate_fake_pg_compatibility` checks only checkpointing, pipeline
>   parallelism and one named FSDP + selective-AC case; it has no communication-backend check.
> - The README says CPU-vs-GPU needs are "encoded by the `unit_tests/cpu/` and `unit_tests/gpu/`
>   directories". Several unit tests still sit at the root of `tests/unit_tests/`
>   (`test_tpu_*.py`, `test_flux_*.py`, `test_moe_aux_loss.py`, `test_tools_utils.py`), so
>   `pytest tests/unit_tests/cpu/` does not run them.
> - The README lists `flux.py` with the integration suites, but it is not a `--test_suite` value
>   (the choices are `features`, `models`, `h100`, `b200`). The README also omits the `--exclude`,
>   `--parallel/--no-parallel` and `--gpu_arch_type` flags.

## Connections
- Code concepts: [job-configuration](../concepts/job-configuration.md) (full configurations,
  `--module`/`--config`), [trainer-lifecycle](../concepts/trainer-lifecycle.md),
  [checkpointing](../concepts/checkpointing.md) (seed checkpoints, resume cases),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [dtensor-and-sharding-types](../concepts/dtensor-and-sharding-types.md),
  [context-parallel](../concepts/context-parallel.md),
  [profiling-and-tooling](../concepts/profiling-and-tooling.md),
  [torchtpu-backend](../concepts/torchtpu-backend.md).
- Module catalogs: [catalog index](../catalog/index.md); entry symbols above
  (`tests/integration_tests/*`, `tests/utils`, `scripts/loss_compare`, `torchtitan_recipes/tests`,
  `experiments/tpu/base_distributed_device_test`).
- Related doc-concepts: [full-configurations-and-recipes](full-configurations-and-recipes.md)
  (where test configs live), [model-config-transforms](model-config-transforms.md),
  [repo-conventions](repo-conventions.md) (numerics bar for every PR),
  [checkpoint-conversion](checkpoint-conversion.md) (numerical conversion tests),
  [rl-batch-invariance](rl-batch-invariance.md) (bitwise parity),
  [transformers-modeling-backend](transformers-modeling-backend.md), [flux-model](flux-model.md).

## Source
Extracted from `tests/README.md` (kept in place). The CI-image notes come from
`.ci/docker/README.md`, and the micro-encoder notes from
`tests/assets/flux_test_encoders/clip-micro/README.md` and `t5-micro/README.md`.
