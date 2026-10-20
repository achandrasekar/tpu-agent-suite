---
title: "TorchTitan (distributed training on TPU)"
type: codebase
tags: [codebase, torchtitan, pytorch, parallelism, training]
created: 2026-09-25
updated: 2026-10-15
sources: 2
human_reviewed: false
aliases: [torchtitan, titan, torchtitan_tpu]
---

# TorchTitan (distributed training on TPU)

`torchtitan` is PyTorch's native platform for large-scale LLM training: a config-driven trainer
over composable parallelism (FSDP, tensor, pipeline, context and expert parallel), distributed
checkpointing, and a model zoo (Llama 3, Qwen3 and Qwen3.5/3.6/3.8, DeepSeek-V3 and V4, Kimi
K2.7 and K3, Muse Glimmer, GPT-OSS, Flux and more). The fork
ingested here carries a TorchTPU backend under `torchtitan/experiments/tpu` that trains on Cloud
TPU through the [torch_tpu runtime](torch-tpu.md). This hub page states the ingested pin and
routes into the grounded subwiki; mechanism and symbol-level detail live there, not here.

> **Which torchtitan.** Everything here describes https://github.com/google-pytorch/torchtitan,
> Google's TorchTitan and the one in current use for TPU training: it is the only one
> with the TorchTPU backend (`torchtitan/experiments/tpu`, `TPUTrainer`, `TpuGraphTrainer`).
> The official https://github.com/pytorch/torchtitan has no TPU backend. The two
> have diverged: they share no git history (upstream changes are ported by hand in periodic
> rebases), and of the 1,037 files at this pin only 353 exist at the same path in upstream `main`
> (2026-10-15), 74 of them identical. Do not answer TPU questions from upstream code or docs, and do not assume a
> symbol, line number or config flag here matches upstream.

## Pin

| | |
|---|---|
| Repository (ingested) | https://github.com/google-pytorch/torchtitan (Google's TorchTitan with the TorchTPU backend; the source is not shipped with this knowledge base) |
| Official upstream | https://github.com/pytorch/torchtitan (no TPU backend; diverged, see above) |
| Ingested commit | `033f131322a8b5b026ac53d734c689d664442bf8` (main HEAD, 2026-10-15); previous pin `63e334d5` (2026-09-23), delta in [`changes/033f131322`](torchtitan/changes/033f131322.md) |
| Ingest config | `config/torchtitan.md` (wikify-repo; re-ingest at a new pin with `wikify prepare torchtitan --ref <commit>`) |
| Provenance | [torchtitan repo snapshot (2026-10)](../sources/2026-10-torchtitan-repo.md) (supersedes [2026-09](../sources/2026-09-torchtitan-repo.md)) |

## Start here (subwiki)
- [Overview](torchtitan/overview.md): the system on one page, architecture diagrams, the
  TPU-relevant knobs, known issues at this pin, and a task table.
- [Concept index](torchtitan/index.md): 24 grounded mechanism pages - the training loop and
  trainer lifecycle, job configuration, the device mesh, each parallelism axis (FSDP, TP, PP,
  CP, EP), activation checkpointing, DTensor layouts, torch.compile, checkpointing and
  Hugging Face state-dict adapters, optimizer and LR schedule, LoRA, quantization, the model
  zoo, metrics and profiling, and the TorchTPU backend.
- `torchtitan/doc-concepts/`: 56 pages extracted from the repo's own docs, including the TPU
  training recipes for TPU7x (Ironwood) and v6e (Trillium), TPU VM provisioning and multi-host
  launch with Cluster Toolkit, libtpu flag precedence, the graph trainer, the JAX and torchax
  lanes, the Grain data pipeline, float8/MXFP8/NVFP4 training, and one page per model family. Each notes
  where its doc disagrees with the code.
- `torchtitan/catalog/`: the symbol index - `symbols.tsv` (13,713 symbols, one row each: anchor,
  path, line, kind, rank, hash, caller count, citing pages, signature, doc line), `edges.tsv`
  (42,812 caller edges) and [`index.md`](torchtitan/catalog/index.md) (module map, 689 modules).
  Look a symbol up with `grep -P '[#.]<Symbol>\t' catalog/symbols.tsv`. Citations link the
  symbol's source line on GitHub at the pin and carry its index key as the link title, so the
  row is one grep away even without the source.

## Training on TPU: where to look first
- Which trainer runs, and how it binds to torch_tpu:
  [TorchTPU backend](torchtitan/concepts/torchtpu-backend.md) - `TPUTrainer` (eager with
  per-block `torch.compile(backend="tpu")`) vs `TpuGraphTrainer` (one whole-step graph lowered
  by torch_tpu's `StaticCompiler`).
- Tested configurations per model and topology:
  [TPU training recipes](torchtitan/doc-concepts/tpu-training-recipes.md), with the
  [TPU7x](torchtitan/doc-concepts/ironwood-tpu7x-recipes.md) and
  [v6e](torchtitan/doc-concepts/trillium-v6e-recipes.md) tables and the
  [scaling patterns](torchtitan/doc-concepts/tpu-recipe-scaling-patterns.md) across them.
- Mesh and collectives: [device mesh](torchtitan/concepts/parallel-dims-and-mesh.md) (including
  the topology-aware mesh option) and [FSDP](torchtitan/concepts/fsdp-sharding.md).
- libtpu flags and how they override each other:
  [flag precedence](torchtitan/doc-concepts/tpu-init-flags-precedence.md).
- What does not run on TPU at this pin, and the bugs found in the source: the overview's
  "Known issues at this pin".

## See also
- [TorchTPU Runtime & OpenXLA Backend](torch-tpu.md) - the runtime underneath the TPU backend.
- [TPU7x (Ironwood)](../hardware/tpu-7x.md) and [TPU v6e (Trillium)](../hardware/tpu-v6e.md) -
  the hardware the recipes target.

## Sources
- [torchtitan repo snapshot (2026-10)](../sources/2026-10-torchtitan-repo.md) - provenance of
  the current pin: what changed upstream, how the re-ingest handled it, verification, hygiene,
  doc-versus-code mismatches and gaps.
- [torchtitan repo snapshot (2026-09)](../sources/2026-09-torchtitan-repo.md) - the first
  ingest's provenance record (previous pin).
