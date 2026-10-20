---
title: "torchtitan repo snapshot (2026-10)"
type: source
tags: [codebase, torchtitan, pytorch, parallelism]
created: 2026-10-15
updated: 2026-10-15
sources: 1
human_reviewed: false
---

Provenance record for the re-ingest of Google's `torchtitan` repository (TorchTitan with the
TorchTPU backend) at the head of its `main` branch into the codebase subwiki
`wiki/codebases/torchtitan/`, produced by wikify-repo. It supersedes
[the 2026-09 snapshot](2026-09-torchtitan-repo.md) at `63e334d5`; the delta between the two pins
is in [`changes/033f131322.md`](../codebases/torchtitan/changes/033f131322.md). Knowledge lives on
the subwiki pages; this page records where it came from and what was left out.

## Identity

| | |
|---|---|
| Repository | https://github.com/google-pytorch/torchtitan: Google's TorchTitan, the only one with the TorchTPU backend. Not the official https://github.com/pytorch/torchtitan, which has no TPU backend; at this pin 353 of 1,037 files exist at the same path in upstream `main` (2026-10-15), 74 of them identical |
| Commit | `033f131322a8b5b026ac53d734c689d664442bf8` (main HEAD, 2026-10-15) |
| Previous pin | `63e334d5ba90602b17b916e9e0c193059a294007` (180 commits earlier, 144 of them non-merge; 868 files changed) |
| Acquired as | the existing clean clone under `raw/code/torchtitan/` (gitignored, never shipped), fetched to the new commit; tracked files only |
| Indexed with | SCIP: scip-python over `torchtitan/*` (14 shards); wikify 0.4.1 |
| Authority | primary source: the implementation, its docstrings and comments, its README files and the repo's agent rules under `.claude/` |

## What changed in the repo, and what the ingest did about it
- **One upstream rebase dominates.** `fa627bf2` brings in pytorch/torchtitan as of 2026-09-11.
  Activations are flattened to one `[T]` token stream (the TPU lane reshapes back to `[B, L]`
  with an unflatten shim), batches are sized in tokens (`num_tokens_per_microbatch_per_dp_rank`,
  `max_context_length`) instead of `local_batch_size` / `seq_len`, per-model TP plans are
  replaced by `ShardingConfig` declarations, context parallel becomes a config transform that
  requires the `spmd_types` backend, the `full_dtensor` backend is gone, and checkpointing,
  optimizers and data loading become packages (`components/checkpointer/`, `components/optimizer/`,
  `components/data/` on Grain). New model packages: DeepSeek-V4, Kimi K2.7 and K3, Muse Glimmer,
  Qwen3.6 and Qwen3.8; new top-level `README.md` and `torchtitan_recipes/`.
- **TPU-side work.** Muse Glimmer on both TPU lanes, the HF transformers backend on TorchTPU,
  a TPU float8 converter (`llama3_70b_float8`), the `tpu_selective` remat policy and block
  layout, ICI all-gather pipelining in the scan recipes, Cluster Toolkit replacing XPK for
  multi-host launch, and new Ironwood recipes (Llama3-70B on 128 devices, Qwen3-32B, Muse
  Glimmer 30B, Mamba3).
- **Config.** New pin and source URL; three moved subsystems remapped
  (`components/checkpointer`, `components/data`, `distributed/context_parallel`); three
  concepts re-seeded because their code was restructured: `tensor-parallel` (now the
  `ShardingConfig` helpers in `models/common/*_sharding.py`), `context-parallel`
  (`ContextParallelTransform` and the CP inner attentions) and `optimizer-and-lr-schedule`
  (the `components/optimizer` package; `build_optimizers` / `build_lr_schedulers` no longer
  exist). The six new model directories are folded into `model-definitions` like the others.
- **Reconcile.** 6,144 symbols changed and 856 were removed, so all 24 concept pages were
  rebuilt, each by updating the existing page in place (wording, structure and diagrams kept
  where still accurate) with a `## Recent changes` section; 262 moved symbols were relinked
  first (43 citations in 9 pages). No concept page was added or removed.
- **Docs.** The doc worklist grew from 48 to 68 documents (23 new, 3 gone: the Forge README, and
  the MXFP8 doc and the Mamba2 recipe, which moved). Doc-concept pages went from 45 to 56: 43
  old slugs reused and updated, 13 new (Grain data pipeline, dataset mixing and packing, NVFP4,
  FlexShard with DistMuon, full configurations and recipes, model config transforms, Llama 3,
  DeepSeek-V4, Kimi K2.7, Kimi K3, Muse Glimmer, DAPO math RL recipe, verifiers RL integration),
  2 removed because their source doc is gone and no current doc covers them (`forge-engine`:
  `experiments/forge/` was deleted upstream; `tpu-model-parity-tests`: its per-model test README
  is gone). Qwen3.6 and Qwen3.8 fold into the Qwen3.5 page.
- **Symbol index.** `catalog/symbols.tsv`: 13,713 symbols with signatures and doc lines;
  `catalog/edges.tsv`: 42,812 caller edges; `catalog/index.md`: module map over 689 modules;
  100% of documentable symbols and all 1,425 classes represented; 843 symbols (6.1%) covered by
  a concept page.
- **Verification.** The citation linter gates every page (`finalize` green, 31 diagrams, no
  warnings). The concept pages were then adversarially verified against the source at the pin
  by reviewers who had not written them: 436 load-bearing claims in the first pass and 32 in the
  incremental re-check; 17 were wrong and are corrected in place (for example: which token
  budget the trainer checks against the TP degree, that `update_from_config` no longer resizes
  RoPE, that the fault-tolerant step does run accumulation groups and SDC replay, and that the
  `torch_checkpointing` backend writes the last step synchronously). Verdicts live in
  `.cache/verify/` (not shipped); each page carries a `verified:` stamp.
- **Public-repository hygiene.** Pages were written without internal identifiers. The symbol
  index's doc column needed no scrub at this pin (no internal bug IDs, short links or usernames
  in the source comments it copies). The generated change page listed an upstream commit that
  removes partner model packages withheld from this wiki, with their file paths; its subject is
  reworded and the withheld paths are dropped from its file lists. A personal reviewer stamp in
  the overview's front matter was removed with the claim changes. A future `finalize`
  regenerates the catalog and the change page, so both checks have to be repeated.

## Doc-versus-code mismatches found
Recorded on the doc-concept and concept pages; the ones a user would hit first:
- The float8 and MXFP8 docs pass `quantization=[...]` to `model_registry`; the parameter is
  `converters=`. The float8 examples filter `"output"`, but the final projection is `lm_head`.
- The TPU README still says only `llama3_tpu` is supported with a 128/512 context cap; five
  model packages are registered and registry defaults go up to 16384. It cites a
  `qwen3_8b_tpu7x` flag example that no longer exists, and its profiling flags
  (`--profiling.*`, `--job.dump_folder`) are not the real names (`--profiler.*`, `dump_folder`).
- The Mamba3 recipe README passes `--training.local_batch_size` and `--training.seq_len`,
  which no longer exist; the Mamba2 README says selective activation checkpointing where the
  recipe sets full AC.
- The config-transform README lists `QuantizationTransform` and `LMHeadCastTransform`; only
  `ContextParallelTransform` exists. The config README says to run `pip install -e .`, but the
  checkout has no `pyproject.toml` or `setup.py`.
- The `.claude/` rules list a `llama4` model, `models/common/rmsnorm.py` and a `moe/`
  directory, none of which exist, and a public `maybe_enable_async_tp` that is now private in
  `distributed/compile.py`.
- The transformers-backend README's train command omits the required
  `--parallelism.spmd_backend spmd_types` and raises as written; the torchft README misspells
  `data_parallel_replicate_degree`; the JAX and torchax quick starts use flags that are not
  config fields.
- The RL GKE script selects Qwen3 1.7B and 4B varlen configs that do not exist in the
  registry; the RL README tells users to set `max_offpolicy_steps`, now a derived property.
- Docstrings in the code still show `Shard(1)` for sequence-parallel outputs that are now
  sharded on the flattened token dim 0.

## Gaps and caveats
- **The TPU backend still needs the explicit-seed workaround** described on the
  [2026-09 page](2026-09-torchtitan-repo.md): wikify's planner treats every `experiments/` path
  as non-library, so `torchtpu-backend` is seeded with symbols. Other `experiments/` subsystems
  (graph_trainer, jax, torchax, rl, torchft, transformers backend) have no concept page; they
  are covered by doc-concepts and the symbol index.
- `torchtitan_recipes/` (new, one Muse Glimmer CP recipe module) is in the symbol index but
  outside the index shards' glob and has no page beyond the config and testing doc-concepts.
- Upstream bugs found from source are listed on the overview under "Known issues at this pin"
  (12 items, e.g. `get_peak_flops` returning `None` for an unrecognized TPU name, the DCP
  backend's multi-file HF export calling a function whose import is commented out, and the
  Mamba2/Mamba3 parallelizers calling `apply_compile` positionally). They are properties of the
  repo at this pin, not of this wiki.
- The generated change page and the silo log use non-ASCII arrows and dashes in their
  templates (wikify output, left as generated). wikify also writes the change page's
  `description:` unquoted around a colon, which is invalid YAML; it is quoted by hand here, and a
  re-`finalize` undoes that until wikify quotes it.
- Source links are GitHub permalinks at the pin; a reader who cannot open them uses the link
  title (the index key) to find the row in `catalog/symbols.tsv`.

## Connections
- Updated: the `wiki/codebases/torchtitan/` subwiki (24 concept pages, the overview, 56
  doc-concept pages (43 updated, 13 new; 2 removed), the regenerated `catalog/`, `index.md`, `log.md`), the new change page
  [`changes/033f131322.md`](../codebases/torchtitan/changes/033f131322.md), and the hub page
  [TorchTitan](../codebases/torchtitan.md).
- Supersedes: [torchtitan repo snapshot (2026-09)](2026-09-torchtitan-repo.md).
- Related subwiki: [torch_tpu](../codebases/torch-tpu.md) - the runtime TorchTitan's TPU
  backend binds to.
