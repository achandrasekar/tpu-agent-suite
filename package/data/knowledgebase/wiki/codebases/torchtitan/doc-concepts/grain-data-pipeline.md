---
title: "Grain data pipeline: source -> dataset -> compose -> loader, DP sharding and resume"
type: doc-concept
provenance: doc
source: torchtitan/components/data/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Grain data pipeline: source -> dataset -> compose -> loader, DP sharding and resume

## Definition
The data README states that "TorchTitan uses one Grain-based data pipeline for text
pretraining, SFT, and image training", and gives a five-step mental model: a **source** yields a
random-access data source or an iterable dataset; a **dataset** applies pre-filter -> process ->
post-filter to one source; optional **composition** nodes mix, concatenate or pack child
datasets; the **dataloader** batches, collates and prefetches; the **trainer** consumes the
resulting `TrainerBatch`. Every stage is a config object with a `build`, so a data recipe is a
tree of configs assigned to `config.dataloader`. SFT "changes the processor and packing policy,
not the loader", and images reuse "the same source, dataset, loader, sharding, and checkpoint
contracts".

## In torchtitan (grounded)
- **Sources** (`components/data/sources`). [`SourceConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/sources.py#L39 "torchtitan/components/data/sources#SourceConfig")
  is the build protocol; random-access sources implement
  [`RandomAccessDataSource`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/sources.py#L29 "torchtitan/components/data/sources#RandomAccessDataSource") (`__len__` / `__getitem__`).
  Shipped sources are [`IndexedJsonlSource`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/sources.py#L50 "torchtitan/components/data/sources#IndexedJsonlSource") (one JSON
  object per non-empty line, glob `patterns`),
  [`HuggingFaceRandomAccessSource`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/sources.py#L94 "torchtitan/components/data/sources#HuggingFaceRandomAccessSource") (a materialized
  dataset) and [`HuggingFaceStreamingSource`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/sources.py#L145 "torchtitan/components/data/sources#HuggingFaceStreamingSource") (when "the
  corpus should not be materialized"). The README's pretokenized-memmap example shows a custom
  source is just a `Configurable` + `RandomAccessDataSource` whose constructor takes a
  [`DatasetIterationPolicy`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/types.py#L28 "torchtitan/components/data/types#DatasetIterationPolicy").
- **One dataset.** [`SingleDatasetConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L84 "torchtitan/components/data/dataset#SingleDatasetConfig") holds `source`,
  `pre_filters`, a `processor` and `post_filters`; its docstring says it "owns leaf shuffle,
  repeat, and effective-DP sharding". Processors subclass
  [`SampleProcessor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L67 "torchtitan/components/data/dataset#SampleProcessor") (called as `(row, rng)`, returning
  a sample or `None`, which the README's examples drop with a `post_filters` predicate) and are
  built with a [`DatasetBuildContext`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/types.py#L17 "torchtitan/components/data/types#DatasetBuildContext"). Text samples are
  [`TextSequence`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L39 "torchtitan/components/data/dataset#TextSequence") (`input_ids`, `labels`). The stock
  processors are [`TextProcessor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/hf_datasets/text_datasets.py#L32 "torchtitan/hf_datasets/text_datasets#TextProcessor") (`text_fn` turns a
  row into text) and [`ChatProcessor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/hf_datasets/text_datasets.py#L61 "torchtitan/hf_datasets/text_datasets#ChatProcessor") (`messages_fn`;
  it applies the tokenizer's chat template and masks prompt labels with `IGNORE_INDEX`).
- **Loader.** [`GrainDataLoader`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L61 "torchtitan/components/data/loader#GrainDataLoader") builds the graph for its
  DP rank in [`__init__`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L82 "torchtitan/components/data/loader#GrainDataLoader.__init__"), converts a
  remaining `MapDataset` to an `IterDataset`, batches with the collator (default
  [`TextCollator`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/collators.py#L52 "torchtitan/components/data/collators#TextCollator"); drop-remainder when repeating), and
  wraps the result in a thread prefetcher. Run-wide knobs live in its
  [`Config`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L65 "torchtitan/components/data/loader#GrainDataLoader.Config"): `seed`, `shuffle`, `repeat`,
  `streaming_shuffle_buffer_size` ("raw rows retained for approximate shuffling", default
  1,000), `read_options` (Grain threads and buffer for **each** map-to-iter conversion; an
  all-map mix converts once, a mix containing a stream converts each map child separately), and
  `num_prefetch_batches` (complete collated batches waiting for the trainer, default 2).
- **Map vs iter.** Random-access sources become Grain `MapDataset`s and stay map-style until a
  stage must read sequentially: packing always returns an `IterDataset`, a mix with any
  iterable child converts its map children, and the loader converts last.
- **Multimodal.** The Qwen multimodal example plugs
  [`MultiModalProcessor`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/hf_datasets/multimodal/mm_datasets.py#L275 "torchtitan/hf_datasets/multimodal/mm_datasets#MultiModalProcessor"),
  [`MMSamplePackingConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/hf_datasets/multimodal/mm_datasets.py#L377 "torchtitan/hf_datasets/multimodal/mm_datasets#MMSamplePackingConfig") and a
  [`MultiModalCollator`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/hf_datasets/multimodal/mm_collator.py#L22 "torchtitan/hf_datasets/multimodal/mm_collator#MultiModalCollator") into the same loader;
  the doc notes `num_packing_bins` "is the number of candidate packed rows kept open, not an
  input sample buffer", and that custom image augmentation belongs in a `SampleProcessor`.

## Distributed and checkpoint behavior
- **Only effective DP selects data.** Effective DP is replicate x shard degree; different
  effective-DP ranks get disjoint rows, while TP/PP/CP peers get the same rows.
- **Ownership is decided before batching.** For a random-access source,
  [`_build_map_dataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L126 "torchtitan/components/data/dataset#SingleDatasetConfig._build_map_dataset") filters and
  processes globally indexed rows, shuffles globally, then gives each rank a contiguous balanced
  slice of the shuffled index space (raising if there are fewer rows than DP ranks); a Hugging
  Face stream is DP-sharded at the source (`split_dataset_by_node`) and window-shuffled in
  [`_build_iter_dataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L173 "torchtitan/components/data/dataset#SingleDatasetConfig._build_iter_dataset"). Mixes
  combine children already owned by the rank; concatenation shuffles and shards after
  concatenating; packing runs locally on each rank.
- **No finite multi-rank data.** With effective DP > 1, `repeat=False` raises ("can exhaust
  ranks at different steps and hang collectives; use repeat=True with a trainer-controlled step
  count").
- **Resume.** [`state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L158 "torchtitan/components/data/loader#GrainDataLoader.state_dict") stores a
  version, the effective DP degree and the whole Grain iterator state under a `dp_rank_<N>` key
  (source cursors, shuffle/repeat progress, mix and packing-buffer state, batching and prefetch
  state). [`load_state_dict`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/loader.py#L165 "torchtitan/components/data/loader#GrainDataLoader.load_state_dict") refuses a
  changed DP degree or a missing rank key. The README adds that resume also requires unchanged
  code, config, source contents and tokenizer, which the code cannot check.

## Why it matters / when it applies
The pipeline replaces per-recipe dataset classes with one composable graph, so changing from
pretraining to SFT or images is a matter of swapping the processor, packing node and collator.
The DP rules matter for correctness: a resume with a different data-parallel layout is refused
rather than silently reshuffled, and finite datasets are only allowed on a single DP rank.
Packing to fixed-length rows also gives every batch the same static shape, which is what a TPU
compile wants; see [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md) for the
flat token batch and its TPU reading.

## Connections
- Code concepts: [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md) (the deep
  page on this subsystem), [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md),
  [checkpointing](../concepts/checkpointing.md),
  [job-configuration](../concepts/job-configuration.md),
  [trainer-lifecycle](../concepts/trainer-lifecycle.md).
- Module catalogs: [module map](../catalog/index.md) (`components/data/*`,
  `hf_datasets/text_datasets`, `hf_datasets/multimodal/*`).
- Related doc-concepts: [dataset-mixing-and-packing](dataset-mixing-and-packing.md),
  [qwen3-5-model](qwen3-5-model.md), [config-system-conventions](config-system-conventions.md).

## Source
Extracted from `torchtitan/components/data/README.md` (kept in place).
