---
title: "Dataset mixing, concatenation and packing order"
type: doc-concept
provenance: doc
source: torchtitan/components/data/README.md
updated: 2026-10-15
generated: {by: wikify/0.4.1, at: 2026-10-15T04:24:09Z}
---
# Dataset mixing, concatenation and packing order

## Definition
The data README separates three ways to combine datasets and makes their **order** the policy
knob. A **mix** draws from children by relative weight ("any positive relative weights are
accepted and normalized internally"); a **concatenation** treats finite datasets as one corpus
in which "every row appears once" and "dataset sizes determine their proportions"; **packing**
turns samples into fixed-length rows. What a mix weight counts depends on what its children
emit: mix documents and then pack when "weights should count documents", or pack each child and
then mix when "weights should count fixed-length rows" (that is, tokens).

## In torchtitan (grounded)
- **Mix.** [`DatasetMixConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L211 "torchtitan/components/data/dataset#DatasetMixConfig") takes a tuple of
  [`WeightedDataset`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L203 "torchtitan/components/data/dataset#WeightedDataset") (dataset + weight, default 1.0).
  [`build`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L226 "torchtitan/components/data/dataset#DatasetMixConfig.build") rejects non-finite or
  non-positive weights, seeds child `i` with `seed + i` ("Inserting or reordering a child
  reseeds every later child"), uses Grain `MapDataset.mix` when all children are map-style and
  `IterDataset.mix` otherwise. Its docstring adds two caveats the README does not: on the
  all-map path a `filter` leaves rejected indices as `None`, so weights count **attempted**
  indices rather than accepted samples; and with `repeat=False` the mix stops at the first
  exhausted child.
- **Concatenate.** [`DatasetConcatConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/dataset.py#L267 "torchtitan/components/data/dataset#DatasetConcatConfig") builds every
  child unshuffled, unrepeated and unsharded, requires all of them to be map-style (`TypeError`
  otherwise, which is why the README sends streaming data to mixing), concatenates, then shuffles
  the combined index space and DP-shards it. Repeating a child in the tuple (`(books_ds,) * 3`)
  makes each of its rows appear three times per epoch. `GrainDataLoader.Config(repeat=True)`
  repeats the whole composed dataset and does not change a child's share.
- **Pack.** [`ConcatThenSplitPackingConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/packing.py#L22 "torchtitan/components/data/packing#ConcatThenSplitPackingConfig")
  ("Concatenates documents, chunking them into fixed-length rows") is the pretraining choice;
  [`FirstFitPackingConfig`](https://github.com/google-pytorch/torchtitan/blob/033f131322a8b5b026ac53d734c689d664442bf8/torchtitan/components/data/packing.py#L225 "torchtitan/components/data/packing#FirstFitPackingConfig") ("Packs document chunks no
  longer than the context window", with `num_packing_bins` candidate rows kept open, default 8)
  is the SFT choice. Both convert a map-style child to an `IterDataset` and pack per DP rank.

> [!warning] Doc vs code
> The README frames mix-before-pack as weights that "count documents". With map-style children
> and filters, the `DatasetMixConfig` docstring says weights count attempted indices (filtered
> rows still consume a draw), so the realized document ratio can drift from the weights.

## Why it matters / when it applies
Choosing the composition order is how a recipe expresses a data ratio. Mixing documents before
packing keeps document counts in proportion but lets long-document sources dominate tokens;
packing first makes the weights token ratios. Concatenation gives exact, finite coverage (and
exact duplication by repetition) but only for map-style data. The README notes that
"automatic adjustment from observed document or supervised-token counts is not implemented"
(the code carries a matching TODO for token-weighted mixing); a custom mix would have to keep
its own running estimates.

## Connections
- Code concepts: [dataloading-and-tokenizer](../concepts/dataloading-and-tokenizer.md) (packing,
  per-document positions and padding in depth),
  [parallel-dims-and-mesh](../concepts/parallel-dims-and-mesh.md).
- Module catalogs: [module map](../catalog/index.md) (`components/data/dataset`,
  `components/data/packing`).
- Related doc-concepts: [grain-data-pipeline](grain-data-pipeline.md).

## Source
Extracted from `torchtitan/components/data/README.md` (kept in place).
