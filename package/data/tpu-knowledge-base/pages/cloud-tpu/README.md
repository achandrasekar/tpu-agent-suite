# Cloud TPU and GKE knowledge

These pages summarize durable concepts from official Google Cloud documentation.
They use short replay queries, one generation call per page and shared-source
footnotes. They are general guidance, not installed-version or live-capacity
answers.

| Topic | Page | Simple replay query |
|---|---|---|
| Slices, hosts and topology | [Concepts](concepts/overview.md) | [Recipe](concepts/overview.recipe.json) |
| GKE resource scheduling | [Deployment](deployment/overview.md) | [Recipe](deployment/overview.recipe.json) |
| Multislice networking and training | [Multislice](multislice/overview.md) | [Recipe](multislice/overview.recipe.json) |
| Capacity choices and queues | [Capacity](capacity/overview.md) | [Recipe](capacity/overview.recipe.json) |
| TPU VM lifecycle | [VM lifecycle](vm-lifecycle/overview.md) | [Recipe](vm-lifecycle/overview.recipe.json) |
| Storage choices | [Storage](storage/overview.md) | [Recipe](storage/overview.recipe.json) |
| Performance principles | [Performance](performance/overview.md) | [Recipe](performance/overview.recipe.json) |
| Troubleshooting workflow | [Troubleshooting](troubleshooting/overview.md) | [Recipe](troubleshooting/overview.recipe.json) |
| gcloud invocation patterns | [CLI](cli/overview.md) | [Recipe](cli/overview.recipe.json) |

The concept pages above deliberately omit command inventories. The CLI page is
the exception: it documents gcloud invocation patterns and their subtleties.
Flags and defaults change between gcloud releases, so that page ages faster
than the others and should be confirmed against the installed CLI.

Prices, quota amounts, current capacity, regions/zones, runtime versions and
minimum GKE versions require a fresh lookup of the official documentation.
Examples are illustrative: choose accelerator type, topology and resource
counts for the actual target. API-specific state and lifecycle behavior must
not be transferred between Cloud TPU and GKE resources.

Each sibling `overview.recipe.json` holds the query the page answers. See
[source and replay instructions](../README.md).
