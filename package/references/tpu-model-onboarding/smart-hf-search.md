---
scripts:
  - scripts/tpu-model-onboarding/smart_hf_search.py
---
# Smart Hugging Face Search Skill

A skill to search for Hugging Face models and libraries using the
`smart_hf_search` tool located at
`{{script:tpu-model-onboarding/smart_hf_search.py}}`.

## Workflow & Execution Rules

When executing this skill:

1.  Read and adhere to `AGENTS.md`.
2.  Use the `smart_hf_search` script via standard `python3`.
3.  Map the user's natural language request to the appropriate command-line
    flags.

### Running the Tool

Run the script standalone (after installing `huggingface_hub` with `pip`):

```bash
python3 {{script:tpu-model-onboarding/smart_hf_search.py}} [flags]
```

--------------------------------------------------------------------------------

## Query Examples and Flag Mappings

The examples below show standalone execution.

### 1. Top-K Hugging Face models (General popular models)

```bash
python3 {{script:tpu-model-onboarding/smart_hf_search.py}} --k=5 --sort=downloads
```

### 2. Top-K popular or trending Hugging Face libraries

```bash
python3 {{script:tpu-model-onboarding/smart_hf_search.py}} --k=5 --mode=libraries
```

### 3. Top-K trending Hugging Face models this week

```bash
python3 {{script:tpu-model-onboarding/smart_hf_search.py}} --k=5 --sort=trending
```

### 4. Top-K trending Hugging Face models with transformer architecture

We filter by the `transformers` library as most transformer models use it:

```bash
python3 {{script:tpu-model-onboarding/smart_hf_search.py}} --k=5 --sort=trending --library=transformers
```

### 5. Top-K trending Hugging Face models with less than 20B parameters

```bash
python3 {{script:tpu-model-onboarding/smart_hf_search.py}} --k=5 --sort=trending --max_params=20B
```

--------------------------------------------------------------------------------

## Available Flags

To view the complete and up-to-date list of available command-line flags and
descriptions, run the script with the `--help` flag:

```bash
python3 {{script:tpu-model-onboarding/smart_hf_search.py}} --help
```

--------------------------------------------------------------------------------

## Time-Based Trending Queries (API Limitations)

The Hugging Face API only supports a weekly window for the `trending` score. Map
time-based queries as follows:

*   **"this week" / "trending" (default)**: Use `--sort=trending`.
*   **"this month" / "this year" / "overall"**: Hugging Face does not support
    monthly trending API queries. Fallback to `--sort=downloads` or
    `--sort=likes` to approximate long-term popularity, and inform the user of
    the fallback.
