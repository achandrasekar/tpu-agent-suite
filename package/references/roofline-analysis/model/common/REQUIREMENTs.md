---
references:
- references/roofline-analysis/COLLECTIVE.md
- references/roofline-analysis/TYPE_PROMOTION.md
- references/roofline-analysis/tpu/V7X.md
---

# Common Requirements for Roofline Performance Estimation

To ensure accuracy and avoid hallucinations, the following requirements must be met when executing roofline analysis:

*   **Use Python for Calculations:** Do NOT perform roofline arithmetic (FLOPS, memory transfer sizes, latencies, or throughput calculations) mentally or within the chat output. You MUST write and execute a Python script to do all numerical calculations and output the exact results.
*   **Consult Type Promotion Rules:** When determining data types (`dtype`), output element sizes (bytes), or the results of operations/projections, you MUST read [TYPE_PROMOTION.md]({{reference:roofline-analysis/TYPE_PROMOTION.md}}) and explicitly justify your dtype and byte-size choices based on JAX Type Promotion and TPU MXU Accumulation Rules. Do not assume default data types without verifying how the input/weight dtypes promote or accumulate.
*   **Standardize Latency Units:** All roofline analysis latency must be calculated and reported in milliseconds (ms).
*   **Standardize FLOPS Units:** All FLOPS values must be reported in Billions (GFLOPS / GFLOPs; Billions of FLOPS).
*   **Standardize Memory Transfer Units:** All memory transfer values must be reported in Megabytes (MB).
*   **TRANSFER and OCCUPANCY are two different numbers. Never label them the same.**
    *   **Transfer** is what the op reads and writes this pass. It is the only thing
        that may divide by HBM bandwidth to give a memory time.
    *   **Occupancy** is what the op's tensors take up in HBM -- the weights it owns,
        the cache it attends over. It is a capacity number and it does not reconcile
        with the memory time.
    Under sparse attention they are **64x apart** (`B * topk` read out of a `B * S`
    cache). **State which one a column holds, in the header**; if a row shows both,
    give them different names.


## Fusion Requirements

*   **"Fused" / "kept in VMEM" / "can be ignored" is a CAPACITY CLAIM — test it.** ~25 places in
    these specs drop an intermediate's HBM cost on those words. Each holds only while the tensor
    fits **8 MiB of pooled VMEM per SparseCore** ([`tpu/V7X.md`]({{reference:roofline-analysis/tpu/V7X.md}}) §VMEM). Compute
    `elements * bytes` and compare before taking the zero.
*   **A streaming reduction may fuse; a global one usually may not.** FlashAttention's online
    softmax genuinely never materialises its scores — it tiles over keys and carries a running
    max and sum, so `attention/mla` §Note is sound at any `S`. **Top-K is different**: ranking
    `K_idx` of `S` wants all `S` scores for a query at once, so `attention/dsa` cannot make the
    same claim for free, and at 128K it does not hold. Ask which kind of reduction it is.
*   **Charge an intermediate zero or twice, never once.** If the producer writes it, the consumer
    reads it; if the producer fuses it, the consumer has nothing to read. Mixing the two is how
    `roofline.py` came to read a score matrix nothing had written.

## MXU Requirements

*   **MXU Padding (Contracting Dimension):** When calculating FLOPS, if the Contracting Dimension is smaller than the minimum MXU size (e.g., 256 for TPU v7x), it must be ceil-padded to that minimum MXU size. The FLOPs calculation must be performed using this padded dimension.

## Collective Requirements

*   **Consult Collective Rules:** When analyzing operations involving collective communications, you MUST read [COLLECTIVE.md]({{reference:roofline-analysis/COLLECTIVE.md}}) and adhere to the domain knowledge and rules provided there.


## Output Format

The agent should output the detailed operation-level roofline analysis in the following format:

```json
{
  "{op_name1}": {
    "GFlops": <number>,
    "memory_transfer (MB)": <number>,
    "memory_occupancy_weights (MB)": <number>,
    "memory_occupancy_kv (MB)": <number>,
    "execution_time (ms)": <number>,
    "memory_transfer_time (ms)": <number>,
    "collective_transfer_time (ms)": <number>,
    "collective_type": <text>,
    ""
  },
  "{op_name2}": {
    ...
  }
}
```