**What transfers:** Techniques that are hardware-agnostic, and fundamentally about reducing data movement and intermediate results, generally carry over:

* Tiling and avoiding materializing large tensors: The FlashAttention idea of computing softmax block by block without storing the full attention matrix maps onto how you partition tiles in Pallas.
* Numerical tricks: Online softmax, log-sum-exp accumulation, mask handling, and the order of dequantization steps are pure math.
* Where to draw fusion boundaries: The CUDA implementation has usually already worked out which ops are worth merging into one kernel.
* Ragged/paged indexing logic: How the block table is looked up and how variable-length sequences are laid out can be borrowed directly at the semantic level.

**What doesn't transfer (or possibly hurts):** TPU's execution model is very different from GPU's, and GPU-specific tricks copied over often make things worse:

* Data-dependent control flow: CUDA kernels often use early exits, loops that break on sequence length, and branches that skip padding. The Pallas docs note that many patterns unnatural to the hardware will still be accepted by the compiler, but may require software emulation and slow down the computation. The agent can easily produce code that runs correctly but slowly.
* Memory hierarchy: GPUs follow an SRAM/HBM cache mindset. TPUs have their own memory spaces and pipelining model. For example, community kernels prefetch scalars from SMEM, use double or triple input buffering to hide memory latency, and use emit_pipeline so compute and memory operations overlap. None of this has an equivalent in CUDA code.
* Block sizes, thread grids, streams, pointer arithmetic, cuBLAS calls: These are simply meaningless on TPU.
