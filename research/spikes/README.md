# spikes — throwaway experiments, kept as evidence

Each spike answered one question before it was allowed into the product. They
are never built into the server image; the findings they produced live in
`docs/findings/`.

Spikes with a `CMakeLists.txt` build and run in one step:

    make spike SPIKE=s0_slot_slab          # CPU
    make spike-vk SPIKE=s0b_backend_slab   # with the iGPU passed through

| Spike | Question it answered | Finding |
|---|---|---|
| `s0_slot_slab` | Does a slot table + id remap work at all? | S0 |
| `s0b_backend_slab` | Does it survive on a real Vulkan backend? | S0b |
| `s1_real_expert_stream` | Can experts be read straight from the GGUF? | S1 |
| `s2_iouring_bw` | What bandwidth can the SSD actually deliver? | S2 |
| `s3_zero_slot` | Is a null expert safe as a miss placeholder? | — |
| `s4_sweep_math` | Expert Sweep: how many passes would it cost? | N4 |
| `s10_p2_accuracy` | How well does a layer-lookahead predictor guess? | S10 |
| `s15_syscall_overhead` | Is syscall count worth batching away? | RESULTS §10.13 |
| `s16_policy_compare` | S3-FIFO against LRU and LFU on real traces | RESULTS §10.15 |
| `n1_cache_replay` | Replay a trace against candidate cache policies | N1 |

The rest are single-file sources kept for reference. They were built by hand
during development and have no `CMakeLists.txt`, so `make spike` does not apply
to them:

| Source | Question it answered | Finding |
|---|---|---|
| `s5_prefill_arena/` | Can a whole layer be staged and imported? | S5, S7 |
| `s6_arena_mulmatid/` | Does `mul_mat_id` accept an arena-backed tensor? | S6 |
| `s9_prefetch_bw/` | Is there bandwidth headroom left for prefetch? | S9 |
| `s17_miss_weight/` | How much decode I/O goes to experts that barely contribute? | S17 |
| `s18_dense_stream/` | Does streaming really do nothing for a dense model? | S18 |
| `s19_pagecache_share/` | Do decode reads actually reach the SSD? | S19 |
| `s20_io_volume_curve/` | What is a byte of decode I/O worth? | S20 |
| `s21_dense_baseline/` | The dense compute ceiling, measured not extrapolated | S21 |
| `s24_skiprank/` | S17 built end to end (patch kept here, not in src/) | S24/S25 |
| `s25_decode_ppl/` | Perplexity through the slab path rather than the arena | S24/S25 |
| `s26_memory_pressure/` | What is the freed memory worth once something uses it? | S26 |
| `s27_dense_streaming/` | Dense FFN streaming, implemented and measured | S27 |
| `s28_mtp/` | Does MTP help or hurt a streaming model? | S28 |
| `s29_dense_tuning/` | Threads, buffers, MTP n_max, batching, ubatch, PPL, stock | S29 |
| `s31_moe_mtp/` | Does MTP help a streaming MoE model? (S13's open half) | S31 |
| `s32_moe_nmax/` | Does a larger MTP draft rescue MoE? | S32 |
| `s33_plain_transformer/` | Dense streaming with no SSM in the way | S33 |
| `s34_like_for_like/` | Every cell with its baseline in the same configuration | S34 |
| `s35_gemma_matrix/` | Dense streaming on a plain transformer, whole matrix | S34 |
| `s36_moe_matrix/` | The MoE half of the same matrix | S34 |
| `s37_batch_freezone/` | Does batching pay for the bytes a dense pass reads? | S37 |
| `s38_moe_batch/` | The same sweep on MoE, where the union grows | S37 |
| `s39_batch_correctness/` | Output and perplexity across the batch sweep | S37 |

`s17` and `s18` are Python and need only `numpy` plus, for `s18`, a GGUF
header. Neither builds or runs a model:

    python3 research/spikes/s17_miss_weight/analyze.py
    python3 research/spikes/s18_dense_stream/analyze.py <dense.gguf>
