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
