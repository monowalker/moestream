# Finding S5 — Feasibility of an anonymous staging arena for prefill

| | |
|---|---|
| Target | an alternative to v2 option A (the first proposal in `docs/findings/V2-idea-analysis.md`) |
| Date | 2026-08-04 |
| Verdict | **feasible. Three gates passed. ggml integration unverified** |
| Code | `research/spikes/s5_prefill_arena/{arena_import.c, arena_read.c, read.comp}` |
| Environment | Ryzen 7 8745HS / Radeon 780M (RADV PHOENIX) / Crucial P310 |
| | BIOS iGPU allocation 512 MB, `ttm.pages_limit=6160384` (GTT 23.5 GiB) |

## Background

Prefill is slow (27–44 tok/s measured) because the slab constraint forces
`ubatch=8`, and of that degradation **3.9x comes from ubatch and only 1.25x from
MoEStream** (finding N4).

V2's first proposal was to mmap the GGUF during prefill and import it into
Vulkan, treating page cache as soft memory. That was **refused by RADV with
`VkResult -13`** and already rejected (`RESULTS.md` §11).

This finding settles the reason for that rejection and verifies the conditions
for an alternative: **an anonymous staging arena**.

```
during prefill only, read one layer's full expert set
(256 x 3 x 450560 B = 330 MiB) into an anonymous arena with O_DIRECT
-> run that layer's FFN at a large ubatch
-> reuse the same arena for the next layer (rotation)
```

The 7.9 GiB resident slab is unchanged, and the increment caps out at 330 MiB
(660 MiB if double buffered).

## Gate 1: can an anonymous arena be imported? → **passed**

```
=== importing an anonymous arena (size scaling) ===
  64 MiB (control)     * success  memtype=5 heap=0  [HOST_VISIBLE HOST_COHERENT HOST_CACHED]
  330 MiB = one layer  * success  memtype=5 heap=0  [HOST_VISIBLE HOST_COHERENT HOST_CACHED]
  660 MiB = doubled    * success  memtype=5 heap=0  [HOST_VISIBLE HOST_COHERENT HOST_CACHED]
  1320 MiB = 4 layers  * success  memtype=5 heap=0  [HOST_VISIBLE HOST_COHERENT HOST_CACHED]
```

`maxMemoryAllocationSize = 4.00 GiB` caps a single allocation. 660 MiB is needed,
so there is room.

### Control: the reason file-backed mmap is refused is now settled

```
  330 MiB SHARED      failed (VkResult -13, memoryTypeBits=0x20)
  330 MiB PRIVATE     failed (VkResult -13, memoryTypeBits=0x20)
```

`memoryTypeBits=0x20` is bit 5 — **the same memtype 5 that succeeded for
anonymous memory**. So it is not a memory-type mismatch; the driver is refusing
**the fact that the pages are file-backed**. `VkResult -13` =
`VK_ERROR_UNKNOWN` is typical of an unexpected error from a kernel ioctl, which
is consistent with amdgpu's userptr accepting only anonymous memory.

> `RESULTS.md` §11's "RADV refuses with `VkResult -13`" is correct, and the cause
> is confirmed as "file-backed VMAs are refused".
> **Showing page cache directly to the GPU is impossible in principle on
> amdgpu.**

## Gate 2: effective SSD → arena bandwidth → **passed (4.48 GB/s)**

**O_DIRECT was mandatory** to avoid mistaking page cache throughput for the real
thing (preventing a recurrence of the error made once in `RESULTS.md` §13.3).

| Threads | Bandwidth |
|---:|---:|
| 1 | 4.36 GB/s |
| 2 | 4.42 GB/s |
| 4 | 4.40 GB/s |
| 8 | 4.48 GB/s |
| 16 | 4.49 GB/s |

Exactly matching finding S2's device limit of 4.48 GB/s.
**One thread already saturates it** (sequential large-block reads).

```
all 40 layers of expert data = 40 x 330 MiB = 12.89 GiB
I/O per forward pass = 12.89 GiB / 4.48 GB/s = 3.09 s
```

## Gate 3: bandwidth as seen from the GPU → **passed (though 2.6x slower)**

Imported memory comes back as `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED` (heap 0),
which is **not** the current slab's `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`
(heap 1). Compared with the same compute shader (a grid-stride XOR convolution
reading everything).

| Condition | Memory type | Bandwidth |
|---|---|---:|
| **A: imported anonymous arena** | memtype 5 / heap 0 / HOST_CACHED | **30.98 GB/s** |
| B: DEVICE_LOCAL only | memtype 0 / heap 1 | 78.18 GB/s |
| C: DEVICE_LOCAL\|HOST_VISIBLE (the current slab) | memtype 3 / heap 1 | 80.58 GB/s |

```
A / C = 0.38      A / B = 0.40
```

**The arena runs at 38% of the current slab path's speed.**

### But it is not the constraint

Per forward pass:

| Path | Time for 12.89 GiB |
|---|---:|
| SSD → arena (4.48 GB/s) | **3.09 s** |
| arena → GPU (30.98 GB/s) | 0.45 s |
| current slab → GPU (80.58 GB/s) | 0.17 s |

**GPU reads are 7x faster than the SSD.** Even 2.6x slower memory leaves the SSD
as the constraint. Gate 3 passes as "slow, but it does not matter".

## Speed estimate

Baseline prefill (no streaming) at ub=512 is 295.1 tok/s = 1.74 s per pass
(finding N4). The arena path adds +0.28 s for the GPU read difference.

| ubatch | I/O (s) | compute (s) | bound by | prefill |
|---:|---:|---:|---|---:|
| 512 | 3.09 | 2.02 | I/O | **166 tok/s** |
| 1024 | 3.09 | 3.75 | compute | **273 tok/s** |
| 2048 | 3.09 | 7.2 | compute | ~285 tok/s |

Against the current 27–44 tok/s that is **4–7x**, consistent with V2's estimate
of ≈250–295 tok/s.

For real use (re-prefilling 20k tokens, measured at 598 s):

| | Passes | Time |
|---|---:|---:|
| current (ub=8) | — | 598 s |
| arena ub=512 | 40 | 124 s |
| arena ub=1024 | 20 | 75 s |

## Memory budget

| | Current | With the arena |
|---|---:|---:|
| resident slab (decode) | 7.9 GiB | **7.9 GiB (unchanged)** |
| prefill arena | — | +330 MiB (+660 MiB if doubled) |
| compute buffer (ubatch increment) | — | **unmeasured** |

**This is a different kind of trade from raising `MOESTREAM_MAX_UBATCH`
(7.9 → 13.5 GiB).** The arena does not add slots, so it lifts the prefill ubatch
constraint without moving the resident memory ceiling. That is the whole point.

## Unverified risks

| # | Item | Impact |
|---|---|---|
| 1 | **ggml integration** — can `mul_mat_id` index 256 expert tensors on the arena | **resolved → bit-identity confirmed in finding S6** |
| 2 | how much the compute buffer grows at ubatch 512–1024 | collides directly with the memory constraint. **Unverified, the biggest unknown** |
| 3 | real expert reads are 450 KB scattered accesses; this was sequential large blocks | 4.48 GB/s is an upper bound; the real rate is lower |
| 4 | SSD writes and GPU reads hitting the same arena concurrently | double buffering is essential; bandwidth contention unmeasured |
| 5 | the shader is an XOR convolution, a different access pattern from matmul | **resolved → S6 measured 1.09–1.13x. The 0.38 here was more than 20x pessimistic** |

> **Gate 3 in this finding erred in the pessimistic direction.** An XOR
> convolution is the worst case of zero weight reuse; in a real `mul_mat_id` the
> expert weights are reused across multiple tokens and the GPU's caches absorb
> the latency. See finding S6.

## Conclusion

**All three gates passed. The approach is feasible.**

- importing an anonymous arena works up to 1320 MiB
- the reason file-backed mmap cannot be used is settled (an amdgpu constraint,
  not avoidable)
- the SSD is the constraint, so the arena being 2.6x slower does not matter
- prefill can plausibly be made 4–7x faster without adding resident memory

The next thing to eliminate is **risk 1 (ggml integration)**. Nothing else
matters if that fails, so it needs a minimal verification before implementation
begins.
