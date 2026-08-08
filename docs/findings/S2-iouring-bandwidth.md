# Finding S2 — io_uring is worth +1.5% in bandwidth; its value is control, not throughput

| | |
|---|---|
| Target | DESIGN.md §13.3 / §13.5 / §18.4 / §23.3 / App. F.2 |
| Date | 2026-08-03 |
| Hardware | Crucial P310 (PCIe4, DRAM-less/HMB), ext4, kernel 7.0, liburing 2.x |
| Reproduce | `make spike SPIKE=s2_iouring_bw` (needs `--security-opt seccomp=unconfined`) |

## ★ Finding 0: Docker's default seccomp blocks io_uring (as App. F.2 predicted)

```
default            : Seccomp: 2 / filters: 1  -> io_uring_setup returns EINVAL
seccomp=unconfined : Seccomp: 0               -> works
```

**The warning written in Appendix F.2 #1 reproduced exactly on real hardware.**
It has to be a check in `moestream doctor`, and the fallback (`pread_pool`) is a
necessity rather than a decoration.

## Bandwidth

### GGUF-equivalent (unaligned 1.42 MiB read-around)

| QD | Bandwidth | p50 | p99 | max |
|---:|---:|---:|---:|---:|
| 1 | 2.82 GB/s | 0.51 ms | 1.06 ms | 1.38 ms |
| **2** | **4.48 GB/s** | 0.66 ms | 1.19 ms | 1.37 ms |
| 4 | 4.46 GB/s | 1.32 ms | 1.94 ms | 2.11 ms |
| 8 | 4.46 GB/s | 2.63 ms | 3.39 ms | 4.60 ms |
| 32 | 4.46 GB/s | 10.53 ms | 17.58 ms | 20.22 ms |
| 128 | 4.41 GB/s | 41.00 ms | **95.02 ms** | 101.54 ms |

### `.msp`-equivalent (4 KiB-aligned 2 MiB records)

| QD | Bandwidth | p50 | p99 |
|---:|---:|---:|---:|
| 1 | 3.20 GB/s | 0.65 ms | 1.02 ms |
| **2** | **4.47 GB/s** | 0.92 ms | 1.55 ms |
| 8 | 4.49 GB/s | 3.69 ms | 4.90 ms |
| 128 | 4.47 GB/s | 57.28 ms | 97.67 ms |

### What io_uring's options contribute (QD=8)

| Configuration | Bandwidth | p99 |
|---|---:|---:|
| default (no flags, no registered buffers) | 4.47 GB/s | 5.18 ms |
| registered buffers only | 4.48 GB/s | **4.42 ms** |
| SINGLE_ISSUER \| DEFER_TASKRUN \| COOP_TASKRUN | 4.48 GB/s | 4.78 ms |
| the above + registered buffers (the design's recommendation) | 4.48 GB/s | 4.88 ms |

## ★ Finding 1: 4.48 GB/s is the device's limit. io_uring buys +1.5%

| Method | Bandwidth |
|---|---:|
| synchronous pread QD=1 | 1.55 GB/s |
| parallel pthread pread QD=8 | 4.42 GB/s |
| **io_uring** | **4.49 GB/s (+1.5%)** |

**The "performance" part of §13.2's case for io_uring does not hold.** The
remaining reasons are still valid but need restating:

| io_uring's real value | Explanation |
|---|---|
| asynchrony without threads | a pthread pool has threads = QD, making QD hard to vary at run time |
| **reordering by priority** | issue order can be controlled from your own priority queue (§22.8's bandwidth governor) |
| cancellation | `IORING_OP_ASYNC_CANCEL` (though §22.9 already decided not to use it) |
| CPU efficiency | registered buffers/files reduce syscalls and page pinning |
| better p99 | 5.18 → 4.42 ms with registered buffers (−15%) |

> Remove the performance claim from §13.2's comparison table and restate it as
> **"io_uring is chosen for control and CPU efficiency; bandwidth is equal to
> parallel pread"**. §23.6's statement that "io_uring is an optimization, not a
> requirement" was correct.

## ★ Finding 2: bandwidth saturates at QD=2; beyond that only latency suffers

```
QD=2   : 4.48 GB/s, p99 1.19 ms
QD=128 : 4.41 GB/s, p99 95.02 ms   <- same bandwidth, 80x the latency
```

§13.5 had already been revised from 64 to 8; **measurement says smaller still,
QD=2–4**.

### → Design change: separate the "software queue" from the "device queue"

§13.5 called for keeping four layers' worth (24 MB) in flight, which QD=2
(4 MiB) cannot satisfy. Reconciling them requires splitting the concept:

| Concept | Value | Role |
|---|---|---|
| **software queue depth** | tens to hundreds | the prefetch plan's inventory, held with priorities |
| **device queue depth (real QD)** | **2–4** | how many SQEs are actually submitted; protects p99 |

`IoGovernor` controls only the latter; the Prefetch Engine owns the former.
**That separation allows deep prefetching and low p99 at the same time.**

| Change | Old | New |
|---|---|---|
| initial device QD | 8 | **2** |
| maximum device QD | 32 | **8** |
| software queue limit | (undefined) | **256 entries** |

## ★ Finding 3: 4 KiB alignment in `.msp` does not improve bandwidth (+0.1%)

The read-around overhead is only **0.27%**, and the bandwidth difference is
within noise.

**The bandwidth-related part of §18.4's case for paying 2.7% more disk for
2 MiB alignment is rejected.**

The other `.msp` advantages remain:

| Advantage | Status |
|---|---|
| one expert = one `readv` (GGUF needs three) | **valid**. A third of the IOPS |
| no bounce buffer (zero-copy) | **valid**. Essential to §14.4's zero-copy path |
| physical layout by popularity (§18.3) | valid (unverified) |
| alignment with HugePage boundaries | valid |
| **bandwidth gain from 4 KiB alignment** | **rejected** |

→ `--pack-align` could default to **4 KiB** instead of 2 MiB (disk overhead
2.7% → nearly 0%). Whether HugePage alignment is worth it should be measured
separately.

## Updated required hit rate

```
BW = 4.49 GB/s, B_act = 455 MiB/token, t_c = 43.7 ms
-> to stay within a 20% slowdown, h >= 50.7%

M0-2 measured h(38%) = 82.4-91.5%   ->  still a wide margin
```

## Remaining work

| ID | Item |
|---|---|
| S2b | can RAID0 across two NVMe drives reach 8–9 GB/s? (corresponding to §2.3's PCIe5 row) |
| S2c | how bandwidth behaves under thermal throttling (§32.6) |
