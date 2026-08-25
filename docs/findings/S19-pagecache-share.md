# Finding S19 — 98.7% of Ornith's decode "I/O" never reaches the SSD

| | |
|---|---|
| Target | `RESULTS.md` §10.12's 12.45 ms/token of decode I/O — is it disk time? |
| Date | 2026-08-22 |
| Verdict | **No. 98.7% of it is served from the page cache.** "I/O" in this system is two different costs wearing one name |
| Measured on | `research/spikes/s19_pagecache_share/measure.sh`, Ornith-1.0-35B-UD-IQ4_NL, `frac=0.25`, 4,000-token window |

## The arithmetic that started it

The project's own published numbers do not close:

```
misses/token at frac 0.25 = 320 fetches x 18.13%  = 58.0 experts
bytes/token               = 58.0 x 1.448 MiB      = 86.5 MB
measured I/O (§10.12)     = 12.45 ms              -> 6.95 GB/s
measured device ceiling (§4.1, O_DIRECT saturated) -> 4.46-4.49 GB/s
```

Reading at 1.55x what the device can deliver is not possible, so the reads had
to be coming from somewhere else. `pread` is buffered here (`open(..., O_RDONLY)`,
no `O_DIRECT`), the expert set is 14.48 GiB and the host has 30.6 GiB, so the
page cache was the obvious suspect. The runtime's own `[prefetch]` decision
comment already reasons about exactly this fit — but nobody had measured it.

## Method

An instrument that does not depend on the runtime's timing at all:
`/proc/<pid>/io`'s `read_bytes`, which counts **only bytes actually fetched from
the storage layer** — a page-cache hit never appears there. Read inside the
container, so it is this process alone. Cross-checked against the device-wide
`/sys/block/nvme0n1/stat`, which agreed to within 0.5%, confirming no other
container polluted the window.

Requested bytes come from the runtime's own `[stats]` miss counter, so both
sides of the ratio are counted rather than modelled.

## Result

4,000 tokens of steady-state decode, page cache in whatever state ordinary use
leaves it:

| | | per token |
|---|---:|---:|
| cache misses | 229,547 | 57.4 |
| **bytes requested** | **324.59 GiB** | **83.1 MiB** |
| bytes actually read from the device (this process) | 4.19 GiB | 1.1 MiB |
| bytes actually read from the device (device-wide) | 4.21 GiB | 1.1 MiB |

```
==> page-cache share of decode reads : 98.7%
==> device share                     :  1.3%
```

83.1 MiB/token against the 86.5 MB predicted above — the model of what is being
requested was right. What was wrong was where it comes from.

## The same code, the other way round

The instance that had been running on this machine before the measurement was
**Ornith-1.5-35B-Q4_K_M** — same architecture, different training and quant, and
crucially a different size relative to RAM. The runtime's own bandwidth
auto-tuner, same code and same machine:

| | expert set | vs ~17 GiB page-cache headroom | `[io]` effective bandwidth |
|---|---:|---|---:|
| Ornith-1.0 UD-IQ4_NL | 14.48 GiB | fits | **9.30 GB/s** (2.08x the device) |
| Ornith-1.5 Q4_K_M | 18.65 GiB | does not | **3.72 GB/s** (0.83x the device) |

A 2.5x difference in effective read bandwidth, from the same code path, with
only one variable: whether the expert set fits in RAM.

## What follows

**1. "I/O" is two different costs sharing a name.** Under the page-cache ceiling
it is `copy_to_user` into GPU-visible memory, bounded by memory bandwidth. Over
it, it is the NVMe. Which one is being paid decides which optimisations are
worth anything, and the answer changes with the model, not with the code.

**2. `O_DIRECT` would make Ornith-1.0 slower**, by forcing 83 MiB/token of real
device reads in place of 1.1. Worth stating explicitly so nobody tries it as an
obvious win.

**3. The memory claim needs a sentence.** Ornith-1.0's honest figure is
7.44 GiB of hard residency **plus** a working set the kernel keeps in
reclaimable page cache. That is still a real result — reclaimable memory is
categorically different from a GTT allocation the GPU driver will not give back
— but a reader on a 16 GiB machine will not reproduce these numbers, and the
docs currently do not warn them.

**4. It explains §10.12's puzzle without inventing anything.** That section found
the built-in blocking-time instrument and the difference method disagreeing by
1.6x and concluded, correctly, that they measure different things. Both are
consistent with a read path that is mostly a memcpy: the blocking time is the
copy, and the extra 0.6x is the cache and memory pressure that copy leaves
behind.

**5. It sharpens where byte-reduction pays.** On a model that fits (Ornith-1.0)
the bytes are cheap and B1's ceiling is small. On one that does not (Laguna,
Ornith-1.5, gpt-oss-120b) every byte is a real device read. **The models this
project exists for are all in the second group.**

## Caveats

- One model, one machine, one page-cache state. The 98.7% is a property of
  "14.48 GiB of experts on a 30.6 GiB host", not of MoEStream.
- The window follows a warm-up that itself populated the cache. A cold-start
  measurement would look completely different, and is not what steady-state
  serving looks like.
- `read_bytes` counts block-layer reads, so readahead is included. That makes
  the device share an over-estimate, not an under-estimate: the true figure is
  at or below 1.3%.

> Reproduce: `research/spikes/s19_pagecache_share/measure.sh 4000`
> Related: `V3-idea-analysis.md` C1, `RESULTS.md` §4.1, §10.12, §10.14
