#!/usr/bin/env python3
"""
MoEStream — cross-domain analysis (§26.4 Affinity Batching, validating §32.2)

A high hit rate within a single domain does not guarantee anything when several
agents interleave different tasks: the working set grows and the cache can
collapse. That is the unverified risk raised in §32.2.

What it checks:
  1. overlap between per-domain hot sets (Jaccard, and how much the union grows)
  2. LRU hit rate when domains alternate, i.e. measured interference
  3. behaviour as the interleaving granularity changes, which is the basis for
     §26.4's "run K steps consecutively"
"""
import sys, struct, itertools
import numpy as np


def load(path):
    with open(path, 'rb') as f:
        magic, ver, n_moe, _, top_k, ntl, nth, n_layer = struct.unpack('<8I', f.read(32))
        assert magic == 0x5254534D
        rec = np.frombuffer(f.read(), dtype=np.uint16)
    per = n_moe * top_k * 2
    rec = rec[:(rec.size // per) * per].reshape(-1, n_moe, 2, top_k)
    return rec[:, :, 0, :].astype(np.int32), n_moe, top_k


def lru_hit(ids, n_expert, frac):
    """Independent LRU per layer. ids: [T, L, K]"""
    T, L, K = ids.shape
    cap = max(1, int(round(frac * n_expert)))
    hits = tot = 0
    for l in range(L):
        last = np.full(n_expert, -1, dtype=np.int64)
        resident = set()
        col = ids[:, l, :]
        for t in range(T):
            for e in col[t]:
                e = int(e); tot += 1
                if e in resident:
                    hits += 1
                else:
                    if len(resident) >= cap:
                        resident.discard(min(resident, key=lambda x: last[x]))
                    resident.add(e)
                last[e] = t
    return hits / tot


def hot_set(ids, n_expert, frac):
    """The union of the most frequent experts per layer, with layer l's expert e
    encoded as l*n_expert+e."""
    T, L, K = ids.shape
    k = max(1, int(round(frac * n_expert)))
    s = set()
    for l in range(L):
        c = np.bincount(ids[:, l, :].ravel(), minlength=n_expert)
        for e in np.argpartition(-c, k - 1)[:k]:
            s.add(l * n_expert + int(e))
    return s


def main():
    paths = sys.argv[1:]
    names = [p.split('/')[-1].split('.')[0] for p in paths]
    traces = {}
    n_expert = 0
    for n, p in zip(names, paths):
        ids, L, K = load(p)
        traces[n] = ids
        n_expert = max(n_expert, int(ids.max()) + 1)
    L = next(iter(traces.values())).shape[1]
    K = next(iter(traces.values())).shape[2]

    print('=' * 78)
    print(' MoEStream — cross-domain analysis (§26.4 / §32.2)')
    print('=' * 78)
    print(f'  domains: {names}   layers={L} top_k={K} n_expert={n_expert}')
    for n in names:
        print(f'    {n}: {traces[n].shape[0]:,} tokens')

    # ---- 1. hot set overlap ----
    print('\n--- 1. overlap between per-domain hot sets (38% cache ratio) ---')
    hs = {n: hot_set(traces[n], n_expert, 0.38) for n in names}
    size = len(next(iter(hs.values())))
    print(f'  hot set size for one domain = {size} (= 38% x {L} layers x {n_expert})')
    for a, b in itertools.combinations(names, 2):
        inter = len(hs[a] & hs[b]); union = len(hs[a] | hs[b])
        print(f'  {a:5s} ∩ {b:5s} : {inter:5d} ({100*inter/size:5.1f}%)   Jaccard={inter/union:.3f}')
    allu = set().union(*hs.values())
    print(f'  union across all domains = {len(allu)} ({len(allu)/size:.2f}x larger)')
    print(f'  -> running 3 domains at once needs {len(allu)/size:.2f}x the cache of one')

    # ---- 2. LRU hit rate when domains alternate ----
    print('\n--- 2. LRU hit rate with domains mixed (the crux of §32.2) ---')
    T = min(t.shape[0] for t in traces.values())
    Tuse = min(T, 6000)
    print(f'  using {Tuse:,} tokens per domain')
    for frac in (0.20, 0.38, 0.50):
        singles = [lru_hit(traces[n][:Tuse], n_expert, frac) for n in names]
        # Sequential: each domain run in one block, i.e. affinity batching works
        seq = np.concatenate([traces[n][:Tuse] for n in names], axis=0)
        h_seq = lru_hit(seq, n_expert, frac)
        # Interleaved: mixed with no regard for affinity
        parts = [traces[n][:Tuse] for n in names]
        inter = np.empty((Tuse * len(parts), L, K), dtype=np.int32)
        for i, p in enumerate(parts):
            inter[i::len(parts)] = p
        h_int = lru_hit(inter, n_expert, frac)
        print(f'  cache ratio {frac*100:4.0f}%: '
              f'single-domain mean {np.mean(singles)*100:5.1f}%  |  '
              f'domains sequential {h_seq*100:5.1f}%  |  '
              f'fully interleaved {h_int*100:5.1f}%  '
              f'(difference {100*(h_seq-h_int):+.1f} pt)')

    # ---- 3. interleaving granularity (§26.4's K consecutive steps) ----
    print('\n--- 3. effect of switching granularity K (§26.4 Affinity Batching) ---')
    frac = 0.38
    parts = [traces[n][:Tuse] for n in names]
    nd = len(parts)
    for Kstep in (1, 4, 16, 64, 256, 1024):
        chunks = []
        pos = 0
        while pos < Tuse:
            for p in parts:
                chunks.append(p[pos:pos + Kstep])
            pos += Kstep
        mixed = np.concatenate(chunks, axis=0)
        h = lru_hit(mixed, n_expert, frac)
        print(f'  K={Kstep:5d} consecutive tokens -> hit rate {h*100:5.1f}%')


if __name__ == '__main__':
    main()
