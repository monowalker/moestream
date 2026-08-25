#!/usr/bin/env python3
"""
Spike S17 -- how much of decode I/O is spent on experts that barely contribute?

Context
-------
DESIGN.md §11.6 defines a `soft` quality mode that skips experts whose router
weight is below tau.  Finding M0-2 measured the weight distribution and
concluded the mechanism was near-inert: tau=0.02 touches 0.1% of selections,
and tau=0.06 costs 2.1-2.9% of the weight mass while skipping only 5-7%.
That evaluation applied the threshold to *every* selection.

This spike asks a different question.  Since RESULTS.md §10.12 the dominant
decode cost is I/O, and I/O is paid **only on a cache miss**.  A hit costs
nothing, so there is no reason to skip it.  The interesting quantity is
therefore the joint distribution:

    P(router weight < tau  AND  the selection missed the cache)

If low-weight experts are over-represented among misses -- which is what one
would expect, since they are chosen more rarely and so are less likely to be
resident -- then skipping *misses only* buys far more bytes per unit of lost
weight mass than the blanket rule M0-2 rejected.

Output: for each tau, the share of read bytes removed and the share of router
weight mass lost, plus the resulting bytes-per-weight exchange rate.

No model, GPU or build required; runs on the recorded traces in research/bench.
"""
import sys, os, struct, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, '..', '..', 'bench')


def load(path):
    with open(path, 'rb') as f:
        magic, ver, n_moe, _pad, top_k, ntl, nth, n_layer = struct.unpack('<8I', f.read(32))
        if magic != 0x5254534D:
            raise ValueError('magic mismatch: ' + path)
        rec = np.frombuffer(f.read(), dtype=np.uint16)
    per_tok = n_moe * top_k * 2
    usable = (rec.size // per_tok) * per_tok
    rec = rec[:usable].reshape(-1, n_moe, 2, top_k)
    ids = rec[:, :, 0, :].astype(np.int32)
    w = np.frombuffer(rec[:, :, 1, :].copy().tobytes(),
                      dtype=np.float16).reshape(ids.shape).astype(np.float32)
    return ids, w


def lru_miss_mask(ids, n_expert, cap):
    """Per-layer independent LRU.  Returns a boolean mask, True where the
    selection was a miss (i.e. where bytes had to be read)."""
    T, L, K = ids.shape
    miss = np.zeros(ids.shape, dtype=bool)
    for l in range(L):
        last = np.full(n_expert, -1, dtype=np.int64)
        resident = set()
        col = ids[:, l, :]
        for t in range(T):
            for k in range(K):
                e = int(col[t, k])
                if e in resident:
                    pass
                else:
                    miss[t, l, k] = True
                    if len(resident) >= cap:
                        victim = min(resident, key=lambda x: last[x])
                        resident.discard(victim)
                    resident.add(e)
                last[e] = t
    return miss


def main():
    fracs = [0.15, 0.25, 0.40]
    taus = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    traces = sys.argv[1:] or [os.path.join(BENCH, n) for n in ('en.trace', 'ja.trace', 'code.trace')]

    for path in traces:
        ids, w = load(path)
        T, L, K = ids.shape
        n_expert = int(ids.max()) + 1
        # The trace records post-softmax router weights; normalise per token/layer
        # so "weight mass" is comparable across rows.
        wsum = w.sum(axis=2, keepdims=True)
        wn = w / np.maximum(wsum, 1e-9)
        total_mass = wn.sum()

        print('=' * 74)
        print(f'{os.path.basename(path)}  tokens={T:,} layers={L} top_k={K} n_expert={n_expert}')
        print(f'  mean weight by rank: ' +
              ' '.join(f'{v:.3f}' for v in np.sort(wn.reshape(-1, K), axis=1)[:, ::-1].mean(axis=0)))

        for frac in fracs:
            cap = max(1, int(round(frac * n_expert)))
            miss = lru_miss_mask(ids, n_expert, cap)
            n_sel = miss.size
            n_miss = int(miss.sum())
            hr = 1.0 - n_miss / n_sel
            print(f'\n  frac={frac:.2f} ({cap} slots)  LRU hit rate {hr*100:.2f}%  '
                  f'misses/token {n_miss/T:.2f}')
            # Is low weight over-represented among misses?
            mw = wn[miss]
            print(f'    mean weight of a HIT  {wn[~miss].mean():.4f}')
            print(f'    mean weight of a MISS {mw.mean():.4f}   '
                  f'(ratio {mw.mean()/wn[~miss].mean():.2f}x)')
            print(f'    {"tau":>6} {"sel skipped":>12} {"of misses":>10} '
                  f'{"bytes saved":>12} {"mass lost":>10} {"bytes/mass":>11}')
            for tau in taus:
                sel = miss & (wn < tau)
                n_sk = int(sel.sum())
                bytes_saved = n_sk / max(n_miss, 1)      # bytes are proportional to misses
                mass_lost = wn[sel].sum() / total_mass
                rate = bytes_saved / mass_lost if mass_lost > 0 else float('inf')
                print(f'    {tau:6.2f} {n_sk/n_sel*100:11.2f}% {n_sk/max(n_miss,1)*100:9.2f}% '
                      f'{bytes_saved*100:11.2f}% {mass_lost*100:9.3f}% {rate:10.1f}x')
            # Reference: the blanket rule M0-2 evaluated (skip regardless of residency)
            print(f'    -- blanket (M0-2 rule, skip hits too) --')
            for tau in (0.06, 0.10):
                sel = (wn < tau)
                n_sk = int(sel.sum())
                bs = float((miss & sel).sum()) / max(n_miss, 1)
                ml = wn[sel].sum() / total_mass
                print(f'    {tau:6.2f} {n_sk/n_sel*100:11.2f}% {"-":>9}  '
                      f'{bs*100:11.2f}% {ml*100:9.3f}% '
                      f'{(bs/ml if ml else float("inf")):10.1f}x')
        print()


if __name__ == '__main__':
    main()
