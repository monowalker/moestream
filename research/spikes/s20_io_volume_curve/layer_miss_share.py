#!/usr/bin/env python3
"""
S20 companion -- convert drop_from=N into the fraction of read bytes it removes.

S20 sweeps MOESTREAM_DROP_FROM, which stops fetching misses at layers >= N.
Reading the resulting curve as "decode time vs read volume" assumes misses are
spread evenly over layers. They are not: early layers route more repetitively,
so they miss less, and the layer index is a poor proxy for bytes.

This computes the real mapping from the recorded traces, so the S20 timings can
be plotted against measured volume without re-running anything on the GPU.

Every miss costs the same bytes (gate+up+down for one expert), so the share of
misses in layers < N is exactly the share of read bytes that survives
drop_from=N.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S17 = os.path.join(HERE, '..', 's17_miss_weight')
sys.path.insert(0, S17)
from analyze import load, lru_miss_mask   # noqa: E402

BENCH = os.path.join(HERE, '..', '..', 'bench')


def main():
    frac = 0.25
    for name in ('en.trace', 'ja.trace', 'code.trace'):
        path = os.path.join(BENCH, name)
        ids, _w = load(path)
        T, L, K = ids.shape
        n_expert = int(ids.max()) + 1
        cap = max(1, int(round(frac * n_expert)))
        miss = lru_miss_mask(ids, n_expert, cap)
        per_layer = miss.sum(axis=(0, 2)).astype(float)      # misses per layer
        total = per_layer.sum()
        print(f'{name}  frac={frac}  layers={L}  total misses/token {total/T:.1f}')
        print(f'  {"drop_from":>10} {"layers kept":>12} {"reads kept":>11} {"reads removed":>14}')
        for n in (0, 10, 20, 30, L):
            kept = per_layer[:n].sum() / total
            print(f'  {n:10d} {f"0..{n-1}" if n else "none":>12} '
                  f'{kept*100:10.1f}% {(1-kept)*100:13.1f}%')
        # per-layer miss rate, to show why the mapping is not linear
        rate = per_layer / (T * K)
        print(f'  miss rate by layer decile: ' +
              ' '.join(f'{rate[i]*100:.0f}%' for i in range(0, L, max(1, L//10))))
        print()


if __name__ == '__main__':
    main()
