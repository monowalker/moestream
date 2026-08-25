#!/usr/bin/env python3
"""
S17 follow-up -- is a RANK-based skip as good as a WEIGHT-based one?

Why this matters for implementation, not just for accuracy:

  The remap op receives `selected_experts`, produced by ggml_argsort_top_k,
  whose output is ordered by descending probability. So the position k within
  top_k already IS the rank, and a rule of the form "skip a miss at k >= K"
  needs no new tensor, no change to graph construction, and no extra CPU<->GPU
  sync. It is a predicate change inside the existing remap_exec.

  A weight-based rule (w < tau) needs the router weights tensor, which in
  llama.cpp is only finalised well after the `ffn_moe_topk` anchor the patch
  attaches to -- so it requires moving where build_remap is called, i.e. a
  graph-order change. This project has already been bitten once by graph
  reordering (Expert Sweep, RESULTS.md §9).

  If rank does nearly as well as weight, the cheap and safe rule is the one to
  build.

Reports both rules on the same miss stream so they can be compared directly.
"""
import sys, os, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, '..', '..', 'bench')
sys.path.insert(0, HERE)
from analyze import load, lru_miss_mask     # noqa: E402


def main():
    fracs = [0.25, 0.40]
    traces = sys.argv[1:] or [os.path.join(BENCH, n) for n in ('en.trace', 'ja.trace', 'code.trace')]
    for path in traces:
        ids, w = load(path)
        T, L, K = ids.shape
        n_expert = int(ids.max()) + 1
        wn = w / np.maximum(w.sum(axis=2, keepdims=True), 1e-9)
        total_mass = wn.sum()
        # rank: position within top_k. Verify argsort ordering actually holds.
        desc = (np.diff(wn, axis=2) <= 1e-6).mean()
        print('=' * 78)
        print(f'{os.path.basename(path)}  tokens={T:,} layers={L} top_k={K}')
        print(f'  weights are in descending rank order for {desc*100:.2f}% of adjacent pairs')

        for frac in fracs:
            cap = max(1, int(round(frac * n_expert)))
            miss = lru_miss_mask(ids, n_expert, cap)
            n_miss = int(miss.sum())
            print(f'\n  frac={frac:.2f} ({cap} slots)  misses/token {n_miss/T:.2f}')
            print(f'    {"rule":>22} {"bytes saved":>12} {"mass lost":>10} {"bytes/mass":>11}')
            # rank rule: skip a miss whose rank k >= Kmin
            rank = np.broadcast_to(np.arange(K), ids.shape)
            for kmin in range(K - 1, K - 5, -1):
                sel = miss & (rank >= kmin)
                bs = int(sel.sum()) / max(n_miss, 1)
                ml = wn[sel].sum() / total_mass
                print(f'    {"rank >= %d" % kmin:>22} {bs*100:11.2f}% {ml*100:9.3f}% '
                      f'{(bs/ml if ml else 0):10.1f}x')
            # weight rule, for comparison
            for tau in (0.06, 0.08, 0.10):
                sel = miss & (wn < tau)
                bs = int(sel.sum()) / max(n_miss, 1)
                ml = wn[sel].sum() / total_mass
                print(f'    {"weight < %.2f" % tau:>22} {bs*100:11.2f}% {ml*100:9.3f}% '
                      f'{(bs/ml if ml else 0):10.1f}x')
        print()


if __name__ == '__main__':
    main()
