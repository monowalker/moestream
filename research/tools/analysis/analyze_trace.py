#!/usr/bin/env python3
"""
MoEStream — expert-activation trace analysis (M0-2, the go/no-go decision)

Quantifies DESIGN.md §32.1's largest risk -- that expert activation may not be
skewed enough -- from a measured trace.

What it computes:
  1. Miss Ratio Curve       : hit rate h(f) as a function of cache ratio f
                              (the most important output), both for an oracle
                              (top-frequency experts pinned) and for LRU /
                              an S3-FIFO approximation
  2. Zipf parameter s       : a measure of skew (tests the §21.5 assumption)
  3. per-layer entropy      : the basis for §12.5's per-layer quotas
  4. reuse between consecutive tokens : the ceiling for predictor P1
                              (temporal locality)
  5. cross-layer coactivation mutual information : the ceiling for predictor P5
  6. router weight distribution : whether §11.6's weight-threshold skipping is sound
  7. go/no-go               : is the required hit rate (51% by default) met?
"""
import sys, os, struct, math, json
import numpy as np
from collections import Counter

# Defaults taken from measurement (Finding S1)
DEFAULT_B_ACT_MIB = 455.0     # active expert bytes per token
DEFAULT_TC_MS     = 43.7      # compute time per token when fully resident
DEFAULT_BW_GBPS   = 4.46      # effective bandwidth, O_DIRECT at QD=8
DEFAULT_SLACK     = 1.20      # slowdown we are willing to accept (20%)


def load(path):
    with open(path, 'rb') as f:
        hdr = struct.unpack('<8I', f.read(32))
        magic, ver, n_moe, _, top_k, ntl, nth, n_layer = hdr
        if magic != 0x5254534D:
            raise ValueError('magic mismatch')
        n_token = ntl | (nth << 32)
        rec = np.frombuffer(f.read(), dtype=np.uint16)
    per_tok = n_moe * top_k * 2          # ids + weights(f16)
    usable = (rec.size // per_tok) * per_tok
    rec = rec[:usable].reshape(-1, n_moe, 2, top_k)
    ids = rec[:, :, 0, :].astype(np.int32)                 # [T, L, K]
    w   = rec[:, :, 1, :].view(np.uint16)
    w   = np.frombuffer(w.tobytes(), dtype=np.float16).reshape(ids.shape).astype(np.float32)
    return ids, w, n_moe, top_k, n_layer


def miss_ratio_curve(ids, n_expert, fracs):
    """Oracle hit rate with the most frequent experts pinned, allocated per layer."""
    T, L, K = ids.shape
    out = []
    # Per-layer frequencies
    freq = np.zeros((L, n_expert), dtype=np.int64)
    for l in range(L):
        c = np.bincount(ids[:, l, :].ravel(), minlength=n_expert)
        freq[l] = c
    total = freq.sum()
    for f in fracs:
        per_layer = max(1, int(round(f * n_expert)))
        hits = 0
        for l in range(L):
            top = np.argpartition(-freq[l], per_layer - 1)[:per_layer]
            hits += freq[l][top].sum()
        out.append(hits / total)
    return out


def lru_hit_rate(ids, n_expert, frac):
    """Hit rate from simulating an independent LRU cache per layer."""
    T, L, K = ids.shape
    cap = max(1, int(round(frac * n_expert)))
    hits = tot = 0
    for l in range(L):
        # Simple LRU, tracked by last access time
        last = np.full(n_expert, -1, dtype=np.int64)
        resident = set()
        for t in range(T):
            for e in ids[t, l, :]:
                e = int(e)
                tot += 1
                if e in resident:
                    hits += 1
                else:
                    if len(resident) >= cap:
                        victim = min(resident, key=lambda x: last[x])
                        resident.discard(victim)
                    resident.add(e)
                last[e] = t
    return hits / max(tot, 1)


def zipf_fit(freq):
    """Estimate the Zipf exponent s by fitting rank vs frequency in log-log space."""
    f = np.sort(freq[freq > 0])[::-1].astype(np.float64)
    if f.size < 10:
        return float('nan')
    r = np.arange(1, f.size + 1, dtype=np.float64)
    lo, hi = 1, f.size
    A = np.polyfit(np.log(r[lo:hi]), np.log(f[lo:hi]), 1)
    return -A[0]


def main():
    path = sys.argv[1]
    bw   = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BW_GBPS
    ids, w, n_moe, top_k, n_layer = load(path)
    T, L, K = ids.shape
    n_expert = int(ids.max()) + 1

    print('=' * 78)
    print(' MoEStream — expert activation distribution (M0-2)')
    print('=' * 78)
    print(f'  tokens={T:,}  MoE layers={L}  top_k={K}  estimated n_expert={n_expert}')
    print(f'  total activation events = {T*L*K:,}')

    # ---- 1. overall distribution and Zipf ----
    flat = ids.reshape(-1, K).ravel()
    gfreq = np.bincount(flat, minlength=n_expert)
    print('\n--- 1. overall expert usage distribution ---')
    print(f'  experts never used : {(gfreq==0).sum()} / {n_expert}')
    sh = np.sort(gfreq)[::-1]
    for p in (0.05, 0.10, 0.20, 0.38, 0.50):
        k = max(1, int(p * n_expert))
        print(f'  the top {p*100:4.0f}% of experts account for {100*sh[:k].sum()/sh.sum():5.1f}% of activations')
    s_hat = zipf_fit(gfreq)
    print(f'  Zipf exponent s = {s_hat:.3f}   (§21.5 assumes s >= 0.45 meets the target)')

    # ---- 2. Miss Ratio Curve ----
    print('\n--- 2. Miss Ratio Curve (allocated independently per layer) ---')
    fracs = [0.05, 0.10, 0.20, 0.30, 0.38, 0.50, 0.60, 0.75]
    orc = miss_ratio_curve(ids, n_expert, fracs)
    print('   cache ratio |  oracle (top-frequency pinned)  |  LRU')
    lru_at = {}
    for f, h in zip(fracs, orc):
        lr = ''
        if f in (0.20, 0.38, 0.50):
            v = lru_hit_rate(ids, n_expert, f)
            lru_at[f] = v
            lr = f'{v*100:6.1f}%'
        print(f'   {f*100:9.0f}% |  {h*100:18.1f}%  |  {lr:>10s}')

    # ---- 3. per-layer entropy ----
    print('\n--- 3. routing concentration per layer (basis for §12.5 quotas) ---')
    ent = []
    for l in range(L):
        c = np.bincount(ids[:, l, :].ravel(), minlength=n_expert).astype(np.float64)
        p = c / c.sum()
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    ent = np.array(ent)
    emax = math.log2(n_expert)
    print(f'  maximum entropy = {emax:.2f} bits (perfectly uniform)')
    print(f'  mean {ent.mean():.2f} bits  min {ent.min():.2f} (layer {int(ent.argmin())})  max {ent.max():.2f} (layer {int(ent.argmax())})')
    q = [0, L//4, L//2, 3*L//4, L-1]
    print('  by layer: ' + '  '.join(f'L{i}={ent[i]:.2f}' for i in q))
    if ent[:L//2].mean() > ent[L//2:].mean():
        print('  -> shallow layers are more diffuse, deep layers more specialized (matches the §12.5 prediction)')
    else:
        print('  -> deep layers are more diffuse (the opposite of the §12.5 prediction; quotas still help, but in reverse)')

    # ---- 4. reuse between consecutive tokens (the ceiling for P1) ----
    print('\n--- 4. expert reuse between consecutive tokens (ceiling for predictor P1) ---')
    reuse = []
    for l in range(L):
        a = ids[:-1, l, :]; b = ids[1:, l, :]
        inter = np.array([len(set(x) & set(y)) for x, y in zip(a[:2000], b[:2000])])
        reuse.append(inter.mean() / K)
    reuse = np.array(reuse)
    print(f'  mean {reuse.mean()*100:.1f}%  (min {reuse.min()*100:.1f}% / max {reuse.max()*100:.1f}%)')
    print(f'  -> prefetching the previous token\'s selection verbatim would hit {reuse.mean()*100:.1f}%')

    # ---- 5. router weight distribution (soundness of the §11.6 threshold) ----
    print('\n--- 5. router weight distribution (§11.6 weight-threshold skipping) ---')
    ws = np.sort(w.reshape(-1, K), axis=1)[:, ::-1]   # descending
    mean_w = ws.mean(axis=0)
    print('  mean weight by rank: ' + ' '.join(f'#{i+1}={mean_w[i]:.3f}' for i in range(K)))
    tail = mean_w[-2:].sum() / mean_w.sum()
    print(f'  the bottom two ranks account for {tail*100:.1f}% of total weight')
    for tau in (0.02, 0.06):
        frac = (w < tau).mean()
        mass = w[w < tau].sum() / w.sum()
        print(f'  w < {tau}: {frac*100:5.1f}% of selections / {mass*100:5.2f}% of weight mass'
              f'  -> skipping them loses {mass*100:.2f}% of the information')

    # ---- 6. go/no-go ----
    print('\n' + '=' * 78)
    print(' go/no-go decision (§34.2)')
    print('=' * 78)
    b_act = DEFAULT_B_ACT_MIB * 1048576
    budget = bw * 1e9 * (DEFAULT_SLACK * DEFAULT_TC_MS / 1000.0)
    need = max(0.0, 1.0 - budget / b_act)
    print(f'  B_act={DEFAULT_B_ACT_MIB:.0f} MiB/token  t_c={DEFAULT_TC_MS} ms  BW={bw:.2f} GB/s')
    print(f'  -> hit rate needed to stay within a {(DEFAULT_SLACK-1)*100:.0f}% slowdown: h >= {need*100:.1f}%')
    h38_oracle = orc[fracs.index(0.38)]
    h38_lru    = lru_at.get(0.38, float('nan'))
    print(f'  measured h(38%) : oracle {h38_oracle*100:.1f}%  /  LRU {h38_lru*100:.1f}%')
    verdict = 'GO' if h38_lru >= need else ('CONDITIONAL' if h38_oracle >= need else 'NO-GO')
    print(f'\n  verdict: {verdict}')
    if verdict == 'GO':
        print('    plain LRU already meets the required hit rate; the design has ample margin.')
    elif verdict == 'CONDITIONAL':
        print('    LRU falls short, but a frequency-based PINNED set (§12.6) can reach it.')
        print('    -> implementing S3-FIFO + PINNED (§12.3) becomes a hard requirement.')
    else:
        print('    caching alone is insufficient; §32.1 mitigations 1-3 are required:')
        print('      1) improve predictive prefetch accuracy (P1 ceiling %.0f%%)' % (reuse.mean()*100))
        print('      2) reduce miss bytes with precision tiers (§18.6)')
        print('      3) skip low-weight experts in soft mode')

    # ---- output ----
    res = {
        'tokens': int(T), 'n_moe_layer': int(L), 'top_k': int(K), 'n_expert': int(n_expert),
        'zipf_s': float(s_hat),
        'mrc_oracle': {str(f): float(h) for f, h in zip(fracs, orc)},
        'mrc_lru': {str(k): float(v) for k, v in lru_at.items()},
        'entropy_per_layer': ent.tolist(),
        'reuse_prev_token': reuse.tolist(),
        'mean_weight_by_rank': mean_w.tolist(),
        'required_hit_rate': float(need), 'bw_gbps': bw, 'verdict': verdict,
    }
    out = os.path.splitext(path)[0] + '_analysis.json'
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\n  → {out}')


if __name__ == '__main__':
    main()
