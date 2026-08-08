#!/usr/bin/env python3
"""Build the cross-layer coactivation table (DESIGN.md §22.4, predictor P5).

For each expert e chosen at layer L, record the top M experts most likely to
be chosen at layer L+1. P2 (layer lookahead) needs the hidden state from the
GPU, which costs about 0.5 ms of synchronization per layer under Vulkan.
P5 predicts from ids that have already been read, so it adds no synchronization.

Output: "MSCA <n_layer> <n_expert> <M>" followed by u16 x M per (layer, expert).
"""
import sys, struct
import numpy as np

def load(path):
    with open(path,'rb') as f:
        h = struct.unpack('<8I', f.read(32)); assert h[0]==0x5254534D
        n_moe, top_k = h[2], h[4]
        rec = np.frombuffer(f.read(), dtype=np.uint16)
    per = n_moe*top_k*2
    rec = rec[:(rec.size//per)*per].reshape(-1, n_moe, 2, top_k)
    return rec[:,:,0,:].astype(np.int32), n_moe, top_k

M = 8
ids_all = []
for p in sys.argv[1:-1]:
    a, L, K = load(p); ids_all.append(a)
ids = np.concatenate(ids_all, axis=0)
T, L, K = ids.shape
E = int(ids.max())+1
print(f"tokens={T:,} layers={L} top_k={K} n_expert={E}")

out = np.zeros((L, E, M), dtype=np.uint16)
for l in range(L-1):
    # co[e, f] = times f was chosen at layer l+1 given e was chosen at layer l
    co = np.zeros((E, E), dtype=np.int32)
    cur, nxt = ids[:, l, :], ids[:, l+1, :]
    for t in range(T):
        c = cur[t]; n = nxt[t]
        co[np.ix_(c, n)] += 1
    for e in range(E):
        out[l, e] = np.argsort(-co[e])[:M].astype(np.uint16)
    if l % 10 == 0: print(f"  layer {l}/{L}", flush=True)

with open(sys.argv[-1], 'wb') as f:
    f.write(struct.pack('<4sIII', b'MSCA', L, E, M))
    f.write(out.tobytes())
print(f"→ {sys.argv[-1]}  ({L}×{E}×{M} u16 = {out.nbytes/1024:.0f} KiB)")
