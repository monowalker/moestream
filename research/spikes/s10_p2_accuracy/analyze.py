#!/usr/bin/env python3
"""Measure the accuracy of the P2 predictor (W_{L+1} . h_L).

  Usage: analyze.py <trace.bin> <model.gguf> <top_k>

  The self-check always runs first: if the top-k of W_L . h_L does not match
  the top-k of logits_L, the captured tensors are not what we assume, and the
  accuracy numbers are meaningless.
"""
import sys, struct, glob, os
import numpy as np

# ---- read F32 tensors from the GGUF -----------------------------------------
def gguf_tensors(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', path
    ver, = struct.unpack('<I', f.read(4))
    nt,  = struct.unpack('<Q', f.read(8))
    nkv, = struct.unpack('<Q', f.read(8))
    def rs():
        n, = struct.unpack('<Q', f.read(8)); return f.read(n)
    def skip(t):
        fx = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: rs(); return
        if t == 9:
            et, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
            for _ in range(n): skip(et)
            return
        f.read(fx[t])
    align = 32
    for _ in range(nkv):
        k = rs(); t, = struct.unpack('<I', f.read(4))
        if k == b'general.alignment':
            align, = struct.unpack('<I', f.read(4))
        else:
            skip(t)
    infos = []
    for _ in range(nt):
        nm = rs().decode(); nd, = struct.unpack('<I', f.read(4))
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        ty, = struct.unpack('<I', f.read(4)); off, = struct.unpack('<Q', f.read(8))
        infos.append((nm, ne, ty, off))
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    return f, infos, data_start

def load_routers(model_path):
    """Collect blk.<L>.ffn_gate_inp.weight from every shard. F32 only."""
    base = model_path
    if '-of-' in os.path.basename(model_path):
        tail = os.path.basename(model_path)[-20:]
        cur, tot = int(tail[1:6]), int(tail[10:15])
        d = os.path.dirname(model_path)
        pre = os.path.basename(model_path)[:-20]
        paths = [os.path.join(d, f"{pre}-{i:05d}-of-{tot:05d}.gguf") for i in range(1, tot+1)]
    else:
        paths = [model_path]
    W = {}
    for p in paths:
        f, infos, ds = gguf_tensors(p)
        for nm, ne, ty, off in infos:
            if '.ffn_gate_inp.weight' not in nm: continue
            if ty != 0:
                print(f"  {nm} is not F32 (type={ty}); this analysis supports F32 only"); sys.exit(1)
            il = int(nm.split('.')[1])
            f.seek(ds + off)
            n = ne[0] * ne[1]
            a = np.frombuffer(f.read(n*4), dtype=np.float32).reshape(ne[1], ne[0])
            W[il] = a.copy()          # [n_expert, n_embd]
        f.close()
    return W

# ---- read the trace ----------------------------------------------------------
def load_trace(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'P2AC'
    n_embd, n_exp, nrec = struct.unpack('<III', b[4:16])
    rec = 4 + n_embd*4 + n_exp*4
    body = b[16:16+rec*nrec]
    arr = np.frombuffer(body, dtype=np.uint8).reshape(nrec, rec)
    il  = arr[:, :4].copy().view(np.uint32).ravel()
    h   = arr[:, 4:4+n_embd*4].copy().view(np.float32).reshape(nrec, n_embd)
    lg  = arr[:, 4+n_embd*4:].copy().view(np.float32).reshape(nrec, n_exp)
    return n_embd, n_exp, il, h, lg

def topk(x, k):
    return np.argpartition(-x, k-1, axis=-1)[..., :k]

def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    trace, model, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_embd, n_exp, il, h, lg = load_trace(trace)
    W = load_routers(model)
    print(f"  trace: {len(il)} records  n_embd={n_embd} n_expert={n_exp} top_k={K}")
    print(f"  router matrices: {len(W)} layers (shape {next(iter(W.values())).shape})\n")

    # A token boundary is where il decreases
    tok = np.zeros(len(il), dtype=np.int32)
    t = 0
    for i in range(1, len(il)):
        if il[i] <= il[i-1]: t += 1
        tok[i] = t
    n_tok = t + 1

    # ---- (1) self-check: does top-k of W_L . h_L equal top-k of logits_L? ----
    ok = tot = 0
    for i in range(len(il)):
        L = int(il[i])
        if L not in W: continue
        pred = topk(W[L] @ h[i], K)
        act  = topk(lg[i], K)
        ok  += len(set(pred.tolist()) & set(act.tolist()))
        tot += K
    v = ok / tot if tot else 0
    print(f"  (1) self-check  W_L.h_L vs logits_L : {v*100:.2f}%  ({tot} records)")
    if v < 0.99:
        print("     mismatch: the captured tensors are not what we assume.")
        print("     the remaining numbers would be untrustworthy, so stopping.")
        sys.exit(1)
    print("     -> the pipeline is correct\n")

    # ---- (2) P2: predict layer L+1 from W_{L+1} . h_L ----
    # ---- (3) P5 (for comparison): predict layer L+1 from layer L's choices ----
    p2 = p5 = tot2 = 0
    per_layer = {}
    for i in range(len(il) - 1):
        if tok[i] != tok[i+1]: continue          # stay within one token
        L, L1 = int(il[i]), int(il[i+1])
        if L1 != L + 1 or L1 not in W: continue
        act = set(topk(lg[i+1], K).tolist())
        pr2 = set(topk(W[L1] @ h[i], K).tolist())
        pr5 = set(topk(lg[i],   K).tolist())
        a2, a5 = len(pr2 & act), len(pr5 & act)
        p2 += a2; p5 += a5; tot2 += K
        d = per_layer.setdefault(L1, [0, 0]); d[0] += a2; d[1] += K
    print(f"  (2) P2  W_(L+1).h_L -> layer L+1 : {p2/tot2*100:.2f}%   ({tot2} records)")
    print(f"  (3) P5  layer L choices -> L+1   : {p5/tot2*100:.2f}%   (for comparison)")
    print(f"      random baseline              : {K/n_exp*100:.2f}%\n")

    ls = sorted(per_layer)
    print("  per-layer P2 accuracy:")
    for a in range(0, len(ls), 8):
        chunk = ls[a:a+8]
        print("    " + "  ".join(f"L{L}:{per_layer[L][0]/per_layer[L][1]*100:.0f}%" for L in chunk))
    print(f"\n  tokens: {n_tok}")

if __name__ == '__main__':
    main()
