#!/usr/bin/env python3
"""
Spike S18 -- is "dense models get no benefit" actually true?

DESIGN.md NG-4 says: "for dense, B_act = total size and streaming cannot work
in principle. Compatibility only."  README repeats it as "MoEStream is inert on
dense models -- it will run them, with no benefit."

`B_act = total size` is true *per forward pass*, not per token.  Prefill puts
U tokens through one pass, so the streamed bytes per token are total/U.  This
spike works out, from a real dense GGUF's tensor index, where that lands:

  1. the per-layer byte budget, split FFN / attention / other
  2. resident footprint under the existing prefill-arena scheme (N arenas of
     one layer each) instead of the whole model
  3. decode cost when every layer is streamed (the case NG-4 describes)
  4. prefill cost as a function of ubatch, and the ubatch above which I/O is
     fully hidden behind compute by the existing async arena (S14)

Empirical constants come from measurements already in RESULTS.md, so nothing
here is a fresh guess:
  BW_SSD    4.48 GB/s   (§4.1, O_DIRECT saturated)
  BW_RAM    ~70 GB/s    (derived: Ornith decode reads ~2.8 GiB in 41.9 ms)
  FLOPS_eff 1.77 TFLOP/s (derived: Ornith prefill 295.6 tok/s at 3B active)

No GPU and no build required; only the GGUF header is read.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'tools', 'analysis'))
import gguf_inspect as gi

BW_SSD    = 4.48e9      # bytes/s
FLOPS_EFF = 1.77e12     # effective FLOP/s during prefill on the Radeon 780M

FFN_PAT  = ('ffn_gate', 'ffn_up', 'ffn_down')
ATTN_PAT = ('attn_', 'ssm_', 'conv1d', 'shortconv')


def classify(name):
    if any(p in name for p in FFN_PAT):
        return 'ffn'
    if any(p in name for p in ATTN_PAT):
        return 'attn'
    if 'token_embd' in name or 'output.weight' in name:
        return 'embd'
    return 'other'


def main(path):
    g = gi.parse(path)
    kv = g['kv']
    arch = kv.get('general.architecture', '?')
    n_layer = next((v for k, v in kv.items() if k.endswith('.block_count')), 0)
    n_param = 0
    tot = {'ffn': 0, 'attn': 0, 'embd': 0, 'other': 0}
    per_layer = {}
    for t in g['tensors']:
        c = classify(t['name'])
        tot[c] += t['nbytes']
        n = 1
        for d in t['dims']:
            n *= d
        n_param += n
        if t['name'].startswith('blk.'):
            il = int(t['name'].split('.')[1])
            per_layer.setdefault(il, {'ffn': 0, 'attn': 0, 'other': 0})
            per_layer[il][c if c in ('ffn', 'attn') else 'other'] += t['nbytes']

    G = 1024 ** 3
    layer_bytes = {il: sum(v.values()) for il, v in per_layer.items()}
    max_layer = max(layer_bytes.values())
    body = sum(layer_bytes.values())
    nonbody = g['file_size'] - g['data_start'] - body

    print('=' * 74)
    print(f'{os.path.basename(path)}   arch={arch}  layers={n_layer}  params={n_param/1e9:.2f} B')
    print(f'  file {g["file_size"]/G:.2f} GiB')
    print(f'    ffn        {tot["ffn"]/G:8.2f} GiB  ({tot["ffn"]/g["file_size"]*100:5.1f}%)')
    print(f'    attn/ssm   {tot["attn"]/G:8.2f} GiB  ({tot["attn"]/g["file_size"]*100:5.1f}%)')
    print(f'    embd/out   {tot["embd"]/G:8.2f} GiB  ({tot["embd"]/g["file_size"]*100:5.1f}%)')
    print(f'    other      {tot["other"]/G:8.2f} GiB')
    print(f'  per layer: mean {body/len(layer_bytes)/1048576:.1f} MiB  max {max_layer/1048576:.1f} MiB')

    # --- residency under an arena scheme ---
    print('\n  -- residency if the body is streamed through N one-layer arenas --')
    for n_arena in (1, 2, 3, 4):
        resident = nonbody + n_arena * max_layer
        print(f'    N={n_arena}: {resident/G:6.2f} GiB resident '
              f'(vs {g["file_size"]/G:.2f} GiB, -{100*(1-resident/g["file_size"]):.0f}%)')

    # --- decode: every layer streamed, one token per pass ---
    t_io = body / BW_SSD
    print(f'\n  -- decode, fully streamed (the case NG-4 describes) --')
    print(f'    bytes/token {body/G:.2f} GiB  -> I/O {t_io*1000:8.0f} ms/token '
          f'({1/t_io:.2f} tok/s)')
    print(f'    FFN only streamed, attn resident: {tot["ffn"]/G:.2f} GiB '
          f'-> {tot["ffn"]/BW_SSD*1000:.0f} ms/token ({BW_SSD/tot["ffn"]:.2f} tok/s)')

    # --- prefill: bytes are per pass, so they amortise over ubatch ---
    flops_tok = 2 * n_param
    t_compute = flops_tok / FLOPS_EFF          # s per token, compute bound
    print(f'\n  -- prefill, fully streamed --')
    print(f'    compute ceiling {1/t_compute:.1f} tok/s ({t_compute*1000:.1f} ms/token)')
    print(f'    {"ubatch":>8} {"I/O ms/tok":>11} {"serial tok/s":>13} '
          f'{"overlapped tok/s":>17} {"% of ceiling":>13}')
    for U in (64, 128, 256, 512, 1024, 2048, 4096):
        io_tok = t_io / U
        ser = 1.0 / (t_compute + io_tok)
        ovl = 1.0 / max(t_compute, io_tok)
        print(f'    {U:8d} {io_tok*1000:11.2f} {ser:13.1f} {ovl:17.1f} '
              f'{ovl*t_compute*100:12.0f}%')
    U_be = t_io / t_compute
    print(f'    break-even ubatch (I/O fully hidden): U >= {U_be:.0f}')

    # --- what activation sparsity would have to deliver for decode ---
    #   Attention/SSM is not sparsifiable, so it stays resident; the whole
    #   read budget goes to the FFN.  Two bandwidths, because the FFN may or
    #   may not fit in the page cache alongside the resident set (30 GiB host).
    resident_attn = nonbody + tot['attn'] + 2 * max_layer
    print(f'\n  -- decode with attention resident ({resident_attn/G:.2f} GiB) '
          f'and the FFN streamed --')
    print(f'    {"target":>8} {"density @4.48 GB/s":>20} {"density @10 GB/s":>18}')
    for target in (1.0, 2.0, 3.0, 5.0, 10.0):
        for bw, col in ((BW_SSD, 'a'), (10e9, 'b')):
            d = (bw / target) / tot['ffn']
            if col == 'a':
                a = f'{d*100:.0f}%' if d <= 1 else 'no sparsity needed'
            else:
                b = f'{d*100:.0f}%' if d <= 1 else 'no sparsity needed'
        print(f'    {target:6.1f} t/s {a:>20} {b:>18}')
    print(f'    (10 GB/s is the page-cache-served figure: RESULTS.md \u00a710.12 measured '
          f'82 MiB/token in 12.45 ms = 6.6 GB/s effective, above the 4.48 GB/s device ceiling)')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
