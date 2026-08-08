#!/usr/bin/env python3
"""
MoEStream — GGUF structure inspector

Purpose:
  Validate the premises of DESIGN.md §13.6 (reading GGUF directly), §18 (.msp
  layout) and §11.3 (Slot/Slab) against a real model. Specifically:

    - architecture and MoE parameters (L, E, top_k, shared, n_ff_exp)
    - expert tensor names, types, shapes and offsets within the file
    - bytes per expert, and their alignment
    - actual size per residency class (RESIDENT / STREAMED / ROW_LOOKUP)
    - active expert bytes per token -> regime determination (§2.2)

The weights themselves are never read; only the header and tensor index.
"""
import sys, os, struct, json, re
from collections import OrderedDict

# --- GGUF metadata value types ---
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)
_FMT = {U8:'<B', I8:'<b', U16:'<H', I16:'<h', U32:'<I', I32:'<i',
        F32:'<f', BOOL:'<B', U64:'<Q', I64:'<q', F64:'<d'}
_SZ  = {U8:1, I8:1, U16:2, I16:2, U32:4, I32:4, F32:4, BOOL:1, U64:8, I64:8, F64:8}


def load_ggml_types(ggml_h):
    """Read the GGML_TYPE_* enum from ggml.h rather than hardcoding it."""
    names = {}
    try:
        src = open(ggml_h, encoding='utf-8', errors='ignore').read()
        m = re.search(r'enum\s+ggml_type\s*\{(.*?)\}', src, re.S)
        if m:
            for line in m.group(1).splitlines():
                mm = re.match(r'\s*GGML_TYPE_(\w+)\s*=\s*(\d+)', line)
                if mm:
                    names[int(mm.group(2))] = mm.group(1)
    except OSError:
        pass
    return names


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError('unexpected EOF')
        return b

    def scalar(self, t):
        return struct.unpack(_FMT[t], self.raw(_SZ[t]))[0]

    def string(self):
        n = self.scalar(U64)
        return self.raw(n).decode('utf-8', errors='replace')

    def value(self, t):
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.scalar(U32)
            n = self.scalar(U64)
            if et == STRING:
                # Vocabulary arrays are huge; keep only the length
                if n > 4096:
                    for _ in range(n):
                        self.f.seek(self.scalar(U64), os.SEEK_CUR)
                    return f'<{n} strings (skipped)>'
                return [self.string() for _ in range(n)]
            if n > 4096:
                self.f.seek(_SZ[et] * n, os.SEEK_CUR)
                return f'<{n} values (skipped)>'
            return [self.scalar(et) for _ in range(n)]
        if t == BOOL:
            return bool(self.scalar(BOOL))
        return self.scalar(t)


def parse(path, ggml_h=None):
    types = load_ggml_types(ggml_h) if ggml_h else {}
    with open(path, 'rb') as f:
        r = Reader(f)
        if r.raw(4) != b'GGUF':
            raise ValueError('GGUF magic mismatch')
        version = r.scalar(U32)
        n_tensor = r.scalar(U64)
        n_kv = r.scalar(U64)

        kv = OrderedDict()
        for _ in range(n_kv):
            k = r.string()
            t = r.scalar(U32)
            kv[k] = r.value(t)

        tensors = []
        for _ in range(n_tensor):
            name = r.string()
            nd = r.scalar(U32)
            dims = [r.scalar(U64) for _ in range(nd)]
            ttype = r.scalar(U32)
            off = r.scalar(U64)
            tensors.append({'name': name, 'dims': dims, 'type': ttype,
                            'type_name': types.get(ttype, f'T{ttype}'), 'offset': off})

        align = kv.get('general.alignment', 32)
        pos = f.tell()
        data_start = (pos + align - 1) // align * align

    # Recover each tensor's real size from offset differences, which avoids
    # needing per-type block sizes
    order = sorted(tensors, key=lambda t: t['offset'])
    file_size = os.path.getsize(path)
    for i, t in enumerate(order):
        end = order[i + 1]['offset'] if i + 1 < len(order) else (file_size - data_start)
        t['nbytes'] = end - t['offset']
        t['abs_offset'] = data_start + t['offset']
    return {'version': version, 'kv': kv, 'tensors': tensors,
            'align': align, 'data_start': data_start, 'file_size': file_size}


def human(n):
    for u in ['B', 'KiB', 'MiB', 'GiB']:
        if abs(n) < 1024 or u == 'GiB':
            return f'{n:,.2f} {u}' if u != 'B' else f'{n:,} B'
        n /= 1024


def main():
    path = sys.argv[1]
    ggml_h = sys.argv[2] if len(sys.argv) > 2 else None
    g = parse(path, ggml_h)
    kv, ts = g['kv'], g['tensors']
    arch = kv.get('general.architecture', '?')

    print('=' * 78)
    print(f' GGUF: {os.path.basename(path)}')
    print('=' * 78)
    print(f'  version={g["version"]}  tensors={len(ts)}  align={g["align"]}')
    print(f'  file={human(g["file_size"])}  data_start=0x{g["data_start"]:x}')
    print(f'  architecture = {arch}')
    print(f'  name         = {kv.get("general.name","?")}')

    print('\n--- key architecture parameters ---')
    keys = [k for k in kv if k.startswith(f'{arch}.') or k.startswith('general.file_type')]
    for k in keys:
        v = kv[k]
        if isinstance(v, str) and v.startswith('<'):
            continue
        print(f'  {k:52s} = {v}')

    # ---- extract expert tensors ------------------------------------------
    exps = [t for t in ts if '_exps.' in t['name']]
    print(f'\n--- expert tensors ({len(exps)}) ---')
    if exps:
        byrole = OrderedDict()
        for t in exps:
            role = t['name'].split('.')[-2]           # e.g. ffn_gate_exps
            byrole.setdefault(role, []).append(t)
        for role, lst in byrole.items():
            s = lst[0]
            total = sum(x['nbytes'] for x in lst)
            tset = sorted({x['type_name'] for x in lst})
            print(f'  {role:18s} n={len(lst):3d}  dims={s["dims"]}  '
                  f'type={"/".join(tset)}  total={human(total)}')
            if len(tset) > 1:
                # UD (Unsloth Dynamic) and similar vary the type per layer
                from collections import Counter
                c = Counter(x['type_name'] for x in lst)
                print(f'  {"":18s} type breakdown: {dict(c)}')

        # Bytes per expert, computed on layer 0
        l0 = [t for t in exps if t['name'].startswith('blk.0.')]
        if l0:
            n_expert = int(l0[0]['dims'][2]) if len(l0[0]['dims']) > 2 else 1
            per_layer = sum(t['nbytes'] for t in l0)
            per_expert = per_layer / n_expert
            print(f'\n  layer 0: n_expert={n_expert}  per layer={human(per_layer)}  '
                  f'1 expert={human(per_expert)}')
            print(f'  4 KiB boundary: {"OK" if per_expert % 4096 == 0 else f"no (remainder {per_expert % 4096:.0f} B); .msp would need padding"}')

    # ---- residency classification ----------------------------------------
    print('\n--- residency classification (§9.5) ---')
    cls = {'STREAMED': [], 'ROW_LOOKUP': [], 'RESIDENT': []}
    for t in ts:
        n = t['name']
        if '_exps.' in n and 'shexp' not in n:
            cls['STREAMED'].append(t)
        elif n.startswith('token_embd'):
            cls['ROW_LOOKUP'].append(t)
        else:
            cls['RESIDENT'].append(t)
    tot = sum(t['nbytes'] for t in ts)
    for k, lst in cls.items():
        b = sum(t['nbytes'] for t in lst)
        print(f'  {k:12s} {len(lst):5d} tensors  {human(b):>14s}  ({100*b/tot:5.1f}%)')
    print(f'  {"total":12s} {len(ts):5d} tensors  {human(tot):>14s}')

    # ---- regime determination (§2.2) -------------------------------------
    n_layer = kv.get(f'{arch}.block_count')
    n_used = kv.get(f'{arch}.expert_used_count')
    if exps and n_layer and n_used:
        l0 = [t for t in exps if t['name'].startswith('blk.0.')]
        n_expert = int(l0[0]['dims'][2])
        # Count the MoE layers directly
        moe_layers = len({t['name'].split('.')[1] for t in exps})
        per_expert = sum(t['nbytes'] for t in l0) / n_expert
        b_act = moe_layers * n_used * per_expert
        print('\n--- dominant variable and regime (§2.2 / §2.3) ---')
        print(f'  MoE layers   = {moe_layers} / {n_layer}')
        print(f'  n_expert     = {n_expert},  top_k = {n_used}')
        print(f'  1 expert     = {human(per_expert)}')
        print(f'  B_act        = {human(b_act)}  / token  (the dominant variable)')
        for bw, label in [(3.5e9, 'PCIe3.0'), (6.0e9, 'PCIe4.0'), (12.0e9, 'PCIe5.0')]:
            t_io = b_act / bw
            print(f'    BW={bw/1e9:4.1f} GB/s ({label:8s}): cold I/O = {t_io*1000:6.1f} ms/token')
        tc = 0.040
        print(f'  assuming compute time t_c = {tc*1000:.0f} ms/token:')
        for bw in [3.5e9, 6.0e9, 12.0e9]:
            need = 1.0 - (1.2 * tc * bw) / b_act
            need = max(need, 0.0)
            print(f'    BW={bw/1e9:4.1f} GB/s -> hit rate for a 20% slowdown: h >= {need*100:5.1f}%')

    if '--json' in sys.argv:
        with open('gguf_index.json', 'w') as fp:
            json.dump({'kv': {k: v for k, v in kv.items() if not isinstance(v, list)},
                       'tensors': ts, 'data_start': g['data_start']}, fp, indent=1)
        print('\n  -> wrote gguf_index.json')


if __name__ == '__main__' and '--layout' not in sys.argv:
    main()


def export_layout(path, ggml_h, layer, out):
    """Dump expert tensor placement as plain text for S1 and the replay harness.
    A negative layer means every layer."""
    g = parse(path, ggml_h)
    if layer < 0:
        return export_all_layers(g, out)
    pref = f'blk.{layer}.'
    roles = {'ffn_gate_exps': 'gate', 'ffn_up_exps': 'up', 'ffn_down_exps': 'down'}
    lines = []
    for t in g['tensors']:
        if not t['name'].startswith(pref):
            continue
        for k, short in roles.items():
            if t['name'].endswith(k + '.weight'):
                d = t['dims'] + [1, 1, 1]
                lines.append(f"TENSOR {short} {t['abs_offset']} {t['nbytes']} "
                             f"{t['type']} {int(d[0])} {int(d[1])} {int(d[2])} {t['type_name']}")
    with open(out, 'w') as f:
        f.write(f"LAYER {layer}\n")
        f.write("\n".join(lines) + "\n")
    print(f"layer {layer} layout -> {out}")
    for l in lines:
        print("  " + l)


def export_all_layers(g, out):
    """Dump expert placement for every layer, read by ExpertIndex in the replay harness."""
    import re as _re
    roles = {'ffn_gate_exps': 'gate', 'ffn_up_exps': 'up', 'ffn_down_exps': 'down'}
    rows = {}
    for t in g['tensors']:
        m = _re.match(r'blk\.(\d+)\.', t['name'])
        if not m: continue
        il = int(m.group(1))
        for k, short in roles.items():
            if t['name'].endswith(k + '.weight'):
                d = list(t['dims']) + [1, 1, 1]
                rows.setdefault(il, {})[short] = (
                    t['abs_offset'], t['nbytes'], t['type'],
                    int(d[0]), int(d[1]), int(d[2]), t['type_name'])
    with open(out, 'w') as f:
        f.write(f"NLAYER {len(rows)}\n")
        for il in sorted(rows):
            for short in ('gate', 'up', 'down'):
                if short not in rows[il]: continue
                o, nb, ty, n0, n1, n2, tn = rows[il][short]
                f.write(f"E {il} {short} {o} {nb} {ty} {n0} {n1} {n2} {tn}\n")
    print(f"expert placement for {len(rows)} layers -> {out}")


if __name__ == '__main__' and '--layout' in sys.argv:
    i = sys.argv.index('--layout')
    export_layout(sys.argv[1], sys.argv[2], int(sys.argv[i+1]), sys.argv[i+2])
    sys.exit(0)
