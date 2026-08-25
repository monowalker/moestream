#!/usr/bin/env bash
# =============================================================================
# MoEStream launcher
#
#   Everything this project can derive, it derives. What is left is the handful
#   of facts that genuinely depend on what you are doing -- which model, how
#   much context, how many concurrent requests -- and those are exactly the ones
#   a config file is bad at, because their consequences depend on each other and
#   on this machine.
#
#   So: read the GGUF headers, read the GPU's limits, and ask only what cannot
#   be worked out. Then show what the answers imply, before starting anything.
#
#   .env keeps two lines: where the models are, and the port.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }
MODEL_DIR="${MODEL_DIR:-}"
MS_PORT="${MS_PORT:-8091}"
GEN=.env.launcher

die() { echo "error: $*" >&2; exit 1; }
[ -n "$MODEL_DIR" ] || die "MODEL_DIR is not set. Put it in .env: MODEL_DIR=/path/to/models"
[ -d "$MODEL_DIR" ] || die "MODEL_DIR does not exist: $MODEL_DIR"
command -v python3 >/dev/null || die "python3 is needed to read GGUF headers"

# ---- machine facts -----------------------------------------------------------
GTT=0; VRAM=0
for d in /sys/class/drm/card*/device; do
  [ -f "$d/mem_info_gtt_total" ] || continue
  GTT=$(cat "$d/mem_info_gtt_total"); VRAM=$(cat "$d/mem_info_vram_total"); break
done
RAM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
BUDGET_GIB=$(python3 -c "print('%.1f' % (($GTT + $VRAM)/1073741824))")
RAM_GIB=$(python3 -c "print('%.1f' % ($RAM_KB/1048576))")

echo
echo "  MoEStream launcher"
echo "  ─────────────────────────────────────────────────────────────"
printf "  GPU-reachable memory  %s GiB      host RAM  %s GiB\n" "$BUDGET_GIB" "$RAM_GIB"
printf "  models                %s\n" "$MODEL_DIR"
echo

# ---- inspect every GGUF ------------------------------------------------------
MANIFEST_FILE=$(mktemp); trap 'rm -f "$MANIFEST_FILE"' EXIT
python3 - "$MODEL_DIR" > "$MANIFEST_FILE" <<'PY'
import sys, os, json, struct
d = sys.argv[1]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'research', 'tools', 'analysis'))
def kvscan(path):
    """Everything the launcher needs from the header, KV geometry included."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF': return None
        struct.unpack('<I', f.read(4))
        n_tensor, n_kv = struct.unpack('<QQ', f.read(16))
        FIX = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        def rd(t):
            if t == 8:
                n, = struct.unpack('<Q', f.read(8))
                return f.read(n).decode('utf-8', 'replace')
            if t == 9:
                et, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
                return [rd(et) for _ in range(n)]
            b = f.read(FIX.get(t, 0))
            if t == 6: return struct.unpack('<f', b)[0]
            if t == 12: return struct.unpack('<d', b)[0]
            if t == 7: return bool(b[0])
            return int.from_bytes(b, 'little', signed=(t in (5,11)))
        WANT = ('.block_count', '.expert_count', '.embedding_length',
                '.attention.head_count', '.attention.head_count_kv',
                '.attention.key_length', '.attention.value_length',
                '.attention.key_length_swa', '.attention.value_length_swa',
                '.attention.sliding_window', '.attention.sliding_window_pattern',
                '.full_attention_interval', '.ssm.conv_kernel', '.ssm.state_size',
                '.ssm.inner_size')
        out = {}
        for _ in range(n_kv):
            kl_, = struct.unpack('<Q', f.read(8))
            k = f.read(kl_).decode('utf-8', 'replace')
            vt, = struct.unpack('<I', f.read(4))
            v = rd(vt)
            if k == 'general.architecture': out['arch'] = v
            elif any(k.endswith(w) for w in WANT):
                out[k.split('.', 1)[1]] = v
            if 'ssm' in k or 'conv_kernel' in k: out['hybrid'] = True
        return out

def kv_geometry(kv):
    """(bytes per token that grow with context, bytes that do not).

    Sliding-window layers stop growing once the window is full, and several
    architectures vary the KV head count per layer, so a single number is wrong
    for exactly the models where the answer matters most."""
    lay = kv.get('block_count', 0)
    if not lay: return 0.0, 0.0
    kvh = kv.get('attention.head_count_kv', 0)
    kl  = kv.get('attention.key_length', 0) or (
          kv.get('embedding_length', 0) // max(kv.get('attention.head_count', 1), 1))
    vl  = kv.get('attention.value_length', 0) or kl
    kls = kv.get('attention.key_length_swa', 0)   or kl
    vls = kv.get('attention.value_length_swa', 0) or vl
    win = kv.get('attention.sliding_window', 0)
    pat = kv.get('attention.sliding_window_pattern', None)
    # A hybrid model keeps a constant-size recurrent state on most layers and
    # real attention only every full_attention_interval-th one. Counting all 65
    # of Qwen3.8's layers as attention overstated its cache four-fold.
    ivl  = kv.get('full_attention_interval', 0)
    conv = kv.get('ssm.conv_kernel', 0)
    st   = kv.get('ssm.state_size', 0)
    inner= kv.get('ssm.inner_size', 0)
    if not kl: return 0.0, 0.0
    grow = fixed = 0
    for i in range(lay):
        if ivl and (i + 1) % ivl != 0:
            fixed += (conv + st) * inner * 4      # f32 recurrent state
            continue
        h = kvh[i] if isinstance(kvh, list) else kvh
        if not h: continue
        is_swa = bool(pat[i]) if isinstance(pat, list) and i < len(pat) else False
        if is_swa and win:
            fixed += h * (kls + vls) * win
        else:
            grow  += h * (kl + vl)
    Q8 = 1.0625                       # q8_0 KV, the project default
    return grow * Q8, fixed * Q8      # close enough; the state term is small

rows = []
for fn in sorted(os.listdir(d)):
    if not fn.endswith('.gguf'): continue
    if '-of-' in fn and '00001-of-' not in fn: continue
    if 'mmproj' in fn.lower(): continue          # a vision projector, not a model
    p = os.path.join(d, fn)
    try: kv = kvscan(p)
    except Exception: kv = None
    if not kv: continue
    if 'clip' in kv.get('arch', '') or not kv.get('block_count'): continue
    size = 0                                     # split GGUFs: sum the shards
    base = fn.split('-00001-of-')[0] if '-00001-of-' in fn else None
    for g in os.listdir(d):
        if g.endswith('.gguf') and ((base and g.startswith(base)) or g == fn):
            size += os.path.getsize(os.path.join(d, g))
    kv_grow, kv_fixed = kv_geometry(kv)
    rows.append({'file': fn, 'gib': size/1073741824,
                 'arch': kv.get('arch','?'), 'layers': kv.get('block_count', 0),
                 'experts': kv.get('expert_count',0),
                 'hybrid': bool(kv.get('hybrid', False)),
                 'kv_per_tok': kv_grow, 'kv_fixed': kv_fixed})
print(json.dumps(rows))
PY
COUNT=$(python3 -c "import json;print(len(json.load(open('$MANIFEST_FILE'))))")
[ "$COUNT" -gt 0 ] || die "no GGUF files found in $MODEL_DIR"

python3 - "$BUDGET_GIB" "$MANIFEST_FILE" <<'PY'
import json, sys
budget = float(sys.argv[1]); rows = json.load(open(sys.argv[2]))
print("   #  model                                          size   type        fits?")
for i, r in enumerate(rows, 1):
    kind = f"MoE {r['experts']}e" if r['experts'] else ("dense/hybrid" if r['hybrid'] else "dense")
    fits = "yes" if r['gib'] + 2.0 < budget else "NO — streaming is the point"
    print(f"  {i:2d}  {r['file'][:44]:<44} {r['gib']:6.2f}G  {kind:<12} {fits}")
PY
echo
FILE=""
while [ -z "$FILE" ]; do
  read -rp "  model number (1-$COUNT, or q to quit): " PICK
  case "$PICK" in [Qq]*|"") echo "  cancelled"; exit 0 ;; esac
  FILE=$(PICK="$PICK" python3 -c "
import json,sys,os
r=json.load(open('$MANIFEST_FILE'))
try: i=int(os.environ['PICK'])-1
except ValueError: i=-1
print(r[i]['file'] if 0<=i<len(r) else '')")
  [ -z "$FILE" ] && echo "  please type a number between 1 and $COUNT"
done

INFO=$(FSEL="$FILE" python3 -c "
import json,os
r=[x for x in json.load(open('$MANIFEST_FILE')) if x['file']==os.environ['FSEL']][0]
print(r['gib'], r['experts'], 1 if r['hybrid'] else 0, r['arch'], r['layers'], r['kv_per_tok'], r['kv_fixed'])")
read -r MGIB MEXP MHYB MARCH MLAY MKVT MKVF <<<"$INFO"

# Which kinds of speculative decoding this model supports is llama.cpp's
# question to answer, not ours: the rule is a property of the tensors a GGUF
# carries and upstream already implements it. moestream-spec-probe links
# llama.cpp's own common library and calls
# common_speculative_types_from_gguf(), so the answer follows upstream with no
# rule duplicated here. Before the image exists there is nothing to ask.
SPEC_TYPES=""
if docker image inspect moestream/server:local >/dev/null 2>&1; then
  SPEC_TYPES=$(docker run --rm -v "$MODEL_DIR":/models:ro \
      --entrypoint /usr/local/bin/moestream-spec-probe moestream/server:local \
      "/models/$FILE" 2>/dev/null | paste -sd, -)
fi

echo
echo "  $FILE"
printf "  %s, %s layers, %.2f GiB\n" "$MARCH" "$MLAY" "$MGIB"
if [ "$MEXP" -gt 0 ]; then
  echo "  -> MoE ($MEXP experts). Expert streaming; MOESTREAM_CACHE_FRAC=learn finds the slot count."
  if [ -n "$SPEC_TYPES" ]; then
    echo "     This model can self-speculate ($SPEC_TYPES). Whether that pays depends"
    echo "     on how slow its decode is, and a MoE model's is fast, so it is left to"
    echo "     measurement rather than assumed: the first few starts try one draft"
    echo "     size each, including none (finding S42)."
  fi
else
  echo "  -> dense. FFN streaming, MOESTREAM_DENSE_FRAC=auto streams the least that still fits."
  echo "     Generation costs ~3x at one request and ~1.2x at 16 (finding S37), so the"
  echo "     concurrency you pick below matters more here than anywhere else."
  if [ -n "$SPEC_TYPES" ]; then
    echo "     llama.cpp reports this model can self-speculate ($SPEC_TYPES), so that"
    echo "     is switched on and tuned by measurement: worth up to 2.3x on a streamed"
    echo "     dense model, which is most of what a single-request workload loses."
    echo "     The first few starts try one draft size each, including none at all"
    echo "     (finding S42)."
  fi
fi
echo

echo "  How will you use it?"
echo "    1  chat / agent, one request at a time     (latency matters)"
echo "    2  batch or API serving, several at once   (throughput matters)"
echo "    3  long prompts, short answers             (summarise, classify, rerank)"
read -rp "  choice [1]: " USE; USE=${USE:-1}
# UBATCH is always learned. The prefill optimum is strongly model-dependent --
# 1024 on Ornith, 4096 on Qwen3-Coder-Next, 8096 on Laguna (RESULTS 10.11) --
# so pinning a value here would be a guess, and worst of all in choice 3, where
# prefill is the whole workload.
case "$USE" in
  2) NPAR=8 ;;
  3) NPAR=2 ;;
  *) NPAR=1 ;;
esac
UB=learn

# Suggest the largest context that still leaves room, from the model's real
# attention geometry rather than a guess. Streaming can shrink the weights, so
# the budget assumes it will if it has to.
read -r DEFCTX KVMSG RESID <<<"$(python3 -c "
mg=float('$MGIB'); bud=float('$BUDGET_GIB'); npar=int('$NPAR'); kvt=float('$MKVT')
exp=int('$MEXP'); kvf=float('$MKVF')
def kvgib(c): return (kvt*c + kvf)*npar/1073741824
# What has to be resident, used to size the context. Expert streaming sizes its
# slab to whatever memory is left, so for MoE the binding constraint is the part
# that cannot be streamed -- the attention and embedding weights, about a fifth
# of the file. Dense 'auto' streams only what it must, so plan for the whole
# model and let auto do better than the plan.
resident = 0.20*mg + 1.0 if exp else mg
free = max(bud - resident - 1.5, 0.5)
if kvt <= 0:
    print(8192, 'unknown', '%.2f' % resident)
else:
    best = 4096
    for c in (4096, 8192, 16384, 32768):        # capped: longer is rarely wanted
        if kvgib(c) <= free: best = c
    print(best, '%.2f' % (kvt*1000/1073741824*npar), '%.2f' % resident)
")"
echo "  Context length is how much the model can see at once."
if [ "$KVMSG" != "unknown" ]; then
  echo "  Costs about ${KVMSG} GiB of memory per 1000 tokens at this concurrency."
fi
read -rp "  context length [$DEFCTX]: " CTX; CTX=${CTX:-$DEFCTX}
read -rp "  port [$MS_PORT]: " PORT; PORT=${PORT:-$MS_PORT}

# ---- what the answers imply --------------------------------------------------
echo
python3 - "$MGIB" "$MEXP" "$BUDGET_GIB" "$CTX" "$NPAR" "$MKVT" "$RESID" "$MKVF" <<'PY'
import sys
gib, exp, budget, ctx, npar, kvt, resid = (float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]),
                                           int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]),
                                           float(sys.argv[7]))
kvf = float(sys.argv[8])
kv = (kvt * ctx + kvf) * npar / 1073741824 if kvt > 0 else 0.0
print("  what this implies")
print("  " + "\u2500"*61)
print(f"  weights on disk        {gib:6.2f} GiB")
if kv > 0: print(f"  KV cache               {kv:6.2f} GiB   at ctx {ctx} x {npar}")
else:      print( "  KV cache                  n/a   (geometry not in the header)")
head = gib + kv + 1.5
if exp:
    # Expert streaming applies whether or not the model fits: that is the point
    # of it. The slab is sized to the memory that is left, so quoting a resident
    # figure derived from the model size would be a guess, and a bad one -- it
    # is the machine that decides. Say what is actually pinned instead.
    print(f"  cannot be streamed     {resid:6.2f} GiB   (attention and embeddings)")
    if head <= budget:
        print(f"  all resident would be  {head:6.2f} GiB of {budget:.1f} -- it fits either way")
    elif resid + kv + 1.5 > budget:
        print(f"  !! that plus the KV cache is {resid+kv+1.5:.1f} GiB of {budget:.1f}, before a")
        print( "     single expert. Choose a shorter context.")
    else:
        print( "  -> the expert slab is sized to the memory that is left, so this runs")
        print(f"     within your {budget:.0f} GiB instead of needing {gib:.1f}. MOESTREAM_CACHE_FRAC")
        print( "     =learn starts low and raises it; the startup log prints the figure.")
else:
    if head <= budget:
        print(f"  everything fits        {head:6.2f} GiB of {budget:.1f}")
        print( "  -> MOESTREAM_DENSE_FRAC=auto will stream nothing. Identical to plain")
        print( "     llama.cpp, at plain llama.cpp speed.")
    else:
        print(f"  will not all fit       {head:6.2f} GiB wanted, {budget:.1f} available")
        print(f"  -> the FFN is streamed, down to roughly {max(gib*0.45, 1.0):.1f} GiB of weights")
        if npar == 1:
            print( "     Dense at one request is the worst case: about 3x slower to")
            print( "     generate (finding S37). Choice 2 above costs far less if you")
            print( "     can batch -- 1.2x at 16 concurrent.")
        if max(gib*0.45,1.0) + kv + 1.5 > budget:
            print(f"  !! even streamed that is {max(gib*0.45,1.0)+kv+1.5:.1f} GiB. Choose a shorter context.")
PY
# This model fits, so streaming is a choice rather than a necessity -- but
# freeing the memory is the whole point of the project and the freed GiB is
# real (finding S26), so it stays the default. What is never right is to decide
# either way silently.
STREAM=1
DFRAC=auto
if python3 -c "
import sys; sys.exit(0 if float('$MGIB')+(float('$MKVT')*$CTX+float('$MKVF'))*$NPAR/1073741824+1.5 <= float('$BUDGET_GIB') else 1)"; then
  echo
  echo "  This model would also fit whole, so you get to choose what to spend."
  if [ "$MEXP" -gt 0 ]; then
    FREED=$(python3 -c "print('%.0f' % (float('$MGIB')*0.35))")
    echo "    1  stream, freeing about ${FREED} GiB for everything else   (about 1.3x slower)"
    echo "    2  keep it all resident                             (fastest, uses the memory)"
    read -rp "  choice [1]: " SM; case "${SM:-1}" in 2) STREAM=0 ;; *) STREAM=1 ;; esac
  else
    # Dense costs far more than MoE to stream, and how much more depends
    # entirely on the concurrency chosen above (finding S37), so quote the
    # figure for the concurrency actually picked rather than a general one.
    FREED=$(python3 -c "print('%.0f' % (float('$MGIB')*0.55))")
    case "$NPAR" in 1) COST="about 3x slower to generate" ;;
                    2) COST="about 2.5x slower to generate" ;;
                    *) COST="about 1.2x slower per token, at 8 concurrent" ;; esac
    echo "    1  stream the FFN, freeing about ${FREED} GiB   ($COST)"
    echo "    2  keep it all resident                  (fastest, uses the memory)"
    read -rp "  choice [1]: " SM
    case "${SM:-1}" in 2) DFRAC=auto ;; *) DFRAC=0.00 ;; esac
  fi
fi

# Speculation on MoE is the one place where this machine's answer should not be
# assumed to be yours. Measured here it loses on an unstreamed model and gains a
# little on a streamed one, and both are small; on a GPU with compute to spare
# the balance moves the other way (finding S42). So: offer it, with the numbers.
# Whether drafting pays, and how many tokens to draft, is not a property of the
# model family -- it is how slow the target pass is. Measured on this machine the
# best setting runs from n_max=5 (a streamed dense model, 2.29x) through
# break-even (a streamed MoE one) to a 65% loss (a fast MoE model at llama.cpp's
# defaults), and even the dense optimum turns over by n_max=8. No value is worth
# hardcoding, so it is learned from the server's own throughput (finding S42).
SPEC_ON=0
if [ -n "$SPEC_TYPES" ] && [ "$STREAM" = "1" ]; then
  if [ "$NPAR" -gt 2 ]; then
    # Batching and drafting are two ways to widen a pass and they do not add.
    # Measured on a streamed dense model: drafting is worth 2.08x at one request
    # and costs 34% at four, for 3.6 GiB more (S45). At this concurrency the
    # batch has already taken what there was to take.
    echo
    echo "  Speculative decoding is available but not switched on: you chose to run"
    echo "  several requests at once, and batching already fills the pass it would"
    echo "  widen. Measured at 4 concurrent it costs 34% and 3.6 GiB (finding S45)."
  else
    SPEC_ON=1
  fi
fi

echo
read -rp "  start? [Y/n]: " GO; case "${GO:-y}" in [Nn]*) echo "  cancelled"; exit 0;; esac

# ---- write only what the launcher decided; everything else stays derived -----
{
  echo "# generated by launcher.sh -- do not edit; rerun 'make launch'"
  echo "MODEL_DIR=$MODEL_DIR"
  echo "MODEL_FILE=$FILE"
  echo "MS_PORT=$PORT"
  echo "CTX_SIZE=$CTX"
  echo "N_PARALLEL=$NPAR"
  echo "UBATCH=$UB"
  echo "MOESTREAM=$STREAM"
  if [ "$MEXP" -gt 0 ]; then echo "MOESTREAM_CACHE_FRAC=learn"
  else                        echo "MOESTREAM_DENSE_FRAC=$DFRAC"; fi
  # Learned, not decided here: which side of the trade a model lands on depends
  # on the machine as much as the architecture (S42).
  if [ "$SPEC_ON" = "1" ] && { [ "$MEXP" -gt 0 ] || [ "$DFRAC" != "auto" ]; }; then
    echo "SPEC_DECODING=learn"
  fi
} > "$GEN"
echo "  wrote $GEN"
if ! docker image inspect moestream/server:local >/dev/null 2>&1; then
  echo
  echo "  The image is not built yet. This first run compiles llama.cpp and takes"
  echo "  roughly 25 minutes. Later starts are seconds."
  read -rp "  build now? [Y/n]: " B; case "${B:-y}" in [Nn]*) echo "  cancelled"; exit 0;; esac
fi
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video  | cut -d: -f3)
if ! RENDER_GID=$RENDER_GID VIDEO_GID=$VIDEO_GID \
     MODEL_DIR="$MODEL_DIR" MODEL_FILE="$FILE" MS_PORT="$PORT" \
     docker compose up -d; then
  echo "  failed to start. 'make logs' will say why."; exit 1
fi
echo
echo -n "  loading the model"
for i in $(seq 1 200); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo; echo "  ready:  http://localhost:$PORT"
    echo "          http://localhost:$PORT/v1   (OpenAI-compatible)"
    echo "  stop:   make down"
    exit 0
  fi
  docker ps -q -f name=moestream | grep -q . || {
    echo; echo "  the server stopped while loading. Last lines:"
    docker compose logs --tail 15 2>/dev/null | sed "s/^/    /"
    exit 1; }
  echo -n "."; sleep 3
done
echo; echo "  still loading after 10 minutes. 'make logs' to watch."
