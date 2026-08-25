#!/usr/bin/env bash
# =============================================================================
# Spike S39 -- is the high-batch result correct, not just fast?
#
# S37 reported dense streaming converging on plain llama.cpp as the batch grows
# (3.15x at K=1 down to 1.20x at K=16) and measured nothing but time. That is
# the same omission that produced a 39.63 ms "fastest result of the session"
# from a broken configuration earlier in this work (RESULTS.md 10.8), and the
# 218 ms dense run that was silently emitting garbage. A speed number without a
# correctness check beside it has already been wrong twice here.
#
# Compared at the SAME K on both sides: batch composition changes matmul tiling
# and therefore the output even in plain llama.cpp (RESULTS.md 10.6), so
# plain@K=1 is not the right reference for streaming@K=16.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1; exec flock -n /tmp/moestream-spike.lock "$0" "$@" || { echo locked >&2; exit 9; }
fi
trap 'docker rm -f ms-s39 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
M=${M:-Qwen3.8-27B-IQ4_NL.gguf}; NAME=ms-s39; PORT=18118

gen() {  # $1=env $2=K -> writes concatenated greedy output to stdout
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add 992 --group-add 44 -v "$MODEL_DIR":/models:ro \
    -e XDG_RUNTIME_DIR=/tmp -e MODEL_FILE="$M" -e CTX_SIZE=8192 -e UBATCH=1024 \
    -e N_PARALLEL="$2" -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 $1 -p "$PORT":8080 moestream/server:local >/dev/null
  for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    docker ps -q -f name="$NAME" | grep -q . || { echo "DIED"; return 1; }; sleep 3; done
  # issue K concurrent requests so the batch is genuinely full, then read one back
  for p in "Explain the CAP theorem in detail." "Write a Python function to merge two sorted lists." \
           "The causes of the First World War were" "Describe how a B-tree insert works."; do
    for _ in $(seq 2 "$2"); do
      curl -s -m 900 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
        -d '{"prompt":"filler","n_predict":80,"temperature":0,"top_k":1}' >/dev/null &
    done
    python3 -c "
import json,sys;print(json.dumps({'prompt':sys.argv[1],'n_predict':100,'temperature':0,'top_k':1,'cache_prompt':False}))" "$p" \
      | curl -s -m 900 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @- \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['content'])"
    wait
    echo "###SEP###"
  done
  docker logs "$NAME" 2>&1 | grep -ac "exhaust\|\[BUG\]" | sed 's/^/EXHAUST_OR_BUG_LINES=/'
  docker rm -f "$NAME" >/dev/null 2>&1
}

echo "## S39 -- correctness at the batch sizes S37 reported   $(date -Iseconds)   $M"
echo "   plain@K vs streaming@K, same K on both sides, greedy, 4 prompts x 100 tokens"
echo
for k in 4 16; do
  gen "-e MOESTREAM=0" "$k" > /tmp/s39_ref_$k.txt 2>&1
  gen "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "$k" > /tmp/s39_str_$k.txt 2>&1
  if diff -q /tmp/s39_ref_$k.txt /tmp/s39_str_$k.txt >/dev/null 2>&1; then
    echo "K=$k : IDENTICAL ($(wc -c < /tmp/s39_ref_$k.txt) bytes compared)"
  else
    echo "K=$k : DIFFERENT"
    python3 - "$k" <<'PY'
import sys
k=sys.argv[1]
a=open(f"/tmp/s39_ref_{k}.txt").read().split("###SEP###")
b=open(f"/tmp/s39_str_{k}.txt").read().split("###SEP###")
for i,(x,y) in enumerate(zip(a,b)):
    if x==y: print(f"   prompt {i}: identical ({len(x)} chars)")
    else:
        d=next((j for j,(cx,cy) in enumerate(zip(x,y)) if cx!=cy), min(len(x),len(y)))
        print(f"   prompt {i}: diverges at char {d} of {len(x)}")
        print(f"      ref : {x[max(0,d-50):d+30]!r}")
        print(f"      strm: {y[max(0,d-50):d+30]!r}")
PY
  fi
  grep -h EXHAUST_OR_BUG /tmp/s39_ref_$k.txt /tmp/s39_str_$k.txt | sed 's/^/   /'
done
