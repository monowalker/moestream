#!/usr/bin/env bash
# =============================================================================
# Spike S24 -- finding S17 implemented and measured end to end
#
# S17 measured offline that skipping a cache MISS whose router weight is low
# removes read bytes far more cheaply than the blanket rule M0-2 rejected.
# The follow-up (rank_rule.py) then found that ggml_argsort_top_k returns
# experts in descending probability order -- verified on 100.00% of adjacent
# pairs across all three traces -- so the position within top_k IS the rank,
# and the rule can be a predicate on k with no new tensor, no graph reordering
# and no extra CPU<->GPU sync.
#
# MOESTREAM_SKIP_RANK=K implements exactly that: a miss at rank >= K is pointed
# at the null expert instead of being fetched. Hits are never skipped.
#
# Read bytes removed at each K, from the traces (en, frac=0.25):
#     K=7  19.6%    K=6  36.8%    K=5  51.6%    K=4  64.4%
# Router weight mass lost, which is an UPPER bound on the error because the
# surviving weights are not renormalised:
#     K=7  2.05%    K=6  3.99%    K=5  5.84%    K=4  7.63%
#
# Needs the dev image, which carries the patch.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || {
    echo "another spike holds /tmp/moestream-spike.lock; refusing to run" >&2; exit 9; }
fi
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
IMG=${IMG:-moestream/devserver:local}
NAME=ms-s24b; PORT=18086

boot() {
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$MODEL_FILE" -e CTX_SIZE="${CTX_SIZE:-32768}" -e UBATCH="${UBATCH:-1024}" \
    -e N_PARALLEL=1 -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 -e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC="${MOESTREAM_CACHE_FRAC:-0.25}" \
    $1 -p "$PORT":8080 "$IMG" >/dev/null || return 1
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done; return 1
}
ask() { curl -s -m 900 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$1"; }

speed() {  # $1=label $2=env
  if ! boot "$2"; then printf "%-26s FAILED TO BOOT\n" "$1"; docker logs "$NAME" 2>&1|tail -6; return; fi
  ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":2200,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local D; D=$(for _ in 1 2 3; do
      ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":100,"temperature":0,"top_k":1,"cache_prompt":true}' \
      | python3 -c "import json,sys;print('%.2f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null
    done | sort -n | sed -n 2p)
  local ST; ST=$(docker logs "$NAME" 2>&1 | grep -a '\[stats\]' | tail -1 | sed 's/moestream: //')
  local OUT; OUT=$(ask '{"prompt":"The capital of France is","n_predict":14,"temperature":0,"top_k":1}' \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['content'][:48].replace(chr(10),' '))" 2>/dev/null)
  printf "%-26s %8s ms/tok   out='%s'\n" "$1" "${D:-FAIL}" "$OUT"
  [ -n "$ST" ] && echo "      $ST"
  docker rm -f "$NAME" >/dev/null 2>&1
}

ppl() {  # $1=label $2=env
  local OUT
  # shellcheck disable=SC2086
  OUT=$(timeout 2400 docker run --rm --entrypoint /opt/llama.cpp/build/bin/llama-perplexity \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -v "$PWD/research/bench":/bench:ro \
    -e XDG_RUNTIME_DIR=/tmp -e MOESTREAM_GGUF="/models/$MODEL_FILE" $2 "$IMG" \
    -m "/models/$MODEL_FILE" -f /bench/ppl.txt --chunks 30 -c 512 -b 512 -ub 512 -ngl 99 2>&1)
  local P; P=$(echo "$OUT" | grep -oE 'Final estimate: PPL = [0-9.]+' | grep -oE '[0-9.]+$')
  local X; X=$(echo "$OUT" | grep -ac 'exhaust\|WARNING')
  printf "%-26s PPL = %-12s (warnings=%s)\n" "$1" "${P:-FAILED}" "$X"
}

# SPEED-ONLY RE-RUN. The first pass of S24 was invalidated: an orphaned
# ms-s21 container (its script died before its cleanup line) held 12.9 GiB
# of GTT throughout. On UMA that is system RAM, and finding S19 showed this
# model's decode is 98.7% page-cache served -- so starving the page cache
# changes exactly what is being measured.
trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT INT TERM
echo "## S24 speed re-run (clean machine)   $(date -Iseconds)"
echo "   $MODEL_FILE / frac ${MOESTREAM_CACHE_FRAC} / ub ${UBATCH} / ctx ${CTX_SIZE}"
echo "   reference today: 58.15 ms/tok full reads, 46.33 ms/tok zero I/O (S20)"
echo
echo "### speed"
speed "off (reference)"        ""
speed "SKIP_RANK=7 (-20% B)"   "-e MOESTREAM_SKIP_RANK=7"
speed "SKIP_RANK=6 (-37% B)"   "-e MOESTREAM_SKIP_RANK=6"
speed "SKIP_RANK=5 (-52% B)"   "-e MOESTREAM_SKIP_RANK=5"
speed "SKIP_RANK=4 (-64% B)"   "-e MOESTREAM_SKIP_RANK=4"
echo
echo "### quality   (docs §8.1: baseline 5.0919, MoEStream frac=0.25 5.1040)"
ppl "MOESTREAM=0 (baseline)"   "-e MOESTREAM=0"
ppl "off (reference)"          "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25"
ppl "SKIP_RANK=7"              "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25 -e MOESTREAM_SKIP_RANK=7"
ppl "SKIP_RANK=6"              "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25 -e MOESTREAM_SKIP_RANK=6"
ppl "SKIP_RANK=5"              "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25 -e MOESTREAM_SKIP_RANK=5"
