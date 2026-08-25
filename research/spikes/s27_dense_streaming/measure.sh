#!/usr/bin/env bash
# =============================================================================
# Spike S27 -- dense FFN streaming, measured
#
# Implements and measures what findings S18/S21 argued for: keep the first K
# layers of a dense model resident and stream the rest of their FFN weights
# through an arena. Accuracy is untouched -- the same bytes are used, just not
# all resident at once -- so this is the project's core trade (memory for
# speed, never for accuracy) applied to a model class the README currently
# calls "inert".
#
# MOESTREAM_DENSE_FRAC is the knob: fraction of layers kept resident.
#   1.00 = off (identical to plain llama.cpp)   0.00 = every FFN streamed
#
# Also compares against the stock llama.cpp image on this machine, including
# its MTP self-speculation (--spec-type draft-mtp), because:
#
#   Finding S13 rejected speculative decoding for MoE: verifying K tokens
#   raises the expert union, so I/O per accepted token goes UP ~4x.
#   **For dense the opposite should hold.** A dense pass reads the same weights
#   regardless of how many tokens are in it, so K tokens per pass divides the
#   streamed bytes per token by K. Speculation and dense streaming should
#   compose, not fight.
#
# Caveat on the cross-comparison: the stock image tracks llama.cpp master while
# MoEStream pins 3581ba0c, so stock-vs-MoEStream mixes two variables. The
# MoEStream rows are internally consistent.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || {
    echo "another spike holds /tmp/moestream-spike.lock; refusing to run" >&2; exit 9; }
fi
trap 'docker rm -f ms-s27 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
MODEL=${MODEL:-Qwen3.8-27B-IQ4_NL.gguf}
CTX=${CTX:-16384}
UB=${UB:-1024}
NAME=ms-s27; PORT=18088
MTP='--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7'

boot() {   # $1 = extra -e args
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$MODEL" -e CTX_SIZE="$CTX" -e UBATCH="$UB" \
    -e N_PARALLEL=1 -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 $1 -p "$PORT":8080 moestream/server:local >/dev/null || return 1
  for i in $(seq 1 500); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done; return 1
}
ask() { curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$1"; }

run() {  # $1=label $2=env
  if ! boot "$2"; then printf "%-34s FAILED TO BOOT\n" "$1"; docker logs "$NAME" 2>&1|tail -8; return; fi
  local V G MEM
  V=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null|head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null|head -1)
  MEM=$(echo "scale=2;($V+$G)/1073741824"|bc)
  local PF; PF=$(python3 -c "
import json;print(json.dumps({'prompt':open('research/bench/prompt_long.txt').read(),
'n_predict':1,'temperature':0,'top_k':1,'cache_prompt':False}))" \
    | curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @- \
    | python3 -c "
import json,sys
try: print('%.1f'%json.load(sys.stdin)['timings']['prompt_per_second'])
except Exception: print('FAIL')" 2>/dev/null)
  ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":40,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local D; D=$(for _ in 1 2 3; do
      ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":40,"temperature":0,"top_k":1,"cache_prompt":true}' \
      | python3 -c "import json,sys;print('%.1f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null
    done | sort -n | sed -n 2p)
  local OUT; OUT=$(ask '{"prompt":"The capital of France is","n_predict":12,"temperature":0,"top_k":1}' \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['content'][:40].replace(chr(10),' '))" 2>/dev/null)
  printf "%-34s %7s GiB  %8s tok/s  %9s ms/tok  '%s'\n" "$1" "$MEM" "${PF:-FAIL}" "${D:-FAIL}" "$OUT"
  docker logs "$NAME" 2>&1 | grep -a '\[dense\]' | head -3 | sed 's/^/      /'
  docker rm -f "$NAME" >/dev/null 2>&1
}

echo "## S27 -- dense FFN streaming   $(date -Iseconds)"
echo "   model $MODEL / ctx $CTX / ub $UB / KV q8_0 / ngl 99"
echo "   MOESTREAM_DENSE_FRAC = fraction of layers kept RESIDENT (1.00 = off)"
echo "   v1 streams the FFN only (~60% of a dense model); attention stays resident"
echo
printf "%-34s %11s %15s %16s\n" "configuration" "memory" "prefill" "decode"
run "plain llama.cpp (MOESTREAM=0)"   "-e MOESTREAM=0"
run "dense_frac=1.00 (streaming off)" "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=1.00"
run "dense_frac=0.80"                 "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.80"
run "dense_frac=0.60"                 "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.60"
run "dense_frac=0.40"                 "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.40"
run "dense_frac=0.00 (all FFN)"       "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00"
echo
echo "### with MTP self-speculation (the stock server's production setting)"
echo "   S13 found speculation makes MoE worse (expert union grows). A dense pass"
echo "   reads the same bytes whatever the token count, so it should divide them."
run "plain + MTP"                     "-e MOESTREAM=0 -e SPEC_DECODING=$(printf '%q' "$MTP")"
run "dense_frac=0.40 + MTP"           "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.40 -e SPEC_DECODING=$(printf '%q' "$MTP")"
run "dense_frac=0.00 + MTP"           "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00 -e SPEC_DECODING=$(printf '%q' "$MTP")"
