#!/usr/bin/env bash
# =============================================================================
# Spike S21 -- the dense compute ceiling, measured instead of extrapolated
#
# Finding S18 argues that a dense model streams for free during prefill above
# ubatch ~105. That argument rests on one extrapolated number: an effective
# 1.77 TFLOP/s for the Radeon 780M, derived from Ornith's MoE prefill, which
# gave a 32.4 tok/s compute ceiling for a 27 B dense model.
#
# Ornith is MoE. Its kernel shapes are not a dense model's, so extrapolating
# across is exactly the kind of step this project keeps catching itself making.
# Qwen3.8-27B-IQ4_NL is dense, 15.22 GiB, and fits in this machine's 23.5 GiB
# GTT -- so the ceiling can simply be measured.
#
# Also checks the README's claim that MoEStream is inert on a dense model.
# =============================================================================
set -uo pipefail
# Only one spike may run at a time: they all contend for the single GPU, and a
# second one starting mid-run silently corrupts both sets of numbers.
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || {
    echo "another spike holds /tmp/moestream-spike.lock; refusing to run" >&2; exit 9; }
fi
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
MODEL_FILE=Qwen3.8-27B-IQ4_NL.gguf
CTX=16384
trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT INT TERM
NAME=ms-s21; PORT=18083

boot() {
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$MODEL_FILE" -e CTX_SIZE="$CTX" -e UBATCH="$2" \
    -e N_PARALLEL=1 -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 -e MOESTREAM="$1" \
    -p "$PORT":8080 moestream/server:local >/dev/null || return 1
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done; return 1
}
ask() { curl -s -m 1800 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' -d "$1"; }

run() {  # $1=label $2=MOESTREAM $3=ubatch
  if ! boot "$2" "$3"; then echo "### $1: FAILED TO BOOT"; docker logs "$NAME" 2>&1|tail -8; return; fi
  local V G MEM
  V=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null|head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null|head -1)
  MEM=$(echo "scale=2;($V+$G)/1073741824"|bc)
  local PF; PF=$(python3 -c "
import json;print(json.dumps({'prompt':open('research/bench/prompt_long.txt').read(),
'n_predict':1,'temperature':0,'top_k':1,'cache_prompt':False}))" \
    | curl -s -m 1800 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @- \
    | python3 -c "
import json,sys
try: print('%.1f'%json.load(sys.stdin)['timings']['prompt_per_second'])
except Exception: print('FAILED')" 2>/dev/null)
  ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":100,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local D; D=$(for _ in 1 2 3; do
      ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":60,"temperature":0,"top_k":1,"cache_prompt":true}' \
      | python3 -c "import json,sys;print('%.2f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null
    done | sort -n | sed -n 2p)
  local MS; MS=$(docker logs "$NAME" 2>&1 | grep -acE '^moestream: (enabled|.*slots/layer)')
  echo "### $1"
  printf "   ubatch %-5s device memory %-7s GiB   prefill %-8s tok/s   decode %-9s ms/tok\n" \
    "$3" "$MEM" "${PF:-FAIL}" "${D:-FAIL}"
  docker logs "$NAME" 2>&1 | grep -a '^moestream:' | head -4 | sed 's/^/     /'
  echo
}

echo "## S21 -- dense model on this machine   $(date -Iseconds)"
echo "   model  $MODEL_FILE (15.22 GiB, 65 layers, dense; FFN 59.8% / attn+ssm 29.0%)"
echo "   ctx $CTX / KV q8_0 / FA on / ngl 99"
echo "   S18 predicted, by extrapolation: compute ceiling 32.4 tok/s prefill, ~0.31 tok/s if fully streamed"
echo
run "plain llama.cpp, ub=1024"  0 1024
run "plain llama.cpp, ub=2048"  0 2048
run "MOESTREAM=1 (should be inert on dense)" 1 1024
docker rm -f "$NAME" >/dev/null 2>&1
