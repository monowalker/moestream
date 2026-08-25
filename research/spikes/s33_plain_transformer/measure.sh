#!/usr/bin/env bash
# =============================================================================
# Spike S33 -- dense streaming on a PLAIN transformer
#
# Every dense model tested so far (Qwen3.5-4B, Qwen3.8-27B) is an
# attention/SSM hybrid, so S27's -56% is a figure for that family and not for
# dense models generally. Two things could not be answered without a plain
# transformer:
#
#   1 how much of a dense model is FFN, when nothing is spent on SSM state
#   2 whether streaming ATTENTION is worth implementing. On Qwen3.8 only 17 of
#     65 layers use ordinary attention and the rest is SSM, so the recoverable
#     remainder was 0.95 GiB and finding S30 rejected the work on that basis.
#     Here every layer's attention goes through build_attn.
#
# gemma-4-31B-it: 60 layers, no expert_count, no ssm.*, no nextn. Attention
# alternates sliding-window and global -- both ordinary attention, no linear
# or state-space path.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || { echo locked >&2; exit 9; }
fi
trap 'docker rm -f ms-s33 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
M=${M:-gemma-4-31B-it-IQ4_NL.gguf}
CTX=${CTX:-16384}
NAME=ms-s33; PORT=18109

echo "## S33 -- plain transformer   $(date -Iseconds)   $M"
echo "### tensor budget (what fraction is FFN, with no SSM to pay for)"
python3 research/spikes/s18_dense_stream/analyze.py "$MODEL_DIR/$M" 2>/dev/null | head -12
echo

boot() {
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$M" -e CTX_SIZE="$CTX" -e UBATCH="${2:-1024}" -e N_PARALLEL=1 \
    -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 -e ENABLE_WEBUI=0 $1 \
    -p "$PORT":8080 moestream/server:local >/dev/null || return 1
  for i in $(seq 1 400); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1; sleep 3; done; return 1
}
ask() { curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$1"; }

row() {  # $1=label $2=env $3=ubatch
  if ! boot "$2" "${3:-1024}"; then printf "%-34s FAILED\n" "$1"; docker logs "$NAME" 2>&1|tail -5; return; fi
  local V G MEM; V=$(cat /sys/class/drm/card*/device/mem_info_vram_used|head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used|head -1)
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
  local D; D=$(ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":40,"temperature":0,"top_k":1,"cache_prompt":true}' \
    | python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print('%.1f'%(1000/t['predicted_per_second']))" 2>/dev/null)
  local OUT; OUT=$(ask '{"prompt":"The capital of France is","n_predict":20,"temperature":0,"top_k":1}' \
    | python3 -c "import json,sys;print(repr(json.load(sys.stdin)['content']))" 2>/dev/null)
  printf "%-34s %7s GiB %8s tok/s %9s ms/tok\n   %s\n" "$1" "$MEM" "${PF:-FAIL}" "${D:-FAIL}" "$OUT"
  docker logs "$NAME" 2>&1 | grep -aE "\[dense\]|\[BUG\]" | head -3 | sed 's/^/   /'
  docker rm -f "$NAME" >/dev/null 2>&1
}

echo "### correctness and the memory/speed curve"
row "plain llama.cpp"        "-e MOESTREAM=0"
row "auto"                   "-e MOESTREAM=1"
row "frac=0.50"              "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.50"
row "frac=0.00 (all FFN)"    "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00"
echo
echo "### is prompt processing still free? (S29 found ub>=1024 on the hybrid)"
row "frac=0.00, ub=256"      "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" 256
row "frac=0.00, ub=2048"     "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" 2048
echo
echo "### perplexity (docs method: ppl.txt, 30 chunks, -c 512 -ub 512)"
for fr in 1.00 0.00; do
  OUT=$(timeout 3600 docker run --rm --entrypoint /opt/llama.cpp/build/bin/llama-perplexity \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -v "$PWD/research/bench":/bench:ro \
    -e XDG_RUNTIME_DIR=/tmp -e MOESTREAM_GGUF="/models/$M" \
    -e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=$fr moestream/server:local \
    -m "/models/$M" -f /bench/ppl.txt --chunks 30 -c 512 -b 512 -ub 512 -ngl 99 2>&1)
  printf "%-34s PPL = %s\n" "dense_frac=$fr" \
    "$(echo "$OUT" | grep -oE 'Final estimate: PPL = [0-9.]+' | grep -oE '[0-9.]+$' || echo FAILED)"
done
