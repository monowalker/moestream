#!/usr/bin/env bash
# =============================================================================
# Spike S37 -- does the free zone appear at a large enough batch?
#
# S27 found no free zone for dense decode and explained why:
#     one layer's read    143 MiB / 9.6 GB/s = 15.6 ms
#     one layer's compute 211 ms / 65 layers =  3.3 ms      -> reads outrun 4.7x
#
# But that ratio is per PASS. Compute per pass scales with the tokens in it and
# the reads do not, so at K sequences the window is ~3.3K ms against the same
# 15.6 ms. The arithmetic says the reads become fully hideable around K = 5, and
# beyond that streaming should converge on plain llama.cpp.
#
# S34 already shows the gap closing: 2.86x at K=1, 1.96x at K=4. If the
# reasoning holds it keeps closing. If it plateaus, the window is not the limit
# and S27's explanation is incomplete.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1; exec flock -n /tmp/moestream-spike.lock "$0" "$@" || { echo locked >&2; exit 9; }
fi
trap 'docker rm -f ms-s37 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
M=Qwen3.8-27B-IQ4_NL.gguf; NAME=ms-s37; PORT=18115
P='{"prompt":"Write about consensus algorithms in detail.","n_predict":40,"temperature":0,"top_k":1,"cache_prompt":true}'
cell() {
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add 992 --group-add 44 -v "$MODEL_DIR":/models:ro \
    -e XDG_RUNTIME_DIR=/tmp -e MODEL_FILE="$M" -e CTX_SIZE=8192 -e UBATCH=1024 \
    -e N_PARALLEL="$3" -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 $2 -p "$PORT":8080 moestream/server:local >/dev/null
  for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    docker ps -q -f name="$NAME" | grep -q . || { printf "%-34s FAILED\n" "$1"; return; }; sleep 3; done
  local V G; V=$(cat /sys/class/drm/card*/device/mem_info_vram_used|head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used|head -1)
  curl -s -m 900 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$P" >/dev/null
  local t0 t1; t0=$(date +%s.%N)
  for _ in $(seq 1 "$3"); do curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$P" >/dev/null & done
  wait; t1=$(date +%s.%N)
  printf "%-34s %7s GiB %10s ms/tok\n" "$1" "$(echo "scale=2;($V+$G)/1073741824"|bc)" \
    "$(python3 -c "print('%.1f'%(($t1-$t0)*1000/(40*$3)))")"
  docker rm -f "$NAME" >/dev/null 2>&1
}
echo "## S37 -- does the gap keep closing with batch size?   $(date -Iseconds)"
echo "   S34: 2.86x at K=1, 1.96x at K=4. Prediction: reads fully hidden around K=5-8."
echo
printf "%-34s %11s %13s\n" "configuration" "memory" "aggregate"
for k in 1 2 4 8 16; do
  cell "plain llama.cpp, K=$k"     "-e MOESTREAM=0" "$k"
  cell "streaming frac=0.00, K=$k" "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "$k"
  echo
done
