#!/usr/bin/env bash
# =============================================================================
# Spike S34 -- the full matrix, every cell measured the same way
#
# Three times in this project's reporting, a streaming result was compared
# against a baseline that had been left in a different configuration:
#
#   N_PARALLEL=4 streaming (190.8 ms) vs plain at K=1 (211 ms)  -> "faster than plain"
#   the same mistake repeated in a summary table after being corrected
#   MTP streaming (329.6 ms) vs plain WITHOUT MTP (211 ms)      -> "1.56x"
#
# Every one of those errors flattered the streaming side. That is not chance,
# so the fix is structural: this spike measures the baseline in *every*
# configuration the streamed side is measured in, and prints them adjacent.
#
# aggregate ms/tok = wall / total tokens across K sequences (throughput)
# per-seq   ms/tok = wall / one sequence's tokens          (latency)
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || { echo locked >&2; exit 9; }
fi
trap 'docker rm -f ms-s34 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
M=${M:-Qwen3.8-27B-IQ4_NL.gguf}
NAME=ms-s34; PORT=18112
MTP='--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7'
P='{"prompt":"Write about consensus algorithms in detail.","n_predict":40,"temperature":0,"top_k":1,"cache_prompt":true}'

cell() {  # $1=label $2=stream-env $3=spec $4=K
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$M" -e CTX_SIZE=16384 -e UBATCH=1024 -e N_PARALLEL="$4" \
    -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 -e ENABLE_WEBUI=0 \
    $2 ${3:+-e SPEC_DECODING="$3"} -p "$PORT":8080 moestream/server:local >/dev/null
  for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    docker ps -q -f name="$NAME" | grep -q . || { printf "%-38s FAILED\n" "$1"; return; }; sleep 3; done
  local V G MEM; V=$(cat /sys/class/drm/card*/device/mem_info_vram_used|head -1)
  G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used|head -1)
  MEM=$(echo "scale=2;($V+$G)/1073741824"|bc)
  curl -s -m 900 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$P" >/dev/null
  local t0 t1; t0=$(date +%s.%N)
  for _ in $(seq 1 "$4"); do curl -s -m 1800 -X POST "http://127.0.0.1:$PORT/completion" \
      -H 'Content-Type: application/json' -d "$P" >/dev/null & done
  wait; t1=$(date +%s.%N)
  printf "%-38s %7s GiB %10s %12s\n" "$1" "$MEM" \
    "$(python3 -c "print('%.1f'%(($t1-$t0)*1000/(40*$4)))")" \
    "$(python3 -c "print('%.1f'%(($t1-$t0)*1000/40))")"
  docker rm -f "$NAME" >/dev/null 2>&1
}

echo "## S34 -- like-for-like matrix   $(date -Iseconds)   $M"
echo "   Every streamed cell has its baseline measured in the SAME configuration."
echo
printf "%-38s %11s %10s %12s\n" "configuration" "memory" "aggregate" "per-seq"
echo "-- no MTP, one sequence ------------------------------------------------"
cell "plain llama.cpp"                 "-e MOESTREAM=0" "" 1
cell "streaming frac=0.00"             "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "" 1
echo "-- MTP, one sequence ---------------------------------------------------"
cell "plain llama.cpp + MTP"           "-e MOESTREAM=0" "$MTP" 1
cell "streaming frac=0.00 + MTP"       "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "$MTP" 1
echo "-- no MTP, four sequences ----------------------------------------------"
cell "plain llama.cpp, K=4"            "-e MOESTREAM=0" "" 4
cell "streaming frac=0.00, K=4"        "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "" 4
echo "-- MTP, four sequences -------------------------------------------------"
cell "plain llama.cpp + MTP, K=4"      "-e MOESTREAM=0" "$MTP" 4
cell "streaming frac=0.00 + MTP, K=4"  "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "$MTP" 4
