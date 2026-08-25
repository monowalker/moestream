#!/usr/bin/env bash
# =============================================================================
# Spike S38 -- the same batch sweep on MoE, where the read set is not constant
#
# S37 found dense streaming converging on plain llama.cpp as the batch grows:
# 3.15x at K=1 down to 1.20x at K=16, because a dense pass reads the same bytes
# however many tokens it carries, so batching divides the reads per token.
#
# A MoE pass does not. The experts K tokens want is their UNION:
#     union(K) = 256 x (1 - (1 - 8/256)^K)
#     K=1  8.0     K=4  30.5     K=16 101.6
# so reads per token fall only from 8.0 to 6.4 across the whole sweep. The
# prediction is that MoE barely improves -- S36 already shows 1.61x -> 1.54x
# from K=1 to K=4 -- and that dense may end up the better trade at large K
# despite starting far worse.
#
# Watch for slot exhaustion: union(16) = 101.6 against 64 slots at frac=0.25.
# If the slab cannot hold a batch's union the output degrades, and that matters
# more than the timing.
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
M=Ornith-1.0-35B-UD-IQ4_NL.gguf; NAME=ms-s38; PORT=18117
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
  local EX OUT
  EX=$(docker logs "$NAME" 2>&1 | grep -ac "exhaust")
  OUT=$(curl -s -m 600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
        -d '{"prompt":"The capital of France is","n_predict":10,"temperature":0,"top_k":1}' \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['content'][:32].replace(chr(10),' '))" 2>/dev/null)
  printf "%-34s %7s GiB %10s ms/tok  exhaust=%-3s '%s'\n" "$1" "$(echo "scale=2;($V+$G)/1073741824"|bc)" \
    "$(python3 -c "print('%.1f'%(($t1-$t0)*1000/(40*$3)))")" "$EX" "$OUT"
  docker rm -f "$NAME" >/dev/null 2>&1
}
echo "## S38 -- the batch sweep on MoE   $(date -Iseconds)"
echo "   S37 (dense): 3.15x -> 1.20x from K=1 to K=16. Prediction here: barely moves."
echo
printf "%-34s %11s %13s\n" "configuration" "memory" "aggregate"
for k in 1 2 4 8 16; do
  cell "plain llama.cpp, K=$k"     "-e MOESTREAM=0" "$k"
  cell "expert streaming, K=$k" "-e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC=0.25" "$k"
  echo
done
