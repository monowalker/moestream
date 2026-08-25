#!/usr/bin/env bash
# S29 part 2 -- batching, prefill ubatch, perplexity, and the stock runtime.
# Split from part 1 only so a failure in one half does not cost the other.
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || { echo "locked" >&2; exit 9; }
fi
trap 'docker rm -f ms-s29 ms-stock >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
M=Qwen3.8-27B-IQ4_NL.gguf
NAME=ms-s29; PORT=18093
MTP3='--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7'

boot() {
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$M" -e CTX_SIZE="${3:-16384}" -e UBATCH="${4:-1024}" \
    -e N_PARALLEL="${5:-1}" -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 $1 ${2:+-e SPEC_DECODING="$2"} \
    -p "$PORT":8080 moestream/server:local >/dev/null || return 1
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done; return 1
}
ask() { curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$1"; }
P40='{"prompt":"Write about consensus algorithms in detail.","n_predict":40,"temperature":0,"top_k":1,"cache_prompt":true}'
mem() { local V G; V=$(cat /sys/class/drm/card*/device/mem_info_vram_used|head -1); G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used|head -1); echo "scale=2;($V+$G)/1073741824"|bc; }

# Aggregate throughput with K concurrent sequences. A per-request ms/tok would
# hide the point: batching does not make one sequence faster, it makes the
# machine produce more tokens for the same reads.
para() {  # $1 = K
  local K=$1 t0 t1
  ask "$P40" >/dev/null
  t0=$(date +%s.%N)
  for _ in $(seq 1 "$K"); do ask "$P40" >/dev/null & done; wait
  t1=$(date +%s.%N)
  python3 -c "print('%.1f'%(($t1-$t0)*1000/(40*$K)))"
}

echo "## S29 part 2   $(date -Iseconds)   model $M"
echo
echo "### 2. batching (frac=0.00). ms per token, aggregated over K sequences."
echo "    Unlike MTP every token here is real, so nothing is discarded."
for k in 1 2 4; do
  if boot "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "" 16384 1024 "$k"; then
    printf "%-32s %7s GiB %9s ms/tok\n" "N_PARALLEL=$k" "$(mem)" "$(para $k)"
  else printf "%-32s FAILED\n" "N_PARALLEL=$k"; fi
done
for k in 2 4; do
  if boot "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" "$MTP3" 16384 1024 "$k"; then
    printf "%-32s %7s GiB %9s ms/tok\n" "N_PARALLEL=$k + MTP" "$(mem)" "$(para $k)"
  else printf "%-32s FAILED\n" "N_PARALLEL=$k + MTP"; fi
done
echo
echo "### 5. prefill vs ubatch. S18 predicted break-even at ub~218."
echo "    13877-token document, cache_prompt=false."
for ub in 128 256 512 2048; do
 for fr in 1.00 0.00; do
  if boot "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=$fr" "" 16384 "$ub" 1; then
    PF=$(python3 -c "
import json;print(json.dumps({'prompt':open('research/bench/prompt_long.txt').read(),
'n_predict':1,'temperature':0,'top_k':1,'cache_prompt':False}))" \
      | curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @- \
      | python3 -c "
import json,sys
try: print('%.1f'%json.load(sys.stdin)['timings']['prompt_per_second'])
except Exception: print('FAIL')" 2>/dev/null)
    printf "%-32s %9s tok/s\n" "ub=$ub frac=$fr" "$PF"
  else printf "%-32s FAILED\n" "ub=$ub frac=$fr"; fi
 done
done
docker rm -f "$NAME" >/dev/null 2>&1
echo
echo "### 6. perplexity on the dense path (ppl.txt, 30 chunks, -c 512 -ub 512)"
for fr in 1.00 0.40 0.00; do
  OUT=$(timeout 3600 docker run --rm --entrypoint /opt/llama.cpp/build/bin/llama-perplexity \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -v "$PWD/research/bench":/bench:ro \
    -e XDG_RUNTIME_DIR=/tmp -e MOESTREAM_GGUF="/models/$M" \
    -e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=$fr moestream/server:local \
    -m "/models/$M" -f /bench/ppl.txt --chunks 30 -c 512 -b 512 -ub 512 -ngl 99 2>&1)
  printf "%-32s PPL = %s\n" "dense_frac=$fr" \
    "$(echo "$OUT" | grep -oE 'Final estimate: PPL = [0-9.]+' | grep -oE '[0-9.]+$' || echo FAILED)"
done
echo
echo "### 7. stock llama.cpp (master) vs MoEStream's pinned 3581ba0c, same model"
docker rm -f ms-stock >/dev/null 2>&1
if docker run -d --name ms-stock --device /dev/dri:/dev/dri --group-add 992 --group-add 44 \
     -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp -p 18094:8080 \
     --entrypoint /usr/local/bin/llama-server llama-vulkan:latest \
     -m "/models/$M" --host 0.0.0.0 --port 8080 -c 16384 -ngl 99 -b 4096 -ub 1024 \
     --cache-type-k q8_0 --cache-type-v q8_0 -fa on >/dev/null 2>&1; then
  for i in $(seq 1 200); do curl -sf http://127.0.0.1:18094/health >/dev/null 2>&1 && break
    docker ps -q -f name=ms-stock | grep -q . || break; sleep 3; done
  D=$(curl -s -m 900 -X POST http://127.0.0.1:18094/completion -H 'Content-Type: application/json' -d "$P40" \
      | python3 -c "import json,sys;print('%.1f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null)
  printf "%-32s %7s GiB %9s ms/tok\n" "stock llama.cpp master" "$(mem)" "${D:-FAIL}"
else echo "stock image failed to start"; fi
docker rm -f ms-stock >/dev/null 2>&1
