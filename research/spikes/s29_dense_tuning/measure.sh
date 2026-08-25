#!/usr/bin/env bash
# =============================================================================
# Spike S29 -- everything still untested about dense streaming
#
# S27/S28 established the shape (memory -56%, prefill free at ub=1024, MTP 2.1x).
# What they did not do is tune it, or check the claims that are about to go into
# the README. Seven questions, all environment knobs, no code changes:
#
#   1 I/O threads   dense_read_range inherits g_nthreads, which the [io] tuner
#                   optimised for scattered 1.4 MiB EXPERT reads. A dense layer
#                   is a 143 MiB contiguous read -- a different shape entirely.
#                   S19 saw 2->8 threads move bandwidth 3.94 -> 6.05 GB/s.
#   2 N_PARALLEL    A dense pass reads the same bytes at any token count, so two
#                   concurrent sequences halve bytes per token. Same mechanism as
#                   MTP, but BOTH tokens are real -- no acceptance loss.
#   3 MTP n_max     Only 3 was tried (acceptance 0.571). More tokens per pass
#                   amortise more but accept less; there should be an optimum.
#   4 arena buffers Only 2 (one layer of lookahead). A layer's read is 15.6 ms
#                   against 3.3 ms of compute, so one layer ahead cannot keep up.
#   5 prefill ub    "prefill is free" rests on a single point, ub=1024. S18
#                   predicted break-even at ub~218; that has never been checked.
#   6 perplexity    "no accuracy loss" rests on byte-identical greedy output over
#                   a few dozen tokens. PPL is the same yardstick RESULTS.md §8
#                   uses, and it has never been run on the dense path.
#   7 stock         llama.cpp master (the stock image) vs the pinned 3581ba0c,
#                   on the same model, to isolate the runtime.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || { echo "locked" >&2; exit 9; }
fi
trap 'docker rm -f ms-s29 >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
M=Qwen3.8-27B-IQ4_NL.gguf
NAME=ms-s29; PORT=18093
MTP3='--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7'

boot() {  # $1=env  $2=spec  $3=ctx  $4=ub  $5=nparallel
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

dec() {  # median-of-2 warm decode, ms/tok
  ask "$P40" >/dev/null
  for _ in 1 2; do ask "$P40" | python3 -c "import json,sys;print('%.1f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null; done | sort -n | sed -n 1p
}
mem() { local V G; V=$(cat /sys/class/drm/card*/device/mem_info_vram_used|head -1); G=$(cat /sys/class/drm/card*/device/mem_info_gtt_used|head -1); echo "scale=2;($V+$G)/1073741824"|bc; }

row() {  # $1=label $2=env $3=spec $4=ctx $5=ub $6=np
  if ! boot "$2" "$3" "$4" "$5" "$6"; then printf "%-32s FAILED\n" "$1"; docker logs "$NAME" 2>&1|tail -4; return; fi
  printf "%-32s %7s GiB %9s ms/tok\n" "$1" "$(mem)" "$(dec)"
  docker rm -f "$NAME" >/dev/null 2>&1
}

echo "## S29 -- dense tuning   $(date -Iseconds)   model $M"
echo "   S27 reference: frac0.00 = 6.90 GiB / 690.8 ms ; +MTP = 7.82 GiB / 329.6 ms"
echo
echo "### 1. I/O threads (frac=0.00, dense reads are 143 MiB contiguous)"
for t in 1 2 4 8 16; do
  row "threads=$t" "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00 -e MOESTREAM_IO_THREADS=$t" "" 16384 1024 1
done
echo
echo "### 4. arena buffers (frac=0.00)"
for b in 3 4; do
  row "bufs=$b" "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00 -e MOESTREAM_DENSE_BUFS=$b" "" 16384 1024 1
done
echo
echo "### 3. MTP n_max (frac=0.00)"
for n in 1 5 7; do
  row "mtp n_max=$n" "-e MOESTREAM=1 -e MOESTREAM_DENSE_FRAC=0.00" \
      "--spec-type draft-mtp --spec-draft-n-max $n --spec-draft-p-min 0.7" 16384 1024 1
done
