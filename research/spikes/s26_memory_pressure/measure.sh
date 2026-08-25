#!/usr/bin/env bash
# =============================================================================
# Spike S26 -- the memory/speed trade-off, actually traded
#
# The project's claim is "runs, and leaves room for everything else": plain
# llama.cpp holds 17.49 GiB of GTT, MoEStream 7.65 GiB, so ~9.8 GiB is freed
# for other applications. That part is not in doubt -- GTT is a hard driver
# allocation and the difference is real.
#
# But finding S19 showed 98.7% of this model's decode reads are served from the
# page cache, which lives in the SAME free memory that was just handed to other
# applications. So the freed memory and the speed are coupled, and nobody has
# measured how tightly:
#
#     "9.8 GiB freed" is measured.
#     "9.8 GiB freed AND still 58 ms/token" is NOT -- it was only ever measured
#     with that 9.8 GiB sitting idle.
#
# This occupies the freed memory with a ballast process, the way a real
# co-tenant would, and measures decode at each level. The result is the actual
# exchange rate of the project's central trade.
#
# The ballast runs in a container with a hard --memory cap, so if anything is
# over-committed Docker kills the ballast rather than the OOM killer picking a
# victim among the machine's other services.
# =============================================================================
set -uo pipefail
if [ -z "${MS_SPIKE_LOCK:-}" ]; then
  export MS_SPIKE_LOCK=1
  exec flock -n /tmp/moestream-spike.lock "$0" "$@" || {
    echo "another spike holds /tmp/moestream-spike.lock; refusing to run" >&2; exit 9; }
fi
trap 'docker rm -f ms-s26 ms-ballast >/dev/null 2>&1' EXIT INT TERM
cd "$(dirname "$0")/../../.."
set -a; . ./.env; set +a
IMG=${IMG:-moestream/server:local}
NAME=ms-s26; PORT=18087

ballast_up() {   # $1 = GiB
  docker rm -f ms-ballast >/dev/null 2>&1; sleep 1
  [ "$1" -eq 0 ] && return 0
  docker run -d --name ms-ballast --memory="$1"g --memory-swap="$1"g \
    python:3-slim python3 -c "
import time
n = int($1 * 1024 * 0.92)          # MiB, leaving headroom under the cap
buf = []
for _ in range(n):
    b = bytearray(1024*1024)
    b[::4096] = b'x' * len(b[::4096])   # touch every page so it is resident
    buf.append(b)
print('ballast resident', n, 'MiB', flush=True)
time.sleep(100000)" >/dev/null || return 1
  for i in $(seq 1 120); do
    docker logs ms-ballast 2>&1 | grep -q 'ballast resident' && return 0
    docker ps -q -f name=ms-ballast | grep -q . || { echo "  ballast died (cap too low)"; return 1; }
    sleep 2
  done; return 1
}

boot() {
  docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
  docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
    --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
    -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
    -e MODEL_FILE="$MODEL_FILE" -e CTX_SIZE="${CTX_SIZE:-32768}" -e UBATCH="${UBATCH:-1024}" \
    -e N_PARALLEL=1 -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
    -e ENABLE_WEBUI=0 -e MOESTREAM="$2" -e MOESTREAM_CACHE_FRAC="${MOESTREAM_CACHE_FRAC:-0.25}" \
    -p "$PORT":8080 "$IMG" >/dev/null || return 1
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$NAME" | grep -q . || return 1
    sleep 3
  done; return 1
}
ask() { curl -s -m 1800 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$1"; }
procio() { docker exec "$NAME" sh -c \
  'p=$(pgrep -n llama-server || echo 1); awk "/^read_bytes:/{print \$2}" /proc/$p/io' 2>/dev/null; }

run() {  # $1=ballast GiB $2=MOESTREAM
  if ! ballast_up "$1"; then printf "%5s GiB   ballast failed\n" "$1"; return; fi
  if ! boot "$NAME" "$2"; then printf "%5s GiB   server failed to boot\n" "$1"; docker rm -f ms-ballast >/dev/null 2>&1; return; fi
  ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":600,"temperature":0,"top_k":1,"cache_prompt":true}' >/dev/null
  local P0 P1 D
  P0=$(procio)
  D=$(for _ in 1 2 3; do
      ask '{"prompt":"Write about consensus algorithms in detail.","n_predict":100,"temperature":0,"top_k":1,"cache_prompt":true}' \
      | python3 -c "import json,sys;print('%.2f'%(1000/json.load(sys.stdin)['timings']['predicted_per_second']))" 2>/dev/null
    done | sort -n | sed -n 2p)
  P1=$(procio)
  local PC FREE
  PC=$(awk '/^Cached:/{printf "%.1f", $2/1048576}' /proc/meminfo)
  FREE=$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo)
  local DEV="n/a"
  [ -n "${P0:-}" ] && [ -n "${P1:-}" ] && [ "$P1" -gt "$P0" ] 2>/dev/null && \
    DEV=$(python3 -c "print('%.1f' % (($P1-$P0)/300/1048576))")
  printf "%5s GiB  %9s ms/tok   page cache %6s GiB   avail %6s GiB   device reads %8s MiB/tok\n" \
    "$1" "${D:-FAIL}" "$PC" "$FREE" "$DEV"
  docker rm -f "$NAME" ms-ballast >/dev/null 2>&1
}

echo "## S26 -- decode speed as the freed memory is actually used   $(date -Iseconds)"
echo "   $MODEL_FILE / frac ${MOESTREAM_CACHE_FRAC} / experts 14.48 GiB / host 30.6 GiB"
echo "   plain llama.cpp holds 17.49 GiB of GTT; MoEStream holds 7.65 GiB"
echo "   'device reads' is the per-process kernel counter (S19): what truly hits the SSD"
echo
printf "%9s  %9s\n" "ballast" "decode"
for g in 0 4 8 12 16; do run "$g" 1; done
echo
echo "note: plain llama.cpp keeps every weight in GTT and needs no page cache, so"
echo "      it degrades LESS than MoEStream under ballast -- until it simply does"
echo "      not fit. 17.49 GiB + ballast exceeds this 30.6 GiB host at about 13 GiB."
echo "      That crossover is arithmetic here, not measured."
