#!/usr/bin/env bash
# =============================================================================
# Spike S19 -- how much of decode "I/O" actually reaches the SSD?
#
# RESULTS.md §10.12 attributes 12.45 ms/token of decode to I/O. The runtime's
# own [io] auto-tuner, however, reports effective read bandwidths above the
# device's saturated 4.48 GB/s (§4.1) on models whose expert set fits in RAM.
# That is physically impossible from the device, so the reads must be served
# by the page cache.
#
# This measures it with an instrument that does not depend on the runtime's
# timing at all: the kernel's block-device counters. /sys/block/<dev>/stat
# field 3 is sectors read, and it counts only I/O that actually reached the
# device -- page-cache hits never appear there.
#
#   requested bytes = (misses from moestream's own [stats]) x bytes-per-expert
#   device bytes    = (delta sectors) x 512
#   page-cache share = 1 - device/requested
#
# Usage: research/spikes/s19_pagecache_share/measure.sh [n_tokens]
# Requires .env to name the model; boots its own container on port 18081.
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

NTOK="${1:-2500}"
trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT INT TERM
NAME=ms-s19
PORT=18081

SRC=$(df --output=source "$MODEL_DIR" | tail -1)
DEV=$(lsblk -ndo PKNAME "$SRC" 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename "$SRC")
STAT=/sys/block/$DEV/stat
[ -r "$STAT" ] || { echo "cannot read $STAT" >&2; exit 1; }
sectors() { awk '{print $3}' "$STAT"; }
# Per-process counter: /proc/<pid>/io read_bytes counts only bytes actually
# fetched from the storage layer, and unlike the device counter it is immune to
# the other containers on this machine. Read it inside the container, where we
# are root.
procio() { docker exec "$NAME" sh -c \
  'p=$(pgrep -n llama-server || echo 1); awk "/^read_bytes:/{print \$2}" /proc/$p/io' 2>/dev/null; }

echo "## S19 -- page-cache share of decode reads"
echo "  model      : $MODEL_FILE"
echo "  frac       : ${MOESTREAM_CACHE_FRAC}"
echo "  block dev  : $DEV  ($STAT)"
echo "  RAM        : $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
echo

docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
docker run -d --name "$NAME" --entrypoint /usr/local/bin/moestream-entrypoint \
  --device /dev/dri:/dev/dri --group-add "${RENDER_GID:-992}" --group-add "${VIDEO_GID:-44}" \
  -v "$MODEL_DIR":/models:ro -e XDG_RUNTIME_DIR=/tmp \
  -e MODEL_FILE="$MODEL_FILE" -e CTX_SIZE="${CTX_SIZE:-32768}" -e UBATCH="${UBATCH:-1024}" \
  -e N_PARALLEL=1 -e FLASH_ATTN=on -e CACHE_TYPE_K=q8_0 -e CACHE_TYPE_V=q8_0 \
  -e ENABLE_WEBUI=0 -e MOESTREAM=1 -e MOESTREAM_CACHE_FRAC="${MOESTREAM_CACHE_FRAC:-0.25}" \
  -p "$PORT":8080 moestream/server:local >/dev/null || exit 1

for i in $(seq 1 400); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  docker ps -q -f name="$NAME" | grep -q . || { echo "container died"; docker logs "$NAME" 2>&1|tail -20; exit 1; }
  sleep 3
done

ask() { curl -s -m 3600 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' -d "$1"; }
misses() { docker logs "$NAME" 2>&1 | grep -a '\[stats\]' | tail -1 \
           | sed 's/.*miss \([0-9]*\)).*/\1/'; }
toks()   { docker logs "$NAME" 2>&1 | grep -a '\[stats\]' | tail -1 \
           | sed 's/.*tokens=\([0-9]*\).*/\1/'; }

# Warm up past the first [stats] line (they print every 2000 tokens) so the
# measured window is steady state, with the page cache in whatever state real
# use leaves it in.
echo "warming up (this also decides the page-cache state under test)..."
while [ -z "$(misses)" ]; do
  ask "{\"prompt\":\"Write a detailed essay about distributed consensus.\",\"n_predict\":600,\"temperature\":0,\"top_k\":1,\"cache_prompt\":false}" >/dev/null
done

M0=$(misses); T0=$(toks); S0=$(sectors); P0=$(procio); W0=$(date +%s.%N)
echo "  window start: tokens=$T0 misses=$M0"

# Generate until the next [stats] boundary past NTOK
target=$((T0 + NTOK))
while [ "$(toks)" -lt "$target" ]; do
  ask "{\"prompt\":\"Write a detailed essay about distributed consensus.\",\"n_predict\":600,\"temperature\":0,\"top_k\":1,\"cache_prompt\":false}" >/dev/null
done

M1=$(misses); T1=$(toks); S1=$(sectors); P1=$(procio); W1=$(date +%s.%N)
echo "  window end  : tokens=$T1 misses=$M1"
echo

# bytes per expert = streamed total / (layers x experts); read it from the log
EB=$(docker logs "$NAME" 2>&1 | grep -aoE 'slots/layer' >/dev/null && echo "")
echo "  raw: M0=$M0 M1=$M1 T0=$T0 T1=$T1 S0=$S0 S1=$S1 P0=${P0:-} P1=${P1:-}"
python3 - "$M0" "$M1" "$T0" "$T1" "$S0" "$S1" "$W0" "$W1" "${P0:-0}" "${P1:-0}" <<'PY'
import sys
m0,m1,t0,t1,s0,s1 = (int(x) for x in sys.argv[1:7])
w0,w1 = float(sys.argv[7]), float(sys.argv[8])
p0,p1 = int(sys.argv[9]), int(sys.argv[10])
EXPERT_BYTES = 14.48*1024**3/(40*256)     # Ornith-1.0: 14.48 GiB / 40 layers / 256 experts
dm, dt, ds, dw = m1-m0, t1-t0, s1-s0, w1-w0
req = dm*EXPERT_BYTES
dev = ds*512
print(f"  tokens in window   : {dt:,}")
print(f"  cache misses       : {dm:,}   ({dm/dt:.1f}/token)")
print(f"  bytes requested    : {req/1024**3:8.2f} GiB   ({req/dt/1024**2:.1f} MiB/token)")
print(f"  bytes from device  : {dev/1024**3:8.2f} GiB   ({dev/dt/1024**2:.1f} MiB/token)   [device-wide, all containers]")
if p1 > p0:
    pio = p1-p0
    print(f"  server read_bytes  : {pio/1024**3:8.2f} GiB   ({pio/dt/1024**2:.1f} MiB/token)   [this process only]")
print()
base = (p1-p0) if p1 > p0 else dev      # prefer the per-process counter
share = 1-base/req if req else 0
print(f"  ==> page-cache share of decode reads: {share*100:.1f}%")
print(f"  ==> device share                    : {(1-share)*100:.1f}%")
print()
print(f"  wall time in window: {dw:.1f} s   ({dw/dt*1000:.1f} ms/token)")
print(f"  requested bandwidth: {req/dw/1e9:.2f} GB/s   (vs 4.48 GB/s device ceiling, §4.1)")
print(f"  device bandwidth   : {base/dw/1e9:.2f} GB/s")
PY

echo
docker kill -s USR1 "$NAME" >/dev/null 2>&1; sleep 4
docker logs "$NAME" 2>&1 | grep -aE '\[io\]|\[prefetch\]|\[stats\]' | tail -12 | sed 's/^/  /'
docker rm -f "$NAME" >/dev/null 2>&1
