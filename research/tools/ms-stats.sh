#!/usr/bin/env bash
# MoEStream — pull measurements out of a running server
#
#   Usage:
#     research/tools/ms-stats.sh              # read from the default container (moestream)
#     research/tools/ms-stats.sh <container>  # name the container explicitly
#
#   What it reports:
#     [stats] current hit rate
#     [mrc]   slots -> hit rate curve, and a recommended MOESTREAM_CACHE_FRAC
#     [ub]    UBATCH -> predicted prefill speed, and a recommended UBATCH
#
#   How it works: send SIGUSR1 and let the server print at the next token
#   boundary. The server must therefore be running and have served at least one
#   request.
set -euo pipefail
C="${1:-moestream}"

if ! docker ps --format '{{.Names}}' | grep -qx "$C"; then
  echo "error: container '$C' is not running" >&2
  echo "  running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" >&2
  exit 1
fi

BEFORE=$(docker logs "$C" 2>&1 | grep -ac '\[mrc\]' || true)

docker kill -s USR1 "$C" >/dev/null 2>&1 || {
  echo "error: could not send SIGUSR1" >&2; exit 1; }

# The report is emitted between tokens, so nudge the server with a tiny request.
PORT=$(docker port "$C" 8080 2>/dev/null | head -1 | sed 's/.*://') || true
if [[ -n "${PORT:-}" ]]; then
  curl -s -m 30 -X POST "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"hi","n_predict":2,"temperature":0,"cache_prompt":false}' >/dev/null 2>&1 || true
fi

for _ in $(seq 1 20); do
  AFTER=$(docker logs "$C" 2>&1 | grep -ac '\[mrc\]' || true)
  [[ "$AFTER" -gt "$BEFORE" ]] && break
  sleep 1
done

# Do not abort when grep finds nothing: some statistics need more traffic first.
section() {   # $1 = heading, $2 = grep pattern, $3 = line count
  local out
  out=$(docker logs "$C" 2>&1 | grep -aE "$2" | tail -"$3" | sed 's/^moestream: //' || true)
  echo "-----------------------------------------------------------"
  if [[ -n "$out" ]]; then echo "$out"; else echo "  (no $1 yet)"; fi
}

echo "================= MoEStream statistics ($C) ================="
docker logs "$C" 2>&1 | grep -aE '^moestream: (enabled|top_k|GGUF =|[0-9]+ layers|first [0-9]+ layer|prefill arena)' | tail -6 || true
section "hit rate" '\[stats\]' 1
section "slot curve — appears after a few thousand tokens" '\[mrc\]' 18
section "UBATCH recommendation — appears after a few long prompts" '\[ub\]' 14
section "prefetch decision" '\[prefetch\]' 6
section "I/O thread auto-tuning" '\[io\]' 8
echo "==========================================================="
echo
echo "  device memory: $(python3 - <<'PY'
import glob
v=g=0
for p in glob.glob('/sys/class/drm/card*/device/mem_info_vram_used'):
    v=int(open(p).read()); break
for p in glob.glob('/sys/class/drm/card*/device/mem_info_gtt_used'):
    g=int(open(p).read()); break
print(f"{(v+g)/1073741824:.2f} GiB")
PY
)"
