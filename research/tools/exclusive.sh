#!/usr/bin/env bash
# =============================================================================
# exclusive.sh — run a measurement so that only one can run at a time
#
#   research/tools/exclusive.sh <command> [args...]
#
#   Why this exists: two measurement runs were started concurrently more than
#   once during development. They fought over the GPU and the same state files,
#   and the numbers that came out were wrong in ways that looked plausible --
#   which is the worst kind of wrong. Discipline did not prevent it; a lock does.
#
#   What it guarantees:
#     1. Only one measurement runs at a time (flock, non-blocking: a second
#        invocation is refused immediately rather than queued, so it cannot
#        silently start later while the first is still going)
#     2. Any container it started is removed on exit, including on Ctrl-C or
#        kill, so a stray server cannot contend with the next run
#     3. It refuses to start if a measurement container is already up, even one
#        left behind by a process this shell cannot signal
# =============================================================================
set -uo pipefail

LOCK_FILE="${MOESTREAM_LOCK:-/tmp/moestream-measure.lock}"
# Container names measurements are allowed to use. Anything matching this
# prefix is treated as "a measurement is in progress".
NAME_PREFIX="${MOESTREAM_MEASURE_PREFIX:-ms-}"

if [ $# -eq 0 ]; then
    sed -n '2,25p' "$0"
    exit 2
fi

# ---- refuse if a measurement container is already running --------------------
# The lock alone is not enough: a previous run may have been killed in a way
# that released the lock while leaving its container up (or may live in a
# different PID namespace, where it cannot be signalled at all).
existing=$(docker ps --format '{{.Names}}' 2>/dev/null | grep "^${NAME_PREFIX}" || true)
if [ -n "$existing" ]; then
    echo "refusing to start: a measurement container is already running:" >&2
    echo "$existing" | sed 's/^/  /' >&2
    echo "" >&2
    echo "  Wait for it, or remove it:  docker rm -f $(echo "$existing" | tr '\n' ' ')" >&2
    exit 1
fi

# ---- clean up whatever we start ---------------------------------------------
cleanup() {
    local rc=$?
    local mine
    mine=$(docker ps -aq --filter "name=^${NAME_PREFIX}" 2>/dev/null || true)
    if [ -n "$mine" ]; then
        echo "[exclusive] removing measurement containers" >&2
        docker rm -f $mine >/dev/null 2>&1 || true
    fi
    exit $rc
}
trap cleanup EXIT INT TERM HUP

# ---- take the lock, or refuse ------------------------------------------------
exec 9>"$LOCK_FILE" || { echo "cannot open $LOCK_FILE" >&2; exit 1; }
if ! flock -n 9; then
    holder=$(cat "$LOCK_FILE" 2>/dev/null)
    echo "refusing to start: another measurement holds $LOCK_FILE" >&2
    [ -n "$holder" ] && echo "  held by: $holder" >&2
    echo "  Measurements must not overlap: they contend for the GPU and for" >&2
    echo "  ./state, and the resulting numbers look plausible but are wrong." >&2
    exit 1
fi
printf 'pid %s  started %s  cmd %s\n' "$$" "$(date -Iseconds)" "$*" >&9
echo "[exclusive] lock acquired ($LOCK_FILE)" >&2

"$@"
