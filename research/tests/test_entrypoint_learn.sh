#!/usr/bin/env bash
# =============================================================================
# UBATCH=learn selection logic
#
#   Extracts just the learn block from src/entrypoint.sh and checks that its
#   decisions are right while swapping the state files underneath it. No
#   container, no GPU, no model; finishes in seconds.
#
#   What this was written to catch is the two bugs actually hit:
#     - on a first start with no state file, awk exited with code 2, so "cannot
#       open" and "already measured" were indistinguishable and it fell back
#     - handling of rows with rate 0. Confusing "could not be measured" with
#       "measured and slow" either retires a candidate forever or retries it
#       forever
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
BLOCK=$(mktemp); trap 'rm -f "$BLOCK" "$OUT"' EXIT
OUT=$(mktemp)
sed -n '/^if \[\[ "\$UBATCH" == "learn" \]\]; then$/,/^fi$/p' src/entrypoint.sh > "$BLOCK"
[ -s "$BLOCK" ] || { echo "FAIL: cannot extract the learn block from src/entrypoint.sh"; exit 1; }

D=$(mktemp -d); trap 'rm -rf "$D" "$BLOCK" "$OUT"' EXIT
export MOESTREAM_STATE_DIR=$D MODEL_FILE=m.gguf
pass=0; fail=0

# run <expected UBATCH> <expected frac> <description> [value of CACHE_FRAC]
run() {
    local want_ub=$1 want_frac=$2 desc=$3
    UBATCH=learn; MOESTREAM_CACHE_FRAC="${4:-learn}"
    . "$BLOCK" > "$OUT" 2>&1
    if [ "$UBATCH" = "$want_ub" ] && [ "$FRAC" = "$want_frac" ]; then
        pass=$((pass+1)); printf '  ok   %s\n' "$desc"
    else
        fail=$((fail+1))
        printf '  FAIL %s\n       want UBATCH=%s frac=%s / got UBATCH=%s frac=%s\n' \
               "$desc" "$want_ub" "$want_frac" "$UBATCH" "$FRAC"
        sed 's/^/       | /' "$OUT"
    fi
}

echo "UBATCH=learn selection logic"

run 1024 0.15 "first start with no state file (awk cannot open it)"

printf 'm.gguf\t0.15\t256:60727296\n' > $D/tuning.tsv
run 1024 0.15 "frac learned, ubatch not yet measured"

printf 'm.gguf\t0.15\t1024\t220.0\n' > $D/ubatch.tsv
run 2048 0.15 "1024 measured -> move on to the next candidate"

printf 'm.gguf\t0.15\t1024\t220.0\nm.gguf\t0.15\t2048\t224.3\nm.gguf\t0.15\t4096\t210.0\nm.gguf\t0.15\t8192\t196.0\n' > $D/ubatch.tsv
run 2048 0.15 "all candidates measured -> take the fastest (2048=224.3)"

printf 'm.gguf\t0.25\t256:60727296\n' > $D/tuning.tsv
run 1024 0.25 "* frac moved 0.15->0.25 -> old measurements no longer match, remeasure"

printf 'm.gguf\t0.15\t2048\t224.3\nm.gguf\t0.25\t1024\t249.1\nm.gguf\t0.25\t2048\t226.8\nm.gguf\t0.25\t4096\t210.4\nm.gguf\t0.25\t8192\t196.8\n' > $D/ubatch.tsv
run 1024 0.25 "* picks the fastest at frac=0.25 (1024), not dragged by the 0.15 rows"

printf 'm.gguf\t0.25\t1024\t249.1\nm.gguf\t0.25\t2048\t0.0\nm.gguf\t0.25\t4096\t210.4\nm.gguf\t0.25\t8192\t196.8\n' > $D/ubatch.tsv
run 1024 0.25 "rate 0 (not comparable on this workload) is not retried"

printf 'other.gguf\t0.25\t8192\t999.9\nm.gguf\t0.25\t1024\t249.1\nm.gguf\t0.25\t2048\t226.8\nm.gguf\t0.25\t4096\t210.4\nm.gguf\t0.25\t8192\t196.8\n' > $D/ubatch.tsv
run 1024 0.25 "rows for another model are ignored (does not pick up other.gguf 999.9)"

run 1024 0.40 "an explicit numeric frac takes precedence" 0.40

printf 'garbage\n\t\t\nm.gguf\t0.25\t1024\t249.1\n' > $D/ubatch.tsv
run 2048 0.25 "a corrupt row mixed in does not crash it"

chmod 000 $D 2>/dev/null
run 1024 0.15 "an unreadable state directory does not stop startup"
chmod 755 $D

echo "  ---- $pass passed / $fail failed"
exit $((fail > 0))
