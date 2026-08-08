#!/usr/bin/env bash
# =============================================================================
# Run every test. No GPU, no model, no Docker; finishes in tens of seconds.
#
#   What this catches is obvious regressions, not performance or output
#   correctness. Those need the real machine -- see make bench and
#   docs/RESULTS.md §12.
#   research/tests/README.md says what is and is not verifiable here.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
fail=0

run() {
    echo
    echo "=== $1 ==="
    shift
    "$@" || fail=1
}

run "entrypoint: UBATCH=learn selection logic" ./test_entrypoint_learn.sh
run "apply.py: patching llama.cpp" python3 ./test_apply_patch.py

echo
echo "=== ExpertCache: S3-FIFO ==="
BIN=$(mktemp -u)
if g++ -std=c++17 -O1 -Wall -Wextra -Wshadow -Werror \
       -fsanitize=address,undefined \
       -o "$BIN" test_expert_cache.cpp ../../src/expert_cache.cpp 2>&1; then
    # LeakSanitizer cannot run where the PID namespace is restricted (some
    # containers). We want bounds violations and undefined behaviour, so only
    # leak detection is turned off.
    ASAN_OPTIONS=detect_leaks=0 "$BIN" || fail=1
    rm -f "$BIN"
else
    echo "  FAIL does not compile"; fail=1
fi

echo
if [ $fail -eq 0 ]; then echo "all tests passed"; else echo "failures"; fi
exit $fail
