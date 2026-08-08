# tests — what can be checked without hardware

    make test        # tens of seconds. No GPU, no model, no Docker

## What this catches

| Test | Subject | Bugs actually hit |
|---|---|---|
| `test_entrypoint_learn.sh` | candidate selection for `UBATCH=learn` | on a first start with no state file, awk exited with code 2 and "cannot open" was indistinguishable from "already measured" / handling of rows with rate 0 |
| `test_expert_cache.cpp` | S3-FIFO invariants | without a ceiling on `freq`, second chances never run out, eviction spins without freeing anything, and the seats starve (the origin of `FREQ_MAX`) |
| `test_apply_patch.py` | patching llama.cpp | only the `src/CMakeLists.txt` substitution was unguarded by an assert, so it passed silently even after its anchor disappeared |

`test_expert_cache.cpp` is built with AddressSanitizer and UndefinedBehaviorSanitizer
enabled. An undefined behaviour of the form `int32_t worst = -1 << 30;` once lurked
here, so this class of thing is left for the compiler to find.

## What this does not catch

**Performance and output correctness need the real machine.** For example, none of
the following is detectable here:

- the bug where every read was being thrown away yet it looked "faster" (§10.8).
  Nothing revealed it until the output was read by eye
- the bug where the prefill measurement stretched and shrank with the micro-batch
  size (§10.16). It took actually pushing 22,828 tokens through and comparing
  against llama.cpp's own figure

Those are checked by `make bench` and the procedure in `docs/RESULTS.md` §12.
**These tests exist to prevent obvious regressions; they are not a substitute for
measurement.**

## About the environment

`ASAN_OPTIONS=detect_leaks=0` is set, because LeakSanitizer cannot start in
environments with a restricted PID namespace (some containers). What we want to
detect is bounds violations and undefined behaviour, so dropping leak detection
still serves the purpose.
