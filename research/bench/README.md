# bench — measurement inputs and summaries

## Corpora

Used to drive expert-selection traces across domains (docs/findings/M0-2).
All text here is either this project's own or comes from llama.cpp (MIT):

| File | Source | License |
|---|---|---|
| `corpus/en.txt`, `ppl.txt` | llama.cpp `README.md` | MIT (ggml-org/llama.cpp) |
| `corpus/code.txt`, `corpus/mixed.txt` | llama.cpp `src/*.cpp` | MIT (ggml-org/llama.cpp) |
| `corpus/ja.txt` | this project's `docs/DESIGN.md` | this repository's license |
| `prompt_long.txt` | assembled from the above | as above |

## Traces

`*_analysis.json` are kept: they are primary data (reuse-distance curves, hit
rates) and more reusable than the prose that quotes them.

The raw `*.trace` files they were computed from are **not** in the repository --
they run to tens of megabytes each. Regenerate them with
`make tool TOOL=expert_trace`; see docs/RESULTS.md §12 for the procedure.

## Scripts

`agent_turns.py` replays an agent-style multi-turn conversation against a
running server, for measuring what repeated tool-use traffic does to the cache.
