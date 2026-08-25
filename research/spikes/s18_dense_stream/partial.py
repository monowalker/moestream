#!/usr/bin/env python3
"""
S18 follow-up -- partial residency for a dense model, and why the LAST layers
are the right ones to stream.

Two things fall out that the all-or-nothing framing in S18 missed:

  1. It is a continuum, exactly like MOESTREAM_CACHE_FRAC. Keep K layers
     resident, stream the other 65-K. Memory and speed move together.

  2. For a dense model, WHICH layers you stream does not change bytes/token --
     every layer is used exactly once, unlike MoE where layer 0 misses on 52%
     of lookups and the rest on 11-19% (finding S20). But it does change how
     much can be HIDDEN: stream the tail, and the whole forward pass through
     the resident head is time you can read them in.

     And here dense beats MoE outright. MoE prefetch failed (findings N2/S10/
     S11/§10.14) because you cannot know which experts layer L+1 wants until
     layer L has run. For dense there is nothing to predict: layer 33 is always
     layer 33. Perfect prefetch, zero prediction cost.

Constants are measured, not assumed:
  compute, fully resident   211.96 ms/token   (S21)
  read bandwidth            4.48 GB/s device (§4.1) / ~9.3 GB/s page-cache (S19)
"""
G = 1024 ** 3
N_LAYER   = 65
BODY_GIB  = 13.57            # S18: FFN + attn/ssm across all layers
NONBODY   = 15.22 - BODY_GIB # embeddings, output, norms
LAYER_GIB = BODY_GIB / N_LAYER
COMPUTE_MS = 211.96          # S21, fully resident
ARENA_GIB = 2 * LAYER_GIB    # double-buffered, one layer each

print(f"dense 27B: {N_LAYER} layers x {LAYER_GIB*1024:.0f} MiB = {BODY_GIB:.2f} GiB body"
      f" + {NONBODY:.2f} GiB non-body")
print(f"fully resident: 15.22 GiB, {COMPUTE_MS:.0f} ms/token ({1000/COMPUTE_MS:.2f} tok/s)\n")

for bw_name, bw in (("page-cache 9.3 GB/s", 9.3e9), ("device 4.48 GB/s", 4.48e9)):
    print(f"--- reads served at {bw_name} ---")
    print(f"{'resident':>9} {'streamed':>9} {'memory':>9} {'read':>9} "
          f"{'serial':>9} {'overlapped':>11} {'tok/s':>7}")
    for k in (65, 60, 56, 48, 32, 16, 0):
        streamed = (N_LAYER - k) * LAYER_GIB
        mem = NONBODY + k * LAYER_GIB + (ARENA_GIB if k < N_LAYER else 0)
        read_ms = streamed * G / bw * 1000
        serial = COMPUTE_MS + read_ms
        # Streaming the TAIL means the resident head's compute overlaps the reads.
        # Perfectly pipelined, a token costs whichever side is larger.
        overlapped = max(COMPUTE_MS, read_ms)
        print(f"{k:>9} {N_LAYER-k:>9} {mem:>8.2f}G {read_ms:>8.0f}ms "
              f"{serial:>8.0f}ms {overlapped:>10.0f}ms {1000/overlapped:>7.2f}")
    free = COMPUTE_MS/1000 * bw / G
    print(f"  -> reads are FREE (fully hidden) while streamed <= {free:.2f} GiB "
          f"= {free/LAYER_GIB:.0f} layers; memory saved at no speed cost: "
          f"{free:.2f} GiB minus {ARENA_GIB:.2f} GiB of arena "
          f"= {free-ARENA_GIB:+.2f} GiB\n")
