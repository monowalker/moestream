# Finding S0 — Slot Table + ID Remap works

| | |
|---|---|
| Target | DESIGN.md §10.4 / ADR-0006 |
| Date | 2026-08-02 |
| Verdict | **PASS** |
| ggml | `78de6069` (2026-07-31) |
| Backend | **CPU only** (Vulkan not verified → S0b) |
| Reproduce | `make spike SPIKE=s0_slot_slab` |

## Hypothesis tested

Whether `ggml_mul_mat_id(as, b, ids)` still works when `as` is not a 3D tensor
holding every expert, but **a slab of cache slots `[n_embd, n_ff, n_slot]`
(n_slot < n_expert)**, with `ids` carrying **slot_id** instead of expert_id.

If it does not, a custom MoE kernel is required and both the design and the
effort change fundamentally.

## What the source says

The assertions at `src/ggml.c:3329` contain **no constraint requiring `as->ne[2]`
to equal the expert count**:

```c
GGML_ASSERT(as->ne[3] == 1);            // as is 3d (one matrix per expert)
GGML_ASSERT(ids->ne[1] == b->ne[2]);
GGML_ASSERT(as->ne[0] == b->ne[0]);
GGML_ASSERT(ids->ne[0] % b->ne[1] == 0);
```

Shape contract:
```
as  : [n_embd, n_ff, n_slot]        <- ne[2] is arbitrary; a slot count is fine
b   : [n_embd, 1, n_tok]
ids : [top_k, n_tok]  (I32)
out : [n_ff, top_k, n_tok]
```

## Results

Shapes: `n_embd=256, n_ff=64, n_expert=8, n_slot=3, n_tok=4, top_k=2`
(**n_slot < n_expert**, reproducing a cache that cannot hold every expert)

| # | Check | Result |
|---|---|---|
| T1 | F32 slab vs naive reference | PASS, `max|diff| = 6.68e-06` |
| T2 | Q4_K slab vs reference | PASS, relative RMS `0.0543` (quantization error only) |
| T3 | Permutation invariance, F32 (same experts, different slots) | PASS, **bit-identical** |
| T3 | Permutation invariance, Q4_K | PASS, **bit-identical** |
| T4 | With unreferenced slots present | PASS, `max|diff| = 6.68e-06` |

**T3 matters most.** Moving the same experts from slots 0,1,2 to 1,2,0 and
rewiring `ids` accordingly produced **bit-identical** output. That is direct
proof that a slot is treated as pure indirection, and it means **numerical
results do not change when eviction rearranges slots**. It is also the basis for
QR-1 (determinism).

T2's relative RMS of 0.0543 is a reasonable level for Q4_K quantization error.
Pulling the wrong expert would give a relative error near 1.0, so the value
itself corroborates that the indexing is correct.

## Conclusion

**The §10.4 design works. ggml's MoE kernel can be used unmodified on top of a
dynamic expert cache.** ADR-0006 is confirmed.

## Remaining work

| ID | Item | Priority |
|---|---|---|
| **S0b** | **The same verification on the Vulkan backend.** Vulkan's `mul_mat_id` is separate code from the CPU path and its shaders may assume an expert count | **high** |
| S0c | Behaviour and overhead at a large `n_slot` (~4,000) | medium |
| S0d | Behaviour when `ids` contains an out-of-range slot_id (is defensive code needed?) | medium |
