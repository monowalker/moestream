# Finding S0b — Slot Table + ID Remap works on Vulkan (integrated GPU) too

| | |
|---|---|
| Target | DESIGN.md §10.4 / §14 / ADR-0006 |
| Date | 2026-08-03 |
| Verdict | **PASS** |
| GPU | AMD Radeon 780M (RADV PHOENIX), Mesa 25.2.8 |
| ggml | `78de6069`, `GGML_VULKAN=ON` |
| Reproduce | `make spike-vk SPIKE=s0b_backend_slab` |

## Background

S0 verified this on the CPU backend. But MoEStream's real target is an
integrated GPU, and Vulkan's `mul_mat_id` is **an entirely separate
implementation in GLSL**. Whether its shaders assume an expert count, or whether
`ids` is bounds-checked, cannot be known without running it. Failing here would
have meant redoing the whole design.

## Device detected

```
ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX) (radv)
             uma: 1 | fp16: 1 | bf16: 0 | warp size: 64
             shared memory: 65536 | matrix cores: KHR_coopmat
```

**`uma: 1`** — ggml also recognises this as UMA, so §14.3's UNIFIED topology
detection applies directly.

## Results

Shapes: `n_embd=256, n_ff=64, n_expert=8, n_slot=3, n_tok=4, top_k=2`
(**n_slot < n_expert**)

| Check | Vulkan0 (780M) | CPU |
|---|---|---|
| T1 F32 slab vs reference | **PASS** `6.676e-06` | PASS `6.676e-06` |
| T2 Q4_K slab vs reference | **PASS** relative RMS `0.0542` | PASS `0.0543` |
| T3 F32 permutation invariance | **PASS, bit-identical** | PASS, bit-identical |
| T3 Q4_K permutation invariance | **PASS, bit-identical** | PASS, bit-identical |
| T4 unreferenced slots | **PASS** `7.629e-06` | PASS `6.676e-06` |

### Cross-checking the two backends

| Comparison | Value | Verdict |
|---|---|---|
| F32 `max|diff|` | `1.907e-06` | effectively identical |
| Q4_K **relative RMS** | **`0.0038`** | **1/14** of the `0.054` quantization error |
| Q4_K `max|diff|` | `6.065e-02` | — |

**Interpretation**: Vulkan's and the CPU's quantized GEMM do not agree exactly,
because of fp16 accumulation and similar differences. What matters is not the
absolute difference but **whether the same expert is being fetched**. Misindexing
would put the relative error near 1.0; the measurement is 0.0038, an order of
magnitude below the quantization noise. This is **a kernel numerical difference
on top of identical weights**.

> The first run reported FAIL against an absolute-difference threshold of `5e-3`.
> That was a flaw in the check, not in the implementation, and the criterion was
> changed to relative error. (Not to make the conclusion convenient: absolute
> difference is simply the wrong metric for the hypothesis under test, which is
> "are we fetching a different expert".)

## Conclusion

**ADR-0006 is confirmed on both CPU and Vulkan backends.** ggml's MoE kernel can
drive a dynamic expert cache on an integrated GPU, unmodified. The design's
premise holds on the primary target (UMA / integrated GPU / Vulkan) on real
hardware.

## A design consequence

### QR-1 (determinism) is limited to "within one backend"

§4.3's QR-1 asks for bit-exactness against llama.cpp, but **bit-exactness across
different backends is unachievable in principle** (fp16 accumulation, workgroup
sizes and reduction orders all differ).

The correct statement of QR-1:

> Within the same backend, quantization and seed, **full-residency execution and
> streaming execution are bit-exact**.

That has already been demonstrated by S0/S0b's T3 (permutation invariance is
bit-identical) and S1's T3 (the MoE FFN is bit-identical on a real model).
**MoEStream introduces zero non-determinism**, which is the substance of QR-1.

## Remaining work

| ID | Item | Priority |
|---|---|---|
| S1b | Re-run the real model (S1) on Vulkan | medium |
| S2 | io_uring bandwidth: can it beat QD=8's 4.46 GB/s? | high |
| M0-2 | Expert activation distribution: can the required 51% hit rate be met? | **highest** |
