// =============================================================================
// MoEStream Spike S0 — does "Slot Table + ID Remap" actually work?
//
// The design hypothesis under test (DESIGN.md §10.4 / ADR-0006):
//
//   Replace `as` in ggml_mul_mat_id(as, b, ids): instead of a 3D tensor
//   holding every expert [n_embd, n_ff, n_expert], pass a slab of cache slots
//   [n_embd, n_ff, n_slot] with n_slot < n_expert, and put slot_ids rather
//   than expert_ids in `ids`. Does it still compute the right thing?
//
//   If it holds, MoEStream can run on a dynamic cache with GGML's MoE kernel
//   completely unmodified. If it does not, a custom kernel is required and the
//   design and effort change fundamentally. This spike settles that fork at
//   minimum cost.
//
// Tests performed:
//   T1  F32 slab   : does it match a naive reference implementation?
//                    (correctness of indexing)
//   T2  Q4_K slab  : does the same hold for a quantized type?
//                    (correctness of row strides)
//   T3  permutation invariance : assign the same experts to *different* slot
//                    numbers and recompute; the result must be bit-identical
//                    -> direct proof that a slot is pure indirection
//   T4  sparse use : does it survive slots that are never referenced?
//                    (always the case with a real cache)
// =============================================================================

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- test shapes ------------------------------------------------------------
// n_embd must be a multiple of the Q4_K super-block size (QK_K=256)
#define N_EMBD    256
#define N_FF       64
#define N_EXPERT    8   // logical expert count
#define N_SLOT      3   // cache slot count; n_slot < n_expert is the whole point
#define N_TOK       4
#define N_TOPK      2

#define CTX_SIZE (64u * 1024u * 1024u)
#define N_THREADS 4

static int g_failures = 0;

static void report(const char * name, int ok, const char * detail) {
    printf("  [%s] %-34s %s\n", ok ? " OK " : "FAIL", name, detail ? detail : "");
    if (!ok) g_failures++;
}

// Deterministic PRNG, written here so results do not depend on the platform
static uint32_t rng_state = 0x13572468u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

// -----------------------------------------------------------------------------
// Logical expert weights, kept in F32 for the reference implementation
//   expert_w[e][f * N_EMBD + i] = element (column i, row f) of expert e
// -----------------------------------------------------------------------------
static float expert_w[N_EXPERT][N_FF * N_EMBD];
static float tok_x[N_TOK][N_EMBD];

static void init_data(void) {
    for (int e = 0; e < N_EXPERT; ++e)
        for (int j = 0; j < N_FF * N_EMBD; ++j)
            expert_w[e][j] = frand();
    for (int t = 0; t < N_TOK; ++t)
        for (int i = 0; i < N_EMBD; ++i)
            tok_x[t][i] = frand();
}

// -----------------------------------------------------------------------------
// Reference: out[f, k, t] = sum_i W[slot2expert[ids[k][t]]][f][i] * x[t][i]
// -----------------------------------------------------------------------------
static void reference_matmul(const int slot2expert[N_SLOT],
                             const int32_t ids[N_TOK][N_TOPK],
                             float out[N_TOK][N_TOPK][N_FF]) {
    for (int t = 0; t < N_TOK; ++t) {
        for (int k = 0; k < N_TOPK; ++k) {
            const int slot = ids[t][k];
            const float * W = expert_w[slot2expert[slot]];
            for (int f = 0; f < N_FF; ++f) {
                float acc = 0.0f;
                for (int i = 0; i < N_EMBD; ++i) acc += W[f * N_EMBD + i] * tok_x[t][i];
                out[t][k][f] = acc;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Run mul_mat_id once.
//   slot2expert : which logical expert occupies slot s (the Slot Table)
//   ids         : slot_ids, not expert_ids (the ID Remap)
//   out         : [N_TOK][N_TOPK][N_FF]
// -----------------------------------------------------------------------------
static int run_mul_mat_id(enum ggml_type slab_type,
                          const int slot2expert[N_SLOT],
                          const int32_t ids[N_TOK][N_TOPK],
                          float out[N_TOK][N_TOPK][N_FF]) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ CTX_SIZE,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return -1; }

    // --- as: the slot slab [N_EMBD, N_FF, N_SLOT] ---------------------------
    // The point of this spike is that ne[2] is N_SLOT, not N_EXPERT
    struct ggml_tensor * as = ggml_new_tensor_3d(ctx, slab_type, N_EMBD, N_FF, N_SLOT);

    // Pack weights into the slab in slot order (mirrors cache placement)
    {
        static float staging[N_SLOT * N_FF * N_EMBD];
        for (int s = 0; s < N_SLOT; ++s)
            memcpy(staging + (size_t)s * N_FF * N_EMBD,
                   expert_w[slot2expert[s]],
                   sizeof(float) * N_FF * N_EMBD);

        if (slab_type == GGML_TYPE_F32) {
            memcpy(as->data, staging, sizeof(staging));
        } else {
            // Quantize per row: N_FF * N_SLOT rows of N_EMBD elements each
            const int64_t nrows = (int64_t)N_FF * N_SLOT;
            ggml_quantize_chunk(slab_type, staging, as->data, 0, nrows, N_EMBD, NULL);
        }
    }

    // --- b: input [N_EMBD, 1, N_TOK] ---------------------------------------
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, N_TOK);
    for (int t = 0; t < N_TOK; ++t)
        memcpy((char *)b->data + (size_t)t * b->nb[2], tok_x[t], sizeof(float) * N_EMBD);

    // --- ids: [N_TOPK, N_TOK] (I32) ----------------------------------------
    struct ggml_tensor * tids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_TOPK, N_TOK);
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < N_TOPK; ++k)
            ((int32_t *)tids->data)[t * N_TOPK + k] = ids[t][k];

    // --- build and run the graph -------------------------------------------
    struct ggml_tensor * res = ggml_mul_mat_id(ctx, as, b, tids);   // [N_FF, N_TOPK, N_TOK]

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);

    if (ggml_graph_compute_with_ctx(ctx, gf, N_THREADS) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        ggml_free(ctx);
        return -1;
    }

    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < N_TOPK; ++k)
            memcpy(out[t][k],
                   (char *)res->data + (size_t)t * res->nb[2] + (size_t)k * res->nb[1],
                   sizeof(float) * N_FF);

    ggml_free(ctx);
    return 0;
}

static float max_abs_diff(const float a[N_TOK][N_TOPK][N_FF],
                          const float b[N_TOK][N_TOPK][N_FF]) {
    float m = 0.0f;
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < N_TOPK; ++k)
            for (int f = 0; f < N_FF; ++f) {
                const float d = fabsf(a[t][k][f] - b[t][k][f]);
                if (d > m) m = d;
            }
    return m;
}

static int bit_identical(const float a[N_TOK][N_TOPK][N_FF],
                         const float b[N_TOK][N_TOPK][N_FF]) {
    return memcmp(a, b, sizeof(float) * N_TOK * N_TOPK * N_FF) == 0;
}

int main(void) {
    printf("=====================================================================\n");
    printf(" MoEStream Spike S0 : Slot Table + ID Remap (DESIGN.md §10.4)\n");
    printf("=====================================================================\n");
    printf("  n_embd=%d  n_ff=%d  n_expert=%d  n_slot=%d  n_tok=%d  top_k=%d\n",
           N_EMBD, N_FF, N_EXPERT, N_SLOT, N_TOK, N_TOPK);
    printf("  n_slot(%d) < n_expert(%d): the cache cannot hold every expert\n\n",
           N_SLOT, N_EXPERT);

    init_data();

    // Slot Table: the logical expert occupying slot s.
    //   Only experts 5, 1 and 6 are resident; 0, 2, 3, 4 and 7 are EVICTED.
    const int slot2expert_A[N_SLOT] = { 5, 1, 6 };

    // The router's chosen experts, mapped to slot_ids (the ID Remap)
    //   t0: expert 5,1 -> slot 0,1
    //   t1: expert 6,5 -> slot 2,0
    //   t2: expert 1,6 -> slot 1,2
    //   t3: experts 5,6 -> slots 0,2   (slot 1 is unused for this token)
    const int32_t ids_A[N_TOK][N_TOPK] = { {0,1}, {2,0}, {1,2}, {0,2} };

    static float ref[N_TOK][N_TOPK][N_FF];
    static float got_f32[N_TOK][N_TOPK][N_FF];
    static float got_q4k[N_TOK][N_TOPK][N_FF];

    reference_matmul(slot2expert_A, ids_A, ref);

    // ---------------- T1: F32 slab ----------------
    printf("T1: numerical agreement with an F32 slab\n");
    if (run_mul_mat_id(GGML_TYPE_F32, slot2expert_A, ids_A, got_f32) != 0) {
        report("F32 slab", 0, "execution failed");
    } else {
        const float d = max_abs_diff(ref, got_f32);
        char buf[128];
        snprintf(buf, sizeof(buf), "max|diff| = %.3e", (double)d);
        report("F32 slab vs reference", d < 1e-3f, buf);
    }

    // ---------------- T2: Q4_K slab ----------------
    printf("\nT2: consistency with a Q4_K slab (the type used in production)\n");
    if (run_mul_mat_id(GGML_TYPE_Q4_K, slot2expert_A, ids_A, got_q4k) != 0) {
        report("Q4_K slab", 0, "execution failed");
    } else {
        // Quantization error rules out exact equality. Fetching the wrong
        // expert would inflate the error by orders of magnitude, so relative
        // error is a sound test of indexing correctness.
        float ref_rms = 0.0f, err_rms = 0.0f;
        for (int t = 0; t < N_TOK; ++t)
            for (int k = 0; k < N_TOPK; ++k)
                for (int f = 0; f < N_FF; ++f) {
                    ref_rms += ref[t][k][f] * ref[t][k][f];
                    const float e = ref[t][k][f] - got_q4k[t][k][f];
                    err_rms += e * e;
                }
        const float rel = sqrtf(err_rms / (ref_rms + 1e-12f));
        char buf[128];
        snprintf(buf, sizeof(buf), "relative RMS = %.4f (expected <0.1 for quantization alone)", (double)rel);
        report("Q4_K slab vs reference", rel < 0.10f, buf);
    }

    // ---------------- T3: permutation invariance ----------------
    // Move the same three experts to different slot numbers and rewrite ids to
    // match. The output must be bit-identical.
    printf("\nT3: permutation invariance (proof that a slot is pure indirection)\n");
    {
        // A: slot 0,1,2 = expert 5,1,6
        // B: slots 0,1,2 = experts 6,5,1  (reordered)
        const int slot2expert_B[N_SLOT] = { 6, 5, 1 };
        // Map A's slot -> expert -> B's slot
        //   A slot0(e5) -> B slot1
        //   A slot1(e1) -> B slot2
        //   A slot2(e6) -> B slot0
        const int a2b[N_SLOT] = { 1, 2, 0 };
        int32_t ids_B[N_TOK][N_TOPK];
        for (int t = 0; t < N_TOK; ++t)
            for (int k = 0; k < N_TOPK; ++k)
                ids_B[t][k] = a2b[ids_A[t][k]];

        static float got_B_f32[N_TOK][N_TOPK][N_FF];
        static float got_B_q4k[N_TOK][N_TOPK][N_FF];
        int ok = 1;
        if (run_mul_mat_id(GGML_TYPE_F32,  slot2expert_B, ids_B, got_B_f32) != 0) ok = 0;
        if (run_mul_mat_id(GGML_TYPE_Q4_K, slot2expert_B, ids_B, got_B_q4k) != 0) ok = 0;

        if (!ok) {
            report("permutation run", 0, "execution failed");
        } else {
            report("F32  : bit-identical after remapping",
                   bit_identical(got_f32, got_B_f32), NULL);
            report("Q4_K : bit-identical after remapping",
                   bit_identical(got_q4k, got_B_q4k), NULL);
        }
    }

    // ---------------- T4: sparse use ----------------
    // A real cache always holds slots that are resident but unused this pass.
    // With ids that never reference slot 1, the other slots must be unaffected.
    printf("\nT4: do unreferenced slots affect anything else?\n");
    {
        const int32_t ids_sparse[N_TOK][N_TOPK] = { {0,2}, {2,0}, {0,2}, {2,0} };
        static float ref_sparse[N_TOK][N_TOPK][N_FF];
        static float got_sparse[N_TOK][N_TOPK][N_FF];
        reference_matmul(slot2expert_A, ids_sparse, ref_sparse);
        if (run_mul_mat_id(GGML_TYPE_F32, slot2expert_A, ids_sparse, got_sparse) != 0) {
            report("sparse reference", 0, "execution failed");
        } else {
            const float d = max_abs_diff(ref_sparse, got_sparse);
            char buf[128];
            snprintf(buf, sizeof(buf), "max|diff| = %.3e", (double)d);
            report("correct even with slot 1 unreferenced", d < 1e-3f, buf);
        }
    }

    printf("\n=====================================================================\n");
    if (g_failures == 0) {
        printf(" verdict: PASS — Slot Table + ID Remap (§10.4) holds\n");
        printf("        GGML's MoE kernel runs unmodified on a dynamic cache\n");
    } else {
        printf(" verdict: FAIL (%d) — the §10.4 premise does not hold\n", g_failures);
        printf("        a custom kernel or a design change is required (revisit ADR-0006)\n");
    }
    printf("=====================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
