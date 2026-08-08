// =============================================================================
// MoEStream Spike S0b — Slot Table + ID Remap across backends
//
// S0 confirmed slot-slab operation of ggml_mul_mat_id on the CPU backend.
// MoEStream's real target, however, is an iGPU via Vulkan, where mul_mat_id is
// a completely separate implementation (a GLSL shader). Whether that shader
// assumes an expert count, or bounds-checks ids, can only be settled by
// running it.
//
// This spike runs the same tests on any available backend and cross-checks
// the CPU results against the GPU results.
//
// Usage:
//   s0b_backend_slab            run on every available backend and compare
//   s0b_backend_slab Vulkan0    run on the named backend only
// =============================================================================

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_EMBD    256      // multiple of the Q4_K super-block size (256)
#define N_FF       64
#define N_EXPERT    8      // logical expert count
#define N_SLOT      3      // cache slot count (n_slot < n_expert)
#define N_TOK       4
#define N_TOPK      2
#define MAX_BE      8

static int g_fail = 0;
static void report(const char *n, int ok, const char *d) {
    printf("    [%s] %-40s %s\n", ok ? " OK " : "FAIL", n, d ? d : "");
    if (!ok) g_fail++;
}

static uint32_t rs = 0x13572468u;
static float frand(void) {
    rs = rs * 1664525u + 1013904223u;
    return ((float)(rs >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static float expert_w[N_EXPERT][N_FF * N_EMBD];
static float tok_x[N_TOK][N_EMBD];

static void init_data(void) {
    for (int e = 0; e < N_EXPERT; ++e)
        for (int j = 0; j < N_FF * N_EMBD; ++j) expert_w[e][j] = frand();
    for (int t = 0; t < N_TOK; ++t)
        for (int i = 0; i < N_EMBD; ++i) tok_x[t][i] = frand();
}

static void reference_matmul(const int s2e[N_SLOT], const int32_t ids[N_TOK][N_TOPK],
                             float out[N_TOK][N_TOPK][N_FF]) {
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < N_TOPK; ++k) {
            const float *W = expert_w[s2e[ids[t][k]]];
            for (int f = 0; f < N_FF; ++f) {
                float a = 0;
                for (int i = 0; i < N_EMBD; ++i) a += W[f * N_EMBD + i] * tok_x[t][i];
                out[t][k][f] = a;
            }
        }
}

// -----------------------------------------------------------------------------
// Run mul_mat_id on an arbitrary backend.
//   Create tensors in a no_alloc=true context, allocate them together in a
//   backend buffer, upload with ggml_backend_tensor_set, and run with
//   ggml_backend_graph_compute. This path is identical for CPU, Vulkan and CUDA.
// -----------------------------------------------------------------------------
static int run_backend(ggml_backend_t be, enum ggml_type slab_type,
                       const int s2e[N_SLOT], const int32_t ids[N_TOK][N_TOPK],
                       float out[N_TOK][N_TOPK][N_FF]) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context *ctx = ggml_init(ip);
    if (!ctx) return -1;

    // ne[2] = N_SLOT, not the logical expert count
    struct ggml_tensor *as   = ggml_new_tensor_3d(ctx, slab_type, N_EMBD, N_FF, N_SLOT);
    struct ggml_tensor *b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, N_TOK);
    struct ggml_tensor *tids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_TOPK, N_TOK);
    ggml_set_name(as, "slab"); ggml_set_name(b, "x"); ggml_set_name(tids, "ids");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return -2; }

    // --- pack weights into the slab in slot order ---
    {
        static float staging[N_SLOT * N_FF * N_EMBD];
        for (int s = 0; s < N_SLOT; ++s)
            memcpy(staging + (size_t)s * N_FF * N_EMBD, expert_w[s2e[s]],
                   sizeof(float) * N_FF * N_EMBD);
        if (slab_type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(as, staging, 0, ggml_nbytes(as));
        } else {
            void *q = malloc(ggml_nbytes(as));
            ggml_quantize_chunk(slab_type, staging, q, 0, (int64_t)N_FF * N_SLOT, N_EMBD, NULL);
            ggml_backend_tensor_set(as, q, 0, ggml_nbytes(as));
            free(q);
        }
    }
    ggml_backend_tensor_set(b, tok_x, 0, sizeof(float) * N_EMBD * N_TOK);
    {
        static int32_t flat[N_TOK * N_TOPK];
        for (int t = 0; t < N_TOK; ++t)
            for (int k = 0; k < N_TOPK; ++k) flat[t * N_TOPK + k] = ids[t][k];
        ggml_backend_tensor_set(tids, flat, 0, sizeof(flat));
    }

    // --- graph ---
    struct ggml_tensor *res = ggml_mul_mat_id(ctx, as, b, tids);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc); ggml_backend_buffer_free(buf); ggml_free(ctx); return -3;
    }
    const enum ggml_status st = ggml_backend_graph_compute(be, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc); ggml_backend_buffer_free(buf); ggml_free(ctx); return -4;
    }

    // --- read back ---
    static float tmp[N_FF * N_TOPK * N_TOK];
    ggml_backend_tensor_get(res, tmp, 0, sizeof(tmp));
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < N_TOPK; ++k)
            memcpy(out[t][k], tmp + ((size_t)t * N_TOPK + k) * N_FF, sizeof(float) * N_FF);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

static float maxdiff(const float a[N_TOK][N_TOPK][N_FF], const float b[N_TOK][N_TOPK][N_FF]) {
    float m = 0;
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < N_TOPK; ++k)
            for (int f = 0; f < N_FF; ++f) {
                float d = fabsf(a[t][k][f] - b[t][k][f]);
                if (d > m) m = d;
            }
    return m;
}
static int biteq(const void *a, const void *b, size_t n) { return memcmp(a, b, n) == 0; }

// The test body, for one backend
static float g_res[MAX_BE][2][N_TOK][N_TOPK][N_FF];   // [be][0=F32,1=Q4K]

static const int   s2e_A[N_SLOT]           = { 5, 1, 6 };
static const int32_t ids_A[N_TOK][N_TOPK]  = { {0,1}, {2,0}, {1,2}, {0,2} };
static const int   s2e_B[N_SLOT]           = { 6, 5, 1 };   // same experts, different slots
static const int   a2b[N_SLOT]             = { 1, 2, 0 };

static void test_backend(ggml_backend_t be, int idx, const char *label) {
    static float ref[N_TOK][N_TOPK][N_FF];
    reference_matmul(s2e_A, ids_A, ref);
    char buf[160];

    // T1: F32
    int rc = run_backend(be, GGML_TYPE_F32, s2e_A, ids_A, g_res[idx][0]);
    if (rc) { snprintf(buf, sizeof buf, "rc=%d", rc); report("F32 slab run", 0, buf); return; }
    float d = maxdiff(ref, g_res[idx][0]);
    snprintf(buf, sizeof buf, "max|diff| = %.3e", (double)d);
    report("T1 F32 slab vs reference", d < 2e-3f, buf);

    // T2: Q4_K
    rc = run_backend(be, GGML_TYPE_Q4_K, s2e_A, ids_A, g_res[idx][1]);
    if (rc) { snprintf(buf, sizeof buf, "rc=%d", rc); report("Q4_K slab run", 0, buf); return; }
    {
        float er = 0, rr = 0;
        for (int t = 0; t < N_TOK; ++t) for (int k = 0; k < N_TOPK; ++k) for (int f = 0; f < N_FF; ++f) {
            float e = ref[t][k][f] - g_res[idx][1][t][k][f];
            er += e * e; rr += ref[t][k][f] * ref[t][k][f];
        }
        float rel = sqrtf(er / (rr + 1e-12f));
        snprintf(buf, sizeof buf, "relative RMS = %.4f", (double)rel);
        report("T2 Q4_K slab vs reference", rel < 0.10f, buf);
    }

    // T3: permutation invariance
    {
        int32_t ids_B[N_TOK][N_TOPK];
        for (int t = 0; t < N_TOK; ++t)
            for (int k = 0; k < N_TOPK; ++k) ids_B[t][k] = a2b[ids_A[t][k]];
        static float gb0[N_TOK][N_TOPK][N_FF], gb1[N_TOK][N_TOPK][N_FF];
        int ok = (run_backend(be, GGML_TYPE_F32,  s2e_B, ids_B, gb0) == 0)
              && (run_backend(be, GGML_TYPE_Q4_K, s2e_B, ids_B, gb1) == 0);
        if (!ok) { report("T3 permutation run", 0, NULL); return; }
        report("T3 F32  bit-identical after remapping",
               biteq(g_res[idx][0], gb0, sizeof gb0), NULL);
        report("T3 Q4_K bit-identical after remapping",
               biteq(g_res[idx][1], gb1, sizeof gb1), NULL);
    }

    // T4: sparse reference (slot 1 is never used)
    {
        const int32_t ids_sp[N_TOK][N_TOPK] = { {0,2}, {2,0}, {0,2}, {2,0} };
        static float rsp[N_TOK][N_TOPK][N_FF], gsp[N_TOK][N_TOPK][N_FF];
        reference_matmul(s2e_A, ids_sp, rsp);
        if (run_backend(be, GGML_TYPE_F32, s2e_A, ids_sp, gsp) != 0) {
            report("T4 sparse reference run", 0, NULL); return;
        }
        float dd = maxdiff(rsp, gsp);
        snprintf(buf, sizeof buf, "max|diff| = %.3e", (double)dd);
        report("T4 correct despite unreferenced slots", dd < 2e-3f, buf);
    }
    (void)label;
}

int main(int argc, char **argv) {
    printf("=====================================================================\n");
    printf(" MoEStream Spike S0b : Slot Table + ID Remap across backends\n");
    printf("=====================================================================\n");
    printf("  n_embd=%d n_ff=%d n_expert=%d n_slot=%d n_tok=%d top_k=%d\n",
           N_EMBD, N_FF, N_EXPERT, N_SLOT, N_TOK, N_TOPK);
    printf("  ★ n_slot(%d) < n_expert(%d)\n\n", N_SLOT, N_EXPERT);

    init_data();

    const size_t ndev = ggml_backend_dev_count();
    printf("--- backends detected (%zu) ---\n", ndev);
    for (size_t i = 0; i < ndev; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf("  [%zu] %-10s  %s\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
    }
    printf("\n");

    const char *only = argc > 1 ? argv[1] : NULL;
    int used[MAX_BE]; const char *names[MAX_BE]; int n_used = 0;

    for (size_t i = 0; i < ndev && n_used < MAX_BE; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char *nm = ggml_backend_dev_name(d);
        if (only && strcmp(only, nm) != 0) continue;
        printf("===== backend: %s (%s) =====\n", nm, ggml_backend_dev_description(d));
        ggml_backend_t be = ggml_backend_dev_init(d, NULL);
        if (!be) { report("initialization", 0, "ggml_backend_dev_init failed"); continue; }
        test_backend(be, n_used, nm);
        ggml_backend_free(be);
        used[n_used] = (int)i; names[n_used] = nm; n_used++;
        printf("\n");
    }

    // ---- cross-check between backends ----
    if (n_used >= 2) {
        printf("===== cross-check between backends =====\n");
        printf("  note: quantized GEMM does not agree exactly across backends\n");
        printf("        (fp16 accumulation and similar). What matters here is\n");
        printf("        whether the wrong expert was fetched, which shows up as\n");
        printf("        relative error well above the ~5%% quantization error.\n");
        printf("        (misindexing would put relative error near 1.0)\n\n");
        for (int i = 1; i < n_used; ++i) {
            char b[224];
            float d0 = maxdiff(g_res[0][0], g_res[i][0]);
            // F32 should be effectively identical
            snprintf(b, sizeof b, "%s vs %s : max|diff|=%.3e", names[0], names[i], (double)d0);
            report("F32 slab: agrees across backends", d0 < 1e-4f, b);

            // Judge Q4_K by relative RMS
            float er = 0, rr = 0, md = 0;
            for (int t = 0; t < N_TOK; ++t) for (int k = 0; k < N_TOPK; ++k) for (int f = 0; f < N_FF; ++f) {
                float e = g_res[0][1][t][k][f] - g_res[i][1][t][k][f];
                if (fabsf(e) > md) md = fabsf(e);
                er += e * e; rr += g_res[0][1][t][k][f] * g_res[0][1][t][k][f];
            }
            float rel = sqrtf(er / (rr + 1e-12f));
            snprintf(b, sizeof b,
                     "%s vs %s : relative RMS=%.4f (max|diff|=%.3e), below the 0.054 quantization error",
                     names[0], names[i], (double)rel, (double)md);
            report("Q4_K slab: the same expert is being fetched", rel < 0.02f, b);
        }
        printf("\n");
    } else {
        printf("(only one backend; skipping the cross-check)\n\n");
    }
    (void)used;

    printf("=====================================================================\n");
    if (g_fail == 0)
        printf(" verdict: PASS — Slot Table + ID Remap holds on every backend tested\n");
    else
        printf(" verdict: FAIL (%d)\n", g_fail);
    printf("=====================================================================\n");
    return g_fail == 0 ? 0 : 1;
}
