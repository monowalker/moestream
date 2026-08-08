// =============================================================================
// MoEStream Spike S4 — verify Expert Sweep's mathematical premise minimally
//
//   FFN(x, ids) = down_ids( silu(gate_ids(x)) * up_ids(x) )
//
//   Split ids into two groups, mapping everything outside the group to the
//   zero slot, and ask whether
//
//       FFN(x, ids_0) + FFN(x, ids_1)  ==  FFN(x, ids_full)
//
//   holds. It does not hold in the llama.cpp integration (PPL 4.4 -> 520801).
//   ids, the zero slot and the addition have each been verified individually,
//   so this isolates what happens at the ggml execution level.
// =============================================================================
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_EMBD  256
#define N_FF    256
#define N_EXP     8      // number of real experts
#define N_SLOT  (N_EXP + 1)   // the last slot is the zero slot
#define ZERO    N_EXP
#define N_TOK    16
#define N_TOPK    4

static int g_fail = 0;
static void report(const char *n, int ok, const char *d) {
    printf("  [%s] %-40s %s\n", ok ? " OK " : "FAIL", n, d ? d : "");
    if (!ok) g_fail++;
}
static uint32_t rs = 0xC0FFEEu;
static float frand(void) {
    rs = rs*1664525u + 1013904223u;
    return ((float)(rs >> 8)/(float)(1u<<24))*2.0f - 1.0f;
}

static float g_gate[N_EXP][N_FF*N_EMBD];
static float g_up  [N_EXP][N_FF*N_EMBD];
static float g_down[N_EXP][N_EMBD*N_FF];
static float g_x[N_TOK][N_EMBD];
static int32_t g_ids[N_TOK][N_TOPK];

// Build and run the FFN once; the caller supplies ids.
static int run_ffn(ggml_backend_t be, enum ggml_type qt,
                   const int32_t ids_in[N_TOK][N_TOPK],
                   float out[N_TOK][N_TOPK][N_EMBD]) {
    struct ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * gate = ggml_new_tensor_3d(ctx, qt, N_EMBD, N_FF,   N_SLOT);
    struct ggml_tensor * up   = ggml_new_tensor_3d(ctx, qt, N_EMBD, N_FF,   N_SLOT);
    struct ggml_tensor * down = ggml_new_tensor_3d(ctx, qt, N_FF,   N_EMBD, N_SLOT);
    struct ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, N_TOK);
    struct ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_TOPK, N_TOK);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return -1; }

    // Slab: slots 0..N_EXP-1 hold real experts, slot N_EXP is all zeros
    {
        const size_t nb_g = ggml_nbytes(gate);
        unsigned char * q = calloc(1, nb_g);
        float * tmp = malloc(sizeof(float)*(size_t)N_FF*N_EMBD*N_EXP);
        for (int e = 0; e < N_EXP; ++e) memcpy(tmp + (size_t)e*N_FF*N_EMBD, g_gate[e], sizeof(float)*N_FF*N_EMBD);
        ggml_quantize_chunk(qt, tmp, q, 0, (int64_t)N_FF*N_EXP, N_EMBD, NULL);
        ggml_backend_tensor_set(gate, q, 0, nb_g);
        for (int e = 0; e < N_EXP; ++e) memcpy(tmp + (size_t)e*N_FF*N_EMBD, g_up[e], sizeof(float)*N_FF*N_EMBD);
        memset(q, 0, nb_g);
        ggml_quantize_chunk(qt, tmp, q, 0, (int64_t)N_FF*N_EXP, N_EMBD, NULL);
        ggml_backend_tensor_set(up, q, 0, nb_g);
        free(q); free(tmp);

        const size_t nb_d = ggml_nbytes(down);
        unsigned char * q2 = calloc(1, nb_d);
        float * t2 = malloc(sizeof(float)*(size_t)N_EMBD*N_FF*N_EXP);
        for (int e = 0; e < N_EXP; ++e) memcpy(t2 + (size_t)e*N_EMBD*N_FF, g_down[e], sizeof(float)*N_EMBD*N_FF);
        ggml_quantize_chunk(qt, t2, q2, 0, (int64_t)N_EMBD*N_EXP, N_FF, NULL);
        ggml_backend_tensor_set(down, q2, 0, nb_d);
        free(q2); free(t2);
    }
    ggml_backend_tensor_set(x, g_x, 0, sizeof(float)*N_EMBD*N_TOK);
    ggml_backend_tensor_set(ids, ids_in, 0, sizeof(int32_t)*N_TOK*N_TOPK);

    struct ggml_tensor * gg  = ggml_mul_mat_id(ctx, gate, x, ids);       // [N_FF, K, T]
    struct ggml_tensor * uu  = ggml_mul_mat_id(ctx, up,   x, ids);
    struct ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, gg), uu);
    struct ggml_tensor * res = ggml_mul_mat_id(ctx, down, act, ids);     // [N_EMBD, K, T]

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    int rc = 0;
    if (!ggml_gallocr_alloc_graph(ga, gf) || ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) rc = -1;
    else {
        static float tmp[N_EMBD*N_TOPK*N_TOK];
        ggml_backend_tensor_get(res, tmp, 0, sizeof(tmp));
        for (int t = 0; t < N_TOK; ++t)
            for (int k = 0; k < N_TOPK; ++k)
                memcpy(out[t][k], tmp + ((size_t)t*N_TOPK + k)*N_EMBD, sizeof(float)*N_EMBD);
    }
    ggml_gallocr_free(ga); ggml_backend_buffer_free(buf); ggml_free(ctx);
    return rc;
}

static float maxdiff(float a[N_TOK][N_TOPK][N_EMBD], float b[N_TOK][N_TOPK][N_EMBD]) {
    float m = 0;
    for (int t = 0; t < N_TOK; ++t) for (int k = 0; k < N_TOPK; ++k) for (int i = 0; i < N_EMBD; ++i) {
        float d = fabsf(a[t][k][i] - b[t][k][i]); if (d > m) m = d; }
    return m;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=====================================================================\n");
    printf(" MoEStream Spike S4 : is FFN(ids_0) + FFN(ids_1) == FFN(ids_full)?\n");
    printf("=====================================================================\n");
    for (int e = 0; e < N_EXP; ++e) {
        for (int j = 0; j < N_FF*N_EMBD; ++j) { g_gate[e][j] = frand(); g_up[e][j] = frand(); }
        for (int j = 0; j < N_EMBD*N_FF; ++j)  g_down[e][j] = frand();
    }
    for (int t = 0; t < N_TOK; ++t) {
        for (int i = 0; i < N_EMBD; ++i) g_x[t][i] = frand()*0.5f;
        for (int k = 0; k < N_TOPK; ++k) g_ids[t][k] = (t*N_TOPK + k*3) % N_EXP;
    }
    // Split into a first half (0..3) and a second half (4..7)
    static int32_t ids0[N_TOK][N_TOPK], ids1[N_TOK][N_TOPK];
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < N_TOPK; ++k) {
            const int32_t e = g_ids[t][k];
            ids0[t][k] = (e < N_EXP/2) ? e : ZERO;
            ids1[t][k] = (e >= N_EXP/2) ? e : ZERO;
        }

    const enum ggml_type QT[] = { GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL };
    const char * QN[] = { "Q4_K", "IQ4_NL" };

    const size_t nd = ggml_backend_dev_count();
    for (size_t d = 0; d < nd; ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        ggml_backend_t be = ggml_backend_dev_init(dev, NULL);
        if (!be) continue;
        printf("\n--- backend: %s ---\n", ggml_backend_dev_name(dev));
        for (size_t q = 0; q < 2; ++q) {
            static float full[N_TOK][N_TOPK][N_EMBD];
            static float p0[N_TOK][N_TOPK][N_EMBD], p1[N_TOK][N_TOPK][N_EMBD];
            if (run_ffn(be, QT[q], (const int32_t (*)[N_TOPK]) g_ids, full) ||
                run_ffn(be, QT[q], (const int32_t (*)[N_TOPK]) ids0, p0)  ||
                run_ffn(be, QT[q], (const int32_t (*)[N_TOPK]) ids1, p1)) {
                report(QN[q], 0, "execution failed"); continue;
            }
            static float sum[N_TOK][N_TOPK][N_EMBD];
            for (int t = 0; t < N_TOK; ++t) for (int k = 0; k < N_TOPK; ++k)
                for (int i = 0; i < N_EMBD; ++i) sum[t][k][i] = p0[t][k][i] + p1[t][k][i];
            const float md = maxdiff(full, sum);
            float mag = 0;
            for (int t = 0; t < N_TOK; ++t) for (int k = 0; k < N_TOPK; ++k)
                for (int i = 0; i < N_EMBD; ++i) { float a = fabsf(full[t][k][i]); if (a > mag) mag = a; }
            char b[128];
            snprintf(b, sizeof b, "max|full - (p0+p1)| = %.3e   (|full|max = %.3e)", (double) md, (double) mag);
            report(QN[q], md < 1e-3f * (mag > 1 ? mag : 1), b);
        }
        ggml_backend_free(be);
    }

    printf("\n=====================================================================\n");
    printf(g_fail == 0 ? " verdict: PASS — split accumulation holds; the fault is in the llama.cpp integration\n"
                       : " verdict: FAIL — split accumulation itself does not hold\n");
    printf("=====================================================================\n");
    return g_fail == 0 ? 0 : 1;
}
