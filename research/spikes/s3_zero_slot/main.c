// =============================================================================
// MoEStream Spike S3 — does an all-zero quantized block decode to zero?
//
// Expert Sweep (§20.2) works by mapping experts outside the current pass to a
// "zero slot" so they contribute nothing. The premise is:
//
//     an all-zero quantized block decodes to all 0.0
//
// The reasoning was that quantized formats carry an fp16 scale d at the start
// (or end) of the block, and d=0 zeroes the whole block. Verify by measurement.
//
// Types used by this model: IQ3_S (gate/up), IQ4_NL (down), Q6_K (some layers)
//
// Tests:
//   T1  CPU dequantization: to_float a zero-filled block and take the max |value|
//   T2  Vulkan mul_mat_id: is the result zero when ids point at the zero slot?
// =============================================================================
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_EMBD 512
#define N_FF   64
#define N_SLOT 2        // slot 0 = real data, slot 1 = the zero slot
#define N_TOK  2
#define N_TOPK 2

static int g_fail = 0;
static void report(const char *n, int ok, const char *d) {
    printf("  [%s] %-42s %s\n", ok ? " OK " : "FAIL", n, d ? d : "");
    if (!ok) g_fail++;
}

static const enum ggml_type TYPES[] = {
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K,
};
static const char * TNAME[] = { "IQ3_S", "IQ4_NL", "Q6_K", "Q4_K" };

// ---------------------------------------------------------------------------
// T1: CPU dequantization
// ---------------------------------------------------------------------------
static void test_dequant(void) {
    printf("T1: CPU dequantization of a zero-filled block\n");
    for (size_t i = 0; i < sizeof TYPES / sizeof TYPES[0]; ++i) {
        const enum ggml_type t = TYPES[i];
        const struct ggml_type_traits * tr = ggml_get_type_traits(t);
        if (!tr || !tr->to_float) {
            report(TNAME[i], 0, "no to_float available");
            continue;
        }
        const size_t rb = ggml_row_size(t, N_EMBD);
        unsigned char * q = calloc(1, rb);          // every byte zero
        float * f = malloc(sizeof(float) * N_EMBD);
        tr->to_float(q, f, N_EMBD);
        float mx = 0;
        for (int k = 0; k < N_EMBD; ++k) { float a = fabsf(f[k]); if (a > mx) mx = a; }
        char buf[96];
        snprintf(buf, sizeof buf, "max|value| = %.6g   (row %zu B)", (double) mx, rb);
        report(TNAME[i], mx == 0.0f, buf);
        free(q); free(f);
    }
}

// ---------------------------------------------------------------------------
// T2: index the zero slot through a real mul_mat_id
// ---------------------------------------------------------------------------
static uint32_t rs = 0x12345678u;
static float frand(void) {
    rs = rs * 1664525u + 1013904223u;
    return ((float)(rs >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static void test_matmul(ggml_backend_t be, const char * bename) {
    printf("\nT2: indexing the zero slot (slot 1) via mul_mat_id — backend=%s\n", bename);
    for (size_t i = 0; i < sizeof TYPES / sizeof TYPES[0]; ++i) {
        const enum ggml_type t = TYPES[i];
        struct ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), NULL, true };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * as   = ggml_new_tensor_3d(ctx, t, N_EMBD, N_FF, N_SLOT);
        struct ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, N_TOK);
        struct ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_TOPK, N_TOK);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) { report(TNAME[i], 0, "buffer allocation failed"); ggml_free(ctx); continue; }

        // slot 0 = random real data / slot 1 = all zero bytes
        {
            const size_t nb = ggml_nbytes(as);
            unsigned char * raw = calloc(1, nb);
            float * src = malloc(sizeof(float) * (size_t) N_FF * N_EMBD);
            for (int k = 0; k < N_FF * N_EMBD; ++k) src[k] = frand();
            ggml_quantize_chunk(t, src, raw, 0, N_FF, N_EMBD, NULL);   // slot 0 only
            // slot 1 stays zeroed from calloc
            ggml_backend_tensor_set(as, raw, 0, nb);
            free(raw); free(src);
        }
        {
            float * x = malloc(sizeof(float) * N_EMBD * N_TOK);
            for (int k = 0; k < N_EMBD * N_TOK; ++k) x[k] = frand();
            ggml_backend_tensor_set(b, x, 0, sizeof(float) * N_EMBD * N_TOK);
            free(x);
        }
        {
            int32_t f[N_TOK * N_TOPK];
            for (int k = 0; k < N_TOK * N_TOPK; ++k) f[k] = 1;   // all point at the zero slot
            ggml_backend_tensor_set(ids, f, 0, sizeof f);
        }

        struct ggml_tensor * res = ggml_mul_mat_id(ctx, as, b, ids);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, res);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf) ||
            ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            report(TNAME[i], 0, "execution failed");
            ggml_gallocr_free(ga); ggml_backend_buffer_free(buf); ggml_free(ctx); continue;
        }

        const size_t n = (size_t) N_FF * N_TOPK * N_TOK;
        float * out = malloc(sizeof(float) * n);
        ggml_backend_tensor_get(res, out, 0, sizeof(float) * n);
        float mx = 0;
        for (size_t k = 0; k < n; ++k) { float a = fabsf(out[k]); if (a > mx) mx = a; }
        char bufs[96];
        snprintf(bufs, sizeof bufs, "max|out| = %.6g", (double) mx);
        report(TNAME[i], mx == 0.0f, bufs);

        free(out);
        ggml_gallocr_free(ga); ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
}

int main(void) {
    printf("=====================================================================\n");
    printf(" MoEStream Spike S3 : is the zero slot really zero?\n");
    printf("=====================================================================\n");
    test_dequant();

    const size_t nd = ggml_backend_dev_count();
    for (size_t i = 0; i < nd; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_t be = ggml_backend_dev_init(d, NULL);
        if (!be) continue;
        test_matmul(be, ggml_backend_dev_name(d));
        ggml_backend_free(be);
    }

    printf("\n=====================================================================\n");
    if (g_fail == 0)
        printf(" verdict: PASS — the zero slot returns 0; the Expert Sweep premise holds\n");
    else
        printf(" verdict: FAIL (%d) — the zero slot is not zero; this is the cause\n", g_fail);
    printf("=====================================================================\n");
    return g_fail == 0 ? 0 : 1;
}
