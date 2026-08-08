// =============================================================================
// MoEStream Spike S6 — does ggml_mul_mat_id work on an imported anonymous arena?
//
// Finding S5 established that:
//   - Vulkan import of an anonymous arena succeeds up to 1320 MiB
//   - amdgpu rejects file-backed mmaps (VkResult -13)
//   - GPU reads from the arena run at 38% of the current slab's rate, which
//     does not matter because the SSD is the bottleneck
//
// The largest remaining unknown is ggml integration: can ggml_mul_mat_id
// index correctly into a 256-expert tensor placed on an imported arena?
//
// If this fails, none of S5's other results matter.
//
// Tests:
//   T1  mul_mat_id runs with `as` on the arena              (feasibility)
//   T2  it is bit-identical to `as` in a normal buffer      (the key result)
//   T3  it stays correct with ids spanning all of 0..255    (proof that the
//                                                            slab limit is gone)
//   T4  it agrees with the CPU backend                      (sanity)
//   T5  it still holds at production size (216 MiB), and by how much speed
//       differs                                             (scale)
// =============================================================================

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>

#define N_EXPERT 256
#define N_TOK     64
#define N_TOPK     8

static int g_fail = 0;
static void report(const char *n, int ok, const char *d) {
    printf("  [%s] %-46s %s\n", ok ? " OK " : "FAIL", n, d ? d : "");
    if (!ok) g_fail++;
}
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Expert e's weights are a function of e, so two runs produce identical data
static void gen_expert(float *dst, size_t n, int e) {
    uint32_t rs = 0x9e3779b9u ^ (uint32_t) (e * 2654435761u);
    for (size_t i = 0; i < n; ++i) {
        rs = rs * 1664525u + 1013904223u;
        dst[i] = ((float) (rs >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    }
}
static void gen_tokens(float *dst, size_t n) {
    uint32_t rs = 0x13572468u;
    for (size_t i = 0; i < n; ++i) {
        rs = rs * 1664525u + 1013904223u;
        dst[i] = ((float) (rs >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    }
}

// -----------------------------------------------------------------------------
// Run mul_mat_id once.
//   With use_arena=1, `as` is placed on the imported anonymous arena.
//   Everything else -- the steps and the data -- is identical.
// -----------------------------------------------------------------------------
static int run(ggml_backend_t be, int use_arena, enum ggml_type type,
               int64_t ne0, int64_t ne1, const int32_t *ids_flat,
               float *out, size_t out_floats, double *best_ms) {
    const size_t ebytes = ggml_row_size(type, ne0) * (size_t) ne1;   // bytes per expert
    const size_t total  = ebytes * N_EXPERT;

    struct ggml_init_params ip = {
        ggml_tensor_overhead() * 32 + ggml_graph_overhead(), NULL, true };
    struct ggml_context *ctx = ggml_init(ip);
    if (!ctx) return -1;

    struct ggml_tensor *as   = ggml_new_tensor_3d(ctx, type, ne0, ne1, N_EXPERT);
    struct ggml_tensor *b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, N_TOPK, N_TOK);
    struct ggml_tensor *tids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_TOPK, N_TOK);
    ggml_set_name(as, "as"); ggml_set_name(b, "x"); ggml_set_name(tids, "ids");

    if (ggml_nbytes(as) != total) {
        printf("    (internal) nbytes mismatch %zu != %zu\n", ggml_nbytes(as), total);
        ggml_free(ctx); return -9;
    }

    // ---- where `as` lives ----
    void *arena = NULL;
    ggml_backend_buffer_t abuf = NULL;
    if (use_arena) {
        const size_t msz = (total + 4095) & ~(size_t) 4095;
        arena = mmap(NULL, msz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (arena == MAP_FAILED) { ggml_free(ctx); return -2; }
        memset(arena, 0, msz);
        ggml_backend_dev_t dv = ggml_backend_get_device(be);
        abuf = ggml_backend_dev_buffer_from_host_ptr(dv, arena, msz, 0);
        if (!abuf) { munmap(arena, msz); ggml_free(ctx); return -3; }
        if (ggml_backend_tensor_alloc(abuf, as, ggml_backend_buffer_get_base(abuf)) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(abuf); munmap(arena, msz); ggml_free(ctx); return -4;
        }
    }
    // Allocate the rest normally: just b/ids with use_arena, otherwise `as` too
    ggml_backend_buffer_t nbuf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!nbuf) {
        if (abuf) ggml_backend_buffer_free(abuf);
        ggml_free(ctx); return -5;
    }

    // ---- upload data (byte-for-byte identical on both paths) ----
    {
        float *f = malloc(sizeof(float) * (size_t) ne0 * ne1);
        void  *q = malloc(ebytes);
        for (int e = 0; e < N_EXPERT; ++e) {
            gen_expert(f, (size_t) ne0 * ne1, e);
            ggml_quantize_chunk(type, f, q, 0, ne1, ne0, NULL);
            ggml_backend_tensor_set(as, q, (size_t) e * ebytes, ebytes);
        }
        free(q); free(f);

        float *x = malloc(sizeof(float) * (size_t) ne0 * N_TOPK * N_TOK);
        gen_tokens(x, (size_t) ne0 * N_TOPK * N_TOK);
        ggml_backend_tensor_set(b, x, 0, sizeof(float) * (size_t) ne0 * N_TOPK * N_TOK);
        free(x);

        ggml_backend_tensor_set(tids, ids_flat, 0, sizeof(int32_t) * N_TOPK * N_TOK);
    }

    // ---- graph ----
    struct ggml_tensor *res = ggml_mul_mat_id(ctx, as, b, tids);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    int rc = 0;
    if (!ggml_gallocr_alloc_graph(ga, gf)) rc = -6;

    double best = 1e30;
    if (!rc) {
        for (int it = 0; it < 4; ++it) {          // the first iteration is a warm-up
            double t0 = now_s();
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { rc = -7; break; }
            ggml_backend_synchronize(be);
            double dt = (now_s() - t0) * 1000.0;
            if (it && dt < best) best = dt;
        }
    }
    if (!rc) {
        if (ggml_nelements(res) != (int64_t) out_floats) rc = -8;
        else ggml_backend_tensor_get(res, out, 0, sizeof(float) * out_floats);
    }
    if (best_ms) *best_ms = best;

    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(nbuf);
    if (abuf) {
        ggml_backend_buffer_free(abuf);
        munmap(arena, (total + 4095) & ~(size_t) 4095);
    }
    ggml_free(ctx);
    return rc;
}

static void rel_stats(const float *a, const float *b, size_t n, float *rel, float *mx) {
    double er = 0, rr = 0; float m = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        if (fabsf(d) > m) m = fabsf(d);
        er += (double) d * d; rr += (double) a[i] * a[i];
    }
    *rel = (float) sqrt(er / (rr + 1e-12)); *mx = m;
}

int main(void) {
    printf("=====================================================================\n");
    printf(" MoEStream Spike S6 : ggml_mul_mat_id on an imported arena\n");
    printf("=====================================================================\n");

    // ids span the whole 0..255 range, proving the slab limit no longer applies
    static int32_t ids[N_TOK * N_TOPK];
    for (int i = 0; i < N_TOK * N_TOPK; ++i) ids[i] = i % N_EXPERT;
    int seen[N_EXPERT] = {0}, ndist = 0;
    for (int i = 0; i < N_TOK * N_TOPK; ++i) if (!seen[ids[i]]++) ndist++;
    printf("  n_expert=%d  n_tok=%d  top_k=%d  distinct experts referenced=%d\n\n",
           N_EXPERT, N_TOK, N_TOPK, ndist);

    ggml_backend_t vk = NULL, cpu = NULL;
    const size_t nd = ggml_backend_dev_count();
    for (size_t i = 0; i < nd; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char *nm = ggml_backend_dev_name(d);
        printf("  found: %-10s %s\n", nm, ggml_backend_dev_description(d));
        if (!vk  && strncmp(nm, "Vulkan", 6) == 0) vk  = ggml_backend_dev_init(d, NULL);
        if (!cpu && strcmp(nm, "CPU") == 0)        cpu = ggml_backend_dev_init(d, NULL);
    }
    printf("\n");
    if (!vk) { printf("no Vulkan backend available\n"); return 1; }

    char msg[200];

    // ---------------- case 1: small (18 MiB), correctness ----------------
    {
        const int64_t ne0 = 512, ne1 = 256;
        const size_t nout = (size_t) ne1 * N_TOPK * N_TOK;
        printf("--- case 1: ne0=%lld ne1=%lld Q4_K  (as = %.1f MiB) ---\n",
               (long long) ne0, (long long) ne1,
               ggml_row_size(GGML_TYPE_Q4_K, ne0) * (double) ne1 * N_EXPERT / 1048576.0);

        float *oa = malloc(sizeof(float) * nout);
        float *on = malloc(sizeof(float) * nout);
        float *oc = malloc(sizeof(float) * nout);
        double ta = 0, tn = 0;

        int rca = run(vk, 1, GGML_TYPE_Q4_K, ne0, ne1, ids, oa, nout, &ta);
        snprintf(msg, sizeof msg, "rc=%d", rca);
        report("T1 mul_mat_id runs on the arena", rca == 0, rca ? msg : NULL);

        int rcn = run(vk, 0, GGML_TYPE_Q4_K, ne0, ne1, ids, on, nout, &tn);
        if (rca == 0 && rcn == 0) {
            int bit = memcmp(oa, on, sizeof(float) * nout) == 0;
            float rel, mx; rel_stats(on, oa, nout, &rel, &mx);
            snprintf(msg, sizeof msg, "max|diff|=%.3e relative RMS=%.3e", (double) mx, (double) rel);
            report("T2 bit-identical to a normal buffer (the key result)", bit, msg);
            report("T3 correct with ids spanning all of 0..255", bit && ndist == N_EXPERT, NULL);
        }
        if (cpu && rca == 0) {
            int rcc = run(cpu, 0, GGML_TYPE_Q4_K, ne0, ne1, ids, oc, nout, NULL);
            if (rcc == 0) {
                float rel, mx; rel_stats(oc, oa, nout, &rel, &mx);
                snprintf(msg, sizeof msg, "relative RMS=%.4f (max|diff|=%.3e)", (double) rel, (double) mx);
                report("T4 agrees with the CPU backend", rel < 0.02f, msg);
            }
        }
        if (ta < 1e29 && tn < 1e29)
            printf("       speed: arena %.2f ms / normal %.2f ms  (ratio %.2f)\n",
                   ta, tn, ta / tn);
        free(oa); free(on); free(oc);
        printf("\n");
    }

    // ---------------- case 2: production size (216 MiB) ----------------
    {
        const int64_t ne0 = 2048, ne1 = 768;
        const size_t nout = (size_t) ne1 * N_TOPK * N_TOK;
        printf("--- case 2: ne0=%lld ne1=%lld Q4_K  (as = %.1f MiB) ---\n",
               (long long) ne0, (long long) ne1,
               ggml_row_size(GGML_TYPE_Q4_K, ne0) * (double) ne1 * N_EXPERT / 1048576.0);

        float *oa = malloc(sizeof(float) * nout);
        float *on = malloc(sizeof(float) * nout);
        double ta = 0, tn = 0;

        int rca = run(vk, 1, GGML_TYPE_Q4_K, ne0, ne1, ids, oa, nout, &ta);
        int rcn = run(vk, 0, GGML_TYPE_Q4_K, ne0, ne1, ids, on, nout, &tn);
        snprintf(msg, sizeof msg, "rc=%d/%d", rca, rcn);
        report("T5 runs at production size", rca == 0 && rcn == 0, (rca || rcn) ? msg : NULL);
        if (rca == 0 && rcn == 0) {
            int bit = memcmp(oa, on, sizeof(float) * nout) == 0;
            float rel, mx; rel_stats(on, oa, nout, &rel, &mx);
            snprintf(msg, sizeof msg, "max|diff|=%.3e relative RMS=%.3e", (double) mx, (double) rel);
            report("T5 bit-identical at production size", bit, msg);
            printf("       speed: arena %.2f ms / normal %.2f ms  (ratio %.2f)\n",
                   ta, tn, ta / tn);
        }
        free(oa); free(on);
        printf("\n");
    }

    printf("=====================================================================\n");
    if (g_fail == 0)
        printf(" verdict: PASS — mul_mat_id works on an imported arena\n");
    else
        printf(" verdict: FAIL (%d)\n", g_fail);
    printf("=====================================================================\n");

    ggml_backend_free(vk);
    if (cpu) ggml_backend_free(cpu);
    return g_fail == 0 ? 0 : 1;
}
