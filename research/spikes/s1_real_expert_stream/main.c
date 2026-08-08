// =============================================================================
// MoEStream Spike S1 — stream a real model's experts straight from SSD and
//                       run the MoE FFN on them
//
// S0 verified ggml_mul_mat_id's indirection on synthetic data. S1 exercises
// the entire MoEStream data path against a real GGUF
// (Ornith-1.0-35B-UD-IQ4_NL, 16.87 GiB).
//
//   [1] compute expert e's byte range from the tensor location in the GGUF
//   [2] pread only the required experts directly from SSD
//   [3] pack them into the slot slab (i.e. fill the Expert Cache)
//   [4] map the router's expert_ids to slot_ids (the ID Remap)
//   [5] run the MoE FFN: three mul_mat_id calls plus SwiGLU
//   [6] compare against the same computation with every expert resident
//
// Tests:
//   T1  expert slices are contiguous (the premise for readv in §13.4)
//   T2  whether O_DIRECT works, and the measured alignment (checks §13.6)
//   T3  the streamed MoE FFN is bit-identical to the fully resident one
//       -- this is the central question
//   T4  reduction in bytes read
//   T5  effective read bandwidth (pins down the BW assumed in §2.3)
//   T6  whether the slab layout copes with UD quantization's mixed types
// =============================================================================

#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#define N_TOK    4
#define TOP_K    8
#define MAX_SLOT 32
#define CTX_SIZE (1600ull * 1024 * 1024)
#define N_THREADS 8

static int g_fail = 0;
static void report(const char *name, int ok, const char *d) {
    printf("  [%s] %-46s %s\n", ok ? " OK " : "FAIL", name, d ? d : "");
    if (!ok) g_fail++;
}
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static uint32_t rs = 0x9e3779b9u;
static float frand(void) {
    rs = rs * 1664525u + 1013904223u;
    return ((float)(rs >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

// ---- expert tensor description, read from layout.txt ------------------------
typedef struct {
    char role[8];
    uint64_t off;       // absolute offset from the start of the file
    uint64_t nbytes;    // size of the whole tensor
    int type;           // ggml_type
    int64_t ne0, ne1, ne2;
    char tname[16];
    uint64_t expert_bytes;   // bytes for one expert (= nb2)
} TensorInfo;

static TensorInfo T_gate, T_up, T_down;

static int load_layout(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("layout.txt"); return -1; }
    char line[512];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        TensorInfo t; memset(&t, 0, sizeof(t));
        if (sscanf(line, "TENSOR %7s %lu %lu %d %ld %ld %ld %15s",
                   t.role, &t.off, &t.nbytes, &t.type,
                   &t.ne0, &t.ne1, &t.ne2, t.tname) == 8) {
            t.expert_bytes = t.nbytes / (uint64_t)t.ne2;
            if      (!strcmp(t.role, "gate")) T_gate = t;
            else if (!strcmp(t.role, "up"))   T_up   = t;
            else if (!strcmp(t.role, "down")) T_down = t;
            n++;
        }
    }
    fclose(f);
    return n == 3 ? 0 : -1;
}

// ---- read expert e's slice from the file ------------------------------------
static uint64_t g_bytes_read = 0;
static int read_expert(int fd, const TensorInfo *t, int expert, void *dst) {
    const off_t off = (off_t)(t->off + (uint64_t)expert * t->expert_bytes);
    size_t left = t->expert_bytes;
    char *p = dst;
    while (left) {
        ssize_t r = pread(fd, p, left, off + (off_t)(t->expert_bytes - left));
        if (r <= 0) { perror("pread"); return -1; }
        p += r; left -= (size_t)r;
    }
    g_bytes_read += t->expert_bytes;
    return 0;
}

// ---- effective queue-depth scaling via parallel O_DIRECT reads --------------
// pread is synchronous, so the thread count is the effective queue depth at the
// device. This measures whether parallelism raises bandwidth at all, before
// committing to io_uring.
#include <pthread.h>
static double bw_par[1];

typedef struct {
    const char *path;
    int n_iter;          // number of experts this thread reads
    int seed;
    uint64_t bytes;
} PArg;

static void *par_worker(void *v) {
    PArg *a = v;
    int fd = open(a->path, O_RDONLY | O_DIRECT);
    if (fd < 0) return NULL;
    const size_t cap = ((T_down.expert_bytes + 8191) & ~(size_t)4095) + 4096;
    void *bb = NULL;
    if (posix_memalign(&bb, 4096, cap) != 0) { close(fd); return NULL; }
    const TensorInfo *ts[3] = { &T_gate, &T_up, &T_down };
    for (int i = 0; i < a->n_iter; ++i) {
        int e = (int)((((uint64_t)(i * 64 + a->seed)) * 2654435761u) % (uint64_t)T_gate.ne2);
        for (int j = 0; j < 3; ++j) {
            const uint64_t off = ts[j]->off + (uint64_t)e * ts[j]->expert_bytes;
            const uint64_t a0 = off & ~(uint64_t)4095;
            const uint64_t a1 = (off + ts[j]->expert_bytes + 4095) & ~(uint64_t)4095;
            if (pread(fd, bb, (size_t)(a1 - a0), (off_t)a0) > 0) a->bytes += ts[j]->expert_bytes;
        }
    }
    free(bb); close(fd);
    return NULL;
}

static double measure_parallel(const char *path, int nthr, int total) {
    pthread_t th[64]; PArg ar[64];
    const int per = total / nthr > 0 ? total / nthr : 1;
    double t0 = now_s();
    for (int i = 0; i < nthr; ++i) {
        ar[i] = (PArg){ path, per, i * 977 + 13, 0 };
        pthread_create(&th[i], NULL, par_worker, &ar[i]);
    }
    uint64_t bytes = 0;
    for (int i = 0; i < nthr; ++i) { pthread_join(th[i], NULL); bytes += ar[i].bytes; }
    double dt = now_s() - t0;
    return dt > 0 ? bytes / dt / 1e9 : 0.0;
}

// ---- read a whole tensor (the fully resident reference) ---------------------
static int read_whole(int fd, const TensorInfo *t, void *dst) {
    size_t left = t->nbytes; char *p = dst;
    while (left) {
        ssize_t r = pread(fd, p, left, (off_t)(t->off + (t->nbytes - left)));
        if (r <= 0) { perror("pread"); return -1; }
        p += r; left -= (size_t)r;
    }
    return 0;
}

// =============================================================================
// Run one layer of the MoE FFN.
//   as_gate/up/down : [ne0, ne1, n_idx]  (n_idx = 256 when fully resident,
//                                         n_slot with the slab)
//   ids             : [TOP_K, N_TOK]     (expert_ids when fully resident,
//                                         slot_ids with the slab)
//   out             : [n_embd, N_TOK]    weighted sum using the router weights
// =============================================================================
static int moe_ffn(struct ggml_context *ctx,
                   struct ggml_tensor *as_gate, struct ggml_tensor *as_up,
                   struct ggml_tensor *as_down, struct ggml_tensor *x,
                   const int32_t ids_data[N_TOK][TOP_K],
                   const float w[N_TOK][TOP_K],
                   float *out, int64_t n_embd)
{
    struct ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOP_K, N_TOK);
    memcpy(ids->data, ids_data, sizeof(int32_t) * TOP_K * N_TOK);

    // gate / up : [n_embd] -> [n_ff]
    struct ggml_tensor *g = ggml_mul_mat_id(ctx, as_gate, x, ids);  // [n_ff, TOP_K, N_TOK]
    struct ggml_tensor *u = ggml_mul_mat_id(ctx, as_up,   x, ids);
    struct ggml_tensor *a = ggml_mul(ctx, ggml_silu(ctx, g), u);    // SwiGLU
    // down : [n_ff] -> [n_embd]
    struct ggml_tensor *y = ggml_mul_mat_id(ctx, as_down, a, ids);  // [n_embd, TOP_K, N_TOK]

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, N_THREADS) != GGML_STATUS_SUCCESS) return -1;

    // Weighted sum by router weight. The sum is order-independent, so reduce
    // in a fixed order on the CPU (§10.6).
    memset(out, 0, sizeof(float) * n_embd * N_TOK);
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < TOP_K; ++k) {
            const float *src = (const float *)((char *)y->data + (size_t)t*y->nb[2] + (size_t)k*y->nb[1]);
            float *dst = out + (size_t)t * n_embd;
            for (int64_t i = 0; i < n_embd; ++i) dst[i] += w[t][k] * src[i];
        }
    return 0;
}

int main(int argc, char **argv) {
    const char *gguf   = argc > 1 ? argv[1] : "/models/Ornith-1.0-35B-UD-IQ4_NL.gguf";
    const char *layout = argc > 2 ? argv[2] : "layout.txt";

    printf("=====================================================================\n");
    printf(" MoEStream Spike S1 : expert streaming from a real GGUF + MoE FFN\n");
    printf("=====================================================================\n");
    printf("  model  : %s\n", gguf);

    if (load_layout(layout) != 0) { fprintf(stderr, "failed to load layout\n"); return 2; }

    const int64_t n_embd = T_gate.ne0;   // 2048
    const int64_t n_ff   = T_gate.ne1;   // 512
    const int64_t n_exp  = T_gate.ne2;   // 256
    printf("  layer0 : n_embd=%ld n_ff=%ld n_expert=%ld\n", n_embd, n_ff, n_exp);
    printf("  types  : gate=%s  up=%s  down=%s   (UD quantization uses a different type per role)\n",
           T_gate.tname, T_up.tname, T_down.tname);
    const uint64_t eb = T_gate.expert_bytes + T_up.expert_bytes + T_down.expert_bytes;
    printf("  1 expert = %llu B (%.3f MiB)  [gate %llu + up %llu + down %llu]\n\n",
           (unsigned long long)eb, eb / 1048576.0,
           (unsigned long long)T_gate.expert_bytes,
           (unsigned long long)T_up.expert_bytes,
           (unsigned long long)T_down.expert_bytes);

    // ---------------- T1: are expert slices contiguous? ----------------
    printf("T1: are expert slices contiguous? (the premise for readv, §13.4)\n");
    {
        int ok = 1; char buf[192];
        const TensorInfo *ts[3] = { &T_gate, &T_up, &T_down };
        for (int i = 0; i < 3; ++i)
            if (ts[i]->nbytes % (uint64_t)ts[i]->ne2 != 0) ok = 0;
        snprintf(buf, sizeof buf, "nb2 = nbytes/n_expert divides evenly (gate %llu / up %llu / down %llu)",
                 (unsigned long long)T_gate.expert_bytes,
                 (unsigned long long)T_up.expert_bytes,
                 (unsigned long long)T_down.expert_bytes);
        report("contiguous per expert", ok, buf);

        int a4 = (T_gate.expert_bytes % 4096 == 0) && (T_up.expert_bytes % 4096 == 0)
              && (T_down.expert_bytes % 4096 == 0);
        report("expert size is a multiple of 4 KiB", a4, a4 ? "slot packing in .msp is natural" : "padding required");
    }

    // ---------------- T2: O_DIRECT and alignment ----------------
    printf("\nT2: does O_DIRECT work, and how are tensor offsets aligned? (§13.6)\n");
    {
        char buf[192];
        snprintf(buf, sizeof buf, "gate off%%4096=%llu  up off%%4096=%llu  down off%%4096=%llu",
                 (unsigned long long)(T_gate.off % 4096),
                 (unsigned long long)(T_up.off % 4096),
                 (unsigned long long)(T_down.off % 4096));
        int aligned = (T_gate.off % 4096 == 0) && (T_up.off % 4096 == 0) && (T_down.off % 4096 == 0);
        // The design predicts misalignment, so matching that prediction is a pass
        printf("  [INFO] %s\n", buf);
        printf("  [%s] %-46s %s\n", "INFO", "GGUF start offset",
               aligned ? "4 KiB aligned; O_DIRECT can be used directly" :
                         "unaligned; O_DIRECT is unusable on the raw GGUF (as the design predicted)");

        int fdd = open(gguf, O_RDONLY | O_DIRECT);
        if (fdd < 0) {
            report("O_DIRECT open", 0, strerror(errno));
        } else {
            void *ab = NULL;
            int rc = posix_memalign(&ab, 4096, 1 << 20);
            ssize_t r = (rc == 0) ? pread(fdd, ab, 1 << 20, 0) : -1;
            char b2[128]; snprintf(b2, sizeof b2, "1 MiB read returned %zd B", r);
            report("the file is readable with O_DIRECT", r == (1 << 20), b2);
            free(ab); close(fdd);
        }
    }

    int fd = open(gguf, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }

    // ---------------- prepare inputs and routing ----------------
    static float xin[N_TOK][2048];
    for (int t = 0; t < N_TOK; ++t)
        for (int i = 0; i < n_embd; ++i) xin[t][i] = frand() * 0.5f;

    // Experts chosen by the router; partially different per token, as in reality
    int32_t exp_ids[N_TOK][TOP_K] = {
        {   5, 130, 200,  77,   3,  44, 180,  99 },
        {   5, 130, 201,  77,   3,  44, 180, 250 },
        {   5, 131, 200,  77,   3,  45, 180,  99 },
        {   5, 130, 200,  78,   3,  44, 181,  99 },
    };
    float rw[N_TOK][TOP_K];
    for (int t = 0; t < N_TOK; ++t) {
        float s = 0;
        for (int k = 0; k < TOP_K; ++k) { rw[t][k] = fabsf(frand()) + 0.05f; s += rw[t][k]; }
        for (int k = 0; k < TOP_K; ++k) rw[t][k] /= s;
    }

    // Take the union to build the Slot Table (what the cache manager does)
    int slot2exp[MAX_SLOT]; int n_slot = 0;
    int32_t slot_ids[N_TOK][TOP_K];
    for (int t = 0; t < N_TOK; ++t)
        for (int k = 0; k < TOP_K; ++k) {
            int e = exp_ids[t][k], s = -1;
            for (int i = 0; i < n_slot; ++i) if (slot2exp[i] == e) { s = i; break; }
            if (s < 0) { s = n_slot; slot2exp[n_slot++] = e; }
            slot_ids[t][k] = s;               // ★ ID Remap
        }
    printf("\n  router: %d tokens x top-%d -> %d distinct experts (union)\n",
           N_TOK, TOP_K, n_slot);

    static float out_full[N_TOK * 2048];
    static float out_slab[N_TOK * 2048];

    // ---------------- reference: every expert resident ----------------
    printf("\n[reference] MoE FFN with every expert resident (reads all of layer 0 = %.1f MiB)\n",
           (T_gate.nbytes + T_up.nbytes + T_down.nbytes) / 1048576.0);
    {
        struct ggml_init_params ip = { CTX_SIZE, NULL, false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *ag = ggml_new_tensor_3d(ctx, (enum ggml_type)T_gate.type, n_embd, n_ff, n_exp);
        struct ggml_tensor *au = ggml_new_tensor_3d(ctx, (enum ggml_type)T_up.type,   n_embd, n_ff, n_exp);
        struct ggml_tensor *ad = ggml_new_tensor_3d(ctx, (enum ggml_type)T_down.type, n_ff, n_embd, n_exp);
        double t0 = now_s();
        if (read_whole(fd, &T_gate, ag->data) || read_whole(fd, &T_up, au->data) ||
            read_whole(fd, &T_down, ad->data)) return 2;
        double t1 = now_s();
        printf("  read %.1f MiB in %.3f s (%.2f GB/s)\n",
               (T_gate.nbytes + T_up.nbytes + T_down.nbytes) / 1048576.0, t1 - t0,
               (T_gate.nbytes + T_up.nbytes + T_down.nbytes) / (t1 - t0) / 1e9);

        struct ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, N_TOK);
        for (int t = 0; t < N_TOK; ++t)
            memcpy((char *)x->data + (size_t)t * x->nb[2], xin[t], sizeof(float) * n_embd);

        if (moe_ffn(ctx, ag, au, ad, x, exp_ids, rw, out_full, n_embd) != 0) return 2;
        ggml_free(ctx);
    }

    // ---------------- MoEStream: read only the needed experts ----------------
    printf("\n[MoEStream] reading only the %d required experts from SSD into the slab\n", n_slot);
    g_bytes_read = 0;
    {
        struct ggml_init_params ip = { CTX_SIZE, NULL, false };
        struct ggml_context *ctx = ggml_init(ip);
        // A separate slab per type (T6: UD quantization varies type by role)
        struct ggml_tensor *sg = ggml_new_tensor_3d(ctx, (enum ggml_type)T_gate.type, n_embd, n_ff, n_slot);
        struct ggml_tensor *su = ggml_new_tensor_3d(ctx, (enum ggml_type)T_up.type,   n_embd, n_ff, n_slot);
        struct ggml_tensor *sd = ggml_new_tensor_3d(ctx, (enum ggml_type)T_down.type, n_ff, n_embd, n_slot);

        double t0 = now_s();
        for (int s = 0; s < n_slot; ++s) {
            const int e = slot2exp[s];
            if (read_expert(fd, &T_gate, e, (char *)sg->data + (size_t)s * T_gate.expert_bytes) ||
                read_expert(fd, &T_up,   e, (char *)su->data + (size_t)s * T_up.expert_bytes)   ||
                read_expert(fd, &T_down, e, (char *)sd->data + (size_t)s * T_down.expert_bytes)) return 2;
        }
        double t1 = now_s();
        printf("  read %.2f MiB in %.3f s (%.2f GB/s, %d preads)\n",
               g_bytes_read / 1048576.0, t1 - t0, g_bytes_read / (t1 - t0) / 1e9, n_slot * 3);

        struct ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, N_TOK);
        for (int t = 0; t < N_TOK; ++t)
            memcpy((char *)x->data + (size_t)t * x->nb[2], xin[t], sizeof(float) * n_embd);

        // ids hold slot_ids here
        if (moe_ffn(ctx, sg, su, sd, x, slot_ids, rw, out_slab, n_embd) != 0) return 2;
        ggml_free(ctx);
    }

    // ---------------- T3: numerical agreement ----------------
    printf("\nT3: does the streamed MoE FFN match the fully resident one? (the central question)\n");
    {
        int bitexact = memcmp(out_full, out_slab, sizeof(float) * n_embd * N_TOK) == 0;
        float md = 0, rms = 0, ref = 0;
        for (int i = 0; i < n_embd * N_TOK; ++i) {
            float d = fabsf(out_full[i] - out_slab[i]);
            if (d > md) md = d;
            rms += d * d; ref += out_full[i] * out_full[i];
        }
        char b[160];
        snprintf(b, sizeof b, "max|diff|=%.3e  relative RMS=%.3e", (double)md,
                 (double)sqrtf(rms / (ref + 1e-20f)));
        report("MoE FFN output is bit-identical", bitexact, b);
    }

    // ---------------- T4/T5: reduction and bandwidth ----------------
    printf("\nT4: reduction in bytes read\n");
    {
        const uint64_t full = T_gate.nbytes + T_up.nbytes + T_down.nbytes;
        char b[160];
        snprintf(b, sizeof b, "fully resident %.1f MiB -> streamed %.2f MiB (1/%.1f)",
                 full / 1048576.0, g_bytes_read / 1048576.0, (double)full / (double)g_bytes_read);
        report("only the required data is read", g_bytes_read < full / 10, b);
    }

    printf("\nT5: effective bandwidth for random expert reads (pins down §2.3's BW)\n");
    printf("     measured with O_DIRECT to bypass the page cache.\n");
    printf("     GGUF offsets are not 4 KiB aligned, so read-around is needed (§13.6).\n");
    {
        const int N = 256;
        double bw = 0.0, bw_cached = 0.0;

        // --- (a) through the page cache: informational, not real bandwidth ---
        {
            void *tmp = malloc(eb);
            double t0 = now_s(); uint64_t bytes = 0;
            for (int i = 0; i < N; ++i) {
                int e = (int)(((uint64_t)i * 2654435761u) % (uint64_t)n_exp);
                read_expert(fd, &T_gate, e, tmp);
                read_expert(fd, &T_up,   e, tmp);
                read_expert(fd, &T_down, e, tmp);
                bytes += eb;
            }
            double dt = now_s() - t0;
            free(tmp);
            bw_cached = bytes / dt / 1e9;
            printf("  [INFO] via page cache            : %.2f GB/s  (informational; not the disk)\n", bw_cached);
        }

        // --- (b) O_DIRECT: bypasses the page cache, the true bandwidth ---
        int fdd = open(gguf, O_RDONLY | O_DIRECT);
        if (fdd < 0) {
            report("O_DIRECT bandwidth measurement", 0, strerror(errno));
        } else {
            // Bounce buffer for read-around, since offsets are not 4 KiB aligned
            const size_t cap = ((eb + 8191) & ~(size_t)4095) + 4096;
            void *bb = NULL;
            if (posix_memalign(&bb, 4096, cap) != 0) { report("bounce alloc", 0, NULL); }
            uint64_t bytes = 0, dev_bytes = 0;
            double t0 = now_s();
            for (int i = 0; i < N; ++i) {
                int e = (int)(((uint64_t)i * 2654435761u) % (uint64_t)n_exp);
                const TensorInfo *ts[3] = { &T_gate, &T_up, &T_down };
                for (int j = 0; j < 3; ++j) {
                    const uint64_t off = ts[j]->off + (uint64_t)e * ts[j]->expert_bytes;
                    const uint64_t a0  = off & ~(uint64_t)4095;
                    const uint64_t a1  = (off + ts[j]->expert_bytes + 4095) & ~(uint64_t)4095;
                    ssize_t r = pread(fdd, bb, (size_t)(a1 - a0), (off_t)a0);
                    if (r < 0) { perror("pread(O_DIRECT)"); break; }
                    bytes     += ts[j]->expert_bytes;
                    dev_bytes += (uint64_t)(a1 - a0);
                }
            }
            double dt = now_s() - t0;
            free(bb); close(fdd);
            bw = bytes / dt / 1e9;
            char b[224];
            snprintf(b, sizeof b,
                     "%d experts / %.3f s = %.2f GB/s (%.3f ms/expert), %.2f%% extra read",
                     N, dt, bw, dt / N * 1000,
                     100.0 * ((double)dev_bytes / (double)bytes - 1.0));
            report("O_DIRECT effective bandwidth", bw > 0.5, b);
            printf("  [INFO] %.1fx apart from the page-cache figure, so the cache-measurement trap is avoided\n",
                   bw_cached / (bw > 0 ? bw : 1));

            // --- (c) scaling as the effective queue depth rises ---
            //   Distinguishes "1.51 GB/s is the device limit" from "it is the
            //   QD=1 limit". This decides whether the io_uring + QD=64 design
            //   in §13.5 / §23 is justified.
            printf("\n  --- bandwidth scaling with parallelism (the basis for the §13.5 QD design) ---\n");
            for (int nthr = 1; nthr <= 32; nthr *= 2) {
                bw_par[0] = 0;
                double r = measure_parallel(gguf, nthr, N);
                printf("    QD=%2d (%2d threads) : %6.2f GB/s%s\n", nthr, nthr, r,
                       nthr == 1 ? "   <- today's synchronous pread" : "");
                if (r > bw) bw = r;   // keep the highest achievable bandwidth
            }
            printf("    -> adopting the parallel maximum, %.2f GB/s, as the design bandwidth\n", bw);
        }

        // Recompute §2.3's required hit rate from the measured values
        const double B_act = 40.0 * TOP_K * (double)eb;   // 40 layers x top-8
        const double t_c = 0.0437;                        // measured: llama-server at 43.7 ms/token
        const double budget = bw * 1e9 * 1.2 * t_c;
        double need = 1.0 - budget / B_act; if (need < 0) need = 0;
        printf("\n  --- regime determination from measurements ---\n");
        printf("    B_act        = %.1f MiB/token  (40 layers x top-8 x %.3f MiB)\n",
               B_act / 1048576.0, eb / 1048576.0);
        printf("    t_c (measured) = 43.7 ms/token (llama-server at 22.88 tok/s)\n");
        printf("    bandwidth      = %.2f GB/s\n", bw);
        printf("    -> hit rate needed to stay within a 20%% slowdown: h >= %.1f%%\n", need * 100);
    }

    printf("\nT6: coping with UD quantization's mixed types\n");
    {
        int mixed = strcmp(T_gate.tname, T_down.tname) != 0;
        char b[160];
        snprintf(b, sizeof b, "gate=%s up=%s down=%s -> a separate slab per role is required",
                 T_gate.tname, T_up.tname, T_down.tname);
        report("works with mixed types via per-role slabs", mixed, b);
    }

    close(fd);
    printf("\n=====================================================================\n");
    if (g_fail == 0) {
        printf(" verdict: PASS — a real model's experts were read directly from SSD\n");
        printf("        and the MoE FFN ran bit-identically on the slot slab\n");
    } else {
        printf(" verdict: FAIL (%d)\n", g_fail);
    }
    printf("=====================================================================\n");
    return g_fail == 0 ? 0 : 1;
}
