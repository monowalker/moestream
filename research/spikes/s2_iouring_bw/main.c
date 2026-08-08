// =============================================================================
// MoEStream Spike S2 — io_uring bandwidth measurement (§13.3 / §13.5 / §23)
//
// What S1 established:
//   synchronous pread (QD=1)     : 1.55 GB/s
//   parallel pthread pread (QD=8): 4.42 GB/s, saturated
//
// Questions:
//   (a) can io_uring beat parallel pread, or is 4.46 GB/s the device limit?
//   (b) are .msp's 2 MiB aligned records faster than GGUF's unaligned
//       read-around? (decides whether §18.4's 2.7% disk overhead is worth it)
//   (c) what does the latency distribution look like? §13.5's adaptive QD
//       control keys off p99.
//
// Relation to the design document:
//   the flags in §23.3 (SINGLE_ISSUER / DEFER_TASKRUN / COOP_TASKRUN) and
//   REGISTER_BUFFERS / REGISTER_FILES are each measured separately.
// =============================================================================

#define _GNU_SOURCE
#include <liburing.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// Deterministic pseudo-random offsets
static uint64_t xs = 0x9e3779b97f4a7c15ull;
static uint64_t xrand(void) {
    xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs;
}

typedef struct {
    const char *label;
    int    qd;
    int    n_req;
    size_t req_bytes;      // bytes per request
    int    aligned;        // 1 = 4 KiB aligned (.msp-like) / 0 = unaligned read-around (GGUF-like)
    int    use_regbuf;     // whether to use registered buffers
    unsigned setup_flags;  // io_uring_setup flags
} Cfg;

typedef struct {
    double bw_gbps;
    double lat_p50_ms, lat_p99_ms, lat_max_ms;
    uint64_t dev_bytes, req_bytes;
    long n_err;
} Res;

// -----------------------------------------------------------------------------
static int run_case(const char *path, uint64_t region_start, uint64_t region_len,
                    const Cfg *c, Res *out)
{
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT"); return -1; }

    struct io_uring ring;
    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    p.flags = c->setup_flags;
    if (io_uring_queue_init_params(c->qd * 2, &ring, &p) < 0) {
        // Retry without flags on kernels that lack the newer ones
        memset(&p, 0, sizeof p);
        if (io_uring_queue_init_params(c->qd * 2, &ring, &p) < 0) {
            perror("io_uring_queue_init"); close(fd); return -1;
        }
    }

    // Read length per request (up to +4 KiB for read-around when unaligned)
    const size_t slot_cap = ((c->req_bytes + 4095) & ~(size_t)4095) + 4096;

    // Allocate buffers; QD slots are reused
    void **bufs = calloc(c->qd, sizeof(void *));
    struct iovec *iov = calloc(c->qd, sizeof(struct iovec));
    for (int i = 0; i < c->qd; ++i) {
        if (posix_memalign(&bufs[i], 4096, slot_cap) != 0) { fprintf(stderr, "memalign\n"); return -1; }
        memset(bufs[i], 0, slot_cap);   // fault the pages in ahead of time
        iov[i].iov_base = bufs[i];
        iov[i].iov_len  = slot_cap;
    }

    // liburing returns -errno; it does not set errno
    int regbuf_ok = 0;
    if (c->use_regbuf) {
        const int rc = io_uring_register_buffers(&ring, iov, c->qd);
        if (rc < 0) fprintf(stderr, "  (register_buffers failed: %s; falling back to normal reads)\n", strerror(-rc));
        else regbuf_ok = 1;
    }
    int reg_fd_ok = (io_uring_register_files(&ring, &fd, 1) == 0);

    double *lat = calloc(c->n_req, sizeof(double));
    double *start_t = calloc(c->qd, sizeof(double));
    // Completions arrive out of order, so slots must be tracked with a free
    // stack. Indexing by inflight count would let buffers collide.
    int    *free_stack = calloc(c->qd, sizeof(int));
    int     free_top = c->qd;
    for (int i = 0; i < c->qd; ++i) free_stack[i] = i;
    uint64_t dev_bytes = 0;
    long n_err = 0, n_short = 0;

    // Generate submission offsets
    const uint64_t span = region_len > c->req_bytes ? region_len - c->req_bytes : 0;

    int issued = 0, done = 0, inflight = 0;
    const double t0 = now_s();

    while (done < c->n_req) {
        // Fill the queue
        while (inflight < c->qd && issued < c->n_req) {
            uint64_t off = region_start + (xrand() % (span ? span : 1));
            size_t len;
            if (c->aligned) {
                off &= ~(uint64_t)4095;                 // .msp: always aligned
                len = (c->req_bytes + 4095) & ~(size_t)4095;
            } else {
                off += 4000;                            // GGUF: reproduce a uniform 4000-byte skew
                const uint64_t a0 = off & ~(uint64_t)4095;
                const uint64_t a1 = (off + c->req_bytes + 4095) & ~(uint64_t)4095;
                len = (size_t)(a1 - a0);
                off = a0;
            }
            if (free_top == 0) break;
            const int s = free_stack[--free_top];        // take a slot from the free stack
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) { free_stack[free_top++] = s; break; }
            if (regbuf_ok)   // use fixed buffers only if registration succeeded
                io_uring_prep_read_fixed(sqe, reg_fd_ok ? 0 : fd, bufs[s], (unsigned)len, off, s);
            else
                io_uring_prep_read(sqe, reg_fd_ok ? 0 : fd, bufs[s], (unsigned)len, off);
            if (reg_fd_ok) sqe->flags |= IOSQE_FIXED_FILE;
            io_uring_sqe_set_data64(sqe, ((uint64_t)s << 32) | (uint32_t)len);
            start_t[s] = now_s();
            dev_bytes += len;
            issued++; inflight++;
        }
        io_uring_submit(&ring);

        // Reap completions
        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) { fprintf(stderr, "wait_cqe: %s\n", strerror(-ret)); break; }
        unsigned head; int cnt = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            const uint64_t d = io_uring_cqe_get_data64(cqe);
            const int s = (int)(d >> 32);
            const uint32_t want = (uint32_t)d;
            if (cqe->res < 0)            n_err++;
            else if ((uint32_t)cqe->res != want) n_short++;
            lat[done] = (now_s() - start_t[s]) * 1000.0;
            free_stack[free_top++] = s;                  // return the slot
            done++; cnt++;
        }
        io_uring_cq_advance(&ring, cnt);
        inflight -= cnt;
    }
    const double dt = now_s() - t0;

    qsort(lat, done, sizeof(double), cmp_d);
    out->bw_gbps   = (double)c->n_req * c->req_bytes / dt / 1e9;
    out->n_err = n_err + n_short;
    out->lat_p50_ms = lat[done / 2];
    out->lat_p99_ms = lat[(int)(done * 0.99)];
    out->lat_max_ms = lat[done - 1];
    out->dev_bytes = dev_bytes;
    out->req_bytes = (uint64_t)c->n_req * c->req_bytes;

    if (n_err || n_short) {
        fprintf(stderr, "  FAULT: %ld errors / %ld short reads; this measurement is invalid\n", n_err, n_short);
        out->bw_gbps = -1.0;
    }
    free(lat); free(start_t); free(free_stack);
    for (int i = 0; i < c->qd; ++i) free(bufs[i]);
    free(bufs); free(iov);
    io_uring_queue_exit(&ring);
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Ornith-1.0-35B-UD-IQ4_NL.gguf";
    const int n_req  = argc > 2 ? atoi(argv[2]) : 2048;

    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); return 1; }
    // Target the middle of the file, where the expert data lives
    const uint64_t region_start = (uint64_t)st.st_size / 8;
    const uint64_t region_len   = (uint64_t)st.st_size / 2;

    printf("=====================================================================\n");
    printf(" MoEStream Spike S2 : io_uring bandwidth (§13.3 / §13.5 / §23)\n");
    printf("=====================================================================\n");
    printf("  file   : %s (%.2f GiB)\n", path, st.st_size / 1073741824.0);
    printf("  region : offset %.2f GiB through %.2f GiB\n",
           region_start / 1073741824.0, region_len / 1073741824.0);
    printf("  baseline: sync pread QD=1 = 1.55 GB/s / parallel pthread QD=8 = 4.42 GB/s (Finding S1)\n");

    const unsigned FLAGS =
#ifdef IORING_SETUP_SINGLE_ISSUER
        IORING_SETUP_SINGLE_ISSUER |
#endif
#ifdef IORING_SETUP_DEFER_TASKRUN
        IORING_SETUP_DEFER_TASKRUN |
#endif
#ifdef IORING_SETUP_COOP_TASKRUN
        IORING_SETUP_COOP_TASKRUN |
#endif
        0;

    // ---- (1) QD sweep: GGUF-like (unaligned read-around, 1.42 MiB) ----
    printf("\n--- (1) direct GGUF reads: unaligned 1.42 MiB read-around ---\n");
    printf("   QD | bandwidth |  p50 lat |  p99 lat |  max lat | extra read\n");
    printf("  ----+-----------+----------+----------+----------+---------\n");
    double best_gguf = 0; int best_qd_gguf = 0;
    for (int qd = 1; qd <= 128; qd *= 2) {
        Cfg c = { "gguf", qd, n_req, 1490944, 0, 1, FLAGS };
        Res r;
        if (run_case(path, region_start, region_len, &c, &r) != 0 || r.n_err) continue;
        printf("  %3d | %6.2f GB/s | %7.3f ms | %7.3f ms | %7.3f ms | %+5.2f%%\n",
               qd, r.bw_gbps, r.lat_p50_ms, r.lat_p99_ms, r.lat_max_ms,
               100.0 * ((double)r.dev_bytes / (double)r.req_bytes - 1.0));
        if (r.bw_gbps > best_gguf) { best_gguf = r.bw_gbps; best_qd_gguf = qd; }
    }

    // ---- (2) QD sweep: .msp-like (4 KiB aligned, 2 MiB records) ----
    printf("\n--- (2) .msp-like: 4 KiB aligned 2 MiB records (§18.4) ---\n");
    printf("   QD | bandwidth |  p50 lat |  p99 lat |  max lat\n");
    printf("  ----+-----------+----------+----------+----------\n");
    double best_msp = 0; int best_qd_msp = 0;
    for (int qd = 1; qd <= 128; qd *= 2) {
        Cfg c = { "msp", qd, n_req, 2097152, 1, 1, FLAGS };
        Res r;
        if (run_case(path, region_start, region_len, &c, &r) != 0 || r.n_err) continue;
        printf("  %3d | %6.2f GB/s | %7.3f ms | %7.3f ms | %7.3f ms\n",
               qd, r.bw_gbps, r.lat_p50_ms, r.lat_p99_ms, r.lat_max_ms);
        if (r.bw_gbps > best_msp) { best_msp = r.bw_gbps; best_qd_msp = qd; }
    }

    // ---- (3) contribution of each setup flag ----
    printf("\n--- (3) contribution of io_uring settings (QD=%d, .msp-like) ---\n", best_qd_msp ? best_qd_msp : 16);
    const int qd = best_qd_msp ? best_qd_msp : 16;
    struct { const char *name; unsigned f; int reg; } variants[] = {
        { "default (no flags, no registered buffers)", 0,     0 },
        { "registered buffers only",                   0,     1 },
        { "SINGLE_ISSUER|DEFER_TASKRUN|COOP_TASKRUN",   FLAGS, 0 },
        { "above + registered buffers (the recommended setup)", FLAGS, 1 },
    };
    for (size_t i = 0; i < sizeof variants / sizeof variants[0]; ++i) {
        Cfg c = { "v", qd, n_req, 2097152, 1, variants[i].reg, variants[i].f };
        Res r;
        if (run_case(path, region_start, region_len, &c, &r) != 0) continue;
        printf("  %-46s : %6.2f GB/s (p99 %.3f ms)\n", variants[i].name, r.bw_gbps, r.lat_p99_ms);
    }

    // ---- summary ----
    printf("\n=====================================================================\n");
    printf("  best GGUF-like : %.2f GB/s (QD=%d)\n", best_gguf, best_qd_gguf);
    printf("  best .msp-like : %.2f GB/s (QD=%d)\n", best_msp,  best_qd_msp);
    printf("  pthread pread  : 4.42 GB/s (QD=8, Finding S1)\n");
    printf("  -> io_uring gain      : %+.1f%% (vs parallel pread)\n",
           100.0 * (best_msp / 4.42 - 1.0));
    printf("  -> .msp alignment gain: %+.1f%% (vs GGUF read-around)\n",
           100.0 * (best_msp / (best_gguf > 0 ? best_gguf : 1) - 1.0));

    // Update the required hit rate
    const double B_act = 455.0 * 1048576.0;
    const double t_c = 0.0437;
    for (int i = 0; i < 2; ++i) {
        const double bw = i ? best_msp : best_gguf;
        double need = 1.0 - (bw * 1e9 * 1.2 * t_c) / B_act;
        if (need < 0) need = 0;
        printf("  -> %-9s at BW=%.2f GB/s, required hit rate h >= %.1f%%\n",
               i ? ".msp" : "GGUF", bw, need * 100);
    }
    printf("=====================================================================\n");
    return 0;
}
