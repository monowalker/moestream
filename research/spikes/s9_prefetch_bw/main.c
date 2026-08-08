// =============================================================================
// MoEStream Spike S9 — does prefetching actually raise SSD bandwidth?
//
// Measured (Laguna-S-2.1 / frac=0.20 / decode):
//   I/O 115836 reads, 166383 MiB in 57.783 s  →  3.02 GB/s
//   device limit (S2, sequential large blocks) -> 4.48 GB/s
//   so only 67% of the available bandwidth is in use.
//
// Hypothesis: repeating "batch read -> wait -> compute" once per layer, 47
//   times, drains the queue each time and leaves the SSD idle. Keeping the
//   queue continuously fed should approach the device limit.
//
// If the hypothesis holds, prefetching is worth 2.35x; if not, only 1.58x.
// Settle this before implementing anything.
//
// Access pattern reproduced here, taken straight from measurements:
//   one token  = 47 layers x 9.75 reads = 458 reads
//   one read   = 1351680 B (= 330 x 4096, O_DIRECT aligned)
//   compute per layer = 2.83 ms, during which the current implementation
//                       leaves the SSD idle
//
//   Usage: ./s9_prefetch_bw <large file>
// =============================================================================
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

#define RSIZE     1351680ull      // bytes per read (the measured expert_bytes)
#define N_LAYER   47              // MoE layer count
#define READS_PL  10              // reads per layer (measured 9.75, rounded up)
#define COMPUTE_MS 2.83           // compute time per layer (measured)
#define N_TOKEN   20              // tokens to measure
#define MAXTH     32

static int      g_fd;
static off_t  * g_off;            // table of read offsets
static size_t   g_total;          // total number of reads
static void   * g_buf[MAXTH];     // per-thread destination buffers

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void spin_ms(double ms) {          // simulate compute; sleep is too coarse
    const double t = now_s() + ms / 1000.0;
    while (now_s() < t) { }
}

// ---- shared-queue variant (continuous queueing) -----------------------------
static volatile size_t g_next;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

struct th_arg { int id; size_t lo, hi; };

static void * worker_queue(void * a) {
    struct th_arg * t = (struct th_arg *) a;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        const size_t k = g_next < g_total ? g_next++ : (size_t) -1;
        pthread_mutex_unlock(&g_mu);
        if (k == (size_t) -1) return NULL;
        if (pread(g_fd, g_buf[t->id], RSIZE, g_off[k]) < 0)
            fprintf(stderr, "pread: %s\n", strerror(errno));
    }
}

// ---- range-split variant (used within a burst) ------------------------------
static void * worker_range(void * a) {
    struct th_arg * t = (struct th_arg *) a;
    for (size_t k = t->lo; k < t->hi; ++k)
        if (pread(g_fd, g_buf[t->id], RSIZE, g_off[k]) < 0)
            fprintf(stderr, "pread: %s\n", strerror(errno));
    return NULL;
}

// Reproduce the current implementation: spawn threads per layer, wait, and
// leave a compute gap
static double run_burst(int nth, int spawn_each_time, int with_gap) {
    pthread_t th[MAXTH]; struct th_arg ar[MAXTH];
    size_t k = 0;
    const double t0 = now_s();
    for (int tok = 0; tok < N_TOKEN; ++tok) {
        for (int l = 0; l < N_LAYER; ++l) {
            const size_t lo = k, hi = k + READS_PL;
            const size_t per = (READS_PL + nth - 1) / nth;
            int used = 0;
            for (int i = 0; i < nth; ++i) {
                const size_t a = lo + (size_t) i * per;
                if (a >= hi) break;
                ar[i].id = i; ar[i].lo = a;
                ar[i].hi = (a + per < hi) ? a + per : hi;
                pthread_create(&th[i], NULL, worker_range, &ar[i]);
                used++;
            }
            for (int i = 0; i < used; ++i) pthread_join(th[i], NULL);
            k = hi;
            if (with_gap) spin_ms(COMPUTE_MS);
        }
    }
    (void) spawn_each_time;
    return now_s() - t0;
}

// Ideal prefetch: never let the queue drain (compute is assumed to run
// concurrently on another thread)
static double run_stream(int nth) {
    pthread_t th[MAXTH]; struct th_arg ar[MAXTH];
    g_next = 0;
    const double t0 = now_s();
    for (int i = 0; i < nth; ++i) { ar[i].id = i; pthread_create(&th[i], NULL, worker_queue, &ar[i]); }
    for (int i = 0; i < nth; ++i) pthread_join(th[i], NULL);
    return now_s() - t0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { puts("usage: s9_prefetch_bw <big file>"); return 1; }

    g_fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (g_fd < 0) { fprintf(stderr, "open: %s\n", strerror(errno)); return 1; }
    struct stat st;
    if (fstat(g_fd, &st) != 0) return 1;

    g_total = (size_t) N_TOKEN * N_LAYER * READS_PL;
    g_off = malloc(sizeof(off_t) * g_total);
    // Scatter across the whole GGUF, matching real expert placement
    const off_t span = (off_t) (st.st_size - (off_t) RSIZE);
    unsigned int seed = 12345;
    for (size_t i = 0; i < g_total; ++i) {
        const double r = (double) rand_r(&seed) / (double) RAND_MAX;
        g_off[i] = ((off_t) (r * (double) span)) & ~(off_t) 4095;
    }
    for (int i = 0; i < MAXTH; ++i)
        if (posix_memalign(&g_buf[i], 4096, RSIZE) != 0) return 1;

    const double gib = (double) g_total * (double) RSIZE / 1073741824.0;
    printf("=====================================================================\n");
    printf(" S9: does prefetching raise SSD bandwidth?\n");
    printf("=====================================================================\n");
    printf("  file          : %s (%.1f GiB)\n", argv[1], st.st_size / 1073741824.0);
    printf("  per read      : %.2f MiB   total reads %zu (%.2f GiB)\n",
           RSIZE / 1048576.0, g_total, gib);
    printf("  pattern       : %d tokens x %d layers x %d reads\n", N_TOKEN, N_LAYER, READS_PL);
    printf("  compute/layer : %.2f ms\n\n", COMPUTE_MS);

    printf("--- (A) reproducing today: per-layer bursts plus a compute gap ---\n");
    for (int nth = 8; nth <= 8; ++nth) {
        const double dt = run_burst(nth, 1, 1);
        printf("  %2d threads : %6.2f s  -> %5.2f GB/s   (effective, including the gap)\n",
               nth, dt, gib * 1073741824.0 / dt / 1e9);
    }

    printf("\n--- (B) gap removed: I/O only, as if compute were hidden ---\n");
    for (int nth = 8; nth <= 8; ++nth) {
        const double dt = run_burst(nth, 1, 0);
        printf("  %2d threads : %6.2f s  -> %5.2f GB/s\n",
               nth, dt, gib * 1073741824.0 / dt / 1e9);
    }

    printf("\n--- (C) ideal prefetch: continuous queueing ---\n");
    for (int nth = 4; nth <= 32; nth *= 2) {
        const double dt = run_stream(nth);
        printf("  %2d threads : %6.2f s  -> %5.2f GB/s\n",
               nth, dt, gib * 1073741824.0 / dt / 1e9);
    }

    close(g_fd);
    printf("\n=====================================================================\n");
    printf(" reading: if (C) clearly beats (A), prefetching improves bandwidth\n");
    printf("          if (C) is about equal to (B), the gain is only from hiding compute\n");
    printf("=====================================================================\n");
    return 0;
}
