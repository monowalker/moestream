// MoEStream Spike S15 — how much of a decode read is syscall overhead?
//
// Decode issues roughly 6 preads per layer (1-2 missed experts x gate/up/down)
// across 40 layers, so about 240 syscalls per token. If a large share of that
// is per-call overhead rather than data movement, batching the submissions
// (io_uring) could pay off. Finding S2 measured io_uring at only +1.5%, but
// that was on prefill-sized reads where per-call overhead is amortized.
//
// This measures the decode-sized case directly.
//
// Method: read the same total bytes two ways from page-cache-warm data,
//   (a) as N separate preads of the real expert size
//   (b) as one large pread
// The difference in per-byte cost is the per-call overhead.
//
//   Usage: ./s15_syscall_overhead <gguf path>
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

#define EXPERT_BYTES 450560ull      // measured on Ornith-1.0-35B
#define N_READ       240            // ~6 per layer x 40 layers = one token

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static unsigned long long rnd(unsigned long long *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gguf>\n", argv[0]); return 2; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }

    // Work inside a window small enough to stay in the page cache.
    const unsigned long long WINDOW = 512ull << 20;      // 512 MiB
    if ((unsigned long long) st.st_size < WINDOW * 2) {
        fprintf(stderr, "file too small\n"); return 1;
    }
    const unsigned long long base = (unsigned long long) st.st_size / 4;

    void *buf = NULL;
    if (posix_memalign(&buf, 4096, (size_t) EXPERT_BYTES * 8)) return 1;

    // Warm the page cache for the window.
    printf("warming page cache over %.0f MiB ...\n", WINDOW / 1048576.0);
    for (unsigned long long o = 0; o < WINDOW; o += EXPERT_BYTES)
        if (pread(fd, buf, EXPERT_BYTES, (off_t)(base + o)) < 0) break;

    unsigned long long seed = 12345;
    off_t offs[N_READ];
    for (int i = 0; i < N_READ; ++i)
        offs[i] = (off_t)(base + (rnd(&seed) % (WINDOW - EXPERT_BYTES)));

    const double total_mib = (double) EXPERT_BYTES * N_READ / 1048576.0;

    // (a) N separate preads, the pattern decode actually uses
    double best_a = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        for (int i = 0; i < N_READ; ++i)
            if (pread(fd, buf, EXPERT_BYTES, offs[i]) < 0) { perror("pread"); return 1; }
        const double dt = now_s() - t0;
        if (dt < best_a) best_a = dt;
    }

    // (b) the same bytes in one call, to isolate pure copy throughput
    const size_t big = (size_t) EXPERT_BYTES * N_READ;
    void *bigbuf = NULL;
    if (posix_memalign(&bigbuf, 4096, big)) { fprintf(stderr, "alloc %zu failed\n", big); return 1; }
    double best_b = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        if (pread(fd, bigbuf, big, (off_t) base) < 0) { perror("pread big"); return 1; }
        const double dt = now_s() - t0;
        if (dt < best_b) best_b = dt;
    }

    // (c) The syscall floor: same call count, minimal data. Comparing (a) with
    //     (b) turned out to measure destination-buffer locality rather than
    //     syscall cost -- (a) rewrites one hot 3.4 MiB buffer while (b) fills a
    //     cold 103 MiB one. Reading 4 KiB per call isolates the fixed part.
    double best_c = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        for (int i = 0; i < N_READ; ++i)
            if (pread(fd, buf, 4096, offs[i]) < 0) { perror("pread small"); return 1; }
        const double dt = now_s() - t0;
        if (dt < best_c) best_c = dt;
    }

    const double per_call_a = best_a / N_READ * 1e6;          // us
    const double per_call_b = best_b / N_READ * 1e6;          // us, same bytes
    const double per_call_c = best_c / N_READ * 1e6;          // us, syscall floor
    const double overhead   = per_call_c;                      // fixed cost per call

    printf("\n  bytes per read      : %.2f MiB\n", EXPERT_BYTES / 1048576.0);
    printf("  reads               : %d  (about one token)\n", N_READ);
    printf("  total               : %.1f MiB\n\n", total_mib);
    printf("  (a) %d separate preads : %8.3f ms   %6.2f GB/s   %7.2f us/call\n",
           N_READ, best_a * 1000, total_mib / 1024.0 / best_a, per_call_a);
    printf("  (b) 1 large pread      : %8.3f ms   %6.2f GB/s   %7.2f us/call-equiv\n",
           best_b * 1000, total_mib / 1024.0 / best_b, per_call_b);
    printf("      NOTE: (b) fills a cold 103 MiB buffer while (a) rewrites one hot\n");
    printf("      3.4 MiB buffer, so (a) vs (b) measures buffer locality, not syscalls.\n");
    printf("  (c) %d preads of 4 KiB : %8.3f ms                  %7.2f us/call  <- syscall floor\n",
           N_READ, best_c * 1000, per_call_c);
    printf("\n  fixed cost per call    : %6.2f us  x %d = %.3f ms per token\n",
           overhead, N_READ, overhead * N_READ / 1000.0);
    printf("  as a share of a 58 ms decode : %.2f%%\n", overhead * N_READ / 1000.0 / 58.0 * 100.0);
    printf("\n  Batching submissions (io_uring) can remove at most that fixed cost.\n");

    free(buf); free(bigbuf); close(fd);
    return 0;
}
