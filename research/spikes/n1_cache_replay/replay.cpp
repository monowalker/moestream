// =============================================================================
// MoEStream — trace replay
//
// Feed the real expert-activation trace captured in M0-2 through the actual
// ExpertCache implementation.
//
//   (a) does it reach the same hit rate as the Python simulation
//       (analyze_trace.py)? -- verifies the implementation
//   (b) by actually reading missed experts from SSD, measure bytes/token
//       (the master metric of §P2) and real elapsed time
//       -- the first real data on the tok/s MoEStream can deliver
//
// Usage:
//   replay <trace> <expert_index.txt> [gguf] [cache_frac] [pin_ratio]
//     Omitting the gguf skips I/O and exercises only the cache behaviour.
// =============================================================================

#define _GNU_SOURCE
#include "expert_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

using namespace moestream;

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---- expert_index.txt (produced by tools/analysis/gguf_inspect.py --layout -1) ----
struct RoleInfo { uint64_t off = 0, nbytes = 0; uint32_t expert_bytes = 0; };
struct LayerInfo { RoleInfo gate, up, down; };

static bool load_index(const char* path, std::vector<LayerInfo>& out) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); return false; }
    char line[512]; int nlayer = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "NLAYER %d", &nlayer) == 1) { out.assign(nlayer, LayerInfo{}); continue; }
        int il; char role[16]; unsigned long off, nb; int ty; long n0, n1, n2; char tn[32];
        if (sscanf(line, "E %d %15s %lu %lu %d %ld %ld %ld %31s",
                   &il, role, &off, &nb, &ty, &n0, &n1, &n2, tn) == 9) {
            if (il < 0 || il >= (int)out.size()) continue;
            RoleInfo r; r.off = off; r.nbytes = nb; r.expert_bytes = uint32_t(nb / n2);
            if      (!strcmp(role, "gate")) out[il].gate = r;
            else if (!strcmp(role, "up"))   out[il].up   = r;
            else if (!strcmp(role, "down")) out[il].down = r;
        }
    }
    fclose(f);
    return nlayer > 0;
}

// ---- trace (produced by tools/analysis/expert_trace) ------------------------
struct Trace {
    uint32_t n_moe = 0, top_k = 0;
    uint64_t n_token = 0;
    std::vector<uint16_t> ids;      // [n_token][n_moe][top_k]
};

static bool load_trace(const char* path, Trace& t) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    uint32_t h[8];
    if (fread(h, sizeof h, 1, f) != 1) { fclose(f); return false; }
    if (h[0] != 0x5254534Du) { fprintf(stderr, "magic mismatch\n"); fclose(f); return false; }
    t.n_moe = h[2]; t.top_k = h[4];
    t.n_token = uint64_t(h[5]) | (uint64_t(h[6]) << 32);
    const size_t per = size_t(t.n_moe) * t.top_k * 2;   // ids + weights
    std::vector<uint16_t> buf(per);
    t.ids.reserve(t.n_token * t.n_moe * t.top_k);
    for (uint64_t i = 0; i < t.n_token; ++i) {
        if (fread(buf.data(), sizeof(uint16_t), per, f) != per) break;
        for (uint32_t l = 0; l < t.n_moe; ++l)
            for (uint32_t k = 0; k < t.top_k; ++k)
                t.ids.push_back(buf[size_t(l) * t.top_k * 2 + k]);
    }
    fclose(f);
    t.n_token = t.ids.size() / (size_t(t.n_moe) * t.top_k);
    return t.n_token > 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <trace> <expert_index.txt> [gguf] [cache_frac] [pin_ratio]\n", argv[0]);
        return 1;
    }
    const char* trace_path = argv[1];
    const char* index_path = argv[2];
    const char* gguf       = argc > 3 && strcmp(argv[3], "-") ? argv[3] : nullptr;
    const double frac      = argc > 4 ? atof(argv[4]) : 0.38;
    const double pin       = argc > 5 ? atof(argv[5]) : 0.05;
    const bool   adapt     = argc > 6 ? atoi(argv[6]) != 0 : true;

    Trace tr;
    std::vector<LayerInfo> idx;
    if (!load_trace(trace_path, tr)) return 2;
    if (!load_index(index_path, idx)) return 2;

    uint32_t n_expert = 0;
    for (auto v : tr.ids) n_expert = std::max<uint32_t>(n_expert, v + 1u);

    printf("=====================================================================\n");
    printf(" MoEStream — trace replay\n");
    printf("=====================================================================\n");
    printf("  trace  : %s  (%llu tokens, %u MoE layers, top-%u, n_expert=%u)\n",
           trace_path, (unsigned long long)tr.n_token, tr.n_moe, tr.top_k, n_expert);
    printf("  cache  : %.0f%%  pin_ratio=%.2f  per-layer quota adaptation=%s\n",
           frac * 100, pin, adapt ? "ON" : "OFF");

    const uint32_t eb = idx[0].gate.expert_bytes + idx[0].up.expert_bytes + idx[0].down.expert_bytes;
    printf("  expert : %u B (%.3f MiB) x %u layers -> B_act=%.1f MiB/token\n",
           eb, eb / 1048576.0, tr.n_moe, double(tr.n_moe) * tr.top_k * eb / 1048576.0);

    int fd = -1;
    std::vector<uint8_t> slab;      // backing store for slots: a flat region
    if (gguf) {
        fd = open(gguf, O_RDONLY);
        if (fd < 0) { perror("open gguf"); return 2; }
        printf("  I/O    : reading from the real SSD (%s)\n", gguf);
    } else {
        printf("  I/O    : none (cache behaviour only)\n");
    }

    ExpertCache::Config cfg;
    cfg.n_layer = tr.n_moe;
    cfg.n_expert = n_expert;
    cfg.cache_frac = frac;
    cfg.pin_ratio = pin;
    cfg.layer_quota_adapt = adapt;
    ExpertCache cache(cfg);

    if (gguf) slab.resize(size_t(cache.n_slot()) * eb);

    // PINNED is built from calibration statistics; the first 10% of the trace
    // is treated as the calibration corpus.
    if (pin > 0) {
        std::vector<uint32_t> freq(size_t(tr.n_moe) * n_expert, 0);
        const uint64_t cal = std::max<uint64_t>(1, tr.n_token / 10);
        for (uint64_t t = 0; t < cal; ++t)
            for (uint32_t l = 0; l < tr.n_moe; ++l)
                for (uint32_t k = 0; k < tr.top_k; ++k)
                    freq[size_t(l) * n_expert + tr.ids[(t * tr.n_moe + l) * tr.top_k + k]]++;
        cache.build_pinned(freq);
        printf("  PINNED : built from %llu calibration tokens\n", (unsigned long long)cal);
    }

    // ---- replay ----
    printf("\n--- running the replay ---\n");
    const double t0 = now_s();
    uint64_t io_calls = 0;
    double io_time = 0;

    for (uint64_t t = 0; t < tr.n_token; ++t) {
        for (uint32_t l = 0; l < tr.n_moe; ++l) {
            const uint16_t* row = &tr.ids[(t * tr.n_moe + l) * tr.top_k];
            uint32_t slots[32];
            for (uint32_t k = 0; k < tr.top_k; ++k) {
                const uint32_t e = row[k];
                uint32_t s = 0;
                const bool hit = cache.acquire(l, e, Origin::Demand, &s);
                slots[k] = s;
                if (!hit && s != 0xFFFFFFFFu) {
                    if (fd >= 0) {
                        // Real I/O: read the gate/up/down regions, as a direct
                        // GGUF read would
                        const double it0 = now_s();
                        uint8_t* dst = slab.data() + size_t(s) * eb;
                        const RoleInfo* rs[3] = { &idx[l].gate, &idx[l].up, &idx[l].down };
                        size_t o = 0;
                        for (int j = 0; j < 3; ++j) {
                            const off_t off = off_t(rs[j]->off + uint64_t(e) * rs[j]->expert_bytes);
                            size_t left = rs[j]->expert_bytes;
                            while (left) {
                                ssize_t r = pread(fd, dst + o, left, off + (rs[j]->expert_bytes - left));
                                if (r <= 0) break;
                                o += size_t(r); left -= size_t(r);
                            }
                            io_calls++;
                        }
                        io_time += now_s() - it0;
                    }
                    cache.mark_resident(l, e, eb);
                }
            }
            for (uint32_t k = 0; k < tr.top_k; ++k) cache.release(l, row[k]);
            (void)slots;
        }
        cache.end_step();
        if ((t & 0x3FF) == 0) {
            fprintf(stderr, "\r  %llu / %llu tokens  hit=%.1f%%",
                    (unsigned long long)t, (unsigned long long)tr.n_token,
                    cache.stats().hit_rate() * 100);
        }
    }
    const double dt = now_s() - t0;
    fprintf(stderr, "\r%60s\r", "");

    // ---- results ----
    const auto& s = cache.stats();
    printf("  hit rate        : %.2f%%  (hit %llu / miss %llu)\n",
           s.hit_rate() * 100, (unsigned long long)s.hit, (unsigned long long)s.miss);
    printf("    breakdown     : PINNED %.1f%%  SMALL %.1f%%  MAIN %.1f%%\n",
           100.0 * s.hit_pinned / std::max<uint64_t>(s.hit, 1),
           100.0 * s.hit_small  / std::max<uint64_t>(s.hit, 1),
           100.0 * s.hit_main   / std::max<uint64_t>(s.hit, 1));
    printf("  ghost hits      : %llu  (would have hit with a slightly larger cache)\n", (unsigned long long)s.ghost_hit);
    printf("  promote / evict : %llu / %llu\n",
           (unsigned long long)s.promote, (unsigned long long)s.evict);
    printf("  layer quotas    : %s  (result of the §12.5 adaptation)\n", cache.quota_summary().c_str());

    const double bytes_per_tok = double(s.bytes_fetched) / double(tr.n_token);
    printf("\n  bytes/token     : %.2f MiB  (%.1f%% of the %.1f MiB fully resident case)\n",
           bytes_per_tok / 1048576.0,
           double(tr.n_moe) * tr.top_k * eb / 1048576.0,
           100.0 * bytes_per_tok / (double(tr.n_moe) * tr.top_k * eb));

    if (fd >= 0) {
        printf("  real I/O        : %llu preads, %.3f s total (%.2f GB/s)\n",
               (unsigned long long)io_calls, io_time,
               double(s.bytes_fetched) / io_time / 1e9);
        const double io_per_tok_ms = io_time / double(tr.n_token) * 1000.0;
        printf("  I/O time/token  : %.2f ms\n", io_per_tok_ms);
        const double t_c = 43.7;    // measured with llama-server (Finding S1)
        printf("\n  --- projected performance (t_c = %.1f ms/token, I/O fully hidden) ---\n", t_c);
        printf("    t_step = max(t_c, t_io) = max(%.1f, %.2f) = %.2f ms\n",
               t_c, io_per_tok_ms, std::max(t_c, io_per_tok_ms));
        printf("    -> %.1f tok/s  (%.0f%% of the %.1f tok/s baseline)\n",
               1000.0 / std::max(t_c, io_per_tok_ms), 1000.0 / t_c,
               100.0 * t_c / std::max(t_c, io_per_tok_ms));
    }
    printf("\n  replay took     : %.2f s\n", dt);
    if (fd >= 0) close(fd);
    return 0;
}
