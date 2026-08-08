// MoEStream Spike S16 — S3-FIFO vs LFU vs LRU on real traces
//
// Pulsar (NVIDIA, 295B-1T) admits experts into its VRAM hot set by LFU
// "touch-count admission". MoEStream uses S3-FIFO. S3-FIFO was compared against
// LRU when it was chosen (Finding N1/M0-2), but never against LFU.
//
// If LFU wins, it is free: a policy change costs no memory.
//
// The S3-FIFO numbers come from linking the product implementation directly
// (llamapatch/expert_cache.cpp) rather than reimplementing it here, so the
// comparison cannot be skewed by a wrong reimplementation.
//
//   Usage: ./s16_policy_compare <trace> [slots_per_layer ...]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <deque>
#include <algorithm>
#include "expert_cache.hpp"

using moestream::ExpertCache;
using moestream::Origin;

struct Trace {
    uint32_t n_moe = 0, top_k = 0, n_expert = 0;
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
    const size_t per = size_t(t.n_moe) * t.top_k * 2;
    std::vector<uint16_t> buf(per);
    for (uint64_t i = 0; i < t.n_token; ++i) {
        if (fread(buf.data(), sizeof(uint16_t), per, f) != per) break;
        for (uint32_t l = 0; l < t.n_moe; ++l)
            for (uint32_t k = 0; k < t.top_k; ++k)
                t.ids.push_back(buf[size_t(l) * t.top_k * 2 + k]);
    }
    fclose(f);
    t.n_token = t.ids.size() / (size_t(t.n_moe) * t.top_k);
    uint16_t mx = 0;
    for (uint16_t v : t.ids) if (v > mx) mx = v;
    t.n_expert = uint32_t(mx) + 1;
    return t.n_token > 0;
}

// ---- LRU, one independent cache per layer -----------------------------------
static double sim_lru(const Trace& t, uint32_t slots) {
    const uint32_t L = t.n_moe, E = t.n_expert, K = t.top_k;
    std::vector<uint64_t> last(size_t(L) * E, 0);
    std::vector<uint32_t> resident(L, 0);
    uint64_t clock = 0, hit = 0, miss = 0;
    for (uint64_t tok = 0; tok < t.n_token; ++tok)
        for (uint32_t l = 0; l < L; ++l)
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t e = t.ids[(tok * L + l) * K + k];
                if (e >= E) continue;
                uint64_t* row = &last[size_t(l) * E];
                ++clock;
                if (row[e]) { hit++; row[e] = clock; continue; }
                miss++;
                if (resident[l] >= slots) {          // evict least recently used
                    uint32_t victim = 0; uint64_t best = UINT64_MAX;
                    for (uint32_t x = 0; x < E; ++x)
                        if (row[x] && row[x] < best) { best = row[x]; victim = x; }
                    row[victim] = 0; resident[l]--;
                }
                row[e] = clock; resident[l]++;
            }
    return double(hit) / double(hit + miss);
}

// ---- LFU with touch counts, one cache per layer (Pulsar's admission) --------
static double sim_lfu(const Trace& t, uint32_t slots) {
    const uint32_t L = t.n_moe, E = t.n_expert, K = t.top_k;
    std::vector<uint64_t> freq(size_t(L) * E, 0);    // touch count, never reset
    std::vector<uint8_t>  in(size_t(L) * E, 0);
    std::vector<uint32_t> resident(L, 0);
    uint64_t hit = 0, miss = 0;
    for (uint64_t tok = 0; tok < t.n_token; ++tok)
        for (uint32_t l = 0; l < L; ++l)
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t e = t.ids[(tok * L + l) * K + k];
                if (e >= E) continue;
                uint64_t* fr = &freq[size_t(l) * E];
                uint8_t*  iv = &in[size_t(l) * E];
                fr[e]++;
                if (iv[e]) { hit++; continue; }
                miss++;
                if (resident[l] >= slots) {
                    // Admit only if hotter than the coldest resident.
                    uint32_t victim = 0; uint64_t best = UINT64_MAX;
                    for (uint32_t x = 0; x < E; ++x)
                        if (iv[x] && fr[x] < best) { best = fr[x]; victim = x; }
                    if (fr[e] <= best) continue;     // not hot enough: bypass
                    iv[victim] = 0; resident[l]--;
                }
                iv[e] = 1; resident[l]++;
            }
    return double(hit) / double(hit + miss);
}

// ---- S3-FIFO: the product implementation, linked directly -------------------
static double sim_s3fifo(const Trace& t, uint32_t slots) {
    const uint32_t L = t.n_moe, E = t.n_expert, K = t.top_k;
    std::vector<ExpertCache*> c(L, nullptr);
    for (uint32_t l = 0; l < L; ++l) {
        ExpertCache::Config cfg;
        cfg.n_layer = 1; cfg.n_expert = E; cfg.n_slot = slots;
        cfg.pin_ratio = 0.0; cfg.layer_quota_adapt = false;
        c[l] = new ExpertCache(cfg);
    }
    uint64_t hit = 0, miss = 0;
    std::vector<uint32_t> held;
    for (uint64_t tok = 0; tok < t.n_token; ++tok)
        for (uint32_t l = 0; l < L; ++l) {
            held.clear();
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t e = t.ids[(tok * L + l) * K + k];
                if (e >= E) continue;
                uint32_t slot = 0;
                const bool h = c[l]->acquire(0, e, Origin::Demand, &slot);
                if (slot == 0xFFFFFFFFu) { miss++; continue; }
                if (h) hit++; else { miss++; c[l]->mark_resident(0, e, 1); }
                held.push_back(e);
            }
            for (uint32_t e : held) c[l]->release(0, e);   // mirrors the runtime
        }
    for (auto* p : c) delete p;
    return double(hit) / double(hit + miss);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <trace> [slots ...]\n", argv[0]); return 2; }
    Trace t;
    if (!load_trace(argv[1], t)) return 1;
    printf("  trace %s : %llu tokens, %u MoE layers, top-%u, n_expert=%u\n\n",
           argv[1], (unsigned long long) t.n_token, t.n_moe, t.top_k, t.n_expert);

    std::vector<uint32_t> slots;
    for (int i = 2; i < argc; ++i) slots.push_back((uint32_t) atoi(argv[i]));
    if (slots.empty()) { uint32_t d[] = {26, 38, 64, 102, 128}; for (uint32_t v : d) slots.push_back(v); }

    printf("  slots   frac    S3-FIFO       LFU       LRU     LFU-S3\n");
    for (uint32_t s : slots) {
        if (s >= t.n_expert) continue;
        const double a = sim_s3fifo(t, s) * 100;
        const double b = sim_lfu(t, s)    * 100;
        const double d = sim_lru(t, s)    * 100;
        printf("  %5u   %4.0f%%   %7.2f%%  %7.2f%%  %7.2f%%   %+6.2f pt\n",
               s, 100.0 * s / t.n_expert, a, b, d, b - a);
    }
    return 0;
}
