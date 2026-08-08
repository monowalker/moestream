// =============================================================================
// Unit tests for ExpertCache (S3-FIFO)
//
//   No GPU, no model; pure logic only.
//   There are three invariants to protect here. Breaking any of them shows up
//   in production as "it looks like it works but the output is broken", which
//   is noticed late.
//
//     1. residents never exceed the seat count -> otherwise it reads past the slab
//     2. nothing in use is evicted             -> otherwise a live seat is overwritten
//     3. eviction always succeeds              -> otherwise the seats starve and it stalls
//
//   Number 3 was actually hit. Without a ceiling on S3-FIFO's freq, second
//   chances are never consumed and eviction spins forever (the origin of
//   FREQ_MAX).
// =============================================================================
#include "../../src/expert_cache.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>

using namespace moestream;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char * what) {
    if (ok) { g_pass++; printf("  ok   %s\n", what); }
    else    { g_fail++; printf("  FAIL %s\n", what); }
}

// One reference (acquire -> make resident -> release)
static bool touch(ExpertCache & c, uint32_t l, uint32_t e, Origin o = Origin::Demand) {
    uint32_t slot = 0;
    const bool hit = c.acquire(l, e, o, &slot);
    if (!hit) c.mark_resident(l, e, 1024);
    c.release(l, e);
    return hit;
}

int main() {
    printf("ExpertCache (S3-FIFO)\n");

    // ---- a cold cache always misses; the second time hits -------------------
    {
        ExpertCache::Config cfg; cfg.n_layer = 2; cfg.n_expert = 32; cfg.n_slot = 16;
        ExpertCache c(cfg);
        check(!touch(c, 0, 5), "the first reference misses");
        check( touch(c, 0, 5), "re-referencing the same one hits");
        check(!touch(c, 1, 5), "a different layer is a different object");
    }

    // ---- residency never exceeds the seat count ----------------------------
    {
        ExpertCache::Config cfg; cfg.n_layer = 1; cfg.n_expert = 256; cfg.n_slot = 32;
        ExpertCache c(cfg);
        std::set<uint32_t> slots;
        bool in_range = true;
        for (uint32_t e = 0; e < 256; ++e) {
            uint32_t slot = 0;
            if (!c.acquire(0, e, Origin::Demand, &slot)) c.mark_resident(0, e, 1024);
            if (slot >= cfg.n_slot) in_range = false;
            slots.insert(slot);
            c.release(0, e);
            c.end_step();
        }
        check(in_range, "assigned seat indices stay within the seat count");
        check(slots.size() <= cfg.n_slot, "total seats used never exceeds the seat count");
    }

    // ---- nothing in use is evicted -----------------------------------------
    {
        ExpertCache::Config cfg; cfg.n_layer = 1; cfg.n_expert = 64; cfg.n_slot = 8;
        ExpertCache c(cfg);
        uint32_t held = 0;
        c.acquire(0, 0, Origin::Demand, &held);
        c.mark_resident(0, 0, 1024);           // no release = still referenced
        bool stolen = false;
        for (uint32_t e = 1; e < 64; ++e) {
            uint32_t slot = 0;
            if (!c.acquire(0, e, Origin::Demand, &slot)) c.mark_resident(0, e, 1024);
            if (slot == held) stolen = true;   // a referenced seat was stolen
            c.release(0, e);
            c.end_step();
        }
        c.release(0, 0);
        check(!stolen, "a seat in use (refcount>0) is never evicted");
    }

    // ---- eviction does not spin forever (what FREQ_MAX is for) --------------
    //   Reference the same few repeatedly to raise freq, then push through more
    //   than the seat count. Without a ceiling on freq, second chances never run
    //   out and the seats starve right here.
    {
        ExpertCache::Config cfg; cfg.n_layer = 1; cfg.n_expert = 128; cfg.n_slot = 16;
        ExpertCache c(cfg);
        for (int rep = 0; rep < 50; ++rep)
            for (uint32_t e = 0; e < 16; ++e) { touch(c, 0, e); c.end_step(); }
        bool ok = true;
        for (uint32_t e = 16; e < 128; ++e) {
            uint32_t slot = 0;
            if (!c.acquire(0, e, Origin::Demand, &slot)) c.mark_resident(0, e, 1024);
            if (slot >= cfg.n_slot) ok = false;   // the seats starved
            c.release(0, e);
            c.end_step();
        }
        check(ok, "eviction still succeeds after freq is raised; seats do not starve");
    }

    // ---- Sweep does not occupy seats (§20.2) --------------------------------
    {
        ExpertCache::Config cfg; cfg.n_layer = 1; cfg.n_expert = 64; cfg.n_slot = 8;
        ExpertCache c(cfg);
        for (uint32_t e = 0; e < 64; ++e) {
            uint32_t slot = 0;
            c.acquire(0, e, Origin::Sweep, &slot);
            c.release(0, e);
        }
        check(c.stats().bypass > 0, "requests originating from Sweep are not admitted");
    }

    // ---- statistics are consistent ------------------------------------------
    {
        ExpertCache::Config cfg; cfg.n_layer = 4; cfg.n_expert = 64; cfg.n_slot = 32;
        ExpertCache c(cfg);
        std::mt19937 rng(1234);
        std::uniform_int_distribution<uint32_t> pick(0, 63);
        uint64_t n = 0;
        for (int step = 0; step < 20000; ++step) {
            for (uint32_t l = 0; l < 4; ++l) { touch(c, l, pick(rng)); n++; }
            c.end_step();
        }
        const auto & st = c.stats();
        check(st.hit + st.miss == n, "hit + miss equals the request count");
        check(st.hit_rate() > 0.0 && st.hit_rate() < 1.0, "the hit rate lies between 0 and 1");
        check(st.hit == st.hit_pinned + st.hit_small + st.hit_main,
              "the hit breakdown (pinned/small/main) sums to hit");
    }

    printf("  ---- %d passed / %d failed\n", g_pass, g_fail);
    return g_fail > 0;
}
