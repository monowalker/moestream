#include "expert_cache.hpp"
#include <cstdint>

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cassert>

namespace moestream {

ExpertCache::ExpertCache(const Config& cfg) : cfg_(cfg) {
    const uint32_t total = cfg_.n_layer * cfg_.n_expert;
    n_slot_ = cfg_.n_slot ? cfg_.n_slot
                          : uint32_t(double(total) * cfg_.cache_frac + 0.5);

    ent_.assign(total, Entry{});
    slot_.assign(total, NO_SLOT);
    lay_.assign(cfg_.n_layer, LayerState{});

    // Per-layer quotas start out even; §12.5 adapts them at run time.
    quota_.assign(cfg_.n_layer, n_slot_ / cfg_.n_layer);
    // Hand the remainder to the leading layers
    for (uint32_t r = n_slot_ - quota_[0] * cfg_.n_layer, l = 0; r > 0; --r, ++l)
        quota_[l % cfg_.n_layer]++;

    free_.resize(n_slot_);
    std::iota(free_.begin(), free_.end(), 0u);
    std::reverse(free_.begin(), free_.end());   // hand out slot 0 first
}

// ---------------------------------------------------------------------------
void ExpertCache::build_pinned(const std::vector<uint32_t>& freq) {
    if (cfg_.pin_ratio <= 0.0 || freq.size() != ent_.size()) return;

    // Pin the top pin_ratio * n_expert experts per layer by frequency.
    // §12.6, measured: a static hot set loses to dynamic LRU, so keep the
    // ratio small (default 0.05).
    const uint32_t per_layer = std::max<uint32_t>(
        1, uint32_t(cfg_.n_expert * cfg_.pin_ratio + 0.5));

    std::vector<uint32_t> order(cfg_.n_expert);
    for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
        std::iota(order.begin(), order.end(), 0u);
        const uint32_t base = l * cfg_.n_expert;
        std::partial_sort(order.begin(), order.begin() + per_layer, order.end(),
                          [&](uint32_t a, uint32_t b) { return freq[base + a] > freq[base + b]; });
        for (uint32_t i = 0; i < per_layer && !free_.empty(); ++i) {
            const uint32_t e = order[i];
            Entry& en = ent_[base + e];
            if (en.seg != Seg::None) continue;
            en.slot = free_.back(); free_.pop_back();
            slot_[base + e] = en.slot;
            en.seg = Seg::Pinned;
            en.resident = true;          // assumed loaded during warm-up
            lay_[l].pinned++;
            lay_[l].resident++;
        }
    }
}

// ---------------------------------------------------------------------------
bool ExpertCache::acquire(uint32_t layer, uint32_t expert, Origin origin,
                          uint32_t* out_slot) {
    const size_t i = idx(layer, expert);
    Entry& en = ent_[i];
    LayerState& ls = lay_[layer];

    en.last = step_;

    if (en.seg != Seg::None) {
        // ---- hit ----
        st_.hit++;
        // Cap freq at 3, as standard S3-FIFO does.
        //   Without a cap, a hot expert's freq grows into the hundreds, the
        //   second chances in evict_one are never used up, and eviction fails.
        if (en.freq < FREQ_MAX) en.freq++;
        en.rc++;
        switch (en.seg) {
            case Seg::Pinned: st_.hit_pinned++; break;
            case Seg::Small:
                st_.hit_small++;
                // S3-FIFO: a second reference while in SMALL promotes to MAIN
                if (en.freq >= 2) {
                    auto it = std::find(ls.small.begin(), ls.small.end(), expert);
                    if (it != ls.small.end()) {
                        ls.small.erase(it);
                        ls.main.push_back(expert);
                        en.seg = Seg::Main;
                        st_.promote++;
                    }
                }
                break;
            case Seg::Main: st_.hit_main++; break;
            default: break;
        }
        if (out_slot) *out_slot = en.slot;
        return true;
    }

    // ---- miss ----
    st_.miss++;
    en.freq = 1;

    // Presence in the ghost list means it would have hit with a bigger cache (§12.5)
    bool was_ghost = false;
    {
        auto it = std::find(ls.ghost.begin(), ls.ghost.end(), expert);
        if (it != ls.ghost.end()) {
            ls.ghost.erase(it);
            ls.ghost_hits++;
            st_.ghost_hit++;
            was_ghost = true;
        }
    }

    // Admission decision (§12.4)
    if (origin == Origin::Sweep) {
        st_.bypass++;                 // a prefill sweep must not pollute the cache
        if (out_slot) *out_slot = NO_SLOT;
        return false;
    }

    const uint32_t s = alloc_slot(layer);
    if (s == NO_SLOT) {               // everything has refcount>0; nothing evictable
        if (out_slot) *out_slot = NO_SLOT;
        return false;
    }

    en.slot = s;
    slot_[i] = s;
    en.rc = 1;
    en.resident = false;
    st_.admit++;

    // Following §12.4:
    //   ghost hit           -> proven useful, so go straight to MAIN (§12.3)
    //   demand / high-conf  -> tail of SMALL
    //   low-confidence PF   -> head of SMALL, i.e. first to be evicted
    if (was_ghost) {
        ls.main.push_back(expert);
        en.seg = Seg::Main;
    } else if (origin == Origin::PrefetchLow) {
        ls.small.push_front(expert);  // the side that gets evicted first
        en.seg = Seg::Small;
    } else {
        ls.small.push_back(expert);
        en.seg = Seg::Small;
    }
    ls.resident++;

    if (out_slot) *out_slot = s;
    return false;
}

void ExpertCache::mark_resident(uint32_t layer, uint32_t expert, uint64_t nbytes) {
    Entry& en = ent_[idx(layer, expert)];
    if (!en.resident) {
        en.resident = true;
        st_.bytes_fetched += nbytes;
    }
}

void ExpertCache::release(uint32_t layer, uint32_t expert) {
    Entry& en = ent_[idx(layer, expert)];
    if (en.rc) en.rc--;
}

// ---------------------------------------------------------------------------
uint32_t ExpertCache::alloc_slot(uint32_t layer) {
    LayerState& ls = lay_[layer];
    // If the layer is over quota, evict from that layer first
    if (ls.resident >= quota_[layer]) {
        if (!evict_one(layer)) return NO_SLOT;
    }
    if (!free_.empty()) {
        const uint32_t s = free_.back(); free_.pop_back();
        return s;
    }
    // Nothing free globally: take from the layer furthest over its quota
    // INT32_MIN as the sentinel. `-1 << 30` is a left shift of a negative value,
    // which is undefined behaviour: it happens to produce the intended constant
    // on gcc and clang, but nothing guarantees that.
    uint32_t victim_layer = layer; int32_t worst = INT32_MIN;
    for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
        const int32_t over = int32_t(lay_[l].resident) - int32_t(quota_[l]);
        if (over > worst && lay_[l].resident > lay_[l].pinned) { worst = over; victim_layer = l; }
    }
    if (!evict_one(victim_layer)) return NO_SLOT;
    if (free_.empty()) return NO_SLOT;
    const uint32_t s = free_.back(); free_.pop_back();
    return s;
}

bool ExpertCache::evict_one(uint32_t layer) {
    LayerState& ls = lay_[layer];
    // pass 0: head of SMALL, to drop one-shot entries quickly
    // pass 1: MAIN, honouring second chances
    // pass 2: MAIN, ignoring second chances -- by now one must be evicted
    for (int pass = 0; pass < 3; ++pass) {
        auto& q = (pass == 0) ? ls.small : ls.main;
        for (size_t n = q.size(); n > 0; --n) {
            const uint32_t e = q.front();
            q.pop_front();
            Entry& en = ent_[idx(layer, e)];
            if (en.rc > 0) { q.push_back(e); continue; }   // in use; cannot evict (§11.2)
            // In MAIN, freq>1 earns exactly one reinsertion (second chance)
            if (pass == 1 && en.freq > 1) { en.freq--; q.push_back(e); continue; }
            // pass 2 does not grant second chances

            free_.push_back(en.slot);
            slot_[idx(layer, e)] = NO_SLOT;
            en.slot = NO_SLOT; en.seg = Seg::None; en.resident = false; en.freq = 0;
            ls.resident--;
            st_.evict++;

            // Record in the ghost list, which drives the quota adjustment (§12.5)
            ls.ghost.push_back(e);
            const size_t gmax = size_t(quota_[layer]) * cfg_.ghost_mult;
            while (ls.ghost.size() > gmax) ls.ghost.pop_front();
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
void ExpertCache::end_step() {
    step_++;
    if (cfg_.layer_quota_adapt && step_ % cfg_.quota_period == 0) rebalance_quota();
}

// §12.5: more ghost hits means more near-misses, i.e. higher marginal utility
void ExpertCache::rebalance_quota() {
    double mean = 0;
    for (auto& l : lay_) mean += l.ghost_hits;
    mean /= double(cfg_.n_layer);
    if (mean <= 0) { for (auto& l : lay_) l.ghost_hits = 0; return; }

    std::vector<double> delta(cfg_.n_layer, 0.0);
    for (uint32_t l = 0; l < cfg_.n_layer; ++l)
        delta[l] = cfg_.quota_eta * (double(lay_[l].ghost_hits) - mean) / mean
                   * double(quota_[l]);

    // Apply while preserving the total
    std::vector<int64_t> nq(cfg_.n_layer);
    int64_t sum = 0;
    for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
        int64_t v = int64_t(quota_[l]) + int64_t(delta[l] + (delta[l] >= 0 ? 0.5 : -0.5));
        v = std::max<int64_t>(v, cfg_.quota_min);
        v = std::min<int64_t>(v, cfg_.n_expert);
        nq[l] = v; sum += v;
    }
    // Normalize so the total never exceeds the slot count
    const double scale = double(n_slot_) / double(sum);
    sum = 0;
    for (uint32_t l = 0; l < cfg_.n_layer; ++l) {
        nq[l] = std::max<int64_t>(cfg_.quota_min, int64_t(nq[l] * scale));
        sum += nq[l];
    }
    for (uint32_t l = 0; sum < int64_t(n_slot_); ++l) { nq[l % cfg_.n_layer]++; sum++; }
    for (uint32_t l = 0; l < cfg_.n_layer; ++l) quota_[l] = uint32_t(nq[l]);

    for (auto& l : lay_) l.ghost_hits = 0;
}

std::string ExpertCache::quota_summary() const {
    char buf[512];
    const uint32_t q[5] = { quota_[0], quota_[cfg_.n_layer / 4], quota_[cfg_.n_layer / 2],
                            quota_[3 * cfg_.n_layer / 4], quota_[cfg_.n_layer - 1] };
    snprintf(buf, sizeof buf, "L0=%u L%u=%u L%u=%u L%u=%u L%u=%u",
             q[0], cfg_.n_layer / 4, q[1], cfg_.n_layer / 2, q[2],
             3 * cfg_.n_layer / 4, q[3], cfg_.n_layer - 1, q[4]);
    return buf;
}

} // namespace moestream
