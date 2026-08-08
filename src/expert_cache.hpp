// =============================================================================
// MoEStream — Expert Cache
//
// Implements the following sections of DESIGN.md:
//   §11.1 expert metadata by dense id (no hashing)
//   §11.2 expert state machine / refcount
//   §11.3 Slot Table + Slab
//   §12.3 S3-FIFO (SMALL / MAIN / GHOST) + PINNED
//   §12.4 admission policy differentiated by request origin
//   §12.5 dynamic per-layer quota adjustment driven by ghost lists
//
// Default revised after measurement (Finding M0-2):
//   §12.6 PINNED ratio alpha 0.20 -> 0.05, because a static hot set was shown
//         to lose to dynamic LRU
//
// Runs single-threaded and holds no locks.
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <deque>
#include <string>

namespace moestream {

// Why execution requested this expert; drives the admission decision (§12.4)
enum class Origin : uint8_t {
    Demand,        // execution is blocked on it: always admit
    PrefetchHigh,  // high-confidence prefetch from P0/P1
    PrefetchLow,   // low-confidence prefetch: goes to the tail of SMALL
    Sweep,         // prefill sweep: never admitted (§20.2)
};

enum class Seg : uint8_t { None, Pinned, Small, Main };

struct CacheStats {
    uint64_t hit          = 0;
    uint64_t miss         = 0;
    uint64_t hit_pinned   = 0;
    uint64_t hit_small    = 0;
    uint64_t hit_main     = 0;
    uint64_t ghost_hit    = 0;   // would have hit with a slightly larger cache
    uint64_t admit        = 0;
    uint64_t evict        = 0;
    uint64_t bypass       = 0;   // not admitted (sweep and similar)
    uint64_t promote      = 0;   // SMALL -> MAIN
    uint64_t bytes_fetched = 0;

    double hit_rate() const {
        const uint64_t t = hit + miss;
        return t ? double(hit) / double(t) : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Expert Cache
//   Each layer has its own segments; per-layer quotas adapt at run time (§12.5).
// ---------------------------------------------------------------------------
class ExpertCache {
public:
    struct Config {
        uint32_t n_layer      = 40;
        uint32_t n_expert     = 256;
        uint32_t n_slot       = 0;      // total slots; 0 derives it from cache_frac
        double   cache_frac   = 0.25;   // fraction of experts kept resident
        double   pin_ratio    = 0.05;   // §12.6, lowered from 0.20 by measurement
        double   small_ratio  = 0.10;   // S3-FIFO SMALL fraction
        uint32_t ghost_mult   = 2;      // ghost length = layer quota * this
        bool     layer_quota_adapt = true;   // dynamic adjustment per §12.5
        uint32_t quota_period = 512;    // steps between reallocations
        double   quota_eta    = 0.10;   // reallocation learning rate
        uint32_t quota_min    = 8;      // minimum slots per layer
    };

    explicit ExpertCache(const Config& cfg);

    // Request (layer, expert).
    //   Returns true on a hit, with the slot in out_slot; false on a miss.
    //   A slot is reserved on a miss too, and the caller calls mark_resident()
    //   once the I/O completes.
    bool acquire(uint32_t layer, uint32_t expert, Origin origin,
                 uint32_t* out_slot);

    // Signal I/O completion, validating the slot reserved on the miss
    void mark_resident(uint32_t layer, uint32_t expert, uint64_t nbytes);

    // End of reference (refcount--)
    void release(uint32_t layer, uint32_t expert);

    // End of one step; the trigger for per-layer quota reallocation (§12.5)
    void end_step();

    // Build the PINNED set from calibration statistics (§12.6; the frequency
    // array is passed in directly)
    void build_pinned(const std::vector<uint32_t>& freq_per_layer_expert);

    const CacheStats& stats() const { return st_; }
    uint32_t n_slot() const { return n_slot_; }
    uint32_t slot_of(uint32_t layer, uint32_t expert) const {
        return slot_[idx(layer, expert)];
    }
    std::vector<uint32_t> layer_quota() const { return quota_; }
    std::string quota_summary() const;

private:
    static constexpr uint32_t NO_SLOT  = 0xFFFFFFFFu;
    // S3-FIFO frequency ceiling. Without a cap, second chances are never used
    // up, eviction fails, and the failure surfaces as slot exhaustion.
    static constexpr uint32_t FREQ_MAX = 3;

    struct Entry {
        uint32_t slot  = NO_SLOT;
        uint16_t rc    = 0;        // refcount
        Seg      seg   = Seg::None;
        bool     resident = false; // has the I/O completed?
        uint32_t freq  = 0;        // reference count, used for S3-FIFO promotion
        uint32_t last  = 0;        // step of the last reference
    };

    // Per-layer queues
    struct LayerState {
        std::deque<uint32_t> small;   // expert id
        std::deque<uint32_t> main;
        std::deque<uint32_t> ghost;   // evicted expert ids (no backing data)
        uint32_t             ghost_hits = 0;
        uint32_t             resident   = 0;   // number with backing data (incl. pinned)
        uint32_t             pinned     = 0;
    };

    size_t idx(uint32_t l, uint32_t e) const { return size_t(l) * cfg_.n_expert + e; }

    uint32_t alloc_slot(uint32_t layer);       // evicts if nothing is free
    bool     evict_one(uint32_t layer);        // evict one; true on success
    void     rebalance_quota();

    Config                 cfg_;
    uint32_t               n_slot_ = 0;
    std::vector<Entry>     ent_;     // [n_layer * n_expert]
    std::vector<uint32_t>  slot_;    // same indexing; the Slot Table (§11.3)
    std::vector<LayerState> lay_;
    std::vector<uint32_t>  quota_;   // per-layer slot ceiling
    std::vector<uint32_t>  free_;    // stack of free slots
    CacheStats             st_;
    uint32_t               step_ = 0;
};

} // namespace moestream
