// =============================================================================
// MoEStream — llama.cpp embedded implementation
//
// Only mechanisms whose benefit was confirmed by measurement are kept here.
// Rejected approaches are recorded in
// docs/findings/N2-speed-optimization.md and N3-graph-remap.md.
//
// How it works:
//   1. The loader allocates expert tensors as a reduced [n_embd, n_ff, n_slot]
//      slab (patch in llama-model-loader.cpp)
//   2. A remap node is inserted into the graph, mapping the router's expert_id
//      to a slot_id (patch in llama-graph.cpp + build_remap)
//   3. While the remap runs, non-resident experts are read from SSD in
//      parallel. On UMA the read goes straight into GPU-visible memory
//      (zero-copy)
//
// Measured (Radeon 780M / Crucial P310 / Ornith-1.0-35B-UD-IQ4_NL):
//   baseline               42.7 ms/token
//   remap only (no I/O)    54.1 ms/token   <- the +11.4 ms is CPU<->GPU sync
//   full (with I/O)        54.6 ms/token   <- I/O itself costs only 0.5 ms
//
// Rejected; do not reimplement:
//   P1 previous-token reuse / P2 layer lookahead / P5 cross-layer coactivation
//   resident prefetch thread / eval-callback path
// =============================================================================
#include "llama-moestream.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <algorithm>
#include <functional>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sys/mman.h>
#include <set>
#include <sys/stat.h>
#include <dirent.h>
#include <csignal>

#include "expert_cache.hpp"

// Accessors added by the ggml-vulkan patch. Non-null only on UMA devices.
extern "C" void * ggml_backend_vk_buffer_host_ptr(ggml_backend_buffer_t buffer, size_t offset)
    __attribute__((weak));
extern "C" ggml_backend_buffer_t ggml_backend_vk_buffer_from_host_ptr(
        ggml_backend_buffer_type_t buft, void * ptr, size_t size) __attribute__((weak));
extern "C" size_t ggml_backend_vk_host_ptr_alignment(ggml_backend_buffer_type_t buft)
    __attribute__((weak));
extern "C" ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num) __attribute__((weak));

namespace moestream {

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct SlabInfo {
    ggml_tensor * t = nullptr;
    uint64_t file_offset  = 0;   // start of the tensor within the GGUF
    uint64_t expert_bytes = 0;   // bytes for one expert
    int64_t  n_expert     = 0;
    uint64_t arena_off    = 0;   // offset inside the prefill arena (prefill path only)
    uint16_t file_idx     = 0;   // shard index for split GGUFs (offset is within that file)
};
struct LayerSlabs { SlabInfo gate, up, down; bool ready = false; };

struct ReadJob {
    const SlabInfo * si;
    uint32_t expert;
    uint32_t slot;
    uint8_t * dst;
    bool      direct;            // true = dst is GPU-visible memory (zero-copy)
};

static bool                      g_init     = false;
static bool                      g_enabled  = false;
static double                    g_frac     = 0.38;
static bool                      g_frac_auto = false;  // cache_frac=auto
static bool                      g_frac_learn = false; // cache_frac=learn (persisted tuning)
static double                    g_slab_gib = 9.0;     // slab budget in GiB when frac=auto
static int64_t                   g_n_expert = 0;       // captured for the tuning-state key
static int                       g_nthreads = 8;
// ---- automatic I/O thread-count tuning ----
//   The best value varies a lot per model (measured: 4 is best for Ornith,
//   8 for Laguna, a 17% spread). The balance between parallelism gain and
//   thread-spawn cost shifts with read size and miss rate, so no fixed
//   constant covers both. The thread count is just a loop bound -- it touches
//   neither the cache nor the graph -- so we cycle through candidates at
//   runtime and keep whichever yields the highest effective bandwidth.
static bool     g_io_auto    = true;      // automatic unless MOESTREAM_IO_THREADS is set
// Candidate "1" (do not spawn any thread) was deliberately left out.
//   In Ornith decode measurements 1 thread was fastest (56.29 vs 56.96 ms with
//   4), but the 1.2% gap is within run-to-run noise (+/-1 ms). Worse, in the
//   tuner's own bandwidth measurement 1 thread ranks last (8.26 vs 9.80 GB/s
//   with 6): the two measurements disagree, and bandwidth cannot explain the
//   difference. Adding a candidate on such weak evidence only adds the risk of
//   choosing it wrongly. On Laguna, 1 thread is 2.3x slower (655.4 vs 285.3 ms
//   with 16).
static const int TH_CAND[]   = { 2, 4, 6, 8, 12, 16 };
static const int TH_N        = 6;
static double   g_th_bytes[TH_N] = {0}, g_th_time[TH_N] = {0};
static int      g_th_cur     = 0;
static uint64_t g_th_calls   = 0;
static bool     g_th_settled = false;
static uint32_t                  g_slots    = 0;
static std::string               g_path;
static std::vector<int>          g_fds;                // one fd per shard
static uint64_t                  g_maxb     = 0;   // largest single-member size in bytes
// Diagnostic knob: at or above this layer index, misses are not fetched and a
// zero-filled "null expert" is substituted instead.
//   default = disabled / 20 = second half only / 0 = every layer
//   Quantized blocks carry an fp16 scale at the start (or end) of the block,
//   so an all-zero block decodes to zero and the expert's contribution simply
//   vanishes rather than producing garbage.
static int      g_drop_from = 1 << 30;
// ---- dense FFN streaming knobs (findings S18/S21/S26); see the block below ----
static double   g_dn_frac    = -1.0;  // fraction of layers kept RESIDENT; 1.0 = off, <0 = auto
static bool     g_dn_auto    = true;  // decide the split from the memory budget
static bool     g_dn_learn   = false; // also record the measured cost curve
static int      g_dn_nlayer  = 0;     // total layers, from GGUF metadata
static int      g_dn_first   = -1;    // first streamed layer (-1 = none)
static int      g_dn_nbuf    = 2;
static bool     g_dn_ready   = false;
static bool     g_dn_seen    = false; // a dense FFN tensor was claimed
static int      g_dn_th      = 4;     // dense read threads (finding S29, end-to-end)
static uint32_t g_zero_slot = 0;         // zero-filled slot reserved at the end of the slab
static uint64_t g_dropped   = 0, g_demand = 0;
static bool     g_zero_filled = false;

// Zero-fill of the null-expert slot.
// finalize() can also run during graph construction, at which point the tensor
// may not be allocated yet (GGML_ASSERT "tensor not allocated"), so this is
// deferred to the first remap execution.
static void ensure_zero_slot();
static void mrc_report();
// Highest device-memory reading seen while the model was live (GiB). See the
// sampling site in the [stats] path for why the value at exit cannot be used.
static double g_dev_used_peak = 0;
static void ub_report();
static void pf_wait();
static void prefetch_verify();
static inline void mrc_touch(int il, uint32_t e);
void report();
// Lower bound on slab size: guarantees "a ubatch up to this many tokens fits
// in the slab". If the number of distinct experts in one micro-batch exceeds
// the slot count, every expert in that batch holds a refcount, eviction cannot
// free anything, and slot allocation fails. Silently filling in a wrong slot
// corrupts the output (this was the cause of PPL 4.97 -> 2365), so the case is
// detected, warned about, and routed to the null expert.
// Since the prefill arena was introduced the threshold is derived from the
// slot count and is self-consistent, so this bound only has to guarantee that
// decode (n_tokens=1) works. Leaving it at 16 forced a floor of 169 slots on a
// 512-expert model, which made MOESTREAM_CACHE_FRAC below 0.33 ineffective.
static int      g_max_ubatch = 2;
static uint64_t g_exhausted  = 0;        // number of slot-exhaustion events
static bool     g_warned_ub  = false;
static int      g_seen_topk  = 0;
// Lowest layer index that actually has experts. Models whose first layer is
// dense (Laguna, DeepSeek family, ...) have no experts in layer 0, so gating
// statistics on il==0 would disable them entirely.
static int      g_first_il   = -1;

// ---- deciding whether prefetching is worthwhile ----
//   What actually matters is whether the expert data fits in the page cache.
//   If it fits, misses are served from RAM, I/O is cheap, and prefetching buys
//   nothing. If it does not, misses reach the SSD, I/O dominates, and
//   prefetching pays off.
//     Ornith-35B  : experts 14.5 GiB / RAM 30 GiB -> fits     -> I/O is 1% of decode
//     Laguna-S-2.1: experts 55 GiB   / RAM 30 GiB -> does not -> I/O is 75% of decode
//   The decision is made at model load time. Prefetching needs layer L's
//   hidden state on the CPU, which requires a graph node, and llama.cpp reuses
//   the decode graph -- nodes cannot be added or removed afterwards.
static int      g_prefetch    = -1;    // -1=auto / 0=off / 1=on
static uint64_t g_expert_bytes_total = 0;
static double   g_ram_gib     = 0;
static double   g_pc_gib      = 0;     // memory expected to be available as page cache
// Self-check for that decision: measure the actual I/O share at runtime and
// report whether the decision held up.
static double   g_dec_wall    = 0;     // cumulative decode wall time
static double   g_dec_last    = 0;
static double   g_dec_read    = 0;     // cumulative I/O time during decode
// ---- overlap headroom (decode) ----
//   A layer-lookahead prefetch could only hide I/O if there is compute to hide
//   it behind. The window is the gap between one layer's remap finishing and
//   the next layer's remap starting -- that is the compute a background read
//   would run underneath. Measuring it says whether hiding is possible at all,
//   before any prefetch machinery is written.
static double   g_lay_prev_exit = 0;   // when the previous layer's remap returned
static double   g_lay_window    = 0;   // cumulative gap between consecutive remaps
static double   g_lay_read      = 0;   // cumulative read time inside remaps
static uint64_t g_lay_n         = 0;
static double   g_lay_win_max   = 0;
static double   g_lay_read_max  = 0;
static uint64_t g_lay_starved   = 0;   // layers whose read exceeded the window

// ---- the P2 predictor was implemented and then removed (Finding S11) ----
//   The fadvise variant measured 19% worse, and even with free predictions the
//   headroom was only 3.6%. Writing into the slab from a private thread offered
//   at most 1.45x while taking on three "fails silently" risks: refcount leaks,
//   slot-reallocation races, and torn writes. Only the decision logic
//   (g_prefetch below) is kept, for diagnostics.
static std::map<int, LayerSlabs> g_layers;
static std::vector<ExpertCache*> g_cache;           // one per layer, independent
static std::map<int, std::vector<uint32_t>> g_held; // experts currently held, per layer
static std::vector<uint8_t>      g_stage;           // staging buffer when zero-copy is unavailable

// ---- prefill path (staging arena + layer rotation) ----
//   Allocate only N arenas holding one layer's worth (gate+up+down) and reuse
//   them across layers. The resident slab used by decode never grows.
//   Filling the arena is a CPU custom op, like the remap, and runs in graph
//   order immediately before that layer's mul_mat_id.
//
//   History; do not reimplement:
//     - mmap the GGUF and import it  -> amdgpu rejects file-backed VMAs (S5)
//     - import anonymous memory      -> works, but merely having the BO exist
//                                       drops decode from 53.8 to 1023 ms/token (S7)
//     - let ggml-vulkan allocate it  -> adopted; same mechanism as the
//                                       zero-copy path
static int                       g_pf_nbuf  = 2;         // arena count (2 = async prefetch)
static std::vector<void *>       g_arena;                // GPU-visible host pointers
static std::vector<ggml_backend_buffer_t> g_arena_buf;   // backing buffers
static std::vector<int>          g_arena_layer;          // layer resident in each arena (-1 = none)
// Because only the union is read, track which experts are actually present.
// The layer index alone is insufficient: the same layer can be loaded with
// different subsets.
static std::vector<std::vector<uint8_t>> g_arena_have;
static std::vector<uint8_t>      g_need;                 // scratch: experts needed this pass
static uint64_t g_pf_experts_read = 0, g_pf_experts_asked = 0;

// ---- asynchronous arena prefetch ----
//   Reading layer L+1 while the GPU computes layer L overlaps I/O with compute.
//   Measured per pass (the [ub] figures in Finding S7):
//     Ornith : I/O 0.97 s / compute 4.08 s   -> serial 5.05 -> overlapped 4.08 (-19%)
//     Laguna : I/O 13.37 s / compute 11.49 s -> serial 24.86 -> overlapped 13.37 (-46%)
//
//   Unlike the rejected prefetcher (S11) this never touches ExpertCache. The
//   arena is a plain buffer with no refcounts and no eviction, so the only
//   synchronization needed is waiting for completion.
//   No prediction is needed either: at large ubatch the union is close to the
//   full expert set, so reading everything without waiting for ids is not waste.
//   There is never more than one thread; state is handed over across join().
static bool             g_pf_async   = true;
static std::thread      g_pf_thread;
static int              g_pf_inflight = -1;      // layer currently being loaded (-1 = none)
static uint64_t         g_pf_async_hit = 0, g_pf_async_miss = 0;

// ---- measurement only: does the previous pass's union predict the next one? ----
//   Async prefetch cannot wait for ids, so it cannot know the union. If "the
//   union of the same layer in the previous pass" were a good approximation,
//   it could be prefetched and the shortfall filled in synchronously.
//   Behaviour is unchanged; these counters only observe:
//     need     : how many were needed this time
//     prev     : size of the previous union (the prefetch candidate set)
//     shortage : needed now but absent from the previous union (synchronous fill)
//     total    : prev union need (what would actually be read)
static std::vector<std::vector<uint8_t>> g_prev_union;   // [layer][expert]
static uint64_t g_u_need = 0, g_u_prev = 0, g_u_short = 0, g_u_total = 0, g_u_n = 0;
static size_t                    g_arena_bytes = 0;      // bytes per arena
static ggml_context *            g_pf_ctx   = nullptr;
static std::map<int, LayerSlabs> g_full;                 // full-size tensors
static int                       g_pf_threshold = -1;    // above this, use the prefill path (-1 = auto)
static bool                      g_pf_ready = false;
static uint64_t                  g_pf_used  = 0, g_slab_used = 0;
static uint64_t                  g_pf_loads = 0, g_pf_hits = 0, g_pf_bytes = 0;
static double                    g_t_pfload = 0;

// ---- UBATCH cost accounting ----
//   The arena re-reads every expert once per pass, so:
//     time per pass = I (I/O, a fixed cost independent of ubatch)
//                   + C (compute, which grows with ubatch)
//   Raising ubatch cuts the pass count and amortizes I.
//
//   ★ C is NOT proportional to ubatch. Attention is quadratic in the tokens
//     within a micro-batch, so per-token compute rises as ubatch grows
//     (measured 1.60 -> 4.07 ms/token from ubatch 512 -> 4096).
//     There is therefore a real optimum rather than an asymptote.
//     A single run only observes C at one ubatch, which is not enough to fit
//     the growth exponent, so this instrumentation reports the split and does
//     not extrapolate. See RESULTS.md §10.11 for what happened when it did.
static int      g_pf_ntok_seen = 0;   // n_tokens observed by prefill_exps
static int      g_pf_first_il  = -1;  // layer index treated as the start of a pass
static double   g_pass_t0 = 0, g_pass_io = 0;
static int      g_pass_ntok = 0;
static double   g_ub_io = 0, g_ub_cmp = 0;
static uint64_t g_ub_passes = 0;
static int      g_ub_ntok = 0;        // ubatch being aggregated (the max, i.e. non-partial)
// Request-level prefill accounting.
//   A pass-level rate cannot rank UBATCH. Excluding the warm-up pass and the
//   partial tail is necessary for a pass to mean anything, but the share of the
//   prompt that falls into them grows with the ubatch: measured on Qwen3-Coder
//   with a 22828-token prompt, the pass rate ran +2.5% above llama.cpp's at
//   ub=1024 and +14.4% at ub=8192, and that drift flipped the winner (llama.cpp
//   makes 4096 fastest, the pass rate picked 8192).
//   A request does not have that problem: its definition does not move with the
//   ubatch, and it is the thing the user actually waits on. Measured from the
//   first prefill pass to the moment decode starts -- the same quantity
//   llama.cpp reports as "prompt eval".
static double   g_req_t0  = 0;    // start of the current request's prefill
static uint64_t g_req_tok = 0;    // prompt tokens seen so far in it
// Aggregate prefill rate: every pass after the first, including the partial
// final one.
//   u0/(I+C) describes one steady-state pass and ignores the work wasted in a
//   partial pass, which is exactly what a larger UBATCH incurs. Measured that
//   way the candidates looked 1.4% apart when they are really 16% apart, so the
//   learned choice would have been close to random. Sum tokens and time instead.
//
//   The first pass of a process is 2.0-2.3% slower than the rest (measured over
//   six runs), so it is dropped. Run-to-run variation is only 0.3-1.0%, well
//   below the 12.4% that separates candidates -- a single run per candidate is
//   therefore enough, provided no other measurement is running at the same time.
//   The 11% error that once picked the wrong candidate came from two concurrent
//   runs fighting over the GPU, not from a cold cache; research/tools/exclusive.sh is
//   what prevents that.
static double   g_pf_time_total = 0;
static uint64_t g_pf_tok_total  = 0;
static uint64_t g_pf_intervals  = 0;
static double   g_pf_first_time = 0;   // the warm-up interval, excluded when possible
static uint64_t g_pf_first_tok  = 0;

// ---- measured MRC (Miss Ratio Curve) ----
//   Accumulating reuse distances (stack distances) yields the hit rate for
//   every possible slot count from a single run -- no need to sweep the slot
//   count and restart.
//     hit rate(C slots) = P(reuse distance < C)      ... exact for LRU
//   The real cache is S3-FIFO, so this is strictly an approximation, but
//   Finding N1 confirmed that LRU tracks the real behaviour closely.
static bool     g_mrc_on = true;
static std::vector<std::vector<uint64_t>> g_mrc_last;  // [layer][expert] last access time
static std::vector<uint64_t> g_mrc_clock;              // [layer] logical clock
static std::vector<uint64_t> g_mrc_hist;               // distance histogram (all layers)
static std::vector<std::vector<uint64_t>> g_mrc_hist_l;// per-layer distance histogram
static std::vector<uint64_t> g_mrc_tot_l;              // per-layer access count
static uint64_t g_mrc_total = 0, g_mrc_cold = 0;
// How many hit-rate points we are willing to lose per GiB saved.
// Converting to time did not match measurements because such a formula cannot
// account for request overlap, so the decision is made on hit rate, a quantity
// that is directly measurable (the measured cliff sits around 2-3 pt/GiB).
static double   g_pt_per_gib = 2.5;
// SIGUSR1 triggers an on-demand report. fprintf is not async-signal-safe, so
// the handler only raises a flag and the report is emitted at the next token
// boundary.
static volatile sig_atomic_t g_mrc_req = 0;
static void mrc_sighandler(int) { g_mrc_req = 1; }

// Instrumentation
static uint64_t g_tokens = 0, g_io_bytes = 0, g_io_calls = 0, g_zerocopy = 0;
static double   g_t_read = 0, g_t_upload = 0, g_t_total = 0;

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Diagnostics: 0=normal / 1=pass the remap through / 3=cache lookup only (no I/O)
//   Bisecting with these three levels is what revealed that synchronization,
//   not I/O, is the bottleneck.
static int noop_level() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("MOESTREAM_NOOP"); v = e ? atoi(e) : 0; }
    return v;
}
static int zerocopy_mode() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("MOESTREAM_ZEROCOPY"); v = e ? atoi(e) : 1; }
    return v;
}
// Startup self-checks stay silent when they pass. Set MOESTREAM_DEBUG=1 to see
// them regardless; failures are always reported.
static bool debug_mode() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("MOESTREAM_DEBUG"); v = e && atoi(e) ? 1 : 0; }
    return v != 0;
}

bool enabled() {
    if (!g_init) {
        g_init = true;
        const char * e = getenv("MOESTREAM");
        g_enabled = e && atoi(e) != 0;
        if (const char * f = getenv("MOESTREAM_CACHE_FRAC")) {
            if      (strcmp(f, "auto")  == 0) g_frac_auto  = true;
            else if (strcmp(f, "learn") == 0) g_frac_learn = true;
            else                              g_frac = atof(f);
        }
        if (const char * b = getenv("MOESTREAM_SLAB_GIB")) g_slab_gib = atof(b);
        if (const char * m = getenv("MOESTREAM_MRC")) g_mrc_on = atoi(m) != 0;
        if (const char * m = getenv("MOESTREAM_PT_PER_GIB")) g_pt_per_gib = atof(m);
        if (const char * f = getenv("MOESTREAM_PREFETCH")) g_prefetch = atoi(f) ? 1 : 0;
        if (const char * t = getenv("MOESTREAM_IO_THREADS")) {
            g_nthreads = std::max(1, atoi(t)); g_io_auto = false;   // explicit value pins it
            g_dn_th = g_nthreads;
        }
        if (const char * d = getenv("MOESTREAM_DROP_FROM")) g_drop_from = atoi(d);
        if (const char * d = getenv("MOESTREAM_DENSE_FRAC")) {
            if      (strcmp(d, "auto")  == 0) { g_dn_auto = true;  g_dn_learn = false; }
            else if (strcmp(d, "learn") == 0) { g_dn_auto = true;  g_dn_learn = true;  }
            else if (strcmp(d, "off")   == 0) { g_dn_auto = false; g_dn_frac = 1.0;    }
            else {
                g_dn_auto = false;
                g_dn_frac = atof(d);
                if (g_dn_frac < 0.0) g_dn_frac = 0.0;
                if (g_dn_frac > 1.0) g_dn_frac = 1.0;
            }
        }
        if (const char * b = getenv("MOESTREAM_DENSE_BUFS")) g_dn_nbuf = std::max(1, atoi(b));
        if (const char * u = getenv("MOESTREAM_MAX_UBATCH")) g_max_ubatch = std::max(1, atoi(u));
        if (const char * t = getenv("MOESTREAM_PREFILL_THRESHOLD")) g_pf_threshold = atoi(t);
        if (g_enabled) {
            if      (g_frac_auto)  fprintf(stderr, "moestream: enabled (cache_frac=auto, slab %.1f GiB)\n", g_slab_gib);
            else if (g_frac_learn) fprintf(stderr, "moestream: enabled (cache_frac=learn)\n");
            else                   fprintf(stderr, "moestream: enabled (cache_frac=%.2f)\n", g_frac);
        }
    }
    return g_enabled;
}
double cache_frac() { enabled(); return g_frac; }
void   set_gguf_path(const char * p) { g_path = p ? p : ""; }

// ---------------------------------------------------------------------------
// Loader-side hooks
// ---------------------------------------------------------------------------
// Parse "blk.<il>.ffn_<role>_exps.weight"
static bool parse_name(const char * name, int & il, int & role) {
    if (strncmp(name, "blk.", 4) != 0) return false;
    char * end = nullptr;
    long v = strtol(name + 4, &end, 10);
    if (!end || *end != '.') return false;
    il = (int) v;
    if      (strstr(end, ".ffn_gate_exps.weight")) role = 0;
    else if (strstr(end, ".ffn_up_exps.weight"))   role = 1;
    else if (strstr(end, ".ffn_down_exps.weight")) role = 2;
    else return false;
    return true;
}

// ---------------------------------------------------------------------------
// Dense FFN streaming (findings S18 / S21 / S26)
// ---------------------------------------------------------------------------
//   The slab does not apply here. It works by shrinking ne[2], the expert
//   dimension, so that mul_mat_id runs against a smaller array. A dense FFN
//   weight is 2D and goes through plain mul_mat -- there is no expert
//   dimension to shrink. Only the arena approach transfers.
//
//   So: keep the first K layers resident, and give the rest an arena holding
//   one layer's FFN, filled by a CPU op immediately before that layer runs.
//   The model's own tensors for streamed layers are allocated as stubs, since
//   allocating them at full size would leave the whole model resident and save
//   nothing.
//
//   Stream the TAIL, not the head. Which layers are streamed does not change
//   bytes/token -- a dense model uses every layer exactly once -- but it
//   changes how much can be hidden: with the tail streamed, the forward pass
//   through the resident head is time the reads happen in.
//
//   And here dense beats MoE outright. Every predictive prefetch scheme in this
//   project failed (N2 / S10 / S11 / §10.14) because layer L+1's experts cannot
//   be known until layer L has run. A dense model has nothing to predict:
//   layer 33 is always layer 33. Perfect prefetch at zero prediction cost.
//
//   v1 streams the FFN only (about 60% of a dense model). Attention would need
//   a second graph anchor and is not attempted here.

struct DenseInfo {
    ggml_tensor * t    = nullptr;     // arena-backed tensor used by the graph
    // Offset within the arena buffer. Reads must go through the HOST pointer
    // (g_dn_host[bi] + arena_off); t->data is a device address on Vulkan and
    // pread into it fails silently, leaving the arena full of garbage while
    // everything still runs at full speed. That is exactly how this was first
    // shipped, and RESULTS.md §10.8 is the standing warning about it.
    uint64_t arena_off = 0;
    uint64_t file_offset = 0;
    uint64_t bytes     = 0;
    uint16_t file_idx  = 0;
    int      type      = 0;
    int      role      = -1;
    int64_t  ne0 = 0, ne1 = 0;
    bool     valid     = false;
};
//   Not just the FFN. Every per-layer weight matrix in llama.cpp is consumed by
//   build_lora_mm(), so claiming any large 2D blk.* weight and substituting at
//   that one choke point covers FFN, attention and SSM alike -- without a single
//   architecture-specific branch (ADR-0036). Which is why the hook moved there
//   from build_ffn: the first version could only reach the FFN.
struct DenseLayer { std::vector<DenseInfo> w; bool ready = false; };

static std::map<int, DenseLayer> g_dn_meta;   // file locations, per streamed layer
// original model tensor -> (layer, index within that layer)
static std::map<const ggml_tensor *, std::pair<int,int>> g_dn_bypointer;
static std::set<std::string> g_dn_claimed;    // names actually stubbed
static uint64_t g_dn_min_bytes = 8u << 20;    // ignore anything smaller; norms and
                                              // biases are not worth a graph node
static std::map<int, DenseLayer> g_dn_view;   // arena-backed tensors, per streamed layer
static std::vector<void *>       g_dn_host;
static std::vector<ggml_backend_buffer_t> g_dn_buf;
static std::vector<int>          g_dn_resident;   // which layer each arena holds
static ggml_context *            g_dn_ctx   = nullptr;
static size_t                    g_dn_bytes_per = 0;
static uint64_t g_dn_loads = 0, g_dn_hits = 0, g_dn_read_bytes = 0;
static double   g_dn_t_read = 0;
// async prefetch of the next streamed layer
static std::thread g_dn_thread;
static int         g_dn_inflight = -1;
static uint64_t    g_dn_pf_hit = 0, g_dn_pf_miss = 0;

// Any per-layer weight matrix large enough to be worth streaming.
//   role is just this weight's slot within the layer, assigned in load order --
//   there is no need to know what the weight IS, only where its bytes live and
//   which tensor to substitute. That is what keeps this architecture-agnostic.
//   Expert tensors are excluded: the slab owns those.
static bool parse_dense_any(const char * name, int & il, int & role, uint64_t nbytes) {
    if (strncmp(name, "blk.", 4) != 0) return false;
    char * end = nullptr;
    long v = strtol(name + 4, &end, 10);
    if (!end || *end != '.') return false;
    if (nbytes < g_dn_min_bytes) return false;
    // FFN only, and deliberately so.
    //
    //   The first attempt claimed every large 2D per-layer weight, on the
    //   reasoning that they all reach ggml_mul_mat through build_lora_mm. They
    //   do -- but that is not the only thing that touches them. A streamed
    //   weight is allocated as a one-row stub, and any code that reads its
    //   SHAPE rather than multiplying by it then sees the stub. Qwen3.5's SSM
    //   path does exactly that:
    //
    //     ggml_concat <- build_conv_state <- build_layer_attn_linear
    //     GGML_ASSERT(a->ne[d] == b->ne[d]) failed
    //
    //   So stubbing cannot generalise. Extending this to attention and SSM
    //   needs the model tensors allocated at their REAL shape but bound to
    //   arena memory, so the shape is always right and only the bytes are
    //   shared. That is a different design, not a wider filter.
    //
    //   Matching on ffn_gate/up/down is not an architecture branch: it is the
    //   same reliance on GGUF's naming convention that identifies experts by
    //   ffn_*_exps.
    if      (strcmp(end, ".ffn_gate.weight") == 0) role = 0;
    else if (strcmp(end, ".ffn_up.weight")   == 0) role = 1;
    else if (strcmp(end, ".ffn_down.weight") == 0) role = 2;
    else return false;
    il = (int) v;
    return true;
}

static bool dense_streamed_layer(int il) {
    return g_dn_first >= 0 && il >= g_dn_first;
}

// ---------------------------------------------------------------------------
// Return the shard list for a split GGUF (gguf-split); a single entry if not split.
//   llama.cpp's llama_tensor_weight carries (idx, offs) where offs is an offset
//   *within that shard*. MoEStream does its own reads, so ignoring idx would
//   read an unrelated region of a different file as if it were an expert.
//   The symptom is plausible-but-wrong output, which is hard to notice, so this
//   must be handled.
//
//   Naming follows the gguf-split convention "<base>-00001-of-00003.gguf".
//   If any shard is missing, MoEStream disables itself -- better than breaking
//   silently.
// ---------------------------------------------------------------------------
static const std::vector<std::string> & shard_paths() {
    static std::vector<std::string> v;
    static bool done = false;
    if (done) return v;
    done = true;

    const char * p = !g_path.empty() ? g_path.c_str() : getenv("MOESTREAM_GGUF");
    if (!p || !*p) return v;
    const std::string path(p);

    // Does the last 20 characters match "-NNNNN-of-MMMMM.gguf"?
    if (path.size() > 20) {
        const std::string tail = path.substr(path.size() - 20);
        int a = 0, b = 0;
        if (sscanf(tail.c_str(), "-%5d-of-%5d.gguf", &a, &b) == 2 &&
            a >= 1 && b >= 1 && a <= b && b < 1000) {
            const std::string base = path.substr(0, path.size() - 20);
            for (int i = 1; i <= b; ++i) {
                char suf[32];
                snprintf(suf, sizeof suf, "-%05d-of-%05d.gguf", i, b);
                v.push_back(base + suf);
            }
            return v;
        }
    }
    v.push_back(path);
    return v;
}

// ---------------------------------------------------------------------------
// Read top_k (<arch>.expert_used_count) from the GGUF metadata.
//   slab_slots() runs during model load, when hparams is not yet available.
//   This used to hardcode 8, but the Qwen3-Next family uses 10. That was
//   harmless while frac dominated the sizing, but once frac is lowered the
//   need term wins, the slab comes up short, and slot exhaustion corrupts the
//   output. Scanning the GGUF keys directly keeps this architecture-agnostic.
// ---------------------------------------------------------------------------
// Read one integer KV from the GGUF header by key suffix. The key is prefixed
// with the architecture name, so only the suffix is matched.
static int gguf_kv_int(const char * suf_in) {
    int cached = 0;

    const auto & sh = shard_paths();
    if (sh.empty()) return 0;
    FILE * f = fopen(sh[0].c_str(), "rb");   // the KV section lives in shard 0
    if (!f) return 0;

    auto rd = [&](void * p, size_t n) { return fread(p, 1, n, f) == n; };
    char magic[4];
    uint32_t ver = 0; uint64_t n_tensor = 0, n_kv = 0;
    if (!rd(magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !rd(&ver, 4) || !rd(&n_tensor, 8) || !rd(&n_kv, 8)) { fclose(f); return 0; }

    // Skip over a value (type 9 = array, handled recursively)
    std::function<bool(uint32_t, int64_t *)> skip = [&](uint32_t t, int64_t * out) -> bool {
        static const size_t fixed[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
        if (t == 8) {                                   // STRING
            uint64_t n; if (!rd(&n, 8)) return false;
            return fseek(f, (long) n, SEEK_CUR) == 0;
        }
        if (t == 9) {                                   // ARRAY
            uint32_t et; uint64_t n;
            if (!rd(&et, 4) || !rd(&n, 8)) return false;
            for (uint64_t i = 0; i < n; ++i) if (!skip(et, nullptr)) return false;
            return true;
        }
        if (t > 12) return false;
        unsigned char buf[8] = {0};
        if (!rd(buf, fixed[t])) return false;
        if (out) {                                       // extract the value for integer types only
            switch (t) {
                case 0: *out = *(uint8_t  *) buf; break;
                case 1: *out = *(int8_t   *) buf; break;
                case 2: *out = *(uint16_t *) buf; break;
                case 3: *out = *(int16_t  *) buf; break;
                case 4: *out = *(uint32_t *) buf; break;
                case 5: *out = *(int32_t  *) buf; break;
                case 10:*out = (int64_t) *(uint64_t *) buf; break;
                case 11:*out = *(int64_t  *) buf; break;
                default: *out = 0; break;
            }
        }
        return true;
    };

    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t klen; if (!rd(&klen, 8) || klen > (1u << 20)) break;
        std::string key(klen, '\0');
        if (!rd(&key[0], klen)) break;
        uint32_t vt; if (!rd(&vt, 4)) break;
        // The key is prefixed with the architecture name, so match the suffix
        //   e.g. qwen3next.expert_used_count / llama.expert_used_count
        const char * suf = suf_in;
        const size_t sl = strlen(suf);
        const bool want = key.size() >= sl &&
                          key.compare(key.size() - sl, sl, suf) == 0;
        int64_t v = 0;
        if (!skip(vt, want ? &v : nullptr)) break;
        if (want && v > 0 && v < 4096) { cached = (int) v; break; }
    }
    fclose(f);
    return cached;
}

static int gguf_topk() {
    static int cached = -1;
    if (cached < 0) cached = gguf_kv_int(".expert_used_count");
    return cached;
}

static int gguf_nlayer() {
    static int cached = -1;
    if (cached < 0) cached = gguf_kv_int(".block_count");
    return cached;
}

// 0 on a dense model: the key is absent. This is what lets the runtime pick the
// expert path or the dense path on its own, without the operator being asked.
static int gguf_nexpert() {
    static int cached = -1;
    if (cached < 0) cached = gguf_kv_int(".expert_count");
    return cached;
}

// Sum the bytes of everything the dense path would claim, by walking the GGUF
// tensor index. Needed before the first tensor is allocated, which is why it
// cannot be accumulated as tensors arrive.
static void gguf_scan_streamable(uint64_t * total, uint64_t * layer_max) {
    *total = 0; *layer_max = 0;
    const auto & sh = shard_paths();
    std::map<int, uint64_t> per_layer;
    for (const auto & path : sh) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) continue;
        auto rd = [&](void * p, size_t n) { return fread(p, 1, n, f) == n; };
        char magic[4]; uint32_t ver = 0; uint64_t n_tensor = 0, n_kv = 0;
        if (!rd(magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
            !rd(&ver, 4) || !rd(&n_tensor, 8) || !rd(&n_kv, 8)) { fclose(f); continue; }
        // skip the KV section
        std::function<bool(uint32_t)> skip = [&](uint32_t t) -> bool {
            static const size_t fixed[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            if (t == 8) { uint64_t n; if (!rd(&n, 8)) return false;
                          return fseek(f, (long) n, SEEK_CUR) == 0; }
            if (t == 9) { uint32_t et; uint64_t n;
                          if (!rd(&et, 4) || !rd(&n, 8)) return false;
                          for (uint64_t i = 0; i < n; ++i) if (!skip(et)) return false;
                          return true; }
            if (t > 12) return false;
            return fseek(f, (long) fixed[t], SEEK_CUR) == 0;
        };
        bool ok = true;
        for (uint64_t i = 0; i < n_kv && ok; ++i) {
            uint64_t kl; if (!rd(&kl, 8) || kl > (1u << 20)) { ok = false; break; }
            if (fseek(f, (long) kl, SEEK_CUR) != 0) { ok = false; break; }
            uint32_t vt; if (!rd(&vt, 4) || !skip(vt)) ok = false;
        }
        if (!ok) { fclose(f); continue; }
        // tensor index: name, n_dims, dims[], type, offset
        struct Ent { std::string name; uint32_t nd; uint64_t off; };
        std::vector<Ent> ents; ents.reserve((size_t) n_tensor);
        for (uint64_t i = 0; i < n_tensor && ok; ++i) {
            uint64_t nl2; if (!rd(&nl2, 8) || nl2 > (1u << 16)) { ok = false; break; }
            std::string nm(nl2, '\0'); if (!rd(&nm[0], nl2)) { ok = false; break; }
            uint32_t nd; if (!rd(&nd, 4) || nd > 4) { ok = false; break; }
            for (uint32_t d = 0; d < nd; ++d) { uint64_t x; if (!rd(&x, 8)) { ok = false; break; } }
            uint32_t ty; uint64_t off;
            if (!ok || !rd(&ty, 4) || !rd(&off, 8)) { ok = false; break; }
            ents.push_back({ nm, nd, off });
        }
        fclose(f);
        if (!ok || ents.empty()) continue;
        struct stat st; uint64_t fsz = 0;
        if (stat(path.c_str(), &st) == 0) fsz = (uint64_t) st.st_size;
        std::vector<size_t> order(ents.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return ents[a].off < ents[b].off; });
        for (size_t k = 0; k < order.size(); ++k) {
            const Ent & e = ents[order[k]];
            const uint64_t end = (k + 1 < order.size()) ? ents[order[k+1]].off : fsz;
            if (end <= e.off) continue;
            const uint64_t nb = end - e.off;
            int il, role;
            if (e.nd == 2 && parse_dense_any(e.name.c_str(), il, role, nb)) {
                *total += nb;
                per_layer[il] += nb;
            }
        }
    }
    for (const auto & kv : per_layer) *layer_max = std::max(*layer_max, kv.second);
}

// ---------------------------------------------------------------------------
// Scan the GGUF tensor table and return the byte cost of one slot across all
// layers (i.e. holding one expert per layer).
//   With that number, "spend N GiB on the slab" can be converted into a slot
//   count. Summing actual sizes keeps this exact even for UD quantization,
//   where the type differs per layer.
//   Returns 0 if it could not be determined.
// ---------------------------------------------------------------------------
static uint64_t gguf_bytes_per_slot() {
    static uint64_t cached = UINT64_MAX;
    if (cached != UINT64_MAX) return cached;
    cached = 0;

    uint64_t total_all = 0; int64_t n_exp_all = 0;
    for (const auto & sp : shard_paths()) {                 // scan every shard
    FILE * f = fopen(sp.c_str(), "rb");
    if (!f) continue;

    auto rd = [&](void * p, size_t n) { return fread(p, 1, n, f) == n; };
    char magic[4]; uint32_t ver = 0; uint64_t n_tensor = 0, n_kv = 0;
    if (!rd(magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !rd(&ver, 4) || !rd(&n_tensor, 8) || !rd(&n_kv, 8)) { fclose(f); continue; }

    std::function<bool(uint32_t)> skipv = [&](uint32_t t) -> bool {
        static const size_t fixed[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
        if (t == 8) { uint64_t n; return rd(&n,8) && fseek(f,(long)n,SEEK_CUR)==0; }
        if (t == 9) {
            uint32_t et; uint64_t n;
            if (!rd(&et,4) || !rd(&n,8)) return false;
            for (uint64_t i = 0; i < n; ++i) if (!skipv(et)) return false;
            return true;
        }
        if (t > 12) return false;
        unsigned char b[8];
        return rd(b, fixed[t]);
    };
    // Skip the KV section
    bool bad = false;
    for (uint64_t i = 0; i < n_kv && !bad; ++i) {
        uint64_t kl; if (!rd(&kl,8) || kl > (1u<<20)) { bad = true; break; }
        if (fseek(f,(long)kl,SEEK_CUR) != 0) { bad = true; break; }
        uint32_t vt; if (!rd(&vt,4) || !skipv(vt)) { bad = true; break; }
    }
    if (bad) { fclose(f); continue; }
    // Tensor table: name / n_dims / dims[] / type / offset
    uint64_t total = 0; int64_t n_exp = 0;
    for (uint64_t i = 0; i < n_tensor; ++i) {
        uint64_t nl; if (!rd(&nl,8) || nl > (1u<<16)) break;
        std::string nm(nl, '\0');
        if (!rd(&nm[0], nl)) break;
        uint32_t nd; if (!rd(&nd,4) || nd > 4) break;
        int64_t ne[4] = {1,1,1,1};
        for (uint32_t d = 0; d < nd; ++d) { uint64_t v; if (!rd(&v,8)) { fclose(f); return 0; } ne[d] = (int64_t) v; }
        uint32_t ty; uint64_t off;
        if (!rd(&ty,4) || !rd(&off,8)) break;

        int il, role;
        if (!parse_name(nm.c_str(), il, role)) continue;    // expert tensors only
        if (ne[2] <= 1) continue;
        n_exp = ne[2];
        // one expert = row size * ne[1]
        total += (uint64_t) ggml_row_size((ggml_type) ty, ne[0]) * (uint64_t) ne[1];
    }
    fclose(f);
    total_all += total; if (n_exp > 0) n_exp_all = n_exp;
    }                                                        // end of shard loop
    cached = (n_exp_all > 0) ? total_all : 0;
    return cached;
}

// ---------------------------------------------------------------------------
// Device memory ceiling and current usage, read from sysfs.
//   A recommendation that does not fit is worse than no recommendation: the
//   server simply fails to start. So the MRC's suggestion is capped by what
//   the machine can actually hold.
//   Returns false when the values are unavailable (non-amdgpu, /sys not mounted),
//   in which case the recommendation is left uncapped.
// ---------------------------------------------------------------------------
static bool read_u64_file(const std::string & path, uint64_t * out) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) return false;
    unsigned long long v = 0;
    const bool ok = fscanf(f, "%llu", &v) == 1;
    fclose(f);
    if (ok) *out = (uint64_t) v;
    return ok;
}

static bool device_mem_gib(double * total, double * used) {
    DIR * d = opendir("/sys/class/drm");
    if (!d) return false;
    bool found = false;
    uint64_t t = 0, u = 0;
    while (struct dirent * e = readdir(d)) {
        if (strncmp(e->d_name, "card", 4) != 0) continue;
        if (strchr(e->d_name, '-')) continue;                 // skip connectors
        const std::string base = std::string("/sys/class/drm/") + e->d_name + "/device/";
        uint64_t gt = 0, gu = 0, vt = 0, vu = 0;
        if (!read_u64_file(base + "mem_info_gtt_total", &gt)) continue;
        read_u64_file(base + "mem_info_gtt_used",   &gu);
        read_u64_file(base + "mem_info_vram_total", &vt);
        read_u64_file(base + "mem_info_vram_used",  &vu);
        t = gt + vt; u = gu + vu; found = true;
        break;
    }
    closedir(d);
    if (!found) return false;
    *total = double(t) / 1073741824.0;
    *used  = double(u) / 1073741824.0;
    return true;
}

// ---------------------------------------------------------------------------
// Learned settings (MOESTREAM_CACHE_FRAC=learn, UBATCH=learn)
//
//   Two files under MOESTREAM_STATE_DIR, both plain TSV so src/entrypoint.sh
//   can read them with awk:
//
//     tuning.tsv   <model file>  <frac>  <geometry>
//     ubatch.tsv   <model file>  <frac>  <ubatch>  <prefill tok/s>
//
//   ** ubatch rows carry the frac they were measured at, and that is essential. **
//   The two settings interact: on Ornith at frac 0.25 the best ubatch is 1024
//   (249 vs 227 tok/s), while at frac 0.15 it is 2048 (224 vs 220). Learning
//   them independently therefore converges on the wrong answer. Keying each
//   ubatch measurement by frac makes it self-correcting: when frac moves, the
//   old rows simply stop matching and the candidates are measured again.
//
//   Geometry (expert count and bytes per slot) guards against a same-named file
//   with a different shape. Only frac is checked against it -- a wrong frac
//   costs memory and can stop the server from starting, whereas a wrong ubatch
//   only costs speed.
// ---------------------------------------------------------------------------
static std::string state_path(const char * name) {
    const char * d = getenv("MOESTREAM_STATE_DIR");
    return std::string((d && *d) ? d : "/state") + "/" + name;
}

static std::string model_file_name() {
    const auto & sh = shard_paths();
    std::string b = sh.empty() ? std::string("unknown") : sh[0];
    const size_t sl = b.find_last_of('/');
    return sl == std::string::npos ? b : b.substr(sl + 1);
}

static std::string model_geometry() {
    char buf[128];
    snprintf(buf, sizeof buf, "%lld:%llu", (long long) g_n_expert,
             (unsigned long long) gguf_bytes_per_slot());
    return buf;
}

// Rewrite one row of a TSV, keeping every row that does not match `match`.
static void state_put(const char * file, const std::string & match, const std::string & row) {
    const std::string path = state_path(file);
    std::vector<std::string> keep;
    if (FILE * f = fopen(path.c_str(), "r")) {
        char line[1024];
        while (fgets(line, sizeof line, f))
            if (strncmp(line, match.c_str(), match.size()) != 0) keep.push_back(line);
        fclose(f);
    }
    const std::string tmp = path + ".tmp";
    FILE * o = fopen(tmp.c_str(), "w");
    if (!o) return;                       // state dir not writable: stay silent
    for (const auto & l : keep) fputs(l.c_str(), o);
    fputs(row.c_str(), o);
    fclose(o);
    if (rename(tmp.c_str(), path.c_str()) != 0) unlink(tmp.c_str());
}

// The frac learned for this model, or 0 when there is none for this geometry.
static double learned_frac() {
    FILE * f = fopen(state_path("tuning.tsv").c_str(), "r");
    if (!f) return 0;
    const std::string key = model_file_name(), geo = model_geometry();
    char line[1024]; double out = 0;
    while (fgets(line, sizeof line, f)) {
        char m[512], g[128]; double v = 0;
        if (sscanf(line, "%511[^\t]\t%lf\t%127s", m, &v, g) == 3 &&
            key == m && geo == g && v > 0 && v <= 1.0) out = v;
    }
    fclose(f);
    return out;
}

static void learned_frac_store(double frac) {
    if (!(frac > 0 && frac <= 1.0)) return;
    const std::string key = model_file_name();
    char row[1024];
    snprintf(row, sizeof row, "%s\t%.2f\t%s\n", key.c_str(), frac, model_geometry().c_str());
    state_put("tuning.tsv", key + "\t", row);
    fprintf(stderr, "moestream: [learn] frac=%.2f recorded; the next start uses it\n", frac);
}

// Close the request whose prefill just ended, decode having started.
//   Everything between the first prefill pass and here is the prompt: the full
//   passes, the partial tail, and the gaps between them. Nothing is dropped, so
//   the number does not shift as the ubatch changes -- which is the whole point.
//   A request that never reaches decode (an error, a cancelled connection) is
//   never closed and its tokens roll into the next one; the guard below drops
//   the result if that made the span implausible.
static void pf_req_close() {
    if (g_req_t0 <= 0 || g_req_tok == 0) return;
    const double   dt  = now_s() - g_req_t0;
    const uint64_t tok = g_req_tok;
    g_req_t0 = 0; g_req_tok = 0;
    if (!(dt > 0 && dt < 3600.0)) return;
    if (g_pf_intervals == 0) {           // remember the warm-up, subtract later
        g_pf_first_time = dt;
        g_pf_first_tok  = tok;
    }
    g_pf_time_total += dt;
    g_pf_tok_total  += tok;
    g_pf_intervals++;
}

// Prefill rate for the run: whole requests, the first one excluded.
//   The first request of a process reads every expert from SSD and is far
//   slower than the rest (measured 97.9 against 109-111 tok/s on Qwen3-Coder).
//   That is a warm-up cost, not a speed, so it is subtracted once a second
//   request exists to fall back on.
static double prefill_rate() {
    double   t = g_pf_time_total;
    uint64_t n = g_pf_tok_total;
    if (g_pf_intervals >= 2) { t -= g_pf_first_time; n -= g_pf_first_tok; }
    return (t > 0 && n > 0) ? double(n) / t : 0.0;
}

// Record this run's ubatch result. Called once, at exit.
//   Measured per request, so the number means the same thing at every ubatch
//   and the candidates are comparable. Written only when at least two requests
//   completed; otherwise nothing is written and the candidate is tried again.
static void ubatch_store(int ub, double frac, double rate) {
    if (ub <= 0 || frac <= 0) return;
    if (g_pf_intervals < 2) {
        // Fewer than two completed requests: there is no warm-up to subtract
        // against, and the first request is dominated by the cold read of every
        // expert. Nothing is written, so the candidate is measured again on a
        // later start. Any prompt length works -- this only needs traffic.
        fprintf(stderr,
            "moestream: [learn] UBATCH=%d at frac=%.2f: %llu request(s) this run, "
            "need 2 so the warm-up one can be excluded; not recorded\n",
            ub, frac, (unsigned long long) g_pf_intervals);
        return;
    }
    char key[640], row[1024];
    snprintf(key, sizeof key, "%s\t%.2f\t%d\t", model_file_name().c_str(), frac, ub);
    snprintf(row, sizeof row, "%s%.1f\n", key, rate);
    state_put("ubatch.tsv", key, row);
    fprintf(stderr, "moestream: [learn] UBATCH=%d at frac=%.2f -> %.1f tok/s prefill\n",
            ub, frac, rate);
}

uint32_t slab_slots(const char * name, int64_t n_expert) {
    if (!enabled()) return 0;
    int il, role;
    if (!parse_name(name, il, role) || n_expert <= 1) return 0;
    g_n_expert = n_expert;
    uint32_t s;
    if (g_frac_learn) {
        static int    resolved = 0;
        static double v = 0;
        if (!resolved) {
            resolved = 1;
            v = learned_frac();
            if (v > 0) fprintf(stderr, "moestream: [learn] using frac=%.2f from a previous run\n", v);
            else { v = 0.15;
                fprintf(stderr, "moestream: [learn] no frac recorded; starting at %.2f "
                                "(low on purpose: too high and the server may not start)\n", v); }
        }
        g_frac = v;                       // so the rest of the code sees the real value
        s = (uint32_t)(double(n_expert) * v + 0.5);
    } else if (g_frac_auto) {
        // auto: derive the slot count from a slab budget in GiB. This is a
        // specification that keeps its meaning when the model changes. It
        // cannot locate the knee, so tune against the live [stats] hit rate.
        const uint64_t per = gguf_bytes_per_slot();
        if (per > 0) {
            s = (uint32_t) std::max<uint64_t>(1,
                    (uint64_t)(g_slab_gib * 1073741824.0) / per);
            s = (uint32_t) std::min<int64_t>(s, n_expert);
            static bool shown = false;
            if (!shown) {
                shown = true;
                fprintf(stderr,
                    "moestream: cache_frac=auto -> slab %.1f GiB = %u slots "
                    "(%.0f%% of %lld, %.1f MiB per slot)\n",
                    g_slab_gib, s, 100.0 * s / double(n_expert), (long long) n_expert,
                    per / 1048576.0);
            }
        } else {
            s = (uint32_t)(double(n_expert) * 0.38 + 0.5);
            static bool warned = false;
            if (!warned) { warned = true;
                fprintf(stderr, "moestream: cache_frac=auto but the GGUF could not be read; falling back to 0.38\n"); }
        }
    } else {
        s = (uint32_t)(double(n_expert) * g_frac + 0.5);
    }
    // Reserve enough slots to always hold one micro-batch's union.
    //   Coming up short corrupts the output, so round toward the safe side.
    // union(ub) = n_expert * (1 - (1 - top_k/n_expert)^ub), times a 1.15 margin.
    // The old ub*top_k estimate ignored duplicates and overshot by about 30%.
    const int tk_meta = gguf_topk();
    const double tk = tk_meta > 0 ? double(tk_meta) : 8.0;
    {   // report once which value was used
        static bool shown = false;
        if (!shown) {
            shown = true;
            fprintf(stderr, "moestream: top_k = %d (%s)\n", (int) tk,
                    tk_meta > 0 ? "from GGUF metadata" : "default; metadata unreadable");
        }
    }
    const double u  = double(n_expert) * (1.0 - std::pow(1.0 - tk / double(n_expert),
                                                         double(g_max_ubatch)));
    const uint32_t need = (uint32_t) std::min<int64_t>(
        n_expert, (int64_t)(u * 1.15) + (int64_t) tk);
    s = std::max(s, need);
    s = std::max<uint32_t>(8, std::min<uint32_t>(s, (uint32_t) n_expert));
    g_slots = s;
    g_zero_slot = s;            // reserve the last slot for the null expert
    return s + 1;               // so the allocation is one slot larger
}

bool skip_load(const char * name) {
    if (!enabled()) return false;
    int il, role;
    if (parse_name(name, il, role)) return true;
    // Streamed dense FFN weights are stubs; their contents come from the arena.
    // Only what was actually claimed. Deciding again from the name would be a
    // different test than the one that ran at allocation time, and the failure
    // mode is a real weight left unread.
    if (g_dn_claimed.count(name)) return true;
    return false;
}

// Decide whether this tensor is a dense FFN weight on a streamed layer.
//   Called from the loader before allocation. The layer split needs the layer
//   count, which is not known from the tensor name, so it is read from the GGUF
//   header the same way top_k is.
bool dense_claim(const char * name, int64_t n_dims, uint64_t nbytes) {
    if (!enabled()) return false;
    if (n_dims != 2) return false;
    int il = -1, role = -1;
    if (!parse_dense_any(name, il, role, nbytes)) return false;

    if (g_dn_nlayer == 0) {
        g_dn_nlayer = -1;                                    // decided, possibly to "off"
        // ---- is this a dense model at all? ----
        //   A MoE model has expert tensors; its dense ffn_* weights (shared
        //   experts, leading dense layers) belong to the expert path and must
        //   not be stubbed. Decided from metadata, not from the operator.
        if (gguf_nexpert() > 0) {
            fprintf(stderr, "moestream: [dense] MoE model (%d experts); "
                            "expert streaming handles this, dense path off\n", gguf_nexpert());
            return false;
        }
        const int nl = gguf_nlayer();
        if (nl <= 0) {
            fprintf(stderr, "moestream: [dense] block_count unreadable; dense path off\n");
            return false;
        }
        g_dn_nlayer = nl;

        if (!g_dn_auto) {
            if (g_dn_frac >= 1.0) { g_dn_nlayer = -1; return false; }   // explicitly off
        } else {
            // ---- decide the split from the memory budget ----
            //   Stream the least that still fits. If the model already fits,
            //   stream nothing and behave exactly like plain llama.cpp.
            //   Unlike the expert cache there is no knee to find: finding S29
            //   measured the cost as linear in bytes streamed, so "as little as
            //   possible" is the whole policy.
            uint64_t scan_total = 0, scan_layer_max = 0;
            gguf_scan_streamable(&scan_total, &scan_layer_max);
            // Extrapolating "three weights per layer x this one's size" was
            // right only for the FFN-only version. Read the tensor index.
            const double ffn_gib  = scan_total
                ? (double) scan_total / 1073741824.0
                : 3.0 * (double) nl * (double) nbytes / 1073741824.0;
            double file_gib = 0;
            for (const auto & sp : shard_paths()) {
                struct stat st;
                if (stat(sp.c_str(), &st) == 0) file_gib += (double) st.st_size / 1073741824.0;
            }
            const double nonffn_gib = std::max(0.0, file_gib - ffn_gib);
            const double layer_gib  = ffn_gib / (double) nl;
            double dev_t = 0, dev_u = 0;
            const bool have_mem = device_mem_gib(&dev_t, &dev_u);
            double reserve = 2.0;
            if (const char * r = getenv("MOESTREAM_MEM_RESERVE_GIB")) reserve = atof(r);
            // Everything but the streamed FFN has to be resident, plus the
            // arena and whatever the KV cache will want.
            const double budget = have_mem ? (dev_t - reserve - dev_u) : 0.0;
            const double for_ffn = budget - nonffn_gib - g_dn_nbuf * layer_gib;
            double frac = 1.0;
            if (have_mem && ffn_gib > 0) {
                frac = for_ffn / ffn_gib;
                if (frac > 1.0) frac = 1.0;
                if (frac < 0.0) frac = 0.0;
            }
            g_dn_frac = frac;
            fprintf(stderr,
                "moestream: [dense] auto: model %.2f GiB (FFN %.2f + other %.2f), "
                "device %.1f GiB free of %.1f\n"
                "moestream: [dense] auto: keeping %.0f%% of the FFN resident\n",
                file_gib, ffn_gib, nonffn_gib, budget, dev_t, frac * 100.0);
            if (frac >= 1.0) {
                fprintf(stderr, "moestream: [dense] auto: the model fits; "
                                "streaming nothing (identical to plain llama.cpp)\n");
                g_dn_nlayer = -1;
                return false;
            }
        }

        int keep = (int) (g_dn_frac * (double) nl + 0.5);
        if (keep < 0) keep = 0;
        if (keep > nl) keep = nl;
        g_dn_first = keep;
        fprintf(stderr,
            "moestream: [dense] %d layers, keeping %d resident, streaming %d..%d "
            "(frac=%.2f%s)\n", nl, keep, keep, nl - 1, g_dn_frac,
            g_dn_learn ? ", learn" : (g_dn_auto ? ", auto" : ""));
    }
    if (g_dn_nlayer < 0) return false;
    if (!dense_streamed_layer(il)) return false;
    g_dn_seen = true;
    g_dn_claimed.insert(name);
    return true;
}

void register_dense(const char * name, const ggml_tensor * stub, uint64_t file_offset,
                    uint64_t bytes, int file_idx, int type, int64_t ne0, int64_t ne1) {
    int il, role;
    if (!parse_dense_any(name, il, role, bytes)) return;
    DenseInfo di;
    di.file_offset = file_offset; di.bytes = bytes;
    di.file_idx = (uint16_t) (file_idx < 0 ? 0 : file_idx);
    di.type = type; di.ne0 = ne0; di.ne1 = ne1; di.valid = true; di.role = role;
    auto & L = g_dn_meta[il];
    L.w.push_back(di);
    L.ready = true;
    // The graph never names these tensors; it hands build_lora_mm the pointer
    // llama.cpp allocated. That pointer is the only usable key.
    (void) stub;
}

void register_slab(const char * name, ggml_tensor * t,
                   uint64_t file_offset, uint64_t expert_bytes, int64_t n_expert,
                   int file_idx) {
    if (!enabled()) return;
    int il, role;
    if (!parse_name(name, il, role)) return;
    SlabInfo si; si.t = t; si.file_offset = file_offset;
    si.expert_bytes = expert_bytes; si.n_expert = n_expert;
    si.file_idx = (uint16_t) (file_idx < 0 ? 0 : file_idx);
    auto & L = g_layers[il];
    if      (role == 0) L.gate = si;
    else if (role == 1) L.up   = si;
    else                L.down = si;
    L.ready = L.gate.t && L.up.t && L.down.t;
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
static void do_read(const ReadJob & j) {
    // For split GGUFs, offs is an offset within that shard: use the matching fd.
    const size_t fi = (size_t) j.si->file_idx;
    if (fi >= g_fds.size() || g_fds[fi] < 0) return;
    const int fd = g_fds[fi];
    const off_t off = (off_t)(j.si->file_offset + uint64_t(j.expert) * j.si->expert_bytes);
    size_t left = (size_t) j.si->expert_bytes, o = 0;
    while (left) {
        ssize_t r = pread(fd, j.dst + o, left, off + (off_t) o);
        if (r <= 0) break;
        o += (size_t) r; left -= (size_t) r;
    }
}

// Simple parallel read that spawns threads per batch.
// A resident pool with a condvar is a rich source of synchronization bugs (it
// did deadlock here), and at roughly 1 ms per read versus about 20 us to spawn
// a thread, the simple approach is the better trade.
static void run_reads_parallel(std::vector<ReadJob> & jobs, int nthreads) {
    if (jobs.empty()) return;
    const int n = std::min<int>(nthreads, (int) jobs.size());
    if (n <= 1) { for (auto & j : jobs) do_read(j); return; }
    std::vector<std::thread> th; th.reserve(n);
    std::atomic<size_t> next{0};
    for (int i = 0; i < n; ++i)
        th.emplace_back([&jobs, &next] {
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= jobs.size()) return;
                do_read(jobs[k]);
            }
        });
    for (auto & t : th) t.join();
}

// Cycle through the candidates measuring effective bandwidth, then settle on
// the best once enough samples have accumulated.
//   The candidate is switched roughly once per token (i.e. per n_layer calls).
static int io_threads_for_this_call(double * out_t0) {
    if (!g_io_auto || g_th_settled) return g_nthreads;
    *out_t0 = now_s();
    return TH_CAND[g_th_cur];
}
static void io_threads_record(size_t bytes, double t0) {
    if (!g_io_auto || g_th_settled || t0 <= 0) return;
    const double dt = now_s() - t0;
    if (dt > 0) { g_th_bytes[g_th_cur] += double(bytes); g_th_time[g_th_cur] += dt; }
    // Advance to the next candidate every 40 calls (about one token)
    if (++g_th_calls % 40 == 0) g_th_cur = (g_th_cur + 1) % TH_N;
    // Settle once every candidate has 20 rounds of samples (800 calls x 6)
    if (g_th_calls >= 40 * TH_N * 20) {
        int best = 0; double bw = 0;
        for (int i = 0; i < TH_N; ++i) {
            const double b = g_th_time[i] > 0 ? g_th_bytes[i] / g_th_time[i] : 0;
            if (b > bw) { bw = b; best = i; }
        }
        g_nthreads = TH_CAND[best];
        g_th_settled = true;
        fprintf(stderr, "moestream: [io] settled on %d threads (measured %.2f GB/s)\n",
                g_nthreads, bw / 1e9);
        for (int i = 0; i < TH_N; ++i)
            fprintf(stderr, "moestream: [io]   %2d threads: %5.2f GB/s\n", TH_CAND[i],
                    g_th_time[i] > 0 ? g_th_bytes[i] / g_th_time[i] / 1e9 : 0.0);
    }
}

// On UMA, return the GPU-visible pointer for a slot; nullptr if unavailable.
static uint8_t * slab_host_ptr(const SlabInfo & si, uint32_t slot) {
    if (!zerocopy_mode() || !si.t || !si.t->buffer) return nullptr;
    if (!ggml_backend_vk_buffer_host_ptr) return nullptr;   // weak symbol unresolved
    // Under Vulkan, tensor->data is an offset relative to a 0x1000 sentinel
    const size_t base_off = (size_t)((uint8_t *) si.t->data - (uint8_t *) 0x1000);
    return (uint8_t *) ggml_backend_vk_buffer_host_ptr(
        si.t->buffer, base_off + (size_t) slot * si.expert_bytes);
}

// ---------------------------------------------------------------------------
// Dense FFN streaming setup. Runs before the expert path, because a dense
// model has no expert tensors and finalize() would otherwise disable MoEStream
// and return before reaching this.
static bool g_dn_done = false;
static void finalize_dense(const char * gguf_path) {
    if (g_dn_done) return;
    g_dn_done = true;
    if (!g_dn_seen || g_dn_meta.empty()) return;

    if (g_path.empty() && gguf_path && *gguf_path) g_path = gguf_path;
    const auto & shards = shard_paths();
    if (shards.empty()) {
        fprintf(stderr, "moestream: [dense] GGUF path unknown; set MOESTREAM_GGUF\n");
        return;
    }
    if (g_fds.empty()) {
        g_fds.assign(shards.size(), -1);
        for (size_t i = 0; i < shards.size(); ++i) {
            g_fds[i] = open(shards[i].c_str(), O_RDONLY);
            if (g_fds[i] < 0) {
                fprintf(stderr, "moestream: [dense] cannot open %s\n", shards[i].c_str());
                for (int fd : g_fds) if (fd >= 0) close(fd);
                g_fds.clear();
                return;
            }
        }
    }
    // ---- dense FFN arena ----
    //   Only reached on a model with dense FFN weights and MOESTREAM_DENSE_FRAC
    //   below 1. Sized to the largest single layer, allocated the same way the
    //   expert arena is (let ggml-vulkan allocate; never import host memory --
    //   finding S7).
    {
        size_t lay_max = 0;
        for (auto & kv : g_dn_meta) {
            if (!kv.second.ready) continue;
            size_t t = 0;
            for (const auto & w : kv.second.w) if (w.valid) t += (size_t) w.bytes;
            lay_max = std::max(lay_max, t);
        }
        g_dn_bytes_per = (lay_max + 4095) & ~(size_t) 4095;
        ggml_backend_buffer_type_t buft = ggml_backend_vk_buffer_type
                                            ? ggml_backend_vk_buffer_type(0) : nullptr;
        bool ok = buft != nullptr && g_dn_bytes_per > 0;
        for (int i = 0; ok && i < g_dn_nbuf; ++i) {
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, g_dn_bytes_per);
            if (!b) { ok = false; break; }
            void * hp = ggml_backend_vk_buffer_host_ptr
                          ? ggml_backend_vk_buffer_host_ptr(b, 0) : nullptr;
            if (!hp) {
                ggml_backend_buffer_free(b);
                fprintf(stderr, "moestream: [dense] the arena requires a UMA device\n");
                ok = false; break;
            }
            g_dn_buf.push_back(b);
            g_dn_host.push_back(hp);
        }
        g_dn_resident.assign(g_dn_nbuf, -1);
        if (ok) {
            size_t n_t = 0;
            for (const auto & kv : g_dn_meta) n_t += kv.second.w.size();
            // Sized from the real count. It used to assume three weights per
            // layer, which was true only while this streamed the FFN alone;
            // adding attention and SSM overflowed the context and tripped
            // GGML_ASSERT(obj_new) during load.
            struct ggml_init_params ip = {
                ggml_tensor_overhead() * (n_t + 16), nullptr, true };
            g_dn_ctx = ggml_init(ip);
            ok = g_dn_ctx != nullptr;
            for (auto & kv : g_dn_meta) {
                if (!ok) break;
                if (!kv.second.ready) continue;
                const int il = kv.first;
                const size_t bi = (size_t) (il % g_dn_nbuf);
                uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(g_dn_buf[bi]);
                DenseLayer v; uint64_t off = 0;
                v.w.resize(kv.second.w.size());
                for (size_t r = 0; r < kv.second.w.size() && ok; ++r) {
                    const DenseInfo & m = kv.second.w[r];
                    if (!m.valid) continue;
                    ggml_tensor * t = ggml_new_tensor_2d(g_dn_ctx, (ggml_type) m.type, m.ne0, m.ne1);
                    if (!t) { ok = false; break; }
                    if (ggml_backend_tensor_alloc(g_dn_buf[bi], t, base + off)
                            != GGML_STATUS_SUCCESS) { ok = false; break; }
                    v.w[r] = m; v.w[r].t = t; v.w[r].arena_off = off;
                    off += m.bytes;
                }
                v.ready = ok && !v.w.empty();
                if (v.ready) g_dn_view[il] = v;
            }
            g_dn_ready = ok && !g_dn_view.empty();
        }
        if (!g_dn_ready) {
            for (auto b : g_dn_buf) ggml_backend_buffer_free(b);
            g_dn_buf.clear(); g_dn_host.clear(); g_dn_view.clear();
        }
        const double streamed_gib = [&]{
            double t = 0; for (auto & kv : g_dn_meta) for (const auto & w : kv.second.w)
                if (w.valid) t += (double) w.bytes;
            return t / 1073741824.0; }();
        fprintf(stderr,
            "moestream: [dense] arena %s (%d x %.0f MiB); streaming %.2f GiB of FFN "
            "across %d layers, %.2f GiB/token\n",
            g_dn_ready ? "ENABLED" : "FAILED", g_dn_nbuf, g_dn_bytes_per / 1048576.0,
            streamed_gib, (int) g_dn_view.size(), streamed_gib);
    }
}

void finalize(const char * gguf_path) {
    if (!enabled() || !g_cache.empty()) return;
    finalize_dense(gguf_path);
    if (g_layers.empty()) {
        if (g_dn_ready) {
            fprintf(stderr, "moestream: no expert tensors; running as a dense "
                            "streaming model (FFN only)\n");
            return;                       // dense path is live; stay enabled
        }
        fprintf(stderr, "moestream: no expert tensors found, disabling\n");
        g_enabled = false; return;
    }

    const int64_t  n_expert = g_layers.begin()->second.gate.n_expert;
    const uint32_t n_layer  = (uint32_t)(g_layers.rbegin()->first + 1);
    g_first_il = g_layers.begin()->first;
    if (g_first_il != 0)
        fprintf(stderr, "moestream: first %d layer(s) are dense (no experts); "
                        "layers with experts: %d-%d\n",
                g_first_il, g_first_il, g_layers.rbegin()->first);

    // One independent cache per layer. A single shared cache would make slot
    // numbers global, which cannot be mapped back to a per-layer local index.
    g_cache.assign(n_layer, nullptr);
    for (uint32_t l = 0; l < n_layer; ++l) {
        ExpertCache::Config cfg;
        cfg.n_layer   = 1;
        cfg.n_expert  = (uint32_t) n_expert;
        cfg.n_slot    = g_slots;
        cfg.pin_ratio = 0.0;            // ADR-0019: a static hot set loses to dynamic LRU
        cfg.layer_quota_adapt = false;  // only one layer here, so not needed
        g_cache[l] = new ExpertCache(cfg);
    }

    if (g_path.empty() && gguf_path && *gguf_path) g_path = gguf_path;
    const auto & shards = shard_paths();
    if (shards.empty()) {
        fprintf(stderr, "moestream: GGUF path unknown; set MOESTREAM_GGUF to the model path\n");
        g_enabled = false; return;
    }
    // Open every shard. If any is missing, disable rather than misread silently.
    g_fds.assign(shards.size(), -1);
    for (size_t i = 0; i < shards.size(); ++i) {
        g_fds[i] = open(shards[i].c_str(), O_RDONLY);
        if (g_fds[i] < 0) {
            fprintf(stderr, "moestream: cannot open shard: %s (%s)\n",
                    shards[i].c_str(), strerror(errno));
            fprintf(stderr, "moestream: a split GGUF needs every shard present; disabling\n");
            for (int fd : g_fds) if (fd >= 0) close(fd);
            g_fds.clear(); g_enabled = false; return;
        }
    }
    if (shards.size() > 1)
        fprintf(stderr, "moestream: GGUF = %s (split across %zu files)\n",
                shards[0].c_str(), shards.size());
    else
        fprintf(stderr, "moestream: GGUF = %s\n", shards[0].c_str());

    // Validation: every expert tensor must fit inside the shard it claims.
    //   A wrong idx is always caught here; if so, disable MoEStream.
    {
        bool ok = true;
        for (auto & kv : g_layers) {
            const SlabInfo * rs[3] = { &kv.second.gate, &kv.second.up, &kv.second.down };
            for (int r = 0; r < 3 && ok; ++r) {
                const SlabInfo & si = *rs[r];
                if (!si.t) continue;
                if ((size_t) si.file_idx >= g_fds.size()) {
                    fprintf(stderr, "moestream: layer %d: shard index %u out of range (%zu files)\n",
                            kv.first, si.file_idx, g_fds.size());
                    ok = false; break;
                }
                struct stat st;
                if (fstat(g_fds[si.file_idx], &st) != 0) { ok = false; break; }
                const uint64_t end = si.file_offset +
                    (uint64_t) si.expert_bytes * (uint64_t) si.n_expert;
                if (end > (uint64_t) st.st_size) {
                    fprintf(stderr,
                        "moestream: layer %d role %d: past the end of shard %u "
                        "(needs %llu / actual size %llu)\n",
                        kv.first, r, si.file_idx,
                        (unsigned long long) end, (unsigned long long) st.st_size);
                    ok = false; break;
                }
            }
            if (!ok) break;
        }
        if (!ok) {
            fprintf(stderr, "moestream: expert offset validation failed; "
                            "disabling, as that is safer than reading wrong data\n");
            for (int fd : g_fds) if (fd >= 0) close(fd);
            g_fds.clear(); g_enabled = false; return;
        }
    }

    g_maxb = 0;
    for (auto & kv : g_layers)
        g_maxb = std::max({g_maxb, kv.second.gate.expert_bytes,
                           kv.second.up.expert_bytes, kv.second.down.expert_bytes});
    // The staging buffer is allocated lazily. On UMA zero-copy covers 100% of
    // reads, so it is never used, yet allocating it unconditionally wasted
    // 41 MiB on Ornith and 124 MiB on Laguna. Allocate it the first time a
    // non-zero-copy environment (a discrete GPU, say) actually needs it.
    g_stage.clear(); g_stage.shrink_to_fit();

    fprintf(stderr, "moestream: %u layers x %ld experts -> %u slots/layer (%.0f%%)\n",
            n_layer, (long) n_expert, g_slots,
            100.0 * double(g_slots) / double(n_expert));
    fprintf(stderr, "moestream: max micro-batch %d (needs <= %u slots)\n",
            g_max_ubatch, g_slots);

    // ---- prefill path: allocate the staging arena ----
    //   With the arena in place, prefill is no longer tied to a small ubatch.
    //   Anything above the largest token count the slab can safely serve is
    //   routed through this path.
    if (g_pf_threshold < 0) {
        // Find the largest n satisfying union(n) * 1.15 + top_k <= g_slots.
        // Any ubatch beyond that would exhaust the slab, so it must use the arena.
        const int    tk_m = gguf_topk();
        const double tk   = double(tk_m > 0 ? tk_m : (g_seen_topk > 0 ? g_seen_topk : 8));
        int n = 1;
        while (n < 4096) {
            const double u = double(n_expert) *
                (1.0 - std::pow(1.0 - tk / double(n_expert), double(n + 1)));
            if (u * 1.15 + tk > double(g_slots)) break;
            ++n;
        }
        g_pf_threshold = n;
    }
    if (const char * e = getenv("MOESTREAM_PREFILL_BUFS")) g_pf_nbuf = std::max(1, atoi(e));
    if (const char * e = getenv("MOESTREAM_PREFILL_ASYNC")) g_pf_async = atoi(e) != 0;

    if (g_pf_threshold > 0 && ggml_backend_vk_buffer_type && zerocopy_mode()) {
        // One arena holds one layer. Under UD quantization the type varies per
        // layer, so size it by the largest layer.
        size_t lay_max = 0;
        for (auto & kv : g_layers) {
            const SlabInfo * src[3] = { &kv.second.gate, &kv.second.up, &kv.second.down };
            size_t s = 0;
            for (int r = 0; r < 3; ++r)
                s += (size_t) src[r]->expert_bytes * (size_t) src[r]->n_expert;
            lay_max = std::max(lay_max, s);
        }
        g_arena_bytes = (lay_max + 4095) & ~(size_t) 4095;

        ggml_backend_buffer_type_t buft = ggml_backend_vk_buffer_type(0);
        bool ok = buft != nullptr && g_arena_bytes > 0;

        // Never import host memory here (Finding S7).
        //   amdgpu revalidates a BO imported via VK_EXT_external_memory_host on
        //   every command submission, so decode drops from 53.8 to
        //   1023 ms/token even if the buffer is never used.
        //   Let ggml-vulkan allocate normally instead. On UMA that yields
        //   DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT (heap 1), which can be
        //   pread into directly just like the zero-copy path, and which the GPU
        //   also reads faster than imported memory (heap 0).
        for (int i = 0; ok && i < g_pf_nbuf; ++i) {
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, g_arena_bytes);
            if (!b) {
                // If only the second (or later) arena fails, carry on with
                // fewer. Running synchronously beats giving up the arena.
                if (i >= 1) {
                    fprintf(stderr, "moestream: could not allocate arena %d; continuing with %d\n",
                            i + 1, i);
                    g_pf_nbuf = i;
                    break;
                }
                ok = false; break;
            }
            void * hp = ggml_backend_vk_buffer_host_ptr
                          ? ggml_backend_vk_buffer_host_ptr(b, 0) : nullptr;
            if (!hp) {   // non-UMA: cannot pread directly, so skip this path
                ggml_backend_buffer_free(b);
                fprintf(stderr, "moestream: the prefill arena requires a UMA device\n");
                ok = false; break;
            }
            g_arena_buf.push_back(b);
            g_arena.push_back(hp);
        }
        g_arena_layer.assign(g_pf_nbuf, -1);
        g_arena_have.assign(g_pf_nbuf, std::vector<uint8_t>());
        g_prev_union.assign(g_layers.rbegin()->first + 1, std::vector<uint8_t>());

        if (ok) {
            struct ggml_init_params ip = {
                ggml_tensor_overhead() * (g_layers.size() * 3 + 8), nullptr, true };
            g_pf_ctx = ggml_init(ip);
            ok = g_pf_ctx != nullptr;

            for (auto & kv : g_layers) {
                if (!ok) break;
                const int il = kv.first;
                const int bi = il % g_pf_nbuf;
                uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(g_arena_buf[bi]);
                const SlabInfo * src[3] = { &kv.second.gate, &kv.second.up, &kv.second.down };
                SlabInfo dst[3];
                uint64_t off = 0;
                for (int r = 0; r < 3 && ok; ++r) {
                    ggml_tensor * st_ = src[r]->t;
                    // ne[2] here is n_expert, not n_slot: the arena holds them all
                    ggml_tensor * t = ggml_new_tensor_3d(g_pf_ctx, st_->type,
                        st_->ne[0], st_->ne[1], src[r]->n_expert);
                    if (!t) { ok = false; break; }
                    if (ggml_backend_tensor_alloc(g_arena_buf[bi], t, base + off)
                            != GGML_STATUS_SUCCESS) { ok = false; break; }
                    dst[r] = *src[r];
                    dst[r].t = t;
                    dst[r].arena_off = off;
                    off += (uint64_t) src[r]->expert_bytes * (uint64_t) src[r]->n_expert;
                }
                if (!ok) break;
                LayerSlabs L; L.gate = dst[0]; L.up = dst[1]; L.down = dst[2]; L.ready = true;
                g_full[il] = L;
            }
            g_pf_ready = ok;
        }

        if (!g_pf_ready) {
            for (auto b : g_arena_buf) ggml_backend_buffer_free(b);
            g_arena_buf.clear(); g_arena.clear(); g_full.clear();
        }
        fprintf(stderr,
                "moestream: prefill arena %s (%d x %.0f MiB, threshold %d tokens, %s)\n",
                g_pf_ready ? "ENABLED" : "FAILED", g_pf_nbuf,
                g_arena_bytes / 1048576.0, g_pf_threshold,
                g_pf_nbuf > 1 ? "double-buffered" : "single");
        if (g_pf_ready)
            fprintf(stderr, "moestream: prefill async prefetch = %s\n",
                    (g_pf_async && g_pf_nbuf >= 2) ? "on" : "off");
    }
    // Whether drafting pays is a property of the machine and the model together,
    // not of Mixture-of-Experts as such. Measured here (S42): a MoE decode pass
    // costs 3.5x more at width 6 than at width 1, because the tokens in it want
    // different experts, while a dense pass is flat; and one drafted token costs
    // ~53% of a MoE forward pass against ~11% of a dense one. Both penalties land
    // on MoE. On a machine where a decode pass is far more expensive, both shrink
    // -- which is why this warns rather than refuses, and why SPEC_DECODING=learn
    // measures it instead of assuming.
    if (!g_layers.empty()) {
        const char * sd = getenv("SPEC_DECODING");
        if (sd && *sd && strcmp(sd, "learn") != 0)
            fprintf(stderr,
                "moestream: NOTE speculative decoding is pinned on, and this is a MoE model.\n"
                "moestream:      Measured here it loses: 48.6 to 51.1 ms/token at the best\n"
                "moestream:      setting found, and 65%% worse at llama.cpp's own defaults.\n"
                "moestream:      A MoE verification pass gets dearer as it widens, and the\n"
                "moestream:      draft head costs about half a forward pass (findings/S42).\n"
                "moestream:      On a bigger GPU the balance moves the other way, so\n"
                "moestream:      SPEC_DECODING=learn measures it on your machine instead --\n"
                "moestream:      with 'off' among the candidates.\n");
    }
    if (g_drop_from < (int) n_layer)
        fprintf(stderr, "moestream: TURBO enabled from layer %d "
                        "(misses are dropped, not fetched)\n", g_drop_from);

    // ---- decide whether prefetching is worthwhile ----
    {
        const uint64_t per = gguf_bytes_per_slot();
        g_expert_bytes_total = per * (uint64_t) n_expert;
        uint64_t memtotal_kb = 0;
        if (FILE * mi = fopen("/proc/meminfo", "r")) {
            char line[256];
            while (fgets(line, sizeof line, mi))
                if (sscanf(line, "MemTotal: %llu kB", (unsigned long long *) &memtotal_kb) == 1) break;
            fclose(mi);
        }
        g_ram_gib = double(memtotal_kb) * 1024.0 / 1073741824.0;
        const double slab_gib = double(g_slots) * double(per) / 1073741824.0;
        // Page cache headroom = RAM - slab - 2 GiB for KV and resident weights
        g_pc_gib = (g_ram_gib - slab_gib - 2.0) * 0.8;
        const double e_gib = double(g_expert_bytes_total) / 1073741824.0;

        const bool forced = g_prefetch >= 0;
        const bool useful = e_gib > g_pc_gib;
        if (!forced) g_prefetch = useful ? 1 : 0;

        if (g_expert_bytes_total == 0 || g_ram_gib <= 0) {
            g_prefetch = 0;
            fprintf(stderr, "moestream: [prefetch] undecidable (cannot read GGUF/RAM); disabled\n");
        } else {
            fprintf(stderr,
                "moestream: [prefetch] decision = %s%s\n"
                "moestream: [prefetch]   experts total %.1f GiB / page cache headroom %.1f GiB "
                "(80%% of RAM %.1f - slab %.1f - 2.0 reserved)\n",
                g_prefetch ? "on" : "off",
                forced ? " (forced via MOESTREAM_PREFETCH)" : "",
                e_gib, g_pc_gib, g_ram_gib, slab_gib);
            fprintf(stderr,
                "moestream: [prefetch]   note: decode prefetch was rejected (Finding S11); "
                "this decision selects the prefill arena's read strategy\n"
                "moestream: [prefetch]   prefill read strategy = %s\n",
                g_prefetch ? "union (read less)" : "read-all (never stall)");
        }
    }

    if (g_mrc_on) {
        g_mrc_last.assign(n_layer, std::vector<uint64_t>((size_t) n_expert, 0));
        g_mrc_clock.assign(n_layer, 0);
        g_mrc_hist.assign((size_t) n_expert, 0);
        g_mrc_hist_l.assign(n_layer, std::vector<uint64_t>((size_t) n_expert, 0));
        g_mrc_tot_l.assign(n_layer, 0);
        signal(SIGUSR1, mrc_sighandler);
        fprintf(stderr, "moestream: MRC measurement enabled (report on demand with "
                        "docker kill -s USR1 <container>)\n");
    }

    // Report on exit; without this there is no way to inspect the hit rate.
    {
        static bool hooked = false;
        if (!hooked) {
            hooked = true;
            atexit([] {
                pf_wait();
                // Record the ubatch result once, now that the run is over.
                // MOESTREAM_UBATCH is set only by the entrypoint's UBATCH=learn
                // path, so its presence is the signal to record -- this must not
                // depend on frac being learned too, or pinning frac to a number
                // would leave the ubatch search measuring the same candidate
                // forever.
                if (const char * e = getenv("MOESTREAM_UBATCH")) {
                    const int ub = atoi(e);
                    if (ub > 0) ubatch_store(ub, g_frac, prefill_rate());
                }
                report();
                for (auto * c : g_cache) delete c;    // plug the leak
                g_cache.clear();
                for (int fd : g_fds) if (fd >= 0) close(fd);
                g_fds.clear();
            });
        }
    }

}

static void ensure_zero_slot() {
    if (g_zero_filled || g_layers.empty()) return;
    g_zero_filled = true;
    // Check that the slab's ne[2] and g_zero_slot agree. Silent when correct;
    // the details are only printed under MOESTREAM_DEBUG or on failure.
    {
        const SlabInfo & g = g_layers.begin()->second.gate;
        const bool bad = (uint64_t) g_zero_slot * g.expert_bytes + g.expert_bytes
                         > ggml_nbytes(g.t);
        if (bad || debug_mode())
            fprintf(stderr, "moestream: [diag] slab ne[2]=%lld  g_slots=%u  g_zero_slot=%u  "
                            "expert_bytes=%llu  nbytes=%zu\n",
                    (long long) g.t->ne[2], g_slots, g_zero_slot,
                    (unsigned long long) g.expert_bytes, ggml_nbytes(g.t));
        if (bad)
            fprintf(stderr, "moestream: ERROR zero slot lies outside the tensor; "
                            "output will be corrupt\n");
    }
    std::vector<uint8_t> zeros(g_maxb, 0);
    for (auto & kv : g_layers) {
        const SlabInfo * rs[3] = { &kv.second.gate, &kv.second.up, &kv.second.down };
        for (int r = 0; r < 3; ++r) {
            if (!rs[r]->t || !rs[r]->t->buffer) continue;
            ggml_backend_tensor_set(rs[r]->t, zeros.data(),
                (size_t) g_zero_slot * rs[r]->expert_bytes, (size_t) rs[r]->expert_bytes);
        }
    }
    // Diagnostic: read back to confirm the slot really is zeroed
    {
        const SlabInfo & g = g_layers.begin()->second.gate;
        std::vector<uint8_t> rb(g.expert_bytes, 0xAA);
        ggml_backend_tensor_get(g.t, rb.data(),
            (size_t) g_zero_slot * g.expert_bytes, (size_t) g.expert_bytes);
        size_t nz = 0;
        for (uint8_t v : rb) if (v) nz++;
        if (nz)
            fprintf(stderr, "moestream: ERROR zero slot is not zeroed "
                            "(%zu / %zu bytes non-zero); output will be corrupt\n",
                    nz, rb.size());
        else if (debug_mode())
            fprintf(stderr, "moestream: [diag] zero-slot readback clean (%zu bytes)\n",
                    rb.size());
    }
}

// ---------------------------------------------------------------------------
// The remap itself, run inside the graph as a CPU custom op.
//   a   : router selection [n_expert_used, n_tokens] (I32, possibly a view)
//   dst : a contiguous tensor of the same shape; slot_ids are written here.
// ---------------------------------------------------------------------------
static void ms_remap_fn(ggml_tensor * dst, const ggml_tensor * a,
                        int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    const intptr_t packed = (intptr_t) ud;
    const int il   = (int) (packed & 0xFFFF);
    const int pass = (int) ((packed >> 16) & 0xFF);
    const int np   = (int) ((packed >> 24) & 0xFF);

    const int tk = (int) a->ne[0], nt = (int) a->ne[1];
    const char * src = (const char *) a->data;
    g_seen_topk = tk;
    int32_t * out = (int32_t *) dst->data;

    // Even when MoEStream is unavailable, leaving dst uninitialized would let
    // mul_mat_id index an invalid slot and corrupt the output. Write identity.
    auto passthrough = [&]() {
        for (int j = 0; j < nt; ++j)
            for (int k = 0; k < tk; ++k)
                memcpy(&out[(size_t) j * tk + k],
                       src + (size_t) j * a->nb[1] + (size_t) k * a->nb[0], sizeof(int32_t));
    };
    if (!enabled() || noop_level() == 1) { passthrough(); return; }
    if (g_cache.empty()) finalize(g_path.c_str());
    if (g_cache.empty() || il < 0 || il >= (int) g_cache.size()) { passthrough(); return; }
    auto it = g_layers.find(il);
    if (it == g_layers.end() || !it->second.ready) { passthrough(); return; }

    ensure_zero_slot();
    const double t0 = now_s();
    // Decode only: the gap since the previous layer's remap is the compute a
    // background prefetch for this layer could have hidden behind.
    double win = 0;
    if (nt == 1 && g_lay_prev_exit > 0) {
        win = t0 - g_lay_prev_exit;
        if (win > 0.5) win = 0;          // a request boundary, not a layer gap
    }

    // Release the references taken by this (layer, pass) last time.
    // Without this, refcounts accumulate and eviction stops entirely.
    const int hkey = il * 8 + pass;
    {
        // Release every pass of this layer, not just pass p.
        //   Passes execute in creation order and the previous pass's mul_mat_id
        //   has already finished, so this is safe. Releasing only pass p leaves
        //   experts held by pass p-1 occupying slots, which exhausts the slab on
        //   the next pass (measured: 86 of 97 slots in use).
        for (int q = 0; q < 8; ++q) {
            auto it2 = g_held.find(il * 8 + q);
            if (it2 == g_held.end()) continue;
            for (uint32_t e : it2->second) g_cache[il]->release(0, e);
            it2->second.clear();
        }
    }

    // ---- Expert Sweep: choose the expert subset this pass owns ----
    //   Sort the distinct experts in the batch, split them into P equal parts,
    //   and let this pass own part number `pass`. Experts outside the subset map
    //   to the zero slot and contribute nothing, so the sum over all passes
    //   reproduces the correct FFN output.
    static std::vector<uint8_t> in_pass;      // expert -> owned by this pass?
    if (np > 1) {
        const int64_t n_exp_total = it->second.gate.n_expert;
        static std::vector<uint8_t> seen;
        seen.assign((size_t) n_exp_total, 0);
        for (int j = 0; j < nt; ++j)
            for (int k = 0; k < tk; ++k) {
                int32_t e;
                memcpy(&e, src + (size_t) j * a->nb[1] + (size_t) k * a->nb[0], sizeof(int32_t));
                if (e >= 0 && e < n_exp_total) seen[(size_t) e] = 1;
            }
        std::vector<uint32_t> distinct;
        for (int64_t e = 0; e < n_exp_total; ++e) if (seen[(size_t) e]) distinct.push_back((uint32_t) e);
        in_pass.assign((size_t) n_exp_total, 0);
        // Diagnostic MOESTREAM_SWEEP_TEST=1:
        //   pass 0 owns every expert and the others own none, so
        //   y = FFN + 0 + ... and the result should be correct.
        //   If it breaks, the loop machinery itself (accumulation, dependencies,
        //   graph structure) is at fault.
        static int sweep_test = -1;
        if (sweep_test < 0) { const char * e2 = getenv("MOESTREAM_SWEEP_TEST"); sweep_test = e2 ? atoi(e2) : 0; }
        if (sweep_test == 2 || sweep_test == 4) {
            // Every pass owns every expert -> y = np * FFN.
            // If accumulation is correct, PPL should stay in a sane range.
            for (uint32_t e : distinct) in_pass[e] = 1;
        } else if (sweep_test == 1) {
            if (pass == 0) for (uint32_t e : distinct) in_pass[e] = 1;
        } else {
            const size_t per = (distinct.size() + np - 1) / (size_t) np;
            const size_t lo = std::min(distinct.size(), (size_t) pass * per);
            const size_t hi = std::min(distinct.size(), lo + per);
            for (size_t i = lo; i < hi; ++i) in_pass[distinct[i]] = 1;
        }
    }

    const double read_before = g_t_read;

    struct Pend { uint32_t e, slot; };
    static std::vector<Pend> pend;
    pend.clear();

    for (int j = 0; j < nt; ++j) {
        for (int k = 0; k < tk; ++k) {
            int32_t e;
            memcpy(&e, src + (size_t) j * a->nb[1] + (size_t) k * a->nb[0], sizeof(int32_t));
            int32_t sid = (int32_t) g_zero_slot;   // default: null expert (contributes 0)
            if (np > 1 && (e < 0 || e >= it->second.gate.n_expert || !in_pass[(size_t) e])) {
                out[(size_t) j * tk + k] = sid;      // not owned by this pass
                continue;
            }
            if (e >= 0 && e < it->second.gate.n_expert) {
                mrc_touch(il, (uint32_t) e);
                uint32_t slot = 0;
                const bool hit = g_cache[il]->acquire(0, (uint32_t) e, Origin::Demand, &slot);
                if (slot == 0xFFFFFFFFu) {
                    // Slot exhaustion. Writing a wrong slot corrupts the
                    // output, so fall back to the null expert (contributes 0).
                    g_exhausted++;
                    if (!g_warned_ub) {
                        g_warned_ub = true;
                        fprintf(stderr,
                            "moestream: WARNING slot exhausted at layer %d "
                            "(ubatch=%d x top_k=%d needs up to %d slots, have %u).\n"
                            "moestream:   Reduce -ub or raise MOESTREAM_MAX_UBATCH / "
                            "MOESTREAM_CACHE_FRAC. Output quality is degraded.\n",
                            il, nt, tk, nt * tk, g_slots);
                    }
                    out[(size_t) j * tk + k] = (int32_t) g_zero_slot;
                    continue;
                }
                {
                    g_demand++;
                    if (!hit && il >= g_drop_from) {
                        // Drop mode: point at the null expert without fetching.
                        // The slot is already reserved, so later accesses load
                        // it normally.
                        g_cache[il]->release(0, (uint32_t) e);
                        out[(size_t) j * tk + k] = (int32_t) g_zero_slot;
                        g_dropped++;
                        continue;
                    }
                    if (!hit) {
                        if (noop_level() != 3) pend.push_back({ (uint32_t) e, slot });
                        g_cache[il]->mark_resident(0, (uint32_t) e,
                            it->second.gate.expert_bytes + it->second.up.expert_bytes +
                            it->second.down.expert_bytes);
                    }
                    sid = (int32_t) slot;
                    g_held[hkey].push_back((uint32_t) e);
                }
            }
            out[(size_t) j * tk + k] = sid;
        }
    }

    // ---- read all misses in parallel ----
    if (!pend.empty()) {
        const SlabInfo * roles[3] = { &it->second.gate, &it->second.up, &it->second.down };
        // How many staging slots this expert (3 members) needs
        auto stage_need = [&](uint32_t slot) {
            size_t n = 0;
            for (int r = 0; r < 3; ++r) if (!slab_host_ptr(*roles[r], slot)) n++;
            return n;
        };
        // Allocate the staging buffer only when a non-zero-copy read appears
        if (g_stage.empty() && g_maxb) {
            bool need_stage = false;
            for (auto & pj : pend) if (stage_need(pj.slot)) { need_stage = true; break; }
            if (need_stage) {
                g_stage.resize(g_maxb * 3 * 32);
                fprintf(stderr, "moestream: zero-copy unavailable; allocating a %.0f MiB staging buffer\n",
                        g_stage.size() / 1048576.0);
            }
        }
        // cap bounds the number of *staging slots*, not the number of reads.
        //   On UMA, where zero-copy covers everything, no staging is needed and
        //   cap == 0. Gating the loop on `si + 3 <= cap` therefore left jobs
        //   permanently empty: not a single miss was read, yet the slot was
        //   recorded as resident -- garbage weights, collapsed output.
        //   Only the staging requirement may bound the batch.
        const size_t cap = g_maxb ? g_stage.size() / g_maxb : 0;
        static std::vector<ReadJob> jobs;

        for (size_t base = 0; base < pend.size(); ) {
            jobs.clear();
            size_t si = 0, p = base;
            for (; p < pend.size(); ++p) {
                const size_t need = stage_need(pend[p].slot);
                if (need && si + need > cap) break;   // out of staging: next batch
                for (int r = 0; r < 3; ++r) {
                    ReadJob j; j.si = roles[r]; j.expert = pend[p].e; j.slot = pend[p].slot;
                    // Read straight into GPU-visible memory when possible (§14.4)
                    uint8_t * hp = slab_host_ptr(*roles[r], pend[p].slot);
                    if (hp) { j.dst = hp; j.direct = true; }
                    else    { j.dst = g_stage.data() + (si++) * g_maxb; j.direct = false; }
                    jobs.push_back(j);
                }
            }
            if (jobs.empty()) {          // nothing readable: staging unavailable.
                static bool warned = false;  // never swallow this; it corrupts output
                if (!warned) { warned = true;
                    fprintf(stderr, "moestream: [BUG] discarded %zu pending reads "
                                    "(cap=%zu stage=%zu). Output will be corrupt\n",
                            pend.size() - base, cap, g_stage.size()); }
                break;
            }

            size_t rb = 0; for (auto & j : jobs) rb += (size_t) j.si->expert_bytes;
            double tt0 = 0;
            const int nth_use = io_threads_for_this_call(&tt0);
            const double r0 = now_s();
            run_reads_parallel(jobs, nth_use);
            const double rd = now_s() - r0;
            io_threads_record(rb, tt0);
            g_t_read += rd;
            if (nt == 1) g_dec_read += rd;      // track the decode share separately

            const double u0 = now_s();
            for (auto & j : jobs) {
                if (!j.direct)
                    ggml_backend_tensor_set(j.si->t, j.dst,
                                            (size_t) j.slot * j.si->expert_bytes,
                                            (size_t) j.si->expert_bytes);
                else g_zerocopy++;
                g_io_bytes += j.si->expert_bytes; g_io_calls++;
            }
            g_t_upload += now_s() - u0;
            base = p;
        }
    }

    // Diagnostic: dump the id distribution on the first call for layer 0
    if (il == 0 && np > 1 && getenv("MOESTREAM_ORDER_TRACE")) {
        static int no = 0;
        if (no++ < 8) fprintf(stderr, "moestream: [order] remap run  L0 pass=%d\n", pass);
    }
    if (il == 0 && np > 1) {
        static std::map<int,int> shown;
        if (shown[pass]++ < 1) {
            size_t nzero = 0, nreal = 0;
            for (int j = 0; j < nt; ++j)
                for (int k = 0; k < tk; ++k) {
                    const int32_t v = out[(size_t) j * tk + k];
                    if (v == (int32_t) g_zero_slot) nzero++; else nreal++;
                }
            fprintf(stderr, "moestream: [ids] L0 pass=%d/%d  real=%zu zero=%zu  "
                            "first8=[%d %d %d %d %d %d %d %d]\n",
                    pass, np, nreal, nzero,
                    out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
        }
    }

    {
        const double t_exit = now_s();
        if (nt == 1) {
            const double rd = g_t_read - read_before;
            if (win > 0) {
                g_lay_window += win; g_lay_read += rd; g_lay_n++;
                if (win > g_lay_win_max)  g_lay_win_max  = win;
                if (rd  > g_lay_read_max) g_lay_read_max = rd;
                if (rd > win) g_lay_starved++;
            }
            g_lay_prev_exit = t_exit;
        }
    }
    g_t_total += now_s() - t0;
    if (il == g_first_il) {
        g_tokens += nt;
        if (nt == 1) {                     // measure decode only
            pf_req_close();                // decode starting means prefill ended
            const double tnow = now_s();
            if (g_dec_last > 0) {
                const double d = tnow - g_dec_last;
                if (d < 5.0) g_dec_wall += d;   // gaps over 5 s are between requests
            }
            g_dec_last = tnow;
        }
        // Sample device memory while the model is live. At exit the buffers are
        // already released -- the reading there shows ~1 GiB used, so the memory
        // cap on the frac recommendation would pass anything and could record a
        // frac the next start cannot allocate. The first sample is taken early
        // (everything is allocated by then) so that even a short session, which
        // may never reach the 2000-token [stats] interval, still has one.
        {   static uint64_t next_mem = 32;
            if (g_tokens >= next_mem) {
                next_mem = g_tokens + 2000;
                double t = 0, u = 0;
                if (device_mem_gib(&t, &u) && u > g_dev_used_peak) g_dev_used_peak = u;
            }
        }
        if (g_mrc_req) { g_mrc_req = 0; mrc_report(); ub_report(); prefetch_verify(); }
        // Emit one line at intervals so the hit rate is visible while running.
        static uint64_t next_at = 2000;
        if (g_tokens >= next_at) {
            next_at = g_tokens + 2000;
            CacheStats st;
            for (auto * c : g_cache) { st.hit += c->stats().hit; st.miss += c->stats().miss; }
            fprintf(stderr, "moestream: [stats] tokens=%llu  hit rate %.2f%%  "
                            "(hit %llu / miss %llu)  arena loads=%llu\n",
                    (unsigned long long) g_tokens, st.hit_rate() * 100,
                    (unsigned long long) st.hit, (unsigned long long) st.miss,
                    (unsigned long long) g_pf_loads);
            static uint64_t next_mrc = 8000;
            if (g_tokens >= next_mrc) { next_mrc = g_tokens + 20000; mrc_report(); ub_report(); }
        }
    }
}

bool prefill_available() { return g_pf_ready; }

// Split into multiple passes when union(n_tokens) exceeds the slab capacity
int n_passes(int n_tokens) {
    // Expert Sweep is disabled by default.
    //   In llama.cpp's graph executor, buffers alias across passes and the
    //   output is corrupted (Finding N4). This was demonstrated by inserting an
    //   identity op, which shifted PPL by a factor of 2800.
    //   MOESTREAM_FORCE_PASSES re-enables it for experiments only.
    {
        static int allow = -1;
        if (allow < 0) { const char * e = getenv("MOESTREAM_FORCE_PASSES"); allow = e ? 1 : 0; }
        if (!allow) return 1;
    }
    // Do not call finalize() here. During graph construction it would write to
    // unallocated tensors and trip a GGML_ASSERT. g_slots and g_layers are
    // already fixed at model load time, which is all this needs.
    if (!enabled() || g_slots == 0 || g_layers.empty()) return 1;
    const int64_t n_expert = g_layers.begin()->second.gate.n_expert;
    if (n_expert <= 0) return 1;
    // Diagnostic: force a pass count (to isolate eviction effects)
    static int force = -1;
    if (force < 0) { const char * e = getenv("MOESTREAM_FORCE_PASSES"); force = e ? atoi(e) : 0; }
    if (force > 0 && n_tokens > 8) return force;
    // No split needed if the slab can hold every expert
    if ((int64_t) g_slots >= n_expert) return 1;
    const int    tk_m = gguf_topk();
    const double tk   = double(tk_m > 0 ? tk_m : (g_seen_topk > 0 ? g_seen_topk : 8));
    const double u  = double(n_expert) *
        (1.0 - std::pow(1.0 - tk / double(n_expert), double(n_tokens)));
    const int p = std::max(1, std::min((int) std::ceil((u * 1.10 + tk) / double(g_slots)), 8));
    static std::map<int,bool> shown;
    if (!shown[n_tokens]) {
        shown[n_tokens] = true;
        fprintf(stderr, "moestream: n_passes(n_tokens=%d) = %d  (union=%.0f, slots=%u)\n",
                n_tokens, p, u, g_slots);
    }
    return p;
}

ggml_tensor * prefill_exps(int il, int role, int n_tokens) {
    if (!enabled()) return nullptr;
    if (!g_pf_ready || n_tokens <= g_pf_threshold) { g_slab_used++; return nullptr; }
    auto it = g_full.find(il);
    if (it == g_full.end() || !it->second.ready) return nullptr;
    g_pf_used++;
    g_pf_ntok_seen = n_tokens;
    return role == 0 ? it->second.gate.t : (role == 1 ? it->second.up.t : it->second.down.t);
}

// ---------------------------------------------------------------------------
// Load every expert of layer `il` into the prefill arena.
//   Does nothing if it is already resident (only possible when the layer count
//   is at most the arena count). The arena is our own memory, so the host
//   pointer is known directly.
// ---------------------------------------------------------------------------
// If `need` is non-null, only the experts marked 1 there are read (union load);
//   null reads every expert.
//   Slots left unfilled still hold the previous layer's data, but ids never
//   reference them, so mul_mat_id never reads them. This is safe.
static void load_layer_arena(int il, const std::vector<uint8_t> * need) {
    if (!g_pf_ready) return;

    // Detect pass boundaries to separate I (I/O) from C (compute)
    if (g_pf_first_il < 0) g_pf_first_il = il;
    if (il == g_pf_first_il) {
        if (g_pass_t0 > 0 && g_pass_ntok > 0) {
            const double tot = now_s() - g_pass_t0;
            const double cmp = tot - g_pass_io;
            // Count only full micro-batches, and not the first one.
            //
            //   An interval is measured from one pass starting to the next pass
            //   starting. For the final, partial pass of a request that spans
            //   the gap until the next request arrives -- the token generation,
            //   the HTTP response, the next tokenization. Its token count is
            //   small while its measured time is large, so it drags the average
            //   down, and by an amount that depends on how big the remainder
            //   happens to be:
            //     13877 tokens at ub=1024 leaves 565  <- hurt most
            //                    at ub=2048 leaves 1589
            //                    at ub=8192 leaves 5685 <- hurt least
            //   That is precisely the bias observed: 1024 measured 10% low while
            //   2048 and 4096 measured correctly, which handed the decision to
            //   the wrong candidate. Skipping partial passes removes it.
            //
            //   The first pass of a process also runs slower (2.0-2.3% measured
            //   on Ornith). Its magnitude is model-dependent, and because it is
            //   a fixed warm-up cost it weighs less on a large UBATCH than on a
            //   small one -- so it too tilts the ranking, toward larger values.
            //   It is therefore recorded separately and subtracted later, but
            //   only once a second interval exists: at a large UBATCH a request
            //   may contain just one full pass, and discarding it would leave
            //   nothing to measure at all.
            // Exclude partial batches and passes that span an idle gap
            if (cmp > 0 && g_pass_io > 0 && tot < 600.0 && g_pass_ntok >= g_ub_ntok) {
                if (g_pass_ntok > g_ub_ntok) {      // a larger ubatch appeared: restart the aggregate
                    g_ub_io = g_ub_cmp = 0; g_ub_passes = 0; g_ub_ntok = g_pass_ntok;
                }
                g_ub_io += g_pass_io; g_ub_cmp += cmp; g_ub_passes++;
            }
        }
        g_pass_t0 = now_s(); g_pass_io = 0; g_pass_ntok = g_pf_ntok_seen;
        if (g_req_t0 == 0) g_req_t0 = g_pass_t0;   // first pass of this request
        g_req_tok += (uint64_t) g_pf_ntok_seen;
    }

    // The arena vectors are sized from the number of buffers that actually got
    // allocated, so bi is always in range -- but that invariant lives in
    // finalize(), several hundred lines away, and nothing here enforces it.
    // Reads write straight into GPU-visible memory, so a stale index would
    // corrupt weights rather than crash. Check it.
    if (g_pf_nbuf <= 0) return;
    const int bi = il % g_pf_nbuf;
    if (bi < 0 || (size_t) bi >= g_arena.size() || !g_arena[bi] ||
        (size_t) bi >= g_arena_layer.size() || (size_t) bi >= g_arena_have.size()) {
        static bool warned = false;
        if (!warned) { warned = true;
            fprintf(stderr, "moestream: [BUG] arena index %d out of range "
                            "(%zu buffers); prefill disabled\n", bi, g_arena.size()); }
        g_pf_ready = false;
        return;
    }

    auto fit = g_full.find(il);
    auto sit = g_layers.find(il);
    if (fit == g_full.end() || sit == g_layers.end()) return;
    const int64_t NE = sit->second.gate.n_expert;

    // Reset the resident set when the arena switches layers
    if (g_arena_layer[bi] != il) {
        g_arena_have[bi].assign((size_t) NE, 0);
        g_arena_layer[bi] = il;
    }

    const double t0 = now_s();
    const SlabInfo * src[3] = { &sit->second.gate, &sit->second.up, &sit->second.down };
    const SlabInfo * dst[3] = { &fit->second.gate, &fit->second.up, &fit->second.down };
    auto & have = g_arena_have[bi];

    std::vector<ReadJob> jobs;
    size_t bytes = 0; uint64_t n_read = 0;
    for (int64_t e = 0; e < NE; ++e) {
        if (need && !(*need)[(size_t) e]) continue;      // unused by this batch
        if (have[(size_t) e]) continue;                  // already resident
        for (int r = 0; r < 3; ++r) {
            uint8_t * hp = (uint8_t *) g_arena[bi] + dst[r]->arena_off;
            ReadJob j;
            j.si     = src[r];
            j.expert = (uint32_t) e;
            j.slot   = 0;
            j.dst    = hp + (size_t) e * (size_t) src[r]->expert_bytes;
            j.direct = true;                 // the arena is GPU-visible
            jobs.push_back(j);
            bytes += (size_t) src[r]->expert_bytes;
        }
        have[(size_t) e] = 1; n_read++;
    }
    // The denominator is what would have been read without the union, i.e. all
    // experts. Using the union's own size makes this always 100% and hides the
    // saving entirely.
    g_pf_experts_asked += (uint64_t) NE;
    g_pf_experts_read += n_read;
    if (jobs.empty()) {
        // Everything is already loaded. The time is zero, but the pass
        // accounting must continue: returning early here would make [ub]
        // underestimate the real I/O cost.
        g_pf_hits++;
        g_arena_layer[bi] = il;
        return;
    }
    run_reads_parallel(jobs, g_nthreads);
    g_pf_loads++;
    g_pf_bytes += bytes;
    const double dt = now_s() - t0;
    g_t_pfload += dt;
    g_pass_io  += dt;
}

// Wait for an in-flight prefetch. join() is a full barrier, so g_arena_* is
// safe to read afterwards.
static void pf_wait() {
    if (g_pf_thread.joinable()) g_pf_thread.join();
    g_pf_inflight = -1;
}

// Load layer `il` in the background.
//   The ids are not known yet, so the union of the same layer in the previous
//   pass is used as a hint. Measured (Ornith/Laguna): 177 experts needed now
//   against 178 previously, with a shortfall of only 8-12.
//   Whatever is missing is fetched synchronously once the ids arrive, so a bad
//   guess never changes the output.
//   With no previous pass (the first one), every expert is read.
static void pf_kick(int il) {
    if (!g_pf_async || g_pf_nbuf < 2) return;
    if (g_layers.find(il) == g_layers.end()) return;
    pf_wait();                       // never more than one thread
    // Which strategy wins flips between models (measured):
    //   Ornith (experts 14.5 GiB <= page cache): read-all is fastest (+9.9%)
    //     reads are cheap (9 GB/s), so stalling to fetch the shortfall costs
    //     more than reading everything
    //   Laguna (experts 50.7 GiB > page cache): union is fastest (+71.2%)
    //     reads hit the real SSD (3.6 GB/s), so reading less beats the stall
    //   The [prefetch] decision (expert total vs page-cache headroom) selects it.
    std::vector<uint8_t> hint;
    if (g_prefetch && (size_t) il < g_prev_union.size() && !g_prev_union[il].empty())
        hint = g_prev_union[il];   // I/O is expensive here: read less
    g_pf_inflight = il;
    g_pf_thread = std::thread([il, hint = std::move(hint)]() mutable {
        load_layer_arena(il, hint.empty() ? nullptr : &hint);
    });
}

// Load op for the prefill path; the ids pass straight through.
//   Using this node as mul_mat_id's ids enforces the ordering
//   "load completes -> mul_mat_id runs".
//   The input (ids) lives on Vulkan, so the scheduler synchronizes the GPU
//   before the CPU op. This is the same mechanism as the decode-side remap.
// ---------------------------------------------------------------------------
// Dense FFN arena
// ---------------------------------------------------------------------------
//   One arena per buffer, each holding a single layer's gate+up+down. Buffer
//   il % nbuf holds layer il, so with nbuf=2 the layer being computed and the
//   layer being prefetched never share a buffer.
static size_t dense_read_range(int fd, uint64_t off, uint8_t * dst, size_t bytes, int nth) {
    if (nth <= 1) {
        size_t left = bytes, o = 0;
        while (left) {
            ssize_t r = pread(fd, dst + o, left, (off_t) (off + o));
            if (r <= 0) break;
            o += (size_t) r; left -= (size_t) r;
        }
        return o;
    }
    // Large contiguous reads split evenly; this is a very different shape from
    // the scattered 1.4 MiB expert reads run_reads_parallel is tuned for.
    const size_t chunk = (bytes + (size_t) nth - 1) / (size_t) nth;
    std::atomic<size_t> total{0};
    std::vector<std::thread> th; th.reserve((size_t) nth);
    for (int i = 0; i < nth; ++i) {
        const size_t beg = (size_t) i * chunk;
        if (beg >= bytes) break;
        const size_t len = std::min(chunk, bytes - beg);
        th.emplace_back([fd, off, dst, beg, len, &total] {
            size_t left = len, o = 0;
            while (left) {
                ssize_t r = pread(fd, dst + beg + o, left, (off_t) (off + beg + o));
                if (r <= 0) break;
                o += (size_t) r; left -= (size_t) r;
            }
            total += o;
        });
    }
    for (auto & t : th) t.join();
    return total.load();
}

// Fill the arena for layer il. Returns false if it was already resident.
static int  dn_threads_for_this_load();
static void dn_threads_record(size_t bytes, double dt);

static bool dense_load_layer(int il) {
    if (!g_dn_ready) return false;
    const size_t bi = (size_t) (il % g_dn_nbuf);
    if (g_dn_resident[bi] == il) { g_dn_hits++; return false; }
    auto mit = g_dn_meta.find(il);
    auto vit = g_dn_view.find(il);
    if (mit == g_dn_meta.end() || vit == g_dn_view.end()) return false;
    const double t0 = now_s();
    if (bi >= g_dn_host.size() || !g_dn_host[bi]) return false;
    const int nth = dn_threads_for_this_load();
    size_t layer_bytes = 0;
    for (size_t r = 0; r < mit->second.w.size() && r < vit->second.w.size(); ++r) {
        const DenseInfo & m = mit->second.w[r];
        const DenseInfo & v = vit->second.w[r];
        if (!m.valid || !v.t) continue;
        const size_t fi = (size_t) m.file_idx;
        if (fi >= g_fds.size() || g_fds[fi] < 0) {
            static bool warned = false;
            if (!warned) { warned = true;
                fprintf(stderr, "moestream: [dense] [BUG] no fd for shard %zu; "
                                "weights are garbage\n", fi); }
            continue;
        }
        uint8_t * hp = (uint8_t *) g_dn_host[bi] + v.arena_off;
        const size_t got = dense_read_range(g_fds[fi], m.file_offset, hp,
                                            (size_t) m.bytes, nth);
        layer_bytes += (size_t) m.bytes;
        if (got != (size_t) m.bytes) {
            static bool warned = false;
            if (!warned) { warned = true;
                fprintf(stderr, "moestream: [dense] [BUG] short read on layer %d slot %zu "
                                "(%zu of %llu bytes); output will be wrong\n",
                        il, r, got, (unsigned long long) m.bytes); }
        }
        g_dn_read_bytes += m.bytes;
    }
    g_dn_resident[bi] = il;
    g_dn_loads++;
    const double dt = now_s() - t0;
    g_dn_t_read += dt;
    dn_threads_record(layer_bytes, dt);
    return true;
}

// ---- dense read threads ----
//   Fixed at 4, and deliberately not auto-tuned.
//
//   A bandwidth-based tuner was built here, in the shape of the [io] tuner
//   above, and it settled on 8 threads at 15.51 GB/s against 14.30 at 4. It was
//   wrong. Measured end to end (finding S29), 4 threads decode in 656.8 ms and
//   8 in 674.4, and with the tuner active decode came out at 685.3 against the
//   676 the fixed value gives.
//
//   The reason is already in RESULTS.md 10.12: time spent inside the read call
//   and the cost the read imposes are different quantities, and they disagreed
//   by 1.6x on the expert path for the same reason. More reader threads finish
//   the read sooner and leave the compute that follows contending for memory
//   bandwidth. Optimising the first makes the second worse.
//
//   Auto-tuning this properly needs decode wall time as the objective, not read
//   bandwidth. Until something measures that, a value from an end-to-end
//   measurement beats a tuner optimising the wrong thing.
static int dn_threads_for_this_load() { return g_dn_th; }
static void dn_threads_record(size_t, double) {}

static void dn_wait() {
    if (g_dn_thread.joinable()) g_dn_thread.join();
    g_dn_inflight = -1;
}

//   Nothing to predict: layer il+1 is always layer il+1. This is the prefetch
//   that could never work for MoE (findings N2/S10/S11, §10.14), because there
//   the ids do not exist until the previous layer has run.
static void dn_kick(int il) {
    if (!g_dn_ready || g_dn_nbuf < 2) return;
    if (!dense_streamed_layer(il) || il >= g_dn_nlayer) return;
    dn_wait();
    g_dn_inflight = il;
    g_dn_thread = std::thread([il] { dense_load_layer(il); });
}

static void ms_dnload_fn(ggml_tensor * dst, const ggml_tensor * a,
                         int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    const int il = (int) ((intptr_t) ud & 0xFFFF);
    if (g_dn_ready) {
        const bool was_mine = (g_dn_inflight == il);
        dn_wait();
        dense_load_layer(il);              // no-op if the prefetch already did it
        if (was_mine) g_dn_pf_hit++; else g_dn_pf_miss++;
        dn_kick(il + 1);
    }
    // Pass the hidden state through unchanged; the FFN consuming it is what
    // orders "arena filled -> mul_mat runs".
    if (!a->data || !dst->data) return;
    if (ggml_is_contiguous(a) && ggml_is_contiguous(dst)) {
        memcpy(dst->data, a->data, ggml_nbytes(a));
    } else {
        const size_t rb = (size_t) a->ne[0] * ggml_type_size(a->type) / ggml_blck_size(a->type);
        for (int64_t i3 = 0; i3 < a->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < a->ne[1]; ++i1)
            memcpy((char *) dst->data + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1],
                   (const char *) a->data + i3*a->nb[3] + i2*a->nb[2] + i1*a->nb[1], rb);
    }
}

static void ms_pfload_fn(ggml_tensor * dst, const ggml_tensor * a,
                         int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    const int il = (int) ((intptr_t) ud & 0xFFFF);

    // `a` may be a view: ne can match while nb is non-contiguous.
    //   A flat memcpy would hand mul_mat_id the wrong expert_ids and leave the
    //   output plausible-looking but different on every run (Finding S7).
    //   Copy with strides, exactly as the passthrough in ms_remap_fn does.
    const int tk = (int) a->ne[0], nt = (int) a->ne[1];
    const char * src = (const char *) a->data;
    int32_t * out = (int32_t *) dst->data;
    if (!src || !out) return;
    g_seen_topk = tk;

    // Collect only the experts this batch actually references (union load).
    //   Reading every expert would incur a full model's worth of I/O even for a
    //   tiny batch.
    auto sit = g_layers.find(il);
    const int64_t NE = (sit != g_layers.end()) ? sit->second.gate.n_expert : 0;
    const bool use_union = NE > 0;
    if (use_union) g_need.assign((size_t) NE, 0);

    for (int j = 0; j < nt; ++j) {
        for (int k = 0; k < tk; ++k) {
            int32_t e;
            memcpy(&e, src + (size_t) j * a->nb[1] + (size_t) k * a->nb[0], sizeof(int32_t));
            out[(size_t) j * tk + k] = e;
            if (use_union) {
                if (e >= 0 && e < NE) g_need[(size_t) e] = 1;
                else {
                    // Out-of-range id: it is not in the union, so it would be
                    // referenced without being loaded. This hole predates the
                    // union change, but warn once rather than pass silently.
                    static bool warned = false;
                    if (!warned) { warned = true;
                        fprintf(stderr, "moestream: WARNING expert_id %d out of range (n_expert=%lld)\n",
                                (int) e, (long long) NE); }
                }
            }
        }
    }
    // ---- measurement: how much of this pass the previous union covers ----
    if (use_union && il >= 0 && (size_t) il < g_prev_union.size() && nt >= 256) {
        auto & pv = g_prev_union[il];
        if (pv.size() == (size_t) NE) {
            uint64_t nd = 0, pr = 0, sh = 0, to = 0;
            for (int64_t e = 0; e < NE; ++e) {
                const bool n_ = g_need[(size_t) e] != 0, p_ = pv[(size_t) e] != 0;
                if (n_) nd++;
                if (p_) pr++;
                if (n_ && !p_) sh++;
                if (n_ || p_) to++;
            }
            g_u_need += nd; g_u_prev += pr; g_u_short += sh; g_u_total += to; g_u_n++;
        }
        pv.assign(g_need.begin(), g_need.end());
    }

    if (g_pf_async && g_pf_nbuf >= 2) {
        const bool was_mine = (g_pf_inflight == il);
        pf_wait();                                   // wait for the background load
        // The prefetch used the previous pass's union and may fall short. Now
        // that the ids are known, whatever is missing is read synchronously.
        // `have` prevents re-reading what is already resident, so a wrong guess
        // never changes the output.
        load_layer_arena(il, use_union ? &g_need : nullptr);
        if (was_mine) g_pf_async_hit++; else g_pf_async_miss++;
        if (nt >= 256) pf_kick(il + 1);              // prefetch the next layer
    } else {
        load_layer_arena(il, use_union ? &g_need : nullptr);
    }
}

// Pack (layer, pass, total passes) into the userdata pointer
static inline intptr_t pack_ud(int il, int pass, int np) {
    return (intptr_t) ((il & 0xFFFF) | ((pass & 0xFF) << 16) | ((np & 0xFF) << 24));
}

ggml_tensor * build_remap(ggml_context * ctx, ggml_tensor * ids, int il) {
    if (!enabled()) return ids;
    return ggml_map_custom1(ctx, ids, ms_remap_fn, 1, (void *) pack_ud(il, 0, 1));
}

// Diagnostic: the remap, plus a forced read of the router input.
//   The second input is never used for anything. Touching every byte forces the
//   scheduler to bring it to the CPU, which is exactly the transfer a
//   layer-lookahead predictor would need. Comparing decode with and without it
//   gives that predictor's synchronization cost without implementing it.
static void ms_probe_fn(ggml_tensor * dst, const ggml_tensor * a,
                        const ggml_tensor * b, int ith, int nth, void * ud) {
    ms_remap_fn(dst, a, ith, nth, ud);
    if (ith != 0 || !b || !b->data) return;
    // Sum the bytes into a sink the compiler cannot discard. Reading it back
    // keeps the whole loop observable; writing alone left it as a dead store
    // that a future compiler would be free to remove, silently turning this
    // diagnostic into a measurement of nothing.
    static volatile double sink = 0;
    const float * h = (const float *) b->data;
    const size_t n = ggml_nelements(b);
    double acc = 0;
    for (size_t i = 0; i < n; ++i) acc += (double) h[i];
    sink = acc;
    (void) sink;
}

static bool probe_hidden() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("MOESTREAM_PROBE_HIDDEN"); v = e && atoi(e) ? 1 : 0; }
    return v != 0;
}

ggml_tensor * build_remap2(ggml_context * ctx, ggml_tensor * ids,
                           ggml_tensor * hidden, int il) {
    if (!enabled()) return ids;
    if (!probe_hidden() || !hidden || hidden->type != GGML_TYPE_F32)
        return build_remap(ctx, ids, il);
    static bool shown = false;
    if (!shown) { shown = true;
        fprintf(stderr, "moestream: [probe] forcing the router input to the CPU each layer "
                        "(%lld floats) -- diagnostic only, measures P2's sync cost\n",
                (long long) ggml_nelements(hidden)); }
    return ggml_map_custom2(ctx, ids, hidden, ms_probe_fn, 1, (void *) pack_ud(il, 0, 1));
}

// Returns the arena-backed weight for a streamed dense layer, or nullptr when
// this layer is resident (or dense streaming is off), in which case the caller
// keeps llama.cpp's own tensor and nothing changes.
// The dense arena has to exist before the FIRST graph is built, because
// build_ffn substitutes its tensors there. The expert path can initialise
// lazily from remap_exec at execution time; dense has no such op, and
// llama.cpp calls graph_reserve() during model load, so waiting would hand
// mul_mat the one-row stub (GGML_ASSERT ggml_can_mul_mat).
//
// The warning against running finalize() during graph construction is about
// writing to model tensors that are not allocated yet (the zero-slot fill).
// finalize_dense touches none: it opens fds and allocates its own buffers and
// its own tensors in its own context.
static void dense_ensure() {
    if (!g_dn_done && enabled() && g_dn_seen) finalize_dense(g_path.c_str());
}

// Look a weight up by the pointer llama.cpp handed the graph. Returns the
// arena-backed replacement, or nullptr when this weight is not streamed.
ggml_tensor * dense_ffn(int il, int role) {
    dense_ensure();
    if (!enabled() || !g_dn_ready) return nullptr;
    if (role < 0 || role > 2) return nullptr;
    auto it = g_dn_view.find(il);
    if (it == g_dn_view.end() || !it->second.ready) return nullptr;
    // w is filled in load order, which for the FFN-only claim is exactly the
    // order the three roles appear in the file. Match on the recorded role.
    for (const auto & e : it->second.w) if (e.valid && e.role == role) return e.t;
    return nullptr;
}

bool dense_active() { dense_ensure(); return enabled() && g_dn_ready; }

// One load op per layer, not per weight. Graph construction walks layers in
// order, so emitting only on the first weight of a new layer gives exactly one
// CPU<->GPU handoff per streamed layer instead of one per matrix.
ggml_tensor * build_dense_load(ggml_context * ctx, ggml_tensor * cur, int il) {
    if (!enabled() || !g_dn_ready) return cur;
    static int last_il = -1;
    static const ggml_context * last_ctx = nullptr;
    if (ctx == last_ctx && il == last_il) return cur;
    last_ctx = ctx; last_il = il;
    return ggml_map_custom1(ctx, cur, ms_dnload_fn, 1, (void *) pack_ud(il, 0, 1));
}

ggml_tensor * build_prefill_load(ggml_context * ctx, ggml_tensor * ids, int il) {
    if (!enabled() || !g_pf_ready) return ids;
    return ggml_map_custom1(ctx, ids, ms_pfload_fn, 1, (void *) pack_ud(il, 0, 1));
}

// Two-input remap. The second input (dep) is never read, but it creates a
// graph dependency that forces the ordering "previous pass finishes computing
// -> next pass overwrites the slab". Without it the scheduler runs passes
// concurrently and the next pass clobbers a slab the previous one is reading.
static void ms_remap_fn2(ggml_tensor * dst, const ggml_tensor * a,
                         const ggml_tensor * b, int ith, int nth, void * ud) {
    (void) b;
    ms_remap_fn(dst, a, ith, nth, ud);
}

ggml_tensor * build_remap_pass(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * dep,
                               int il, int pass, int np) {
    if (!enabled()) return ids;
    if (!dep) return ggml_map_custom1(ctx, ids, ms_remap_fn, 1, (void *) pack_ud(il, pass, np));
    // One element of dep is enough to establish the dependency (minimal transfer)
    ggml_tensor * tiny = ggml_view_1d(ctx, dep, 1, 0);
    return ggml_map_custom2(ctx, ids, tiny, ms_remap_fn2, 1, (void *) pack_ud(il, pass, np));
}

// ---------------------------------------------------------------------------
static void ms_marker_fn(ggml_tensor * dst, const ggml_tensor * a,
                         int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    // Identity copy (dst and a share shape and type)
    memcpy(dst->data, a->data, ggml_nbytes(a));
    const intptr_t p = (intptr_t) ud;
    const int il = (int)(p & 0xFFFF), pass = (int)((p >> 16) & 0xFF);
    if (il == g_first_il) {
        static int n = 0;
        if (n++ < 8) fprintf(stderr, "moestream: [order] FFN done   L0 pass=%d\n", pass);
    }
}

ggml_tensor * build_marker(ggml_context * ctx, ggml_tensor * t, int il, int pass) {
    if (!getenv("MOESTREAM_ORDER_TRACE")) return t;
    return ggml_map_custom1(ctx, t, ms_marker_fn, 1,
                            (void *)(intptr_t)((il & 0xFFFF) | ((pass & 0xFF) << 16)));
}

// Record a single access. O(n_expert), but only about 0.3% of a token.
static inline void mrc_touch(int il, uint32_t e) {
    if (!g_mrc_on || il < 0 || il >= (int) g_mrc_last.size()) return;
    auto & last = g_mrc_last[il];
    if ((size_t) e >= last.size()) return;
    g_mrc_total++;
    if ((size_t) il < g_mrc_tot_l.size()) g_mrc_tot_l[il]++;
    const uint64_t t = last[e];
    if (t == 0) {
        g_mrc_cold++;                       // a first access misses at any slot count
    } else {
        size_t r = 0;
        for (uint64_t v : last) if (v > t) ++r;   // reuse distance = experts touched more recently
        if (r < g_mrc_hist.size()) {
            g_mrc_hist[r]++;
            if ((size_t) il < g_mrc_hist_l.size()) g_mrc_hist_l[il][r]++;
        }
    }
    last[e] = ++g_mrc_clock[il];
}

// Turn measured reuse distances into slots -> hit rate -> cost, and recommend
// a slot count.
static void mrc_report() {
    if (!g_mrc_on || g_mrc_hist.empty() || g_layers.empty()) return;
    if (g_mrc_total < 200000) {
        fprintf(stderr, "moestream: [mrc] not enough samples (%llu). "
                        "Keep using it normally; a few thousand tokens suffice\n",
                (unsigned long long) g_mrc_total);
        return;
    }
    const int64_t  ne  = g_layers.begin()->second.gate.n_expert;
    const uint64_t per = gguf_bytes_per_slot();          // bytes per slot, summed over layers
    const int      tk  = g_seen_topk > 0 ? g_seen_topk : 8;
    if (per == 0) return;

    // Cumulative sum -> hit[C] = hit rate at C slots
    std::vector<double> hit((size_t) ne + 1, 0.0);
    uint64_t acc = 0;
    for (size_t c = 0; c < g_mrc_hist.size() && c < (size_t) ne; ++c) {
        acc += g_mrc_hist[c];
        hit[c + 1] = double(acc) / double(g_mrc_total);
    }
    for (size_t c = g_mrc_hist.size() + 1; c <= (size_t) ne; ++c) hit[c] = hit[c - 1];

    // Measured bandwidth; fall back to the Finding S2 value if unavailable.
    const double bw = (g_t_read > 0 && g_io_bytes > 0) ? double(g_io_bytes) / g_t_read : 4.48e9;
    const double b_act = double(tk) * double(per);       // bytes touched per token

    static const double fr[] = { 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.75,1.00 };
    const int nf = (int) (sizeof(fr) / sizeof(fr[0]));

    fprintf(stderr,
        "moestream: [mrc] slots -> hit rate on the live workload (from %llu reuse distances)\n"
        "moestream: [mrc]   slots   frac   hit rate   memory   marginal   I/O upper\n",
        (unsigned long long) g_mrc_total);

    int rec = -1;
    for (int i = nf - 1; i >= 0; --i) {
        const int c = std::max(1, (int) (fr[i] * double(ne) + 0.5));
        const double mem_gib = double(c) * double(per) / 1073741824.0;
        // Upper bound on I/O time: it assumes zero overlap, so reality is lower.
        //   In practice more slots (fewer misses) means more of it is hidden;
        //   at 195 slots, 93% was hidden. Do not use this column to decide --
        //   it is only an order-of-magnitude hint.
        const double ms_ub = (1.0 - hit[c]) * b_act / bw * 1000.0;
        // This is the column to decide on: hit-rate points lost per GiB saved
        double marg = 0.0;
        if (i > 0) {
            const int c2 = std::max(1, (int) (fr[i-1] * double(ne) + 0.5));
            const double dm = (double(c) - double(c2)) * double(per) / 1073741824.0;
            const double dh = (hit[c] - hit[c2]) * 100.0;
            if (dm > 0) marg = dh / dm;
        }
        const bool cur = (c == (int) g_slots);
        fprintf(stderr, "moestream: [mrc]  %5d  %5.0f%%   %6.2f%%  %6.2f GiB  %6.2f pt/GiB  (<%.0f ms)%s\n",
                c, fr[i] * 100, hit[c] * 100, mem_gib, marg, ms_ub, cur ? "  <- current" : "");
        if (rec < 0 && marg > 0 && marg > g_pt_per_gib) rec = c;
    }
    // ---- how much would optimizing the per-layer allocation gain? ----
    //   Slots are currently split evenly across layers (slab_slots ignores the
    //   layer index). If layers differ in how skewed their expert usage is,
    //   an even split should not be optimal.
    //
    //   A naive greedy allocation does not work here. Reuse-distance histograms
    //   are not monotonically decreasing, so marginal gains are non-monotonic
    //   and greedy loses optimality -- it actually produced an answer worse than
    //   the even split. Take the upper concave hull of each layer's cumulative
    //   curve first, then allocate.
    if (!g_mrc_hist_l.empty()) {
        const size_t NL = g_mrc_hist_l.size();
        uint64_t hit_uniform = 0, tot_all = 0;
        for (size_t l = 0; l < NL; ++l) {
            tot_all += g_mrc_tot_l[l];
            for (size_t c = 0; c < (size_t) g_slots && c < g_mrc_hist_l[l].size(); ++c)
                hit_uniform += g_mrc_hist_l[l][c];
        }
        if (tot_all > 1000) {
            // Build each layer's cumulative V_l(c) and the slope sequence of
            // its upper concave hull
            struct AllocSeg { double slope; size_t n; size_t layer; };   // named to avoid clashing with ExpertCache's Seg
            std::vector<AllocSeg> segs;
            std::vector<size_t> maxc(NL, 0);
            for (size_t l = 0; l < NL; ++l) {
                const auto & h = g_mrc_hist_l[l];
                const size_t n = h.size();
                maxc[l] = n;
                std::vector<double> V(n + 1, 0.0);
                for (size_t c = 0; c < n; ++c) V[c + 1] = V[c] + double(h[c]);
                // Upper concave hull (monotone chain)
                std::vector<size_t> hull; hull.push_back(0);
                for (size_t c = 1; c <= n; ++c) {
                    while (hull.size() >= 2) {
                        const size_t a = hull[hull.size() - 2], b = hull.back();
                        const double s1 = (V[b] - V[a]) / double(b - a);
                        const double s2 = (V[c] - V[b]) / double(c - b);
                        if (s2 >= s1) hull.pop_back(); else break;
                    }
                    hull.push_back(c);
                }
                for (size_t k = 1; k < hull.size(); ++k) {
                    const size_t a = hull[k - 1], b = hull[k];
                    const double sl = (V[b] - V[a]) / double(b - a);
                    if (sl > 0) segs.push_back({sl, b - a, l});
                }
            }
            std::sort(segs.begin(), segs.end(),
                      [](const AllocSeg & a, const AllocSeg & b) { return a.slope > b.slope; });
            const size_t budget = (size_t) g_slots * NL;
            std::vector<size_t> cur(NL, 0);
            double hit_opt = 0; size_t used = 0;
            for (const auto & sg : segs) {
                if (used >= budget) break;
                const size_t take = std::min(sg.n, budget - used);
                hit_opt += sg.slope * double(take);
                cur[sg.layer] += take; used += take;
            }
            size_t mn = SIZE_MAX, mx = 0;
            for (size_t l = 0; l < NL; ++l) {
                if (g_mrc_tot_l[l] == 0) continue;
                mn = std::min(mn, cur[l]); mx = std::max(mx, cur[l]);
            }
            const double hu = 100.0 * double(hit_uniform) / double(tot_all);
            const double ho = 100.0 * hit_opt / double(tot_all);
            fprintf(stderr,
                "moestream: [mrc] per-layer allocation (total slots fixed at %zu):\n"
                "moestream: [mrc]   even split (current)  hit rate %.2f%%\n"
                "moestream: [mrc]   optimal upper bound   hit rate %.2f%%  (%+.2f pt)\n"
                "moestream: [mrc]   optimal slot counts   min %zu / max %zu (even = %u)\n",
                budget, hu, ho, ho - hu, mn == SIZE_MAX ? 0 : mn, mx, g_slots);
            if (ho - hu < 1.0)
                fprintf(stderr, "moestream: [mrc]   -> the even split is near optimal; not worth building\n");
        }
    }

    if (rec > 0) {
        // ---- cap the recommendation by what actually fits in device memory ----
        //   pt/GiB alone will happily suggest a slot count that does not start.
        //   Reserve headroom so the machine can still host anything else; this
        //   project's value is coexistence, not filling the device.
        double dev_total = 0, dev_used = 0;
        int    fits = 0;
        bool   capped = false;
        const int rec_ideal = rec;
        if (device_mem_gib(&dev_total, &dev_used) && per > 0) {
            double reserve = 1.0;
            if (const char * r = getenv("MOESTREAM_MEM_RESERVE_GIB")) reserve = atof(r);
            const double per_slot_gib = double(per) / 1073741824.0;
            const double slab_now     = double(g_slots) * per_slot_gib;
            // Everything that is not our slab: weights, KV, the arena, and any
            // other process sharing the device. Treating that as fixed is the
            // conservative choice.
            // Everything that is not our slab. Clamp at zero: sysfs can report
            // less in use than the slab we believe we hold (different accounting,
            // or a transient read), and a negative value here would inflate the
            // budget -- producing exactly the too-large recommendation this cap
            // exists to prevent.
            if (g_dev_used_peak > dev_used) dev_used = g_dev_used_peak;
            double non_slab = dev_used - slab_now;
            if (non_slab < 0) non_slab = 0;
            const double budget = dev_total - reserve - non_slab;
            fits = budget > 0 ? (int) (budget / per_slot_gib) : 0;
            if (fits > (int) ne) fits = (int) ne;   // never beyond the expert count
            if (fits > 0 && rec > fits) { rec = fits; capped = true; }
        }

        const double rec_frac = double(rec) / double(ne);
        fprintf(stderr,
            "moestream: [mrc] recommended %d slots (%.2f) -- below this, hit-rate loss exceeds %.1f pt/GiB\n",
            rec, rec_frac, g_pt_per_gib);
        if (dev_total > 0)
            fprintf(stderr,
                "moestream: [mrc]   device memory %.1f / %.1f GiB used; at most %d slots fit\n",
                dev_used, dev_total, fits);
        if (capped)
            fprintf(stderr,
                "moestream: [mrc]   ** capped by memory: %d slots would be better for hit rate,\n"
                "moestream: [mrc]      but would not fit. Recommending %d. **\n",
                rec_ideal, rec);
        else if (fits > 0 && rec_ideal >= fits)
            fprintf(stderr, "moestream: [mrc]   (this is at the memory limit)\n");
        fprintf(stderr,
            "moestream: [mrc]   to apply: set MOESTREAM_CACHE_FRAC=%.2f in .env and restart\n"
            "moestream: [mrc]   note: the I/O upper column assumes zero overlap and overstates; decide on pt/GiB\n",
            rec_frac);

        // In learn mode, persist it; the next start applies it automatically.
        if (g_frac_learn) {
            const double prev = learned_frac();
            if (std::fabs(rec_frac - prev) >= 0.02) learned_frac_store(rec_frac);
        }
    }
}

// From the measured I and C, predict prefill speed at any UBATCH and recommend
// a value.
static void ub_report() {
    if (!g_pf_ready) return;

    // Record first. The aggregate rate exists even when there are too few passes
    // to separate I from C -- which is exactly the case at large UBATCH. Without
    // this, a large candidate could never be measured and `learn` would retry it
    // on every start, forever.
    //   Key the record on the *configured* UBATCH passed in by entrypoint.sh.
    //   Using an observed n_tokens instead records short requests under their
    //   own token count (a 4-token nudge was once stored as "UBATCH=4").

    // Raw accumulators, so the derived rate can be checked against what the
    // server itself reports rather than trusted.
    if (debug_mode())
        fprintf(stderr, "moestream: [ub] raw: full passes=%llu tokens=%llu time=%.3f s"
                        "  warm-up excluded=%s  -> %.1f tok/s  (ubatch seen %d)\n",
                (unsigned long long) g_pf_intervals, (unsigned long long) g_pf_tok_total,
                g_pf_time_total, g_pf_intervals >= 2 ? "yes" : "no",
                prefill_rate(), g_ub_ntok);

    if (g_ub_passes < 4 || g_ub_ntok <= 0) {
        fprintf(stderr, "moestream: [ub] not enough samples (%llu passes). "
                        "Run a few long prompts and this will appear\n",
                (unsigned long long) g_ub_passes);
        return;
    }
    const double I  = g_ub_io  / double(g_ub_passes);   // I/O time per pass
    const double C  = g_ub_cmp / double(g_ub_passes);   // compute time per pass
    const int    u0 = g_ub_ntok;
    if (I <= 0 || C <= 0) return;
    const double asym = double(u0) / C;                 // kept for reference only

    fprintf(stderr,
        "moestream: [ub] prefill cost split (measured over %llu passes, UBATCH=%d)\n"
        "moestream: [ub]   I/O per pass       %6.2f s  <- fixed cost, independent of UBATCH\n"
        "moestream: [ub]   compute per pass   %6.2f s  (%.2f ms per token)\n",
        (unsigned long long) g_ub_passes, u0, I, C, C / double(u0) * 1000.0);

    // Raising UBATCH cuts the number of passes, so the fixed I/O cost is
    // amortized -- that part is simple. What is not simple is the compute term.
    //
    // This used to extrapolate assuming compute grows linearly with UBATCH, and
    // therefore always recommended going higher. That was wrong. Measured on
    // Ornith-35B (RESULTS.md §10.11), compute per pass grows *super-linearly*:
    //
    //     UBATCH  512 -> 1024 -> 2048 -> 4096
    //     compute 0.82   2.40   7.32   16.68 s     (2.3-3.0x per doubling)
    //     per tok 1.60   2.34   3.57    4.07 ms
    //
    // Attention is quadratic in the tokens within a micro-batch, so per-token
    // compute rises with UBATCH. There is a genuine optimum, and the linear
    // model could not see it: it recommended 4096-8192 where 1024 measured
    // fastest and 4096 was 16% slower.
    //
    // One run observes compute at exactly one UBATCH, which is not enough to
    // fit the growth exponent. Rather than guess it, report the measured split
    // and say plainly that it cannot extrapolate.
    fprintf(stderr,
        "moestream: [ub]   raising UBATCH amortizes the %.2f s I/O over fewer passes,\n"
        "moestream: [ub]   but per-token compute grows with UBATCH (attention is\n"
        "moestream: [ub]   quadratic within a micro-batch), so higher is not always\n"
        "moestream: [ub]   better. There is an optimum and it cannot be extrapolated\n"
        "moestream: [ub]   from a single UBATCH -- measure it:\n"
        "moestream: [ub]     research/tools/ms-bench.sh --ub 512   / --ub 1024 / --ub 2048\n"
        "moestream: [ub]   Measured optimum on Ornith-35B: 1024 (RESULTS.md §10.11)\n",
        I);
    (void) asym;
}

// Verify the prefetch decision against measurement. This works whether or not
// prefetching is enabled.
static void prefetch_verify() {
    if (g_dec_wall < 2.0) return;              // under 2 s is too little to judge
    const double share = g_dec_read / g_dec_wall;
    // NOTE ON WHAT THIS MEASURES.
    //   g_dec_read is the time spent *inside* run_reads_parallel, i.e. blocked
    //   waiting on reads. It is not the end-to-end cost of doing I/O: filling a
    //   slot writes ~1.4 MiB into GPU-visible memory, and the resulting memory
    //   and cache pressure slows the compute that follows, after the timer has
    //   already stopped.
    //   Measured end to end by differencing against MOESTREAM_NOOP=3 (identical
    //   cache behaviour, no reads), the real cost is about 1.6x this figure:
    //     frac 0.38: this timer 5.3 ms vs differential 7.8 ms
    //     frac 0.25: this timer 7.2 ms vs differential 12.5 ms
    //   Both numbers are correct for what they measure. Only the label was
    //   misleading, so it now says what it is. See RESULTS.md §10.12.
    fprintf(stderr, "moestream: [prefetch] decode spends %.1f%% blocked on reads "
                    "(%.1f s of %.1f s)\n"
                    "moestream: [prefetch]   end-to-end I/O cost is roughly 1.6x that; "
                    "measure it with MOESTREAM_NOOP=3 as the control\n",
            share * 100, g_dec_read, g_dec_wall);
    if (g_lay_n > 100) {
        const double w = g_lay_window / double(g_lay_n) * 1000.0;
        const double r = g_lay_read   / double(g_lay_n) * 1000.0;
        fprintf(stderr,
            "moestream: [overlap] per layer: %.3f ms of compute between remaps, "
            "%.3f ms of reads inside them\n"
            "moestream: [overlap]   ratio %.1fx -- a background prefetch could hide "
            "the reads only if this is comfortably above 1\n"
            "moestream: [overlap]   worst layer: window %.3f ms / read %.3f ms; "
            "reads exceeded the window in %llu of %llu layers (%.1f%%)\n",
            w, r, r > 0 ? w / r : 0.0,
            g_lay_win_max * 1000.0, g_lay_read_max * 1000.0,
            (unsigned long long) g_lay_starved, (unsigned long long) g_lay_n,
            100.0 * double(g_lay_starved) / double(g_lay_n));
    }
    if (g_prefetch && share < 0.10) {
        fprintf(stderr, "moestream: [prefetch] NOTE: decided \"on\", but I/O is under 10%%; "
                        "prefetching is unlikely to help much\n");
    } else if (!g_prefetch && share > 0.30) {
        fprintf(stderr, "moestream: [prefetch] NOTE: decided \"off\", but I/O accounts for %.0f%%; "
                        "the decision was wrong\n"
                        "moestream: [prefetch]   restart with MOESTREAM_PREFETCH=1\n", share * 100);
    } else {
        fprintf(stderr, "moestream: [prefetch] the decision holds up\n");
    }
}

void report() {
    if (!g_enabled || g_cache.empty()) return;
    CacheStats s;
    for (auto * c : g_cache) { s.hit += c->stats().hit; s.miss += c->stats().miss; }
    fprintf(stderr,
        "\nmoestream: hit rate %.2f%%  (hit %llu / miss %llu)\n"
        "moestream: I/O %llu reads, %.2f MiB in %.3f s (%.2f GB/s)\n"
        "moestream: bytes/token %.2f MiB  (tokens=%llu)\n"
        "moestream: zero-copy %llu / %llu (%.0f%%)\n"
        "moestream: remap total %.2f s (read %.2f s / upload %.2f s)\n"
        "moestream: experts dropped %llu / %llu (%.2f%%)\n"
        "moestream: slot exhaustion events %llu%s\n"
        "moestream: graph builds  prefill-path %llu / slab-path %llu\n"
        "moestream: arena loads %llu (%.2f GiB in %.3f s = %.2f GB/s), reuse hits %llu\n"
        "moestream: arena union read %llu / %llu experts (%.0f%% saved)\n"
        "moestream: arena async prefetch hit %llu / miss %llu\n"
        "moestream: union predictability (%llu samples): needed %.1f / previous %.1f / shortfall %.1f / total read %.1f (%.0f%% of all experts)\n",
        s.hit_rate() * 100, (unsigned long long) s.hit, (unsigned long long) s.miss,
        (unsigned long long) g_io_calls, g_io_bytes / 1048576.0,
        g_t_read, g_t_read > 0 ? g_io_bytes / g_t_read / 1e9 : 0.0,
        g_tokens ? double(g_io_bytes) / double(g_tokens) / 1048576.0 : 0.0,
        (unsigned long long) g_tokens,
        (unsigned long long) g_zerocopy, (unsigned long long) g_io_calls,
        g_io_calls ? 100.0 * double(g_zerocopy) / double(g_io_calls) : 0.0,
        g_t_total, g_t_read, g_t_upload,
        (unsigned long long) g_dropped, (unsigned long long) g_demand,
        g_demand ? 100.0 * double(g_dropped) / double(g_demand) : 0.0,
        (unsigned long long) g_exhausted,
        g_exhausted ? "  <-- OUTPUT WAS DEGRADED, reduce -ub" : "",
        (unsigned long long) g_pf_used, (unsigned long long) g_slab_used,
        (unsigned long long) g_pf_loads, g_pf_bytes / 1073741824.0, g_t_pfload,
        g_t_pfload > 0 ? g_pf_bytes / g_t_pfload / 1e9 : 0.0,
        (unsigned long long) g_pf_hits,
        (unsigned long long) g_pf_experts_read, (unsigned long long) g_pf_experts_asked,
        g_pf_experts_asked ? 100.0 * (1.0 - double(g_pf_experts_read) / double(g_pf_experts_asked)) : 0.0,
        (unsigned long long) g_pf_async_hit, (unsigned long long) g_pf_async_miss,
        (unsigned long long) g_u_n,
        g_u_n ? double(g_u_need)/double(g_u_n) : 0.0,
        g_u_n ? double(g_u_prev)/double(g_u_n) : 0.0,
        g_u_n ? double(g_u_short)/double(g_u_n) : 0.0,
        g_u_n ? double(g_u_total)/double(g_u_n) : 0.0,
        (g_u_n && !g_layers.empty()) ? 100.0*double(g_u_total)/double(g_u_n)
                                       /double(g_layers.begin()->second.gate.n_expert) : 0.0);
    if (g_dn_ready && g_dn_learn && g_tokens > 0 && g_dn_read_bytes > 0) {
        // Dense has no knee to learn. Finding S29 measured decode cost as
        // linear in bytes streamed, so there is no interior optimum the way the
        // expert cache's miss-ratio curve has one -- "auto" already picks the
        // only defensible point (stream the least that fits). What is worth
        // learning is the exchange RATE on this machine, so the operator can
        // price any other point themselves.
        const double gib_per_tok = double(g_dn_read_bytes) / 1073741824.0 / double(g_tokens);
        fprintf(stderr,
            "moestream: [dense] [learn] %.2f GiB/token streamed over %llu tokens\n"
            "moestream: [dense] [learn]   read %.2f s at %.2f GB/s; prefetch hid %llu of %llu layer loads\n"
            "moestream: [dense] [learn]   there is no knee here: cost is linear in bytes streamed,\n"
            "moestream: [dense] [learn]   so auto streams the least that fits and that is the whole policy\n",
            gib_per_tok, (unsigned long long) g_tokens,
            g_dn_t_read, g_dn_t_read > 0 ? g_dn_read_bytes / g_dn_t_read / 1e9 : 0.0,
            (unsigned long long) g_dn_pf_hit,
            (unsigned long long) (g_dn_pf_hit + g_dn_pf_miss));
    }
    if (g_dn_ready)
        fprintf(stderr,
            "moestream: [dense] layers %d-%d streamed (%zu), %d resident\n"
            "moestream: [dense]   arena loads %llu, reuse hits %llu, %.2f GiB read in %.2f s (%.2f GB/s)\n"
            "moestream: [dense]   prefetch hit %llu / miss %llu\n",
            g_dn_first, g_dn_nlayer - 1, g_dn_view.size(), g_dn_first,
            (unsigned long long) g_dn_loads, (unsigned long long) g_dn_hits,
            g_dn_read_bytes / 1073741824.0, g_dn_t_read,
            g_dn_t_read > 0 ? g_dn_read_bytes / g_dn_t_read / 1e9 : 0.0,
            (unsigned long long) g_dn_pf_hit, (unsigned long long) g_dn_pf_miss);
    mrc_report();
    ub_report();
    prefetch_verify();
}

} // namespace moestream
