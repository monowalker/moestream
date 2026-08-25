// MoEStream — expert streaming embedded into llama.cpp
//
// Approach (DESIGN.md §10.4 / ADR-0006):
//   Allocate expert tensors as [n_embd, n_ff, n_slot] with n_slot << n_expert
//   instead of [n_embd, n_ff, n_expert], and rewrite the router's expert_id to
//   a slot_id at run time. That makes streaming possible without changing any
//   llama.cpp graph or kernel code.
//
// Each layer gets its own slab, so a per-layer type difference (as produced by
// UD quantization) is not a problem.
#pragma once
#include <cstdint>
#include <cstddef>

struct ggml_tensor;
struct ggml_context;

namespace moestream {

// Environment variables:
//   MOESTREAM=1                enable
//   MOESTREAM_GGUF=<path>      GGUF to read experts from (required)
//   MOESTREAM_CACHE_FRAC=0.25  fraction of experts kept resident
//   MOESTREAM_IO_THREADS=8     parallel read threads (default: auto-tuned)
//   MOESTREAM_ZEROCOPY=1       read straight into GPU-visible memory on UMA
//   MOESTREAM_NOOP=0|1|3       diagnostics (1 = pass through / 3 = no I/O)
bool   enabled();
double cache_frac();
void   set_gguf_path(const char * path);

// Called from the loader
uint32_t slab_slots(const char * name, int64_t n_expert);
// file_idx: shard index for split GGUFs (llama_tensor_weight::idx).
//   file_offset is an offset within that shard, so both are required.
void     register_slab(const char * name, ggml_tensor * t,
                       uint64_t file_offset, uint64_t expert_bytes, int64_t n_expert,
                       int file_idx);
bool     skip_load(const char * name);
void     finalize(const char * gguf_path);

// Called during graph construction: insert a node mapping ids to slot_ids and
// return it. When disabled, returns ids unchanged.
ggml_tensor * build_remap(ggml_context * ctx, ggml_tensor * ids, int il);

// Same, but also given the router input (the hidden state) so that a diagnostic
// can measure what it costs to bring that state to the CPU.
//   That is the synchronization the P2 layer-lookahead predictor would pay, and
//   the reason P2 was rejected (Finding N2/S11). The measurement it rested on
//   predates union loading, async prefetch and zero-copy, after which the
//   per-layer sync fell about 5x (§10.12) -- so the rejection deserves a fresh
//   number before it is trusted again.
//   Normally identical to build_remap; MOESTREAM_PROBE_HIDDEN=1 adds the transfer.
ggml_tensor * build_remap2(ggml_context * ctx, ggml_tensor * ids,
                           ggml_tensor * hidden, int il);

// ---- separating the prefill and decode paths ----
//   During prefill (many tokens at once) the union of experts referenced by a
//   single mul_mat_id exceeds the slab capacity, which forces a small ubatch
//   and is slow. Prefill therefore uses a staging arena holding one layer's
//   full expert set.
//
//   The original design mmap'd the GGUF and imported it, but amdgpu rejects
//   file-backed VMAs (VkResult -13, Finding S5). Anonymous memory is accepted,
//   and mul_mat_id runs bit-identically on top of it (Finding S6).
//   The arena holds one layer and is reused across layers, so the resident
//   slab used by decode never grows.
//
//   Returns the tensor to use for the given n_tokens, or nullptr when the
//   prefill path is unavailable.
//   role: 0=gate 1=up 2=down
ggml_tensor * prefill_exps(int il, int role, int n_tokens);
bool          prefill_available();

// Insert the CPU op that loads layer il's experts into the prefill arena.
// It returns ids unchanged, so using it as mul_mat_id's ids enforces the
// ordering "load completes -> mul_mat_id runs".
ggml_tensor * build_prefill_load(ggml_context * ctx, ggml_tensor * ids, int il);

// ---- Expert Sweep (§20.2) ----
//   When the union of experts referenced by one mul_mat_id exceeds the slab
//   capacity, split the FFN into P passes and map every expert outside the
//   current pass to the zero slot. The output is an order-independent sum, so
//   adding the per-pass results is correct:
//     y = sum_p FFN(x, ids_p)
//   For decode (n_tokens=1) P is 1, which is identical to the normal path.
int           n_passes(int n_tokens);
//   dep: a dependency on the previous pass's output (its contents are unused).
//        Without it, the previous pass's GPU compute races against the next
//        pass overwriting the slab.
ggml_tensor * build_remap_pass(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * dep,
                               int il, int pass, int n_passes);

// Diagnostics: identity marker op used to observe execution order
ggml_tensor * build_marker(ggml_context * ctx, ggml_tensor * t, int il, int pass);

// ---- dense FFN streaming (findings S18 / S21 / S26) ----
//   MOESTREAM_DENSE_FRAC = auto (default) | learn | off | a fraction.
//   "auto" picks the split from the memory budget and streams the least that
//   still fits, so a model that already fits streams nothing and behaves
//   exactly like plain llama.cpp. Whether a model is dense or MoE is read from
//   the GGUF (expert_count), so nothing has to be selected by hand.
//   A fraction keeps the first f of a dense model's layers resident
//   and streams the rest of their FFN weights through an arena, one layer at a
//   time. Accuracy is untouched -- the same bytes are used, they just are not
//   all resident at once. 1.0 (default) disables it, so MoE models and
//   unconfigured runs behave exactly as before.
//
//   The tail is streamed rather than the head: which layers are chosen does not
//   change bytes/token, but with the tail streamed the resident head's compute
//   is time the reads are hidden behind. Unlike MoE there is nothing to predict
//   -- layer 33 is always layer 33 -- so the prefetch that never worked for
//   experts works here.
//
//   Called from the loader: claim the tensor (so it is allocated as a stub and
//   not read from the file) and record where its real bytes live.
bool dense_claim(const char * name, int64_t n_dims, uint64_t nbytes);
void register_dense(const char * name, const ggml_tensor * stub, uint64_t file_offset,
                    uint64_t bytes, int file_idx, int type, int64_t ne0, int64_t ne1);

//   Called during graph construction, from build_lora_mm -- the single point
//   every per-layer weight matrix in llama.cpp passes through, which is what
//   lets FFN, attention and SSM all be streamed with no architecture-specific
//   code (ADR-0036). Returns the arena-backed replacement for this weight, or
//   nullptr when it is resident and llama.cpp's own tensor should be used.
ggml_tensor * dense_ffn(int il, int role);
bool          dense_active();
//   Insert the CPU op that fills the arena for layer il. It returns the hidden
//   state unchanged, so using its output as the FFN input orders
//   "arena filled -> mul_mat runs".
ggml_tensor * build_dense_load(ggml_context * ctx, ggml_tensor * cur, int il);

void report();

} // namespace moestream
