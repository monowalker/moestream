#!/usr/bin/env python3
"""Patch MoEStream into llama.cpp.

Every replacement is guarded by an assert, so an upstream change that moves
the anchor text fails the build immediately rather than producing a silently
wrong binary.
"""
import sys, pathlib, re
root = pathlib.Path(sys.argv[1])
n = 0

# --- 1. add the sources to CMakeLists ---
p = root / 'src/CMakeLists.txt'
s = p.read_text()
if 'llama-moestream.cpp' not in s:
    old_cmake = '            llama-model-loader.cpp'
    assert old_cmake in s, 'CMakeLists source list anchor not found'
    s = s.replace(old_cmake, '            llama-moestream.cpp\n            expert_cache.cpp\n            llama-model-loader.cpp')
    p.write_text(s); n += 1

# --- 2. create_tensor: allocate expert tensors at slab size and register them ---
p = root / 'src/llama-model-loader.cpp'
s = p.read_text()
if 'moestream' not in s:
    s = s.replace('#include "llama-model-loader.h"',
                  '#include "llama-model-loader.h"\n#include "llama-moestream.h"', 1)
    # create_tensor has two upstream spellings, and both are supported.
    #
    #   Until llama.cpp 1269cb1f ("model : allow reshape of tensors during load",
    #   #26531) the tensor was duplicated straight from the file metadata `cur`.
    #   That commit introduced a local `ggml_tensor t_meta = *cur;` whose ne/nb
    #   are rewritten when TENSOR_ALLOW_RESHAPE is set, and duplicates from that
    #   instead. `cur` still exists, but it is no longer the shape the tensor is
    #   created with, so the slab has to be sized from whichever one upstream
    #   itself uses -- otherwise a reshaped expert tensor would get a slab built
    #   from the pre-reshape dimensions.
    #
    #   MS_META below is that expression; the body is otherwise identical.
    old = """    const bool duplicated = flags & TENSOR_DUPLICATED;

    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
    ggml_set_name(tensor, ggml_get_name(cur));"""
    old_reshape = """    const bool duplicated = flags & TENSOR_DUPLICATED;

    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, &t_meta);
    ggml_set_name(tensor, ggml_get_name(&t_meta));"""
    new = """    const bool duplicated = flags & TENSOR_DUPLICATED;

    struct ggml_tensor * tensor = nullptr;
    {
        // ---- MoEStream: allocate expert tensors as a reduced slot slab ----
        const struct ggml_tensor * ms_meta = MS_META;
        const uint32_t n_slot = moestream::slab_slots(ggml_get_name(ms_meta), ms_meta->ne[2]);
        if (n_slot > 0 && ggml_n_dims(ms_meta) == 3) {
            tensor = ggml_new_tensor_3d(ctx, ms_meta->type, ms_meta->ne[0], ms_meta->ne[1], (int64_t) n_slot);
            ggml_set_name(tensor, ggml_get_name(ms_meta));
            const auto * w = get_weight(ggml_get_name(ms_meta));
            if (w) {
                const uint64_t ebytes = ggml_nbytes(ms_meta) / (uint64_t) ms_meta->ne[2];
                moestream::register_slab(ggml_get_name(ms_meta), tensor,
                                         w->offs, ebytes, ms_meta->ne[2], (int) w->idx);
            }
        }
        // ---- MoEStream: dense FFN weights on a streamed layer ----
        //   Allocated as a one-row stub. The real bytes are read into the arena
        //   at graph time; allocating the full tensor here would leave the whole
        //   model resident and save nothing. Nothing reads the stub, because
        //   build_ffn substitutes the arena tensor for it.
        if (!tensor && moestream::dense_claim(ggml_get_name(ms_meta), ggml_n_dims(ms_meta), ggml_nbytes(ms_meta))) {
            tensor = ggml_new_tensor_2d(ctx, ms_meta->type, ms_meta->ne[0], 1);
            ggml_set_name(tensor, ggml_get_name(ms_meta));
            const auto * w = get_weight(ggml_get_name(ms_meta));
            if (w) {
                moestream::register_dense(ggml_get_name(ms_meta), tensor, w->offs,
                                          ggml_nbytes(ms_meta), (int) w->idx,
                                          (int) ms_meta->type, ms_meta->ne[0], ms_meta->ne[1]);
            }
        }
    }
    if (!tensor) {
        tensor = ggml_dup_tensor(ctx, MS_META);
        ggml_set_name(tensor, ggml_get_name(MS_META));
    }"""
    # (anchor, the expression upstream duplicates from)
    variants = [(old, 'cur'), (old_reshape, '&t_meta')]
    hit = next((v for v in variants if v[0] in s), None)
    assert hit is not None, ("create_tensor anchor not found "
                             "(neither the pre-#26531 `cur` nor the post-#26531 `&t_meta` spelling)")
    anchor, ms_meta = hit
    s = s.replace(anchor, new.replace('MS_META', ms_meta), 1)

    # --- load_all_data: never read the slab from the file ---
    old2 = """        const auto * weight = get_weight(ggml_get_name(cur));"""
    new2 = """        if (moestream::skip_load(ggml_get_name(cur))) { continue; }
        const auto * weight = get_weight(ggml_get_name(cur));"""
    assert old2 in s, "load_all_data anchor not found"
    s = s.replace(old2, new2, 1)
    p.write_text(s); n += 1

# --- 3. llama-graph.cpp: insert the node mapping ids to slot_ids ---
p = root / 'src/llama-graph.cpp'
s = p.read_text()
if 'moestream' not in s:
    s = s.replace('#include "llama-graph.h"', '#include "llama-graph.h"\n#include "llama-moestream.h"', 1)
    old = '    cb(selected_experts, "ffn_moe_topk", il);'
    new = """    cb(selected_experts, "ffn_moe_topk", il);

    // MoEStream: select the path.
    //   prefill (many tokens): load one layer's full expert set into the arena
    //                          and keep ids as expert_ids (no remap needed).
    //   decode  (few tokens):  slab plus a remap to slot_ids.
    ggml_tensor * ms_g = moestream::prefill_exps(il, 0, (int) n_tokens);
    ggml_tensor * ms_u = moestream::prefill_exps(il, 1, (int) n_tokens);
    ggml_tensor * ms_d = moestream::prefill_exps(il, 2, (int) n_tokens);
    const bool ms_prefill = ms_g && ms_u && ms_d;
    // On the prefill path, insert a load op that passes ids through. Using it
    // as mul_mat_id's ids enforces "arena filled -> mul_mat_id runs".
    ggml_tensor * ms_ids = ms_prefill
                             ? moestream::build_prefill_load(ctx0, selected_experts, il)
                             : moestream::build_remap2(ctx0, selected_experts, cur, il);
    if (ms_prefill) { gate_exps = ms_g; up_exps = ms_u; down_exps = ms_d; }"""
    assert old in s, "ffn_moe_topk cb anchor not found"
    s = s.replace(old, new, 1)
    # Replace only the ids passed to mul_mat_id
    import re as _re
    n_mm = 0
    def repl(m):
        global n_mm
        n_mm += 1
        return m.group(0).replace('cur, selected_experts', 'cur, ms_ids')
    s = _re.sub(r'build_lora_mm_id\((?:gate_up_exps|up_exps|gate_exps|down_exps), cur, selected_experts',
                repl, s)
    assert n_mm >= 3, f"expected at least 3 mul_mat_id replacements, got {n_mm}"

    # ---- dense FFN streaming: substitute the arena tensors in build_ffn ----
    #   This hook was briefly moved to build_lora_mm, which every per-layer
    #   weight passes through, in order to reach attention and SSM as well.
    #   That failed twice over: stubbing cannot survive code that reads a
    #   weight's SHAPE (Qwen3.5's SSM path does), and the wider hook then broke
    #   the FFN case that had been working. Reverted to the anchor that S27
    #   measured. See docs/findings for why the generalisation needs a
    #   different design, not a wider filter.
    old_ffn = "    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;"
    new_ffn = """    // MoEStream: on a streamed dense layer the FFN weights live in the arena.
    //   build_dense_load fills it and returns `cur` unchanged, so consuming its
    //   output below is what orders "arena filled -> mul_mat runs".
    if (moestream::dense_active()) {
        ggml_tensor * ms_du = moestream::dense_ffn(il, 1);
        ggml_tensor * ms_dd = moestream::dense_ffn(il, 2);
        if (ms_du && ms_dd) {
            ggml_tensor * ms_dg = moestream::dense_ffn(il, 0);
            cur  = moestream::build_dense_load(ctx0, cur, il);
            up   = ms_du;
            down = ms_dd;
            if (gate && ms_dg) gate = ms_dg;
        }
    }

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;"""
    assert old_ffn in s, "build_ffn anchor not found"
    s = s.replace(old_ffn, new_ffn, 1)

    # ---- Expert Sweep: wrap the FFN body in a loop of P passes ----
    #   Splits when the union exceeds the slab capacity (prefill).
    #   For decode P is 1, which is identical to the unpatched path.
    old_loop = "    ggml_tensor * experts = nullptr;"
    new_loop = """    ggml_tensor * experts = nullptr;

    // MoEStream Expert Sweep (§20.2): split the FFN when union > slab capacity.
    //   y = sum_p FFN(x, ids_p); experts outside a pass map to the zero slot
    //   and contribute nothing.
    //   For decode (n_tokens=1), ms_np is 1 and this matches the normal path.
    ggml_tensor * ms_ffn_in = cur;
    ggml_tensor * ms_acc    = nullptr;
    ggml_tensor * ms_prev   = nullptr;   // previous pass's output (dependency target)
    const int ms_np = ms_prefill ? 1 : moestream::n_passes((int) n_tokens);
    for (int ms_p = 0; ms_p < ms_np; ++ms_p) {
    cur = ms_ffn_in;
    if (!ms_prefill && ms_np > 1) {
        ms_ids = moestream::build_remap_pass(ctx0, selected_experts, ms_prev, il, ms_p, ms_np);
    }"""
    assert old_loop in s, "experts declaration not found"
    s = s.replace(old_loop, new_loop, 1)

    old_end = '''    cb(experts, "ffn_moe_down", il);'''
    new_end = '''    cb(experts, "ffn_moe_down", il);

    // Expert Sweep: accumulate per-pass results (the sum is order-independent)
    //   ms_prev points at the previous pass's own output. Depending on the add
    //   chain instead fails to force the previous pass's GPU work to complete
    //   from pass 2 onward, and the slab gets overwritten.
    experts = moestream::build_marker(ctx0, experts, il, ms_p);
    // Materialize the pass output. Without this, later passes reuse the buffer
    // for their intermediates and the contents are corrupt by the final add
    // (found by inserting an identity CPU op, which moved PPL from 520801 to 185).
    if (ms_np > 1) experts = ggml_cont(ctx0, experts);
    // Register each pass with the graph explicitly. Without this, nothing is
    // registered until the final add, gallocr reuses buffers across passes, and
    // the result is corrupted -- the identity-op experiment (PPL 520801 -> 185)
    // identified this as aliasing.
    ggml_build_forward_expand(gf, experts);
    ms_prev = experts;
    {   // Diagnostic MOESTREAM_SWEEP_TEST=3: keep only pass 0's result and add
        //   nothing else. The other passes leave the graph and are not even
        //   executed, which tests whether the loop machinery is itself harmless.
        static const char * st = getenv("MOESTREAM_SWEEP_TEST");
        const int stv = st ? atoi(st) : 0;
        ggml_tensor * contrib = experts;
        // SWEEP_TEST=4: every pass owns every expert, but passes above 0 are
        //   scaled to zero with ggml_scale, so y = FFN + 0*FFN = FFN.
        //   Comparing against zeroing via the zero slot isolates which of the
        //   two mechanisms is broken.
        if (stv == 4 && ms_p > 0) contrib = ggml_scale(ctx0, experts, 0.0f);
        if (stv != 3 || ms_p == 0) {
            ms_acc = ms_acc ? ggml_add(ctx0, ms_acc, contrib) : contrib;
        }
    }
    }  // end MoEStream pass loop
    experts = ms_acc;'''
    assert old_end in s, "ffn_moe_down cb not found"
    s = s.replace(old_end, new_end, 1)
    print("  inserted the Expert Sweep pass loop")
    p.write_text(s); n += 1
    print(f"  replaced build_lora_mm_id in {n_mm} places")


# --- 4. ggml-vulkan: expose the already-mapped host pointer on UMA ---
#     On UMA, ggml-vulkan allocates with eDeviceLocal|eHostVisible|eHostCoherent
#     and already holds a mapMemory'd pointer. Exposing it allows preading from
#     SSD straight into GPU-visible memory, removing one copy (§14.4).
p = root / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
s = p.read_text()
if 'ggml_backend_vk_buffer_host_ptr' not in s:
    old = 'static void * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer) {'
    new = """// MoEStream: on host-visible UMA, return a direct write target for tensor data.
//   Returns nullptr when unavailable (e.g. a discrete GPU); callers then fall
//   back to the normal tensor_set path.
extern "C" void * ggml_backend_vk_buffer_host_ptr(ggml_backend_buffer_t buffer, size_t offset) {
    if (!buffer || !buffer->context) return nullptr;
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *) buffer->context;
    if (!ctx->dev_buffer || !ctx->dev_buffer->ptr) return nullptr;
    if (!(ctx->dev_buffer->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible))
        return nullptr;
    if (offset + 1 > ctx->dev_buffer->size) return nullptr;
    return (char *) ctx->dev_buffer->ptr + offset;
}

static void * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer) {"""
    assert old in s, "get_base anchor not found"
    s = s.replace(old, new, 1)
    p.write_text(s); n += 1
    print("  added the host-pointer accessor to ggml-vulkan")


# --- 5. ggml-vulkan: import host memory as a Vulkan buffer ---
#     On UMA, VK_EXT_external_memory_host can import a host pointer, which would
#     let the GPU read an mmap'd region of the GGUF without copying. Living in
#     the page cache would also make it reclaimable by the kernel.
#     NOTE: this path is not used by the product; see Finding S7.
p = root / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
s = p.read_text()
if 'ggml_backend_vk_buffer_from_host_ptr' not in s:
    old = 'static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {'
    new = """// MoEStream: import host memory (e.g. an mmap) as a ggml_backend_buffer.
//   Returns nullptr where VK_EXT_external_memory_host is unavailable.
//   ptr must be aligned to minImportedHostPointerAlignment.
extern "C" ggml_backend_buffer_t ggml_backend_vk_buffer_from_host_ptr(
        ggml_backend_buffer_type_t buft, void * ptr, size_t size) {
    if (!buft || !ptr || !size) return nullptr;
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    if (!ctx->device->external_memory_host) return nullptr;
    vk_buffer dev_buffer = nullptr;
    try {
        dev_buffer = ggml_vk_create_buffer(ctx->device, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent},
            ptr);
    } catch (const vk::SystemError & e) {
        return nullptr;
    }
    if (!dev_buffer) return nullptr;
    ggml_backend_vk_buffer_context * bufctx =
        new ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);
    return ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, bufctx, size);
}

extern "C" size_t ggml_backend_vk_host_ptr_alignment(ggml_backend_buffer_type_t buft) {
    if (!buft) return 0;
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->external_memory_host ? 4096 : 0;
}

static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {"""
    assert old in s, "buffer_type_get_alignment anchor not found"
    s = s.replace(old, new, 1)
    p.write_text(s); n += 1
    print("  added the host-pointer import API to ggml-vulkan")

print(f"applied {n} patches to llama.cpp")
