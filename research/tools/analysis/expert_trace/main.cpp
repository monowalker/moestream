// =============================================================================
// MoEStream — expert-activation trace capture tool (M0-2)
//
// Purpose (DESIGN.md §32.1 / §33.4):
//   Measure the project's largest unverified risk: is the expert-activation
//   distribution skewed enough for caching to work?
//
//   From the §2.3 analysis, on this machine (B_act=455 MiB/token, t_c=43.7 ms,
//   BW=4.46 GB/s) a 51% hit rate is required to stay within a 20% slowdown.
//   Whether a 38% cache ratio reaches that is the go/no-go decision.
//
// Method:
//   With llama.cpp unpatched, capture the router's selection tensor
//   `ffn_moe_topk-<layer>` and the weights `ffn_moe_weights_*-<layer>` through
//   llama_context_params.cb_eval. Nothing upstream is modified, so this keeps
//   working across llama.cpp updates.
//
// Output (binary):
//   header : "MSTR" u32ver u32n_layer u32n_expert u32top_k u64n_token
//   record : per MoE layer, per token: u16 ids[top_k] + f16 w[top_k]
// =============================================================================

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// ---- routing captured for one batch -----------------------------------------
struct LayerRouting {
    std::vector<int32_t> ids;     // [n_tokens * top_k]
    std::vector<float>   w;       // [n_tokens * top_k]
    int n_tokens = 0;
    bool has_w   = false;
};

struct TraceCtx {
    int n_layer = 0, n_expert = 0, top_k = 0;
    std::map<int, LayerRouting> batch;   // layer -> routing for this batch
    FILE * out = nullptr;
    uint64_t n_token_written = 0;
    // Several tensors can carry the weights; prefer the later-stage ones
    std::map<int, int> w_priority;
};

static TraceCtx g_tc;

// Extract the layer index from a name of the form "<prefix>-<layer>"
static bool parse_layer(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) return false;
    if (name[n] != '-') return false;
    char * end = nullptr;
    long v = strtol(name + n + 1, &end, 10);
    if (end == name + n + 1) return false;
    il = (int) v;
    return true;
}

// Priority among weight tensors: later transforms are closer to the weights
// actually used
static int weight_rank(const char * name) {
    if (strncmp(name, "ffn_moe_weights_scaled", 22) == 0) return 4;
    if (strncmp(name, "ffn_moe_weights_norm",   20) == 0) return 3;
    if (strncmp(name, "ffn_moe_weights_softmax",23) == 0) return 2;
    if (strncmp(name, "ffn_moe_weights",        15) == 0) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Important: ggml_top_k returns a *view* onto the ggml_argsort result.
//   ne[0] is top_k, but nb[1] remains the original argsort row stride
//   (n_expert * 4). Computing the size naively from ne therefore under-allocates
//   and corrupts the heap. Always allocate ggml_nbytes() and index via nb[].
// ---------------------------------------------------------------------------
static std::vector<uint8_t> g_raw;

static bool eval_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    const char * name = ggml_get_name(t);
    int il = -1;

    if (ask) {
        if (parse_layer(name, "ffn_moe_topk", il)) return true;
        return weight_rank(name) > 0;
    }

    if (parse_layer(name, "ffn_moe_topk", il)) {
        if (t->type != GGML_TYPE_I32) return true;
        const int tk = (int) t->ne[0];
        const int nt = (int) t->ne[1];
        g_raw.resize(ggml_nbytes(t));                     // real size, including the view's stride
        ggml_backend_tensor_get(t, g_raw.data(), 0, g_raw.size());

        auto & lr = g_tc.batch[il];
        lr.ids.assign((size_t) tk * nt, 0);
        lr.n_tokens = nt;
        for (int j = 0; j < nt; ++j)
            for (int k = 0; k < tk; ++k) {
                const size_t off = (size_t) j * t->nb[1] + (size_t) k * t->nb[0];
                int32_t v; memcpy(&v, g_raw.data() + off, sizeof(int32_t));
                lr.ids[(size_t) j * tk + k] = v;
            }
        if (g_tc.top_k == 0) g_tc.top_k = tk;
        return true;
    }

    const int r = weight_rank(name);
    if (r > 0) {
        if (t->type != GGML_TYPE_F32) return true;
        const char * dash = strrchr(name, '-');
        if (!dash) return true;
        il = atoi(dash + 1);
        auto it = g_tc.w_priority.find(il);
        if (it != g_tc.w_priority.end() && it->second >= r) return true;

        // llama.cpp reshapes weights to [1, top_k, n_tokens].
        // Handle [top_k, 1, n_tokens] as well, just in case.
        int tk, nt; size_t stride_k, stride_t;
        if (t->ne[0] == 1) {                     // [1, top_k, n_tokens]
            tk = (int) t->ne[1]; nt = (int) t->ne[2];
            stride_k = t->nb[1]; stride_t = t->nb[2];
        } else if (t->ne[1] == 1) {              // [top_k, 1, n_tokens]
            tk = (int) t->ne[0]; nt = (int) t->ne[2];
            stride_k = t->nb[0]; stride_t = t->nb[2];
        } else {                                 // [top_k, n_tokens]
            tk = (int) t->ne[0]; nt = (int) t->ne[1];
            stride_k = t->nb[0]; stride_t = t->nb[1];
        }
        if (g_tc.top_k && tk != g_tc.top_k) return true;

        g_raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, g_raw.data(), 0, g_raw.size());

        auto & lr = g_tc.batch[il];
        lr.w.assign((size_t) tk * nt, 0.0f);
        for (int j = 0; j < nt; ++j)
            for (int k = 0; k < tk; ++k) {
                const size_t off = (size_t) j * stride_t + (size_t) k * stride_k;
                float v; memcpy(&v, g_raw.data() + off, sizeof(float));
                lr.w[(size_t) j * tk + k] = v;
            }
        lr.has_w = true;
        g_tc.w_priority[il] = r;
        return true;
    }
    return true;
}

static inline uint16_t f32_to_f16_bits(float f) {
    return ggml_fp32_to_fp16(f);
}

// Write out one batch
static void flush_batch(int n_tokens) {
    if (g_tc.batch.empty()) return;
    std::vector<int> layers;
    for (auto & kv : g_tc.batch) layers.push_back(kv.first);
    std::sort(layers.begin(), layers.end());

    const int tk = g_tc.top_k;
    std::vector<uint16_t> ids(tk), wq(tk);
    for (int t = 0; t < n_tokens; ++t) {
        for (int il : layers) {
            const auto & lr = g_tc.batch[il];
            if (t >= lr.n_tokens) continue;
            for (int k = 0; k < tk; ++k) {
                ids[k] = (uint16_t) lr.ids[(size_t) t * tk + k];
                float wv = 0.0f;
                if (lr.has_w && lr.w.size() >= (size_t)(t + 1) * tk)
                    wv = lr.w[(size_t) t * tk + k];
                wq[k] = f32_to_f16_bits(wv);
            }
            fwrite(ids.data(), sizeof(uint16_t), tk, g_tc.out);
            fwrite(wq.data(),  sizeof(uint16_t), tk, g_tc.out);
        }
        g_tc.n_token_written++;
    }
    g_tc.batch.clear();
    g_tc.w_priority.clear();
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <model.gguf> <corpus.txt> <out.trace> [n_gpu_layers] [max_tokens] [batch]\n", argv[0]);
        return 1;
    }
    const char * model_path  = argv[1];
    const char * corpus_path = argv[2];
    const char * out_path    = argv[3];
    const int n_gpu_layers   = argc > 4 ? atoi(argv[4]) : 0;
    const int max_tokens     = argc > 5 ? atoi(argv[5]) : 20000;
    const int n_batch        = argc > 6 ? atoi(argv[6]) : 512;

    // ---- load the corpus ----
    std::string text;
    {
        FILE * f = fopen(corpus_path, "rb");
        if (!f) { perror("corpus"); return 1; }
        char buf[1 << 16]; size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        fclose(f);
    }
    fprintf(stderr, "corpus: %zu bytes\n", text.size());

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load the model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    g_tc.n_layer  = llama_model_n_layer(model);

    // ---- tokenize ----
    std::vector<llama_token> toks(text.size() + 8);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                            toks.data(), (int) toks.size(), true, false);
    if (nt < 0) { toks.resize(-nt);
        nt = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                            toks.data(), (int) toks.size(), true, false); }
    toks.resize(std::max(0, nt));
    if ((int) toks.size() > max_tokens) toks.resize(max_tokens);
    fprintf(stderr, "tokens: %zu\n", toks.size());

    // ---- context, with the eval callback installed ----
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = (uint32_t) std::min<size_t>(toks.size() + 16, 8192);
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch= (uint32_t) n_batch;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = nullptr;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create the context\n"); return 1; }

    // ---- output file; the header is rewritten at the end ----
    g_tc.out = fopen(out_path, "wb");
    if (!g_tc.out) { perror("out"); return 1; }
    uint32_t hdr[8] = { 0x5254534DU /*"MSTR"*/, 1, 0, 0, 0, 0, 0, 0 };
    fwrite(hdr, sizeof hdr, 1, g_tc.out);

    // ---- run; prefill alone is enough for router statistics ----
    const size_t total = toks.size();
    size_t pos = 0;
    int n_moe_layer = 0;
    while (pos < total) {
        const int n = (int) std::min<size_t>(n_batch, total - pos);
        // Reset the KV cache each batch so the context is never exhausted,
        // treating the run as a sequence of independent contexts. That is
        // sufficient for distribution statistics.
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_get_one(toks.data() + pos, n);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at %zu\n", pos);
            break;
        }
        if (n_moe_layer == 0) {
            n_moe_layer = (int) g_tc.batch.size();
            if (!g_tc.batch.empty()) {
                // n_expert cannot be derived from the top-k maximum, and is not
                // available from the model here, so downstream analysis
                // estimates it as max(ids) + 1
                fprintf(stderr, "MoE layers = %d, top_k = %d\n", n_moe_layer, g_tc.top_k);
            }
        }
        flush_batch(n);
        pos += n;
        if ((pos / n_batch) % 8 == 0)
            fprintf(stderr, "\r  %zu / %zu tokens (%.1f%%)", pos, total, 100.0 * pos / total);
    }
    fprintf(stderr, "\n");

    // ---- finalize the header ----
    hdr[2] = (uint32_t) n_moe_layer;
    hdr[3] = 0;                       // n_expert is estimated during analysis
    hdr[4] = (uint32_t) g_tc.top_k;
    hdr[5] = (uint32_t) (g_tc.n_token_written & 0xffffffffU);
    hdr[6] = (uint32_t) (g_tc.n_token_written >> 32);
    hdr[7] = (uint32_t) g_tc.n_layer;
    fseek(g_tc.out, 0, SEEK_SET);
    fwrite(hdr, sizeof hdr, 1, g_tc.out);
    fclose(g_tc.out);

    fprintf(stderr, "trace: %s  tokens=%llu  moe_layers=%d  top_k=%d\n",
            out_path, (unsigned long long) g_tc.n_token_written, n_moe_layer, g_tc.top_k);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
