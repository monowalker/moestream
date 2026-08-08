// =============================================================================
// MoEStream Spike S10 — measure P2 predictor accuracy (Laguna, or any MoE model)
//
// What P2 is: multiply layer L's hidden state h_L by layer L+1's router matrix
//          W_{L+1} to predict which experts layer L+1 will select.
//          It rests on the assumption that the residual stream changes only
//          gradually across layers.
//          It reached 81.4% accuracy on Ornith-35B (Finding N2); whether that
//          holds for other models such as Laguna was unmeasured.
//
// Method: capture and dump the following via cb_eval, without patching
//       llama.cpp at all:
//         ffn_norm-<L>        layer L's router input h_L   [n_embd]
//         ffn_moe_logits-<L>  layer L's router output      [n_expert]
//       Accuracy is computed in Python (W is readable as F32 from the GGUF).
//
// Self-check: Python verifies that W_L . h_L == logits_L. If they disagree,
//   the captured tensors are not what we assume they are, and no accuracy
//   number derived from them can be trusted.
//
// Output (binary):
//   header : "P2AC" u32 n_embd u32 n_expert u32 n_layer_slots u32 reserved
//   record : one per MoE layer, per token
//              u32 il, f32 h[n_embd], f32 logits[n_expert]
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

static FILE * g_out    = nullptr;
static int    g_n_embd = 0, g_n_expert = 0;
static long   g_records = 0;
static bool   g_capture = false;   // capture only during decode

// Accumulate h and logits per layer for this batch; write once both are present
static std::map<int, std::vector<float>> g_h, g_lg;
static std::vector<uint8_t> g_raw;

static bool parse_layer(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) return false;
    if (name[n] != '-') return false;
    char * end = nullptr;
    const long v = strtol(name + n + 1, &end, 10);
    if (end == name + n + 1) return false;
    il = (int) v;
    return true;
}

static void flush_layer(int il) {
    auto ih = g_h.find(il), ig = g_lg.find(il);
    if (ih == g_h.end() || ig == g_lg.end()) return;
    if ((int) ih->second.size() != g_n_embd || (int) ig->second.size() != g_n_expert) return;
    const uint32_t u = (uint32_t) il;
    fwrite(&u, sizeof u, 1, g_out);
    fwrite(ih->second.data(), sizeof(float), g_n_embd,  g_out);
    fwrite(ig->second.data(), sizeof(float), g_n_expert, g_out);
    g_records++;
    g_h.erase(ih); g_lg.erase(ig);
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    const char * nm = ggml_get_name(t);
    int il = -1;
    const bool want_h  = parse_layer(nm, "ffn_norm", il);
    const bool want_lg = !want_h && parse_layer(nm, "ffn_moe_logits", il);
    if (ask) return want_h || want_lg;
    if (!g_capture || (!want_h && !want_lg)) return true;
    // Decode only (n_tokens == 1)
    const int64_t ntok = t->ne[1];
    if (ntok != 1) return true;

    const size_t nb = ggml_nbytes(t);
    if (g_raw.size() < nb) g_raw.resize(nb);
    ggml_backend_tensor_get(t, g_raw.data(), 0, nb);
    if (t->type != GGML_TYPE_F32) return true;      // discard unexpected types

    const float * f = (const float *) g_raw.data();
    const int n = (int) t->ne[0];
    if (want_h) {
        if (g_n_embd == 0) g_n_embd = n;
        if (n != g_n_embd) return true;
        g_h[il].assign(f, f + n);
    } else {
        if (g_n_expert == 0) g_n_expert = n;
        if (n != g_n_expert) return true;
        g_lg[il].assign(f, f + n);
    }
    flush_layer(il);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: s10_p2_accuracy <model.gguf> <out.bin> <n_tokens>\n");
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    const int    n_want     = atoi(argv[3]);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "cannot load the model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 2048;
    // Use ubatch=1 to avoid slot exhaustion. Exhaustion substitutes the null
    // expert and yields a trace of already-degraded output. The prompt is short,
    // so slow prefill does not matter here.
    cp.n_batch  = 512;   // keep the logical batch large
    cp.n_ubatch = 1;     // only the micro-batch drops to 1
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create the context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ---- run the prompt through, without capturing ----
    const char * prompt =
        "Write a complete Python implementation of a thread-safe LRU cache "
        "with a doubly linked list, type hints, docstrings and unit tests, "
        "then explain the design trade-offs in detail.";
    std::vector<llama_token> toks(512);
    const int n_prompt = llama_tokenize(vocab, prompt, (int) strlen(prompt),
                                        toks.data(), (int) toks.size(), true, true);
    if (n_prompt <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n_prompt);

    g_out = fopen(out_path, "wb");
    if (!g_out) { fprintf(stderr, "cannot open the output file\n"); return 1; }
    uint8_t hdr[16] = {0};
    memcpy(hdr, "P2AC", 4);
    fwrite(hdr, sizeof hdr, 1, g_out);          // rewritten at the end

    llama_batch batch = llama_batch_get_one(toks.data(), n_prompt);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }

    // ---- capture while decoding ----
    g_capture = true;
    llama_token cur = llama_vocab_bos(vocab);
    {   // Pick one token from the previous logits
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0; float bv = lg[0];
        const int nv = llama_vocab_n_tokens(vocab);
        for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        cur = best;
    }
    for (int i = 0; i < n_want; ++i) {
        llama_batch b1 = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b1) != 0) break;
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0; float bv = lg[0];
        const int nv = llama_vocab_n_tokens(vocab);
        for (int k = 1; k < nv; ++k) if (lg[k] > bv) { bv = lg[k]; best = k; }
        cur = best;
        if (llama_vocab_is_eog(vocab, cur)) break;
        if ((i + 1) % 25 == 0) fprintf(stderr, "  %d tokens / %ld records\n", i + 1, g_records);
    }

    // Rewrite the header
    fseek(g_out, 0, SEEK_SET);
    memcpy(hdr, "P2AC", 4);
    *(uint32_t *) (hdr + 4)  = (uint32_t) g_n_embd;
    *(uint32_t *) (hdr + 8)  = (uint32_t) g_n_expert;
    *(uint32_t *) (hdr + 12) = (uint32_t) g_records;
    fwrite(hdr, sizeof hdr, 1, g_out);
    fclose(g_out);

    fprintf(stderr, "\ndone: n_embd=%d n_expert=%d %ld records -> %s\n",
            g_n_embd, g_n_expert, g_records, out_path);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
