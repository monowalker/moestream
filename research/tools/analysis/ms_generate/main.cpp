// =============================================================================
// MoEStream — minimal text generation, demonstrating streaming operation
//
// With MOESTREAM=1 the expert weights are streamed from SSD rather than kept
// resident in RAM. The same binary compares against MOESTREAM=0 (fully resident).
// =============================================================================
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <time.h>

namespace moestream { void report(); void set_gguf_path(const char *); }

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> [n_predict] [n_gpu_layers] [n_ubatch]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string prompt = argv[2];
    const int n_predict = argc > 3 ? atoi(argv[3]) : 48;
    const int n_gpu     = argc > 4 ? atoi(argv[4]) : 99;
    const int n_ub      = argc > 5 ? atoi(argv[5]) : 512;

    moestream::set_gguf_path(model_path);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;

    const double t_load0 = now_s();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const double t_load = now_s() - t_load0;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(prompt.size() + 16);
    int nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                            toks.data(), (int) toks.size(), true, true);
    if (nt < 0) { toks.resize(-nt);
        nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                            toks.data(), (int) toks.size(), true, true); }
    toks.resize(nt);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 4096;
    cp.n_batch = 4096;
    cp.n_ubatch = (uint32_t) n_ub;
    cp.n_batch = 4096;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));

    fprintf(stderr, "\n--- prompt (%d tokens) ---\n%s\n--- generation ---\n", nt, prompt.c_str());

    // prefill
    const double t_pf0 = now_s();
    llama_batch batch = llama_batch_get_one(toks.data(), nt);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
    const double t_prefill = now_s() - t_pf0;

    // decode
    const double t_dec0 = now_s();
    std::vector<double> tok_ms;
    tok_ms.reserve(n_predict);
    int n_gen = 0;
    for (; n_gen < n_predict; ++n_gen) {
        const double tt0 = now_s();
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[256];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof buf, 0, true);
        if (n > 0) { fwrite(buf, 1, n, stdout); fflush(stdout); }
        llama_batch b1 = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, b1) != 0) break;
        tok_ms.push_back((now_s() - tt0) * 1000.0);
    }
    const double t_decode = now_s() - t_dec0;

    printf("\n");
    fprintf(stderr, "\n--- measurements ---\n");
    fprintf(stderr, "  model load     : %.2f s\n", t_load);
    fprintf(stderr, "  prefill        : %.2f s (%d tok, %.1f tok/s)\n",
            t_prefill, nt, nt / t_prefill);
    fprintf(stderr, "  decode         : %.2f s (%d tok, %.2f tok/s)\n",
            t_decode, n_gen, n_gen / t_decode);
    if (tok_ms.size() >= 8) {
        // Steady state: look only at the second half, excluding cold start
        const size_t h = tok_ms.size() / 2;
        double sum = 0; for (size_t i = h; i < tok_ms.size(); ++i) sum += tok_ms[i];
        const double steady = sum / double(tok_ms.size() - h);
        std::vector<double> srt(tok_ms.begin() + h, tok_ms.end());
        std::sort(srt.begin(), srt.end());
        fprintf(stderr, "  steady state   : %.2f ms/token = %.2f tok/s"
                        "  (p50 %.1f / p99 %.1f ms)\n",
                steady, 1000.0 / steady, srt[srt.size()/2], srt[(size_t)(srt.size()*0.99)]);
        fprintf(stderr, "    first half   : %.2f ms/token (incl. cold start)\n",
                [&]{ double s2=0; for (size_t i=0;i<h;++i) s2+=tok_ms[i]; return s2/double(h); }());
    }

    // Peak RSS: this project's primary objective (§4.2, PR-1)
    {
        FILE * f = fopen("/proc/self/status", "r");
        if (f) { char l[256];
            while (fgets(l, sizeof l, f))
                if (!strncmp(l, "VmHWM:", 6)) { fprintf(stderr, "  peak RSS       : %s", l + 6); break; }
            fclose(f); }
    }
    moestream::report();

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
