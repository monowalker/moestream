// MoEStream — ask llama.cpp which kinds of speculative decoding a GGUF supports.
//
//   The rule is not ours. Whether a model can self-speculate depends on the
//   tensors it carries, and upstream already decides that in
//   common_speculative_types_from_gguf(). Re-implementing the test in the
//   launcher would mean maintaining a copy of someone else's rule and silently
//   drifting from it — the metadata key `nextn_predict_layers` and the tensor
//   `blk.N.nextn.eh_proj.weight` already disagree on real quantizations.
//
//   So this links llama.cpp's own `common` and calls the function. When
//   upstream changes what it detects, this follows on the next build with no
//   edit here.
//
//   Prints one type name per line; nothing at all when the model supports none.
#include "speculative.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    for (const auto type : common_speculative_types_from_gguf(argv[1])) {
        printf("%s\n", common_speculative_type_to_str(type).c_str());
    }
    return 0;
}
