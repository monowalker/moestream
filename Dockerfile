# MoEStream — patched llama-server. Self-contained; built directly by compose.
FROM ubuntu:24.04

ARG LLAMA_COMMIT=b0539c43ed13b16bf0d8a0840646faea65469702
ENV DEBIAN_FRONTEND=noninteractive

# Toolchain (identical command to research/docker/Dockerfile.dev, so layers are shared)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config python3 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan-dev glslc glslang-tools spirv-headers spirv-tools \
        mesa-vulkan-drivers vulkan-tools libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Node is required to build llama.cpp's bundled Web UI
RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ggml-org/llama.cpp.git llama.cpp \
    && cd llama.cpp && git checkout ${LLAMA_COMMIT}

# Apply the MoEStream patches. Only what apply.py actually needs is copied here:
# entrypoint.sh and spec_probe.cpp arrive later, so editing either of them does
# not invalidate this layer and force a 25-minute rebuild of llama.cpp.
COPY src/apply.py src/llama-moestream.h src/llama-moestream.cpp \
     src/expert_cache.hpp src/expert_cache.cpp /opt/moestream-src/
COPY src/expert_cache.hpp src/expert_cache.cpp /opt/llama.cpp/src/
RUN cp /opt/moestream-src/llama-moestream.h /opt/moestream-src/llama-moestream.cpp /opt/llama.cpp/src/ \
    && python3 /opt/moestream-src/apply.py /opt/llama.cpp

RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=ON \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON \
    && cmake --build /opt/llama.cpp/build -j"$(nproc)" --target llama llama-server llama-bench llama-perplexity \
    && rm -rf /opt/llama.cpp/build/CMakeFiles

# Ask llama.cpp itself which speculative types a GGUF supports, rather than
# keeping our own copy of its rule. Links the common library it just built.
COPY src/spec_probe.cpp /opt/moestream-src/spec_probe.cpp
RUN c++ -O2 -std=c++17 -o /usr/local/bin/moestream-spec-probe \
        /opt/moestream-src/spec_probe.cpp \
        -I/opt/llama.cpp/common -I/opt/llama.cpp/include -I/opt/llama.cpp/ggml/include \
        -I/opt/llama.cpp/vendor \
        -L/opt/llama.cpp/build/bin -lllama-common -lllama -lggml -lggml-base \
        -Wl,-rpath,/opt/llama.cpp/build/bin

COPY src/entrypoint.sh /usr/local/bin/moestream-entrypoint
RUN chmod +x /usr/local/bin/moestream-entrypoint

ENV XDG_RUNTIME_DIR=/tmp
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/moestream-entrypoint"]
