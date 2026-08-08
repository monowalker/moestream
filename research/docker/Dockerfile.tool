ARG BASE=moestream-llama:local
FROM ${BASE}
ARG TOOL=expert_trace
COPY tools/analysis /work/tools
RUN cmake -S /work/tools/${TOOL} -B /work/build/${TOOL} -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /work/build/${TOOL} -j"$(nproc)"
ENV TOOL=${TOOL}
CMD ["/bin/sh","-c","/work/build/${TOOL}/${TOOL}"]
