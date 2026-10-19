# Build: docker build -t llama-cpp-539f245-xppl:server-cuda12.8 -f eval/llama_cpp_xppl.Dockerfile .
# Context: src-539f245.tar.gz (git archive 539f245) + xppl.patch (eval/llama_cpp_prompt_logprobs.patch)
# Built and verified on gpus-vm (192.168.10.11): ~/llama-cpp-xppl-docker/
# llama.cpp @ 539f245 + xppl patch (prompt_logprobs) — CUDA 12.8, sm_86 (RTX 3090)
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake patch python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src-539f245.tar.gz .
RUN tar xzf src-539f245.tar.gz && rm src-539f245.tar.gz
COPY xppl.patch .
RUN patch -p1 < xppl.patch
# expose the CUDA driver-API stub so ggml-cuda links (real libcuda comes from the
# nvidia container runtime at inference time); the .so.1 symlink + rpath-link let
# the final llama-server link resolve the transitive DT_NEEDED
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1
RUN cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64"
RUN cmake --build build --config Release --target llama-server -j"$(nproc)"

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/build/bin/llama-server /app/
COPY --from=build /app/build/bin/*.so* /app/
ENV LD_LIBRARY_PATH=/app
EXPOSE 8080
ENTRYPOINT ["/app/llama-server"]
