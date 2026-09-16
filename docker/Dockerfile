# CUDA image with mamba-ssm (Mamba-3) and causal-conv1d built from source.
#
# Default target is SM_120 (RTX 5090 / Blackwell). For Hopper pass --build-arg CUDA_ARCH=90,
# for Ada --build-arg CUDA_ARCH=89, for Ampere (A100) --build-arg CUDA_ARCH=80. Only the
# SM_120 build has been exercised by the authors.
#
#   docker build -f docker/Dockerfile -t stock-embed-mamba3 .
#   docker run --rm --gpus all stock-embed-mamba3 pytest -q -rs
#
# The image the released weights were validated in cloned both projects on 2026-08-08, at the
# upstream commits pinned below, with torch 2.12.0.dev20260408+cu128.

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
ARG CUDA_ARCH=120
ARG MAX_JOBS=8
ARG TORCH_SPEC="torch"
ARG CAUSAL_CONV1D_REF=4f6ae4e26ae5fe8af9372f8d312ab25cc4595223
ARG MAMBA_REF=e9594ce1c732d97440f0332fdc43170a2294dbfa

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        git ninja-build gcc g++ curl && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

# PyTorch nightly with CUDA 12.8 (SM_120 runtime support). Nightlies are pruned from the
# index, so the default is unpinned; pass --build-arg TORCH_SPEC="torch==<version>" to pin.
RUN pip install --no-cache-dir "${TORCH_SPEC}" --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir packaging ninja wheel einops transformers

# Both setup.py files emit -gencode flags for several architectures. Rewrite every entry
# to the requested one so the build compiles a single architecture, and fail loudly if
# the upstream pattern has drifted.
ENV MAX_JOBS=${MAX_JOBS}
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
RUN git init /tmp/causal-conv1d && cd /tmp/causal-conv1d && \
    git remote add origin https://github.com/Dao-AILab/causal-conv1d.git && \
    git fetch --depth 1 origin ${CAUSAL_CONV1D_REF} && git checkout FETCH_HEAD && \
    sed -i -E "s/arch=compute_[0-9]+,code=sm_[0-9]+/arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}/g" setup.py && \
    grep -q "sm_${CUDA_ARCH}" setup.py && \
    pip install --no-cache-dir --no-build-isolation --no-deps . && \
    rm -rf /tmp/causal-conv1d

ENV MAMBA_FORCE_BUILD=TRUE
# Upstream stopped shipping the Mamba-1 selective_scan_cuda extension by default
# (2026-07). The flag keeps the full CUDA build so every Mamba class imports.
ENV MAMBA_KEEP_CUDA_BUILD=TRUE
# --no-deps on purpose: upstream also lists tilelang, quack-kernels and apache-tvm-ffi,
# which are optional fast paths that fall back to the Triton kernel at import time and
# are not exercised by this model.
RUN git init /tmp/mamba && cd /tmp/mamba && \
    git remote add origin https://github.com/state-spaces/mamba.git && \
    git fetch --depth 1 origin ${MAMBA_REF} && git checkout FETCH_HEAD && \
    sed -i -E "s/arch=compute_[0-9]+,code=sm_[0-9]+/arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}/g" setup.py && \
    grep -q "sm_${CUDA_ARCH}" setup.py && \
    pip install --no-cache-dir --no-build-isolation --no-deps . && \
    rm -rf /tmp/mamba

COPY . /workspace/
RUN pip install --no-cache-dir ".[model,test,examples]"
