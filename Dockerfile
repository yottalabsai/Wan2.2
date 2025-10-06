# Use the devel version: includes nvcc/headers, useful for compiling flash_attn and others
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1

# Base dependencies (including ffmpeg, libsndfile, ninja, cmake, and other build & audio dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git tini ffmpeg libsndfile1 libgl1 libglib2.0-0 python3 python3-pip python3-dev \
    build-essential cmake ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Set PYTHONPATH to include the workspace directory
ENV PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

# Copy current directory contents to workdir
COPY . .

# Default sm_90 (H200 Hopper). For RTX 4090 (Ada), pass 8.9 at build time.
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 12.0" 
# Pre-set build-time environment for PyTorch CUDA extensions
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Install Python dependencies
RUN pip install --no-cache-dir setuptools wheel packaging && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_animate.txt && \
    pip install --no-cache-dir "huggingface_hub[cli]"

# Align UID/GID with the host to avoid write permission issues
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app && chown -R app:app /workspace
USER app

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash", "entrypoint.sh"]
