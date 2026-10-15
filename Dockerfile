# BeyondWaveform reproduction development environment.
# These versions are engineering choices, not versions reported in the paper.
# Build on an x86_64 Linux host with: docker build -t beyondwaveform:dev .
# CUDA execution requires an NVIDIA driver and NVIDIA Container Toolkit on the host.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    MPLBACKEND=Agg

# Ubuntu 22.04 supplies Python 3.10. ffmpeg/libsndfile cover audio IO;
# libgomp supports CPU numerical/ONNX execution. No CUDA compiler is required
# for the intended initial pure-PyTorch implementation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        build-essential pkg-config git ca-certificates curl \
        ffmpeg libsndfile1 libsndfile1-dev libgomp1 \
        tini \
        openssh-server \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @openai/codex@latest
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

RUN printf '%s\n' \
        'Port 22' \
        'PubkeyAuthentication yes' \
        'PasswordAuthentication no' \
        'PermitRootLogin no' \
        'AllowUsers codex' \
        > /etc/ssh/sshd_config.d/codex.conf

# Install matching CUDA wheels from the official PyTorch index.
# torchvision is deliberately omitted: the planned attack has no image model.
RUN python -m pip install \
        torch==2.4.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# A constraint file prevents later dependency installation from replacing the
# CUDA build or the core numerical stack. Direct dependencies are pinned below;
# this is not yet a complete hash-locked transitive dependency environment.
RUN printf '%s\n' \
        'torch==2.4.1+cu121' \
        'torchaudio==2.4.1+cu121' \
        'numpy==1.26.4' \
        'numba==0.60.0' \
        'llvmlite==0.43.0' \
        'huggingface-hub==0.26.5' \
        'tokenizers==0.20.3' \
        'fsspec==2024.9.0' \
        > /opt/reproduction-constraints.txt
ENV PIP_CONSTRAINT=/opt/reproduction-constraints.txt

# Models: Hugging Face WavLM/Whisper/CTC, plus native OpenAI Whisper.
# Evaluation: edit-distance scoring and TorchMetrics DNSMOS/NISQA dependencies.
# ONNX Runtime runs quality scoring on CPU, avoiding a second GPU/cuDNN stack.
# No checkpoints or datasets are downloaded while building the image.
RUN python -m pip install \
        numpy==1.26.4 scipy==1.14.1 \
        numba==0.60.0 llvmlite==0.43.0 \
        transformers==4.46.3 \
        huggingface-hub==0.26.5 tokenizers==0.20.3 safetensors==0.4.5 \
        accelerate==1.1.1 datasets==3.1.0 fsspec==2024.9.0 \
        tiktoken==0.8.0 \
        librosa==0.10.2.post1 soundfile==0.12.1 soxr==0.5.0.post1 \
        jiwer==3.0.5 torchmetrics==1.6.1 onnxruntime==1.20.1 \
        requests==2.32.3 \
        pandas==2.2.3 pyarrow==18.1.0 openpyxl==3.1.5 \
        matplotlib==3.9.3 seaborn==0.13.2 \
        PyYAML==6.0.2 tqdm==4.67.1 tensorboard==2.18.0 \
        jupyterlab==4.3.2 ipykernel==6.29.5 \
        pytest==8.3.4 \
    && python -m pip check

RUN python -c "import pkg_resources" \
    && python -m pip install --no-build-isolation openai-whisper==20240930 \
    && python -m pip check

    # Build-time CPU/import smoke check: no host GPU or network model download needed.
# This checks imports and a torchaudio operation, not full research reproduction.
RUN python -c "import torch, torchaudio, whisper, librosa, soundfile, jiwer, onnxruntime; from transformers import WavLMModel, WhisperForConditionalGeneration, HubertForCTC, Wav2Vec2ForCTC; from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score; from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment; assert torch.__version__ == '2.4.1+cu121'; assert torchaudio.__version__ == '2.4.1+cu121'; assert torch.version.cuda == '12.1'; x = torch.zeros(16000); y = torchaudio.functional.resample(x, 16000, 8000); assert y.numel() == 8000; print('Core imports and CPU audio smoke check passed')"

# Record the environment actually resolved by this build for the experiment log.
RUN python -m pip freeze --all > /opt/environment.freeze.txt

WORKDIR /workspace
RUN mkdir -p /cache/huggingface /cache/torch /workspace

ARG CODEX_UID=1011
ARG CODEX_GID=1011

RUN groupadd --gid "${CODEX_GID}" codex \
    && useradd --uid "${CODEX_UID}" --gid "${CODEX_GID}" \
       -m -s /bin/bash codex \
    && mkdir -p /home/codex/.ssh /home/codex/.codex /run/sshd \
    && chmod 700 /home/codex/.ssh \
    && chown -R codex:codex /home/codex \
    && chmod 755 /run/sshd
RUN chown -R codex:codex /cache
 
# Mount the future implementation at /workspace and persist /cache separately.
# Nothing is copied into this image, so changing research code needs no rebuild.
# Example interactive run:
# docker run --rm -it --gpus all --shm-size=8g \
#   -v "$(pwd):/workspace" -v beyondwaveform-cache:/cache beyondwaveform:dev
# Start Jupyter explicitly when needed; its authentication remains enabled:
# docker run --rm -it --gpus all --shm-size=8g -p 127.0.0.1:8888:8888 \
#   -v "$(pwd):/workspace" -v beyondwaveform-cache:/cache beyondwaveform:dev \
#   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# For a remote Docker host, forward that host's localhost:8888 through VS Code/SSH.
# HiFi-GAN/UTMOS weights and resolved defense repositories must be supplied later.
# This container intentionally uses root for a simple remote-development setup;
# files written into a Linux bind mount may therefore be owned by root.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 22 8888

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/docker-entrypoint.sh"]