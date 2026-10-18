# syntax=docker/dockerfile:1
# The Whisper model repo is private, so instead of downloading it we copy the
# weights from the existing public image that already has them baked in.
FROM dezheen89/whisper-turbo-ctranslate:v3 AS whisper_model

FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Shared base ────────────────────────────────────────────────────────────
# torch is installed once here and shared with every service's virtualenv
# (--system-site-packages below), so the three environments differ only in
# the libraries that actually conflict (mainly the transformers version).
RUN pip install "setuptools<81" wheel cython numpy==1.26.4
# torch 2.7.1 + CUDA 12.8: the first builds with kernels for Blackwell GPUs (sm_120).
# The old torch 2.3.1/cu121 cannot start on them ("CUDA initialization failed").
RUN pip install --default-timeout=300 --retries 5 torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
# The build machine has no GPU, so this only inspects what was compiled into the wheel. It always
# prints the details, and fails only on a wrong CUDA version or an arch list that lacks sm_120.
RUN python - <<'PY'
import torch
cuda = torch.version.cuda
try:
    flags = torch._C._cuda_getArchFlags() or ""
except Exception as e:
    flags = ""
    print("could not read arch flags:", repr(e))
archs = torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", cuda, "| arch flags:", flags, "| arch list:", archs)
assert cuda and cuda.startswith("12.8"), f"expected a CUDA 12.8 torch build, got {cuda}"
if flags or archs:
    assert "sm_120" in flags or "sm_120" in archs, "torch build has no Blackwell (sm_120) kernels"
PY
RUN pip install runpod huggingface_hub

# CosyVoice isn't pip-installable; it's vendored as source (see upstream docs).
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice \
 && find /app/CosyVoice -name .git -prune -exec rm -rf {} +

# onnxruntime-gpu 1.18 needs cuDNN 8 and has no Blackwell kernels. Run CosyVoice's two small ONNX
# models (speaker embedding, speech tokenizer) on the CPU instead; they only process the short
# reference clip once per request.
RUN python - <<'PY'
import re
p = "/app/CosyVoice/cosyvoice/cli/frontend.py"
s = open(p, encoding="utf-8").read()
pat = r'providers=\["CUDAExecutionProvider" if torch\.cuda\.is_available\(\) else\s*"CPUExecutionProvider"\]'
s2 = re.sub(pat, 'providers=["CPUExecutionProvider"]', s)
assert s2 != s, "frontend.py patch did not match"
open(p, "w", encoding="utf-8").write(s2)
print("patched frontend.py: ONNX sessions on CPU")
PY

RUN python -m venv --system-site-packages /venvs/tts \
 && python -m venv --system-site-packages /venvs/w2v \
 && python -m venv --system-site-packages /venvs/whisper

# ── Text-to-speech: CosyVoice3 (transformers 4.51.3) ──────────────────────
COPY reqs/tts.txt /app/reqs/tts.txt
RUN /venvs/tts/bin/pip install --no-build-isolation --default-timeout=200 -r /app/reqs/tts.txt
RUN /venvs/tts/bin/python -c "import sys; sys.path.extend(['/app/CosyVoice', '/app/CosyVoice/third_party/Matcha-TTS']); import runpod; from cosyvoice.cli.cosyvoice import AutoModel; print('tts import check ok')"

# ── Speech-to-text: w2v-bert (transformers 4.41.2) ────────────────────────
COPY reqs/w2v.txt /app/reqs/w2v.txt
RUN /venvs/w2v/bin/pip install --default-timeout=200 -r /app/reqs/w2v.txt
RUN /venvs/w2v/bin/python -c "import transformers, librosa, soundfile, runpod; print('w2v import check ok', transformers.__version__)"

# ── Speech-to-text: Whisper Turbo (transformers 4.46.3 + faster-whisper) ──
COPY reqs/whisper.txt /app/reqs/whisper.txt
RUN /venvs/whisper/bin/pip install --default-timeout=200 -r /app/reqs/whisper.txt
RUN /venvs/whisper/bin/python -c "import transformers, faster_whisper, ctranslate2, runpod; print('whisper import check ok', transformers.__version__)"

# ── Model weights, baked in at build time ─────────────────────────────────
# HF_TOKEN comes from a BuildKit secret (not a build arg), so it never ends
# up in this public image's layers or `docker history`.
ARG TTS_MODEL_ID=computeram/cosyvoice3-badini-tts
ARG W2V_MODEL_ID=computeram/badini-w2v-bert-normalized
COPY downloads/ /app/downloads/

RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" MODEL_ID="$TTS_MODEL_ID" \
    python /app/downloads/download_tts.py

RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" MODEL_ID="$W2V_MODEL_ID" \
    /venvs/w2v/bin/python /app/downloads/download_w2v.py

# Whisper weights come from the stage at the top of this file. Same path as in
# that image (/app/model_cache), because its MODEL_PATH file records absolute
# paths.
COPY --from=whisper_model /app/model_cache /app/model_cache

# Bundled TTS reference voices (natural + 5 emotions).
COPY prompt-*.wav prompt-*.txt /app/

COPY services/ /app/services/
COPY service_runner.py dispatcher.py /app/

CMD ["python", "-u", "/app/dispatcher.py"]
