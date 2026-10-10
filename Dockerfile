FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
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
RUN pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install runpod huggingface_hub

# CosyVoice isn't pip-installable; it's vendored as source (see upstream docs).
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice

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
ARG WHISPER_MODEL_ID=BadiniSpeechNLP/whisper-turbo-ctranslate
COPY downloads/ /app/downloads/

RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" MODEL_ID="$TTS_MODEL_ID" \
    python /app/downloads/download_tts.py

RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" MODEL_ID="$W2V_MODEL_ID" \
    /venvs/w2v/bin/python /app/downloads/download_w2v.py

RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" MODEL_ID="$WHISPER_MODEL_ID" \
    python /app/downloads/download_whisper.py

# Bundled TTS reference voices (natural + 5 emotions).
COPY prompt-*.wav prompt-*.txt /app/

COPY services/ /app/services/
COPY service_runner.py dispatcher.py /app/

CMD ["python", "-u", "/app/dispatcher.py"]
