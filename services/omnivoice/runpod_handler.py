"""Service "omnivoice" of the combined image: RunPod Serverless handler for the Badini Kurdish OmniVoice fine-tune (zero-shot TTS, 3 voices).

Model: Dezheen89/omnivoice (fine-tune of k2-fsa/OmniVoice), baked into the image at build time
(/app/models/omnivoice). Loaded on the first job (the dispatcher starts this service lazily).
OmniVoice clones a voice from a short reference clip + its exact transcript. "Emotion" simply selects
which real recording of the speaker is used as the reference (bundled as /app/prompt-*.wav/.txt).

Input (send with "service": "omnivoice"):
    {"text": "...", "voice": "hayfa"|"abdullah"|"bafreen"?, "emotion": "natural"|"angry"|"sad"|"happy"|"fear"|"surprise"?,
     "speed": 1.0?,                              # 0.5 - 2.0
     "normalize": true?,                        # numbers / dates / abbreviations -> Badini words first
     "ref_audio": "<base64 wav>"?, "ref_text": "..."?}   # your own reference clip instead of a bundled voice
Output:
    {"ok": true, "audio_base64": "...", "sample_rate": N, "format": "wav", "duration_s": S}
"""
import base64
import io
import os
import re
import tempfile
import threading
import time
import traceback

import runpod
import soundfile as sf
import torch

from badini_text import prepare_text

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/omnivoice")
APP = os.getenv("APP_DIR", "/app")
LANGUAGE = "kmr"
VOICES = ("hayfa", "abdullah", "bafreen")
EMOTIONS = ("natural", "angry", "sad", "happy", "fear", "surprise")
MAX_CHARS = 6000

_model = None
_lock = threading.Lock()
_prompts = {}          # (voice, emotion) -> cached voice-clone prompt
_load_report = None    # returned once with the first job of a fresh worker


def prompt_base(voice, emotion):
    """Path stem of the bundled reference clip (hayfa's emotions have no voice name in the file)."""
    if voice == "hayfa":
        return f"{APP}/prompt-hayfa" if emotion == "natural" else f"{APP}/prompt-{emotion}"
    return f"{APP}/prompt-{voice}-{emotion}"


def get_model():
    global _model, _load_report
    with _lock:
        if _model is None:
            from omnivoice import OmniVoice
            t = time.perf_counter()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[LOAD] OmniVoice from {MODEL_DIR} on {device} ...", flush=True)
            _model = OmniVoice.from_pretrained(MODEL_DIR, device_map=device,
                                               dtype=torch.float16 if device == "cuda" else torch.float32)
            _load_report = {"model_load_s": round(time.perf_counter() - t, 1)}
            print(f"[LOAD] ready in {_load_report['model_load_s']} s", flush=True)
    return _model


def get_prompt(model, voice, emotion):
    key = (voice, emotion)
    if key not in _prompts:
        base = prompt_base(voice, emotion)
        with open(base + ".txt", encoding="utf-8") as f:
            ref_text = f.read().strip()
        _prompts[key] = model.create_voice_clone_prompt(ref_audio=base + ".wav", ref_text=ref_text)
    return _prompts[key]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")


def chunk_text(text, soft_words=60, hard_words=90):
    """Split long text at sentence ends (a hard word-count cut if a run has none) so no piece is huge."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return [text]
    chunks, cur, cur_words = [], [], 0
    for sent in sentences:
        words = sent.split()
        if len(words) > hard_words:
            if cur:
                chunks.append(" ".join(cur)); cur, cur_words = [], 0
            chunks += [" ".join(words[i:i + hard_words]) for i in range(0, len(words), hard_words)]
            continue
        if cur_words and (cur_words + len(words) > hard_words or cur_words >= soft_words):
            chunks.append(" ".join(cur)); cur, cur_words = [], 0
        cur.extend(words); cur_words += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def generate_piece(model, piece, prompt, speed=None):
    """One piece of text -> numpy audio. The model now and then fails or returns almost empty audio for a
    normal sentence (the same request works next time): try up to 3 times and keep the longest result."""
    min_seconds = max(0.3, 0.03 * len(piece) / (speed or 1.0))
    best, best_s, last_error = None, -1.0, None
    for attempt in range(3):
        try:
            audio = model.generate(text=piece, language=LANGUAGE, voice_clone_prompt=prompt, speed=speed)[0]
        except Exception as e:
            last_error = e
            print(f"[tts] attempt {attempt + 1} failed: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            continue
        seconds = len(audio) / model.sampling_rate
        if seconds > best_s:
            best, best_s = audio, seconds
        if seconds >= min_seconds:
            break
    if best is None:
        raise last_error
    return best


def handler(job):
    inp = job.get("input") or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "Missing 'text'."}
    if len(text) > MAX_CHARS:
        return {"ok": False, "error": f"Text is too long ({len(text)} characters, max {MAX_CHARS})."}
    try:
        speed = min(2.0, max(0.5, float(inp.get("speed") or 1.0)))
    except (TypeError, ValueError):
        return {"ok": False, "error": "'speed' must be a number between 0.5 and 2."}
    speed = None if abs(speed - 1.0) < 1e-6 else speed   # None = the model's own natural pace
    voice = inp.get("voice", "hayfa")
    emotion = inp.get("emotion", "natural")
    custom_wav = None
    if not inp.get("ref_audio"):
        if voice not in VOICES:
            return {"ok": False, "error": f"Unknown voice '{voice}'. Use one of: {list(VOICES)}."}
        if emotion not in EMOTIONS:
            return {"ok": False, "error": f"Unknown emotion '{emotion}'. Use one of: {list(EMOTIONS)}."}
    elif not (inp.get("ref_text") or "").strip():
        return {"ok": False, "error": "'ref_audio' needs its exact transcript in 'ref_text'."}

    try:
        t0 = time.perf_counter()
        model = get_model()
        if inp.get("ref_audio"):
            fd, custom_wav = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64decode(inp["ref_audio"]))
            prompt = model.create_voice_clone_prompt(ref_audio=custom_wav, ref_text=inp["ref_text"].strip())
        else:
            prompt = get_prompt(model, voice, emotion)

        spoken = prepare_text(text) if inp.get("normalize", True) else text
        pieces = [generate_piece(model, p, prompt, speed) for p in chunk_text(spoken)]
        import numpy as np
        audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

        buf = io.BytesIO()
        sf.write(buf, audio, model.sampling_rate, format="WAV")
        result = {
            "ok": True,
            "audio_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "sample_rate": model.sampling_rate,
            "format": "wav",
            "duration_s": round(len(audio) / model.sampling_rate, 2),
            "generate_s": round(time.perf_counter() - t0, 2),
        }
        global _load_report
        if _load_report:
            result.update(_load_report)
            _load_report = None
        return result
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        return {"ok": False, "error": "Internal server error", "detail": type(e).__name__}
    finally:
        if custom_wav and os.path.exists(custom_wav):
            os.remove(custom_wav)


runpod.serverless.start({"handler": handler})
