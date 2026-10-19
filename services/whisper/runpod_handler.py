"""RunPod serverless handler for a Hugging Face Whisper model.

Works with either model format:
  * transformers  (model.safetensors / pytorch_model.bin)
  * CTranslate2   (model.bin, via faster-whisper)

The format is detected at build time and recorded in /app/model_cache/FORMAT,
so no guessing happens at request time.

The model is lazy-loaded as a thread-safe singleton and unloaded again after
an idle period, so serverless workers don't hold GPU memory while idle.
"""

import base64
import gc
import io
import os
import tempfile
import threading
import time
import traceback

import numpy as np
import runpod

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION — override any of these via RunPod environment variables
# ════════════════════════════════════════════════════════════════════
MODEL_ID  = os.getenv("MODEL_ID", "").strip()
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model_cache")

BEAM_SIZE      = int(os.getenv("BEAM_SIZE", "1"))       # 1=greedy (fast); 4-5=better
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
LANGUAGE       = os.getenv("LANGUAGE", "").strip() or None   # e.g. "ku"
TASK           = os.getenv("TASK", "transcribe")             # transcribe | translate

TARGET_SR       = 16_000
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))  # 25 MB

# Batched multi-chunk transcription (CTranslate2 format only): several
# independent 20s-ish chunks processed in one GPU forward pass instead of
# one RunPod job per chunk. Whisper's decoder max length is 448 regardless
# of model size, so that's the ceiling here rather than MAX_NEW_TOKENS
# (which is tuned for the single-chunk streaming path instead).
BATCH_MAX_LENGTH = int(os.getenv("BATCH_MAX_LENGTH", "448"))

# Idle deactivation: free the model from memory after this many idle seconds.
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "180"))
IDLE_CHECK_INTERVAL  = int(os.getenv("IDLE_CHECK_INTERVAL", "30"))

# ── Resolve format + device ──────────────────────────────────────────
def _read_format() -> str:
    path = os.path.join(MODEL_DIR, "FORMAT")
    if os.path.isfile(path):
        with open(path) as fh:
            return fh.read().strip()
    return os.getenv("MODEL_FORMAT", "transformers").strip()


MODEL_FORMAT = _read_format()

import torch  # noqa: E402  (imported after config so the log order reads well)

DEVICE       = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") or ("float16" if DEVICE == "cuda" else "int8")

torch.set_num_threads(1)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def _model_path() -> str:
    # download_model.py records exactly where the weights ended up, since
    # some repos nest them in a subfolder rather than the root.
    recorded = os.path.join(MODEL_DIR, "MODEL_PATH")
    if os.path.isfile(recorded):
        with open(recorded) as fh:
            path = fh.read().strip()
        if path and os.path.isdir(path):
            return path

    root = os.path.join(MODEL_DIR, MODEL_ID.split("/")[-1]) if MODEL_ID else ""
    if root and os.path.isdir(root):
        return root
    print("[WARN] Local snapshot not found — falling back to Hub id.", flush=True)
    return MODEL_ID


def _log_gpu(tag: str):
    if DEVICE == "cuda":
        mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[MEM] {tag}: {mb:.1f} MB on GPU", flush=True)


# ════════════════════════════════════════════════════════════════════
#  AUDIO HELPERS (transformers path only — faster-whisper reads files itself)
# ════════════════════════════════════════════════════════════════════

def read_audio(raw: bytes):
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio
    n = int(len(audio) * target / sr)
    if n <= 0:
        raise ValueError("Audio too short after resampling.")
    return np.interp(
        np.linspace(0, 1, n, endpoint=False),
        np.linspace(0, 1, len(audio), endpoint=False),
        audio,
    ).astype(np.float32)


def clean(text: str) -> str:
    for tok in ("<|endoftext|>", "<|startoftranscript|>", "<|nospeech|>"):
        text = text.replace(tok, "")
    return " ".join(text.split())


# ════════════════════════════════════════════════════════════════════
#  SINGLETON SERVICE
# ════════════════════════════════════════════════════════════════════

class TranscribeService:
    _instance = None
    _lock = threading.Lock()
    last_request_time = time.time()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("[COLD START] No live model — loading now...", flush=True)
                    cls._instance = cls._create()
        return cls._instance

    @classmethod
    def _create(cls):
        self = object.__new__(cls)
        self._load_model()
        return self

    # ---- Load ------------------------------------------------------
    def _load_model(self):
        path = _model_path()
        print(f"[LOAD] Format : {MODEL_FORMAT}", flush=True)
        print(f"[LOAD] Path   : {path}", flush=True)
        print(f"[LOAD] Device : {DEVICE}", flush=True)

        t0 = time.perf_counter()
        if MODEL_FORMAT == "ct2":
            self._load_ct2(path)
        else:
            self._load_transformers(path)
        print(f"[LOAD] Ready in {time.perf_counter() - t0:.1f}s", flush=True)
        _log_gpu("after load")

        self._warmup()
        print("[READY] Service ready for requests.", flush=True)

    def _load_ct2(self, path: str):
        from faster_whisper import WhisperModel
        print(f"[LOAD] compute_type: {COMPUTE_TYPE}", flush=True)
        self.model = WhisperModel(path, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.processor = None

        # ---- Batched multi-chunk transcription setup ----------------
        # Reuses the SAME loaded CTranslate2 weights that faster_whisper
        # just loaded above (it wraps one internally as `.model`) rather
        # than loading the model a second time into GPU memory. Only
        # needs an additional WhisperProcessor (tokenizer + feature
        # extractor config, no weights) to build batched feature tensors
        # the same way faster_whisper does internally for a single file.
        self.batch_ct2_model = getattr(self.model, "model", None)
        self.batch_processor = None
        if self.batch_ct2_model is not None:
            try:
                from transformers import WhisperProcessor
                repo_root = os.path.dirname(path.rstrip("/"))
                transformer_config_dir = os.path.join(repo_root, "transformer-config")
                if os.path.isdir(transformer_config_dir):
                    self.batch_processor = WhisperProcessor.from_pretrained(transformer_config_dir)
                    print("[LOAD] Batch transcription ready (WhisperProcessor loaded).", flush=True)
                else:
                    print(
                        f"[LOAD] No transformer-config/ found at {transformer_config_dir} "
                        "— batch transcription unavailable, single-chunk path unaffected.",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[LOAD] Batch transcription setup failed (non-fatal): {exc}", flush=True)
                self.batch_processor = None
        else:
            print("[LOAD] Could not access underlying CTranslate2 model — batch transcription unavailable.", flush=True)

    def _load_transformers(self, path: str):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        print(f"[LOAD] dtype: {TORCH_DTYPE}", flush=True)

        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        model.eval()
        self.model = model

        self.forced_decoder_ids = None
        if LANGUAGE:
            try:
                self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=LANGUAGE, task=TASK
                )
                print(f"[LOAD] Forced language={LANGUAGE}, task={TASK}", flush=True)
            except Exception as exc:
                print(f"[LOAD] forced_decoder_ids failed: {exc}", flush=True)

        # Batch transcription is CTranslate2-only for now.
        self.batch_ct2_model = None
        self.batch_processor = None

    # ---- Warmup ----------------------------------------------------
    def _warmup(self):
        print("[WARMUP] Running dummy inference...", flush=True)
        try:
            silence = np.zeros(TARGET_SR, dtype=np.float32)
            if MODEL_FORMAT == "ct2":
                segments, _ = self.model.transcribe(silence, beam_size=1)
                list(segments)
            else:
                feats = self.processor(
                    silence, sampling_rate=TARGET_SR, return_tensors="pt"
                )["input_features"].to(DEVICE, dtype=TORCH_DTYPE)
                with torch.inference_mode():
                    self.model.generate(feats, max_new_tokens=4, num_beams=1)
            print("[WARMUP] Done — kernels ready.", flush=True)
        except Exception as exc:
            print(f"[WARMUP] Skipped: {exc}", flush=True)

    # ---- Transcribe (single chunk — unchanged) ----------------------
    def transcribe(self, raw: bytes, *, lang, task, beam_size, max_tokens) -> dict:
        if MODEL_FORMAT == "ct2":
            return self._transcribe_ct2(raw, lang, task, beam_size)
        return self._transcribe_transformers(raw, lang, task, beam_size, max_tokens)

    def _transcribe_ct2(self, raw, lang, task, beam_size) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()

            t0 = time.perf_counter()
            segments, info = self.model.transcribe(
                tmp.name, language=lang, task=task,
                beam_size=beam_size, vad_filter=True,
            )
            seg_list, parts = [], []
            for seg in segments:
                parts.append(seg.text.strip())
                seg_list.append({
                    "start": round(seg.start, 2),
                    "end":   round(seg.end, 2),
                    "text":  seg.text.strip(),
                })
            elapsed = round((time.perf_counter() - t0) * 1000, 2)

        text = " ".join(p for p in parts if p).strip()
        if not text:
            return {"ok": True, "text": "", "message": "No speech detected."}

        return {
            "ok": True, "text": text, "segments": seg_list,
            "language": info.language,
            "language_prob": round(info.language_probability, 4),
            "duration_s": round(info.duration, 2),
            "device": DEVICE, "model": MODEL_ID, "inference_ms": elapsed,
        }

    def _transcribe_transformers(self, raw, lang, task, beam_size, max_tokens) -> dict:
        audio, sr = read_audio(raw)
        if audio.size == 0:
            return {"ok": False, "error": "Audio is empty."}

        audio = resample(audio, sr, TARGET_SR)
        duration = len(audio) / TARGET_SR

        feats = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        )["input_features"].to(DEVICE, dtype=TORCH_DTYPE)

        gen_kw = {"max_new_tokens": max_tokens, "num_beams": beam_size}
        fdi = getattr(self, "forced_decoder_ids", None)
        if lang:
            try:
                fdi = self.processor.get_decoder_prompt_ids(language=lang, task=task)
            except Exception as exc:
                print(f"[handler] forced_decoder_ids failed: {exc}", flush=True)
        if fdi:
            gen_kw["forced_decoder_ids"] = fdi

        t0 = time.perf_counter()
        with torch.inference_mode():
            ids = self.model.generate(feats, **gen_kw)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        text = clean(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
        if not text:
            return {"ok": True, "text": "", "message": "No speech detected."}

        return {
            "ok": True, "text": text,
            "duration_s": round(duration, 2),
            "device": DEVICE, "model": MODEL_ID, "inference_ms": elapsed,
        }

    # ---- Transcribe (batched multi-chunk, CTranslate2 only) ---------
    def _build_prompt_tokens(self, lang: str, task: str):
        forced_decoder_ids = self.batch_processor.get_decoder_prompt_ids(language=lang, task=task)
        tokenizer = self.batch_processor.tokenizer
        tokens = ["<|startoftranscript|>"]
        tokens.extend(tokenizer.convert_ids_to_tokens(tid) for _, tid in forced_decoder_ids)
        # de-dupe while preserving order, same as the reference script
        seen = []
        for t in tokens:
            if t not in seen:
                seen.append(t)
        return seen

    def _detect_batch_language(self, first_audio: np.ndarray) -> str:
        import ctranslate2
        inputs = self.batch_processor(first_audio, sampling_rate=TARGET_SR, return_tensors="np")
        feat = inputs.input_features[0].astype(np.float32)
        probe = ctranslate2.StorageView.from_array(feat[np.newaxis, :, :])
        results = self.batch_ct2_model.detect_language(probe)
        if results and results[0]:
            token = results[0][0][0]
            return token.replace("<|", "").replace("|>", "")
        return "en"

    def transcribe_batch(self, raw_chunks, *, lang, task, beam_size):
        if MODEL_FORMAT != "ct2" or self.batch_processor is None or self.batch_ct2_model is None:
            raise RuntimeError("Batch transcription is not available for this model.")

        import ctranslate2

        audios = []
        for raw in raw_chunks:
            audio, sr = read_audio(raw)
            audios.append(resample(audio, sr, TARGET_SR))

        detected_lang = lang or self._detect_batch_language(audios[0])
        prompt_tokens = self._build_prompt_tokens(detected_lang, task)
        prompts = [prompt_tokens for _ in audios]

        features_list = []
        for audio in audios:
            inputs = self.batch_processor(audio, sampling_rate=TARGET_SR, return_tensors="np")
            feat = inputs.input_features[0].astype(np.float32)
            if feat.shape[0] != 128:
                raise ValueError(f"Unexpected feature shape: {feat.shape}. Expected (128, time).")
            features_list.append(feat)
        stacked = np.ascontiguousarray(np.stack(features_list, axis=0), dtype=np.float32)
        features_view = ctranslate2.StorageView.from_array(stacked)

        t0 = time.perf_counter()
        outputs = self.batch_ct2_model.generate(
            features_view, prompts=prompts, beam_size=beam_size, max_length=BATCH_MAX_LENGTH,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        tokenizer = self.batch_processor.tokenizer
        results = []
        for audio, out in zip(audios, outputs):
            text = tokenizer.decode(out.sequences_ids[0], skip_special_tokens=True).strip()
            results.append({
                "ok": True,
                "text": text,
                "duration_s": round(len(audio) / TARGET_SR, 2),
                "language": detected_lang,
                "device": DEVICE,
                "model": MODEL_ID,
            })
        return results, elapsed_ms

    # ---- Teardown --------------------------------------------------
    @classmethod
    def deactivate(cls):
        with cls._lock:
            if cls._instance is None:
                return
            print("[IDLE] Deactivating worker — freeing model...", flush=True)
            inst = cls._instance
            for attr in ("model", "processor", "batch_ct2_model", "batch_processor"):
                try:
                    delattr(inst, attr)
                except Exception:
                    pass
            cls._instance = None

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        _log_gpu("after deactivation")
        print("[IDLE] Memory freed. Will reload on next request.", flush=True)


# ════════════════════════════════════════════════════════════════════
#  IDLE MONITOR
# ════════════════════════════════════════════════════════════════════

def _idle_monitor():
    while True:
        time.sleep(IDLE_CHECK_INTERVAL)
        if TranscribeService._instance is None:
            continue
        idle_for = time.time() - TranscribeService.last_request_time
        if idle_for >= IDLE_TIMEOUT_SECONDS:
            print(f"[IDLE] No requests for {idle_for:.0f}s.", flush=True)
            TranscribeService.deactivate()


threading.Thread(target=_idle_monitor, daemon=True).start()


# ════════════════════════════════════════════════════════════════════
#  RUNPOD HANDLER
# ════════════════════════════════════════════════════════════════════

def handler(job):
    try:
        TranscribeService.last_request_time = time.time()
        print("[REQUEST] Received.", flush=True)

        inp = job.get("input", {})

        # ---- Batch mode: several chunks in one RunPod job -----------
        # New, additive path — only triggers when the caller sends a
        # "chunks" array. Anything sending the existing "wav_base64"
        # field (the whole rest of this handler) is completely
        # unaffected and behaves exactly as before.
        chunk_list = inp.get("chunks")
        if chunk_list:
            lang      = inp.get("language", LANGUAGE)
            task      = inp.get("task", TASK)
            beam_size = int(inp.get("beam_size", inp.get("num_beams", BEAM_SIZE)))

            raw_chunks = []
            for b64 in chunk_list:
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    return {"ok": False, "error": "Invalid base64 data in chunks."}
                if not raw:
                    return {"ok": False, "error": "Empty audio chunk in chunks."}
                if len(raw) > MAX_AUDIO_BYTES:
                    mb = MAX_AUDIO_BYTES // (1024 * 1024)
                    return {"ok": False, "error": f"A chunk exceeds {mb} MB limit."}
                raw_chunks.append(raw)

            service = TranscribeService.get_instance()
            TranscribeService.last_request_time = time.time()
            try:
                results, elapsed_ms = service.transcribe_batch(
                    raw_chunks, lang=lang, task=task, beam_size=beam_size
                )
            except RuntimeError as exc:
                return {"ok": False, "error": str(exc)}
            TranscribeService.last_request_time = time.time()

            print(f"[REQUEST] Batch of {len(raw_chunks)} done in {elapsed_ms} ms.", flush=True)
            return {"ok": True, "results": results, "inference_ms": elapsed_ms}

        # ---- Single-chunk mode (unchanged) ---------------------------
        audio_b64 = inp.get("wav_base64") or inp.get("audio_base64")
        if not audio_b64:
            return {"ok": False, "error": "Missing 'wav_base64' in input."}

        lang       = inp.get("language", LANGUAGE)
        task       = inp.get("task", TASK)
        beam_size  = int(inp.get("beam_size", inp.get("num_beams", BEAM_SIZE)))
        max_tokens = int(inp.get("max_new_tokens", MAX_NEW_TOKENS))

        try:
            raw = base64.b64decode(audio_b64)
        except Exception:
            return {"ok": False, "error": "Invalid base64 data."}

        if not raw:
            return {"ok": False, "error": "Audio is empty."}
        if len(raw) > MAX_AUDIO_BYTES:
            mb = MAX_AUDIO_BYTES // (1024 * 1024)
            return {"ok": False, "error": f"Audio exceeds {mb} MB limit."}

        service = TranscribeService.get_instance()

        TranscribeService.last_request_time = time.time()
        result = service.transcribe(
            raw, lang=lang, task=task, beam_size=beam_size, max_tokens=max_tokens
        )
        TranscribeService.last_request_time = time.time()

        if result.get("inference_ms") is not None:
            print(f"[REQUEST] Done in {result['inference_ms']} ms.", flush=True)
        return result

    except Exception:
        print(traceback.format_exc(), flush=True)
        return {"ok": False, "error": "Internal server error."}


print(f"[BOOT] format={MODEL_FORMAT} device={DEVICE} model={MODEL_ID}", flush=True)
runpod.serverless.start({"handler": handler})
