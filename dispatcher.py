"""RunPod serverless entrypoint for the combined image.

One worker, three services — text-to-speech (CosyVoice3), and two speech-to-
text models (w2v-bert, Whisper Turbo). Each service runs in its own
virtualenv as a child process that is started the first time a job for it
arrives, so a worker only pays the load time and GPU memory for the
services it is actually asked to run.

Routing: job["input"]["service"] is "tts", "w2v" or "whisper". For backwards
compatibility, input with a "text" field and no "service" is treated as TTS.
"""

import json
import os
import subprocess
import threading
import traceback

import runpod

APP = "/app"
PRELOAD_TOKEN_PLACEHOLDER = "models-are-baked-in-no-network-needed"

SERVICES = {
    "tts": {
        "python": "/venvs/tts/bin/python",
        "env": {
            "MODEL_DIR": "/app/models/tts",
            "PYTHONPATH": "/app/services/tts:/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS",
        },
    },
    "w2v": {
        "python": "/venvs/w2v/bin/python",
        "env": {
            "MODEL_DIR": "/app/models/w2v",
            "HF_HOME": "/app/models/w2v",
            "MODEL_ID": "computeram/badini-w2v-bert-normalized",
            "PYTHONPATH": "/app/services/w2v",
            # The w2v handler refuses to start without a token even though it
            # loads with local_files_only=True; the weights are baked in.
            "HF_TOKEN": os.environ.get("HF_TOKEN") or PRELOAD_TOKEN_PLACEHOLDER,
        },
    },
    "whisper": {
        "python": "/venvs/whisper/bin/python",
        "env": {
            "MODEL_DIR": "/app/model_cache",
            "HF_HOME": "/app/model_cache",
            "MODEL_ID": "BadiniSpeechNLP/whisper-turbo-ctranslate",
            "PYTHONPATH": "/app/services/whisper",
        },
    },
}

_procs = {}
_locks = {name: threading.Lock() for name in SERVICES}


def _spawn(name):
    cfg = SERVICES[name]
    env = {**os.environ, **cfg["env"], "PYTHONUNBUFFERED": "1"}
    print(f"[DISPATCH] starting service '{name}'", flush=True)
    proc = subprocess.Popen(
        [cfg["python"], "-u", f"{APP}/service_runner.py", name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit: service logs go straight to the worker log
        env=env,
        cwd=f"{APP}/services/{name}",
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    first = proc.stdout.readline()
    if not first or not json.loads(first).get("ready"):
        raise RuntimeError(f"service '{name}' failed to start")
    print(f"[DISPATCH] service '{name}' ready", flush=True)
    return proc


def _get_proc(name):
    proc = _procs.get(name)
    if proc is None or proc.poll() is not None:
        _procs[name] = proc = _spawn(name)
    return proc


def _pick_service(inp):
    service = inp.get("service")
    if service:
        return service
    if "text" in inp:
        return "tts"
    return None


def handler(job):
    inp = job.get("input") or {}
    name = _pick_service(inp)
    if name not in SERVICES:
        return {"ok": False, "error": f"Unknown or missing 'service'. Use one of: {sorted(SERVICES)}."}

    with _locks[name]:
        try:
            proc = _get_proc(name)
            proc.stdin.write(json.dumps({"id": job.get("id"), "input": inp}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                _procs.pop(name, None)
                return {"ok": False, "error": f"Service '{name}' stopped unexpectedly. Please retry."}
            return json.loads(line)
        except Exception:
            print(traceback.format_exc(), flush=True)
            _procs.pop(name, None)
            return {"ok": False, "error": "Internal server error"}


runpod.serverless.start({"handler": handler})
