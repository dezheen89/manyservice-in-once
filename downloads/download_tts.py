"""Build-time model download. HF_TOKEN (optional) comes from a BuildKit secret,
not a build arg, so it never lands in the published image's layers/history."""
import os
import sys
import time

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
token = os.environ.get("HF_TOKEN", "").strip() or None
print(f"Downloading {model_id} (authenticated: {bool(token)})", flush=True)

# Anonymous requests from shared CI runners get 429s; retry with backoff too.
for attempt in range(1, 6):
    try:
        # flow.decoder.estimator.fp32.onnx (1.3 GB) is only read when CosyVoice is started with
        # load_trt=True, which this worker never does, so it is left out of the image.
        path = snapshot_download(
            model_id, local_dir="/app/models/tts", token=token,
            ignore_patterns=["flow.decoder.estimator.fp32.onnx"],
        )
        print(f"Downloaded {model_id} to {path}", flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        if attempt == 5:
            sys.exit(1)
        time.sleep(30 * attempt)
