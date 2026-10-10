"""Build-time download of the w2v-bert ASR model into its own cache dir.
HF_TOKEN comes from a BuildKit secret, never a build arg, so it does not end
up in the published image's layers/history."""
import glob
import os
import sys
import time

from transformers import AutoModelForCTC, AutoProcessor

token = os.environ.get("HF_TOKEN", "").strip() or None
model_id = os.environ.get("MODEL_ID", "computeram/badini-w2v-bert-normalized").strip()
cache = "/app/models/w2v"

print(f"Downloading {model_id} (authenticated: {bool(token)})", flush=True)
for attempt in range(1, 6):
    try:
        AutoProcessor.from_pretrained(model_id, token=token, cache_dir=cache)
        AutoModelForCTC.from_pretrained(model_id, token=token, cache_dir=cache)
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        if attempt == 5:
            sys.exit(1)
        time.sleep(30 * attempt)

print(f"Done - {len(glob.glob(cache + '/**/*', recursive=True))} files in cache.", flush=True)
