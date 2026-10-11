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

# The worker loads this model as float16 anyway (torch_dtype=float16), so store the
# weights as float16 now: the cast is the same round-to-nearest, results are identical,
# and the image is about 1.2 GB smaller. The file is rewritten in place (it sits behind a
# symlink in the Hugging Face cache) so the worker's loading code does not change.
import os
import torch
from safetensors.torch import load_file, save_file

for path in glob.glob(cache + "/models--*/snapshots/*/model.safetensors"):
    real = os.path.realpath(path)
    before = os.path.getsize(real)
    tensors = load_file(real)
    half = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in tensors.items()}
    save_file(half, real, metadata={"format": "pt"})
    print(f"Converted {path}: {before/1e9:.2f} GB -> {os.path.getsize(real)/1e9:.2f} GB", flush=True)

print(f"Done - {len(glob.glob(cache + '/**/*', recursive=True))} files in cache.", flush=True)
