"""Build-time script: pull the model into the image and detect its format.

Runs once during `docker build`. HF_TOKEN is passed as a build arg and is
cleared from the final image afterwards (see Dockerfile).

Writes two files next to the cache:
  FORMAT     -> "ct2" or "transformers"
  MODEL_PATH -> absolute path to the directory that actually holds the weights

The weights are not always at the repo root; some repos keep them in a
subfolder (e.g. model-final-ctranslate/). This script walks the tree to find
them rather than assuming a layout.
"""

import json
import os
import sys
import traceback

from huggingface_hub import model_info, snapshot_download

print("=== download_model.py started ===", flush=True)

token    = os.environ.get("HF_TOKEN", "").strip()
model_id = os.environ.get("MODEL_ID", "").strip()
cache    = "/app/models/whisper"

if not model_id:
    print("ERROR: MODEL_ID is empty!", flush=True)
    sys.exit(1)

print(f"MODEL_ID : {model_id}", flush=True)
print(f"HF_TOKEN : {'SET' if token else 'NOT SET'}", flush=True)

local_dir = os.path.join(cache, model_id.split("/")[-1])

# -- Can we see the repo at all? --------------------------------------
try:
    info = model_info(model_id, token=token or None)
    print(f"Model found : {info.id}", flush=True)
    print(f"Private     : {info.private}", flush=True)
except Exception as exc:
    print(f"ERROR accessing model: {exc}", flush=True)
    print("If the model is private, pass a valid HF_TOKEN build arg.", flush=True)
    traceback.print_exc()
    sys.exit(1)

# -- Download everything ----------------------------------------------
try:
    print("Downloading model snapshot...", flush=True)
    path = snapshot_download(
        repo_id=model_id,
        token=token or None,
        local_dir=local_dir,
    )
    print(f"Downloaded to: {path}", flush=True)
except Exception as exc:
    print(f"ERROR downloading model: {exc}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# -- Show the full tree so failures are diagnosable -------------------
print("\nRepository contents:", flush=True)
for root, dirs, files in os.walk(local_dir):
    dirs[:] = [d for d in dirs if d != ".cache"]
    rel = os.path.relpath(root, local_dir)
    prefix = "" if rel == "." else rel + "/"
    for name in sorted(files):
        size = os.path.getsize(os.path.join(root, name)) / (1024 ** 2)
        print(f"  {prefix + name:<48} {size:9.1f} MB", flush=True)


def classify(directory):
    """Return 'ct2', 'transformers', or None for a single directory."""
    try:
        names = os.listdir(directory)
    except OSError:
        return None

    has_ct2 = "model.bin" in names
    has_hf  = any(
        n.startswith(("model.safetensors", "pytorch_model", "model-00"))
        for n in names
    )

    if has_ct2 and not has_hf:
        return "ct2"
    if has_hf and not has_ct2:
        return "transformers"
    if has_ct2 and has_hf:
        # Ambiguous -- a CT2 config has no "architectures" key, a HF one does.
        cfg = os.path.join(directory, "config.json")
        if os.path.isfile(cfg):
            try:
                with open(cfg) as fh:
                    if "architectures" in json.load(fh):
                        return "transformers"
            except Exception:
                pass
        return "ct2"
    return None


# -- Find the directory holding the weights (root first, then subdirs) --
model_path = None
fmt = classify(local_dir)
if fmt:
    model_path = local_dir
else:
    for root, dirs, _ in os.walk(local_dir):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for d in sorted(dirs):
            candidate = os.path.join(root, d)
            found = classify(candidate)
            if found:
                fmt = found
                model_path = candidate
                break
        if model_path:
            break

if not model_path:
    print("\nERROR: no recognisable model weights found.", flush=True)
    print("Looked in the repo root and every subfolder for either", flush=True)
    print("model.bin (CTranslate2) or model.safetensors /", flush=True)
    print("pytorch_model.bin (transformers).", flush=True)
    sys.exit(1)

rel_path = os.path.relpath(model_path, local_dir)
print(f"\nWeights directory : {'<repo root>' if rel_path == '.' else rel_path}", flush=True)
print(f"Detected format   : {fmt}", flush=True)

with open(os.path.join(cache, "FORMAT"), "w") as fh:
    fh.write(fmt)

with open(os.path.join(cache, "MODEL_PATH"), "w") as fh:
    fh.write(model_path)

print("\n=== Done ===", flush=True)
