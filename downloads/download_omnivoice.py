"""Bake the Badini OmniVoice fine-tune into the image (public repo, no token needed)."""
import os

import torch
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models/omnivoice")
REPO = os.environ.get("MODEL_ID", "Dezheen89/omnivoice")

snapshot_download(REPO, local_dir=MODEL_DIR, allow_patterns=[
    "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])

# Loading once on the CPU also downloads the helper models OmniVoice fetches at start-up into HF_HOME,
# so the worker needs no network access at runtime.
from omnivoice import OmniVoice  # noqa: E402

OmniVoice.from_pretrained(MODEL_DIR, device_map="cpu", dtype=torch.float32)
print("omnivoice model baked in:", MODEL_DIR)
