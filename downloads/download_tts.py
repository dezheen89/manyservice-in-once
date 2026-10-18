"""Build-time model download. HF_TOKEN (optional) comes from a BuildKit secret,
not a build arg, so it never lands in the published image's layers/history."""
import os
import sys
import time

from huggingface_hub import hf_hub_download, snapshot_download

# The fine-tuned LLM trained on all three speakers (Hayfa, Abdullah, Bafreen) lives in
# Dezheen89/cosyvoice (public). Pinned to the upload commit; it replaces the llm.pt of MODEL_ID.
LLM_REPO = "Dezheen89/cosyvoice"
LLM_FILE = "model_all/llm.pt"
LLM_REV = "2625f534e1b95f7f9859e93f7652b5793bf69313"
LLM_BYTES = 2024669583

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
            ignore_patterns=["flow.decoder.estimator.fp32.onnx", "llm.pt"],
        )
        print(f"Downloaded {model_id} to {path}", flush=True)
        llm = hf_hub_download(LLM_REPO, LLM_FILE, revision=LLM_REV, local_dir="/tmp/llm_dl", token=token)
        assert os.path.getsize(llm) == LLM_BYTES, f"llm.pt size {os.path.getsize(llm)} != {LLM_BYTES}"
        os.replace(llm, "/app/models/tts/llm.pt")
        print(f"Installed fine-tuned llm.pt from {LLM_REPO}@{LLM_REV[:7]}", flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        if attempt == 5:
            sys.exit(1)
        time.sleep(30 * attempt)
