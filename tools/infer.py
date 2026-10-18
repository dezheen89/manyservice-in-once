"""Badini Kurdish zero-shot TTS with CosyVoice3 — three voices, six emotions.

The fine-tuned model (model_all/llm.pt, trained on Hayfa, Abdullah and Bafreen) is in this
repo. The remaining files (flow, hift, tokenizer, ...) are the unchanged base-model files
and are fetched from computeram/cosyvoice3-badini-tts; its old llm.pt is NOT used.
This repo also holds the reference clips (prompt-*.wav + exact transcripts) and this script.

Setup:
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    pip install -r CosyVoice/requirements.txt huggingface_hub

Run:
    python infer.py "text to speak" --voice bafreen --emotion happy \
        --cosyvoice-dir ./CosyVoice --out output.wav

Voices: hayfa (female, has "natural"), abdullah (male), bafreen (female).
Emotions: natural (hayfa only), angry, sad, happy, fear, surprise.
"""
import argparse
import os
import shutil
import sys

import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_REPO = "computeram/cosyvoice3-badini-tts"   # base files only (llm.pt is skipped)
LLM_REPO = "Dezheen89/cosyvoice"                   # fine-tuned llm.pt
LLM_FILE = "model_all/llm.pt"
LLM_BYTES = 2024669583
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
EMOTIONS = ["natural", "angry", "sad", "happy", "fear", "surprise"]


def prompt_base(voice, emotion):
    """File stem of the reference clip for a voice + emotion."""
    if voice == "hayfa":
        return "prompt-hayfa" if emotion == "natural" else f"prompt-{emotion}"
    if emotion == "natural":
        sys.exit(f"Voice '{voice}' has no 'natural' recording; pick another emotion.")
    return f"prompt-{voice}-{emotion}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--cosyvoice-dir", default="CosyVoice",
                    help="clone of github.com/FunAudioLLM/CosyVoice (cloned with --recursive)")
    ap.add_argument("--voice", default="hayfa", choices=["hayfa", "abdullah", "bafreen"])
    ap.add_argument("--emotion", default="natural", choices=EMOTIONS)
    ap.add_argument("--prompt-wav", help="use your own reference clip instead of the bundled one")
    ap.add_argument("--prompt-text-file", help="exact transcript of --prompt-wav")
    ap.add_argument("--out", default="output.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    base = os.path.join(HERE, prompt_base(args.voice, args.emotion))
    prompt_wav = args.prompt_wav or base + ".wav"
    prompt_text_file = args.prompt_text_file or base + ".txt"

    sys.path.append(args.cosyvoice_dir)
    sys.path.append(os.path.join(args.cosyvoice_dir, "third_party", "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = os.path.join(HERE, "model_cache")
    snapshot_download(BASE_REPO, local_dir=model_dir, ignore_patterns=["llm.pt", "flow.decoder.estimator.fp32.onnx"])
    llm = hf_hub_download(LLM_REPO, LLM_FILE, local_dir=os.path.join(HERE, "llm_dl"))
    if os.path.getsize(llm) != LLM_BYTES:
        sys.exit("llm.pt download is incomplete; run again.")
    shutil.copyfile(llm, os.path.join(model_dir, "llm.pt"))
    cosy = AutoModel(model_dir=model_dir)
    prompt_text = open(prompt_text_file, encoding="utf-8").read().strip()

    # The model yields one piece per internal sentence chunk: join them all.
    pieces = [
        out["tts_speech"]
        for out in cosy.inference_zero_shot(
            args.text, INSTRUCT + prompt_text, prompt_wav, stream=False, speed=args.speed
        )
    ]
    torchaudio.save(args.out, torch.cat(pieces, dim=1), cosy.sample_rate)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
