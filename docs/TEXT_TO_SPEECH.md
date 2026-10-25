# Text-to-speech: everything in one place

Badini Kurdish TTS on the BLSP site (badiniai.uod.ac/blsp), built on CosyVoice3 (zero-shot voice cloning).

## Voices and emotions
| Voice | Gender | Emotions |
|---|---|---|
| Hayfa | female | natural, angry, sad, happy, fear, surprise |
| Abdullah | male | natural, angry, sad, happy, fear, surprise |
| Bafreen | female | natural, angry, sad, happy, fear, surprise |

CosyVoice3 has no emotion switch. "Emotion" = which real recording of that speaker is used as the
reference clip. Reference clips: `prompt-<voice>-<emotion>.wav` + `.txt` (exact transcript) in the repo root
(Hayfa: `prompt-hayfa` = natural, `prompt-<emotion>` for the others). Abdullah's and Bafreen's `natural` clips are calm sentences
from their training data (not from the emotion set).

## Model weights
Only `llm.pt` is fine-tuned (trained on all three speakers). It is in
[`Dezheen89/cosyvoice`](https://huggingface.co/Dezheen89/cosyvoice) at `model_all/llm.pt`
(pinned in `downloads/download_tts.py`: revision `2625f534e1b9`). The other model files are the unchanged
base files from `computeram/cosyvoice3-badini-tts`; the build skips that repo's old `llm.pt`.

## Where each piece lives in this repo
| Path | What |
|---|---|
| `services/tts/runpod_handler.py` | The TTS worker: `voice` + `emotion` pick the reference clip, long text is chunked, output is Opus |
| `downloads/download_tts.py` | Build-time model download (base files + new `llm.pt`) |
| `prompt-*.wav/.txt` | Reference clips for all voices and emotions |
| `dispatcher.py` | One RunPod endpoint serves TTS, Fast STT (w2v) and Whisper; `input.service` routes |
| `frontend/tts/` | Website code: `tts.js` (API client), `TTS.jsx` (page), `dictionary-tts-keys.txt` (English/Kurdish strings) |
| `proxy/tts-proxy.php` | Server proxy with voice + emotion checks (**not yet live** on the server; RunPod key replaced by a placeholder) |
| `proxy/tts-proxy.live.php` | The proxy currently on the server (Hayfa only), key removed |
| `training/` | Test set evaluation (`eval_testset.py`, `evaluate_test_set.ipynb`) and the 3-speaker inference notebook |
| `tools/infer.py` | Stand-alone inference script (same file as in the Hugging Face repo) |

## Request format
`POST tts-proxy.php?action=run` with `{"text": "...", "voice": "hayfa|abdullah|bafreen", "emotion": "...", "speed": 1.0}`
(max 2,500 characters). Poll `?action=status&id=...`. Without `voice` it is Hayfa.

## Status (2026-10-18)
The image with the three voices and the new `llm.pt` is built from this repo but is **not yet verified or live**.
The live endpoint still runs the earlier image (Hayfa only). Test on a separate endpoint before switching.

## Secrets
Never commit tokens. The RunPod key is in `proxy/*.php` as `YOUR_RUNPOD_API_KEY`; the build's Hugging Face token is a
GitHub Actions secret passed as a BuildKit secret.
