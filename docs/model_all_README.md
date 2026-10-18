# Badini CosyVoice3 — fine-tuned LLM for three speakers (Hayfa, Abdullah, Bafreen)

`llm.pt` is the only fine-tuned file (trained 2026-10-04 on the combined Badini data of all speakers).
It is a drop-in replacement for the `llm.pt` of the base model.

All other files needed to run it come from the public base model
`Fun-CosyVoice3-0.5B` (FunAudioLLM): `flow.pt`, `hift.pt`, `campplus.onnx`,
`speech_tokenizer_v3.onnx`, `flow.decoder.estimator.fp32.onnx` and the `CosyVoice-BlankEN` folder.
`cosyvoice3.yaml`, `config.json` and `configuration.json` here are copies of the base model's.

To run: download the base model, copy this `llm.pt` over its `llm.pt`, then load the folder with
`AutoModel(model_dir=...)` as in `infer.py` (zero-shot: pass a reference clip of the speaker and its exact transcript).
