// Real Text-to-Speech integration — talks to the production CosyVoice3 proxy
// (Badini Kurdish, RunPod-hosted, zero-shot cloned from real "Hayfa"
// reference clips) through our own auth server's /gpu/tts route (not
// directly), so usage can be attributed to a signed-in user and checked
// against their daily GPU-time quota.

const TTS_PROXY = "/gpu/tts";

// CosyVoice3 is zero-shot voice cloning, so there's one voice (Hayfa) — but
// the backend now bundles a real labeled recording of her per emotion below
// and picks the matching one as the clone target, so "emotion" genuinely
// changes which performance is cloned rather than being a no-op.
// Three speakers. "backend" is the name the server knows the voice by.
// Abdullah and Bafreen have no neutral recording, so they have no "natural".
const ALL_EMOTIONS = ["natural", "angry", "sad", "happy", "fear", "surprise"];
const EMOTIONS_NO_NATURAL = ["angry", "sad", "happy", "fear", "surprise"];
export const TTS_VOICES = [
  { id: "hayfa-female", backend: "hayfa", nameKey: "tts.vHayfa", descKey: "tts.vHayfaD", gender: "F", emotions: ALL_EMOTIONS },
  { id: "abdullah-male", backend: "abdullah", nameKey: "tts.vAbdullah", descKey: "tts.vAbdullahD", gender: "M", emotions: EMOTIONS_NO_NATURAL },
  { id: "bafreen-female", backend: "bafreen", nameKey: "tts.vBafreen", descKey: "tts.vBafreenD", gender: "F", emotions: EMOTIONS_NO_NATURAL },
];
export const TTS_AVAILABLE_EMOTIONS = ALL_EMOTIONS;

const POLL_INTERVAL_MS = 2000;
// CosyVoice3 is far slower than the old F5-TTS backend it replaced, and a
// cold worker plus several minutes of real generation for long text is
// normal (measured: up to ~226s of execution alone for a 5,000-char input,
// on top of however long a cold start takes to queue a worker) — RunPod's
// own execution-timeout cap on this endpoint is 10 minutes, so give the
// client enough budget to actually see a job through to that.
const MAX_POLL_ATTEMPTS = 600; // 20 minutes

async function pollUntilDone(jobId, token, onTick) {
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    const res = await fetch(`${TTS_PROXY}/status?id=${encodeURIComponent(jobId)}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    if (data.status === "COMPLETED") {
      if (data.output?.error) throw new Error(data.output.error);
      return data.output;
    }
    if (data.status === "FAILED") throw new Error(data.error || "Generation failed.");
    onTick?.(data.status, attempt);
    // IN_QUEUE or IN_PROGRESS — keep polling
  }
  throw new Error("Generation timed out.");
}

// Returns { audioBase64, sampleRate, format }. onTick(status, attempt) is
// called after each poll so callers can show "still working" progress
// during what can be a multi-minute wait for long text.
export async function generateSpeech({ text, speed = 1.0, emotion = "natural", voice = "hayfa" }, token, onTick) {
  if (!token) throw new Error("You need to be signed in to generate speech.");
  const res = await fetch(`${TTS_PROXY}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ text, speed, emotion, voice }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  if (!data.id) throw new Error("No job ID returned.");

  const output = await pollUntilDone(data.id, token, onTick);
  return {
    audioBase64: output.audio_base64,
    sampleRate: output.sample_rate,
    format: output.format || "wav",
  };
}

// Same split rules as the worker's chunk_text: break at sentence-end punctuation
// (including Arabic-script ؟), keep parts between soft and hard word caps, and
// hard-cut any run that never reaches a sentence end.
export function splitTextForTts(text, softWords = 60, hardWords = 90) {
  const sentences = text.trim().split(/(?<=[.!?؟])\s+/).filter((s) => s.trim());
  if (sentences.length === 0) return [text];
  const parts = [];
  let cur = [];
  const flush = () => {
    if (cur.length) parts.push(cur.join(" "));
    cur = [];
  };
  for (const sent of sentences) {
    const words = sent.trim().split(/\s+/);
    if (words.length > hardWords) {
      flush();
      for (let i = 0; i < words.length; i += hardWords) parts.push(words.slice(i, i + hardWords).join(" "));
      continue;
    }
    if (cur.length && (cur.length + words.length > hardWords || cur.length >= softWords)) flush();
    cur.push(...words);
  }
  flush();
  return parts;
}

export function audioBase64ToUrl(audioBase64, format = "wav") {
  const bytes = atob(audioBase64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  const blob = new Blob([arr], { type: `audio/${format}` });
  return URL.createObjectURL(blob);
}

// Fire-and-forget: nudges the backend to start booting the TTS RunPod worker
// before the user has actually generated anything. Doesn't count against
// GPU quota and never throws — the caller shouldn't care whether it worked.
export function warmupTTS(token) {
  if (!token) return;
  fetch(`${TTS_PROXY}/warmup`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  }).catch(() => {});
}

// Hand-rolled 16-bit PCM WAV encoder — no library needed for a client-side
// concatenation this small. Mirrors the same "write it ourselves" approach
// already used elsewhere in this app for audio (see asr.js's WAV parser).
function encodeWavPCM16(float32, sampleRate) {
  const numSamples = float32.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 16-bit)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);
  let offset = 44;
  for (let i = 0; i < numSamples; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// Stitches several already-generated clips (any of our clips' formats —
// wav or ogg/opus, doesn't matter, the browser's own decoder handles both)
// into one WAV, with a short silence gap between each so one emotion's
// clip doesn't crash straight into the next. Returns { url, format: "wav" }.
export async function combineAudioClips(urls, { gapMs = 350 } = {}) {
  if (urls.length === 0) throw new Error("No segments to combine.");
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  try {
    const buffers = [];
    for (const url of urls) {
      const arrBuf = await fetch(url).then((r) => r.arrayBuffer());
      buffers.push(await ctx.decodeAudioData(arrBuf));
    }
    const sampleRate = ctx.sampleRate;
    const gapSamples = Math.round((gapMs / 1000) * sampleRate);
    let total = 0;
    buffers.forEach((b, i) => {
      total += b.length;
      if (i < buffers.length - 1) total += gapSamples;
    });
    const out = new Float32Array(total);
    let pos = 0;
    buffers.forEach((b, i) => {
      out.set(b.getChannelData(0), pos); // source is always mono
      pos += b.length;
      if (i < buffers.length - 1) pos += gapSamples; // left as silence (zeros)
    });
    const blob = encodeWavPCM16(out, sampleRate);
    return { url: URL.createObjectURL(blob), format: "wav" };
  } finally {
    ctx.close();
  }
}
