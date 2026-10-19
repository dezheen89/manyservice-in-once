import React from "react";
import Icon from "../components/Icon.jsx";
import BackendNotice from "../components/BackendNotice.jsx";
import { useT } from "../context/LangContext.jsx";
import { useAuth } from "../context/AuthContext.jsx";
import { TTS_VOICES, TTS_AVAILABLE_EMOTIONS, generateSpeech, audioBase64ToUrl, combineAudioClips, splitTextForTts, warmupTTS } from "../lib/tts.js";

const EMOTIONS = [
  ["natural", "🙂", "tts.emNatural"],
  ["happy", "😊", "tts.emHappy"],
  ["sad", "😢", "tts.emSad"],
  ["angry", "😠", "tts.emAngry"],
  ["fear", "😨", "tts.emFear"],
  ["surprise", "😲", "tts.emSurprise"],
].filter(([id]) => TTS_AVAILABLE_EMOTIONS.includes(id));

const MAX_CHARS = 2500;

const SAMPLE =
  "سەروکاتیا زانکۆیا دهۆک رادگەهینیت کو دێ رابیت ب راگەهاندنا پوستێ رێڤەبەرێ سنتەرێ ڤەکولینان ل کولیژا کارگێری و ئابووری، هەر ماموستایەکێ حەز هەبیت مافێ پێشکێشکرنێ یێ هەی بو پوستێن دیارکری برێیا (لینکێ) پوستێن هەڤپێچ کری.";

export default function TTS() {
  const t = useT();
  const { token } = useAuth();
  const [text, setText] = React.useState("");
  const [voiceId, setVoiceId] = React.useState(TTS_VOICES[0].id);
  const [speed, setSpeed] = React.useState(1.0);
  const [fontSize, setFontSize] = React.useState(22);
  const [state, setState] = React.useState("idle"); // idle | generating | ready | error
  const [errorMsg, setErrorMsg] = React.useState(null);
  const [audioUrl, setAudioUrl] = React.useState(null);
  const [audioFormat, setAudioFormat] = React.useState("wav");
  const [history, setHistory] = React.useState([]);
  const audioRef = React.useRef(null);
  const [playing, setPlaying] = React.useState(false);
  const [elapsedS, setElapsedS] = React.useState(0);
  const [genPhase, setGenPhase] = React.useState("queue"); // queue | progress
  const [partProgress, setPartProgress] = React.useState({ i: 0, n: 1 });
  const [playingHistory, setPlayingHistory] = React.useState(null);
  const historyAudioRefs = React.useRef({});
  const [emotion, setEmotion] = React.useState("natural");
  const textareaRef = React.useRef(null);
  const [selRange, setSelRange] = React.useState({ start: 0, end: 0 });
  const [segments, setSegments] = React.useState([]); // { id, text, emotion, emotionLabel, audioUrl, format }
  const [segState, setSegState] = React.useState("idle"); // idle | generating | error
  const [segError, setSegError] = React.useState(null);
  const [segElapsedS, setSegElapsedS] = React.useState(0);
  const [playingSegment, setPlayingSegment] = React.useState(null);
  const segmentAudioRefs = React.useRef({});
  const [combinedUrl, setCombinedUrl] = React.useState(null);
  const [combining, setCombining] = React.useState(false);

  const voice = TTS_VOICES.find((v) => v.id === voiceId);
  const voiceEmotions = EMOTIONS.filter(([id]) => voice.emotions.includes(id));
  const pickVoice = (v) => {
    setVoiceId(v.id);
    // Keep the chosen emotion if the new voice has it, else use its first one.
    if (!v.emotions.includes(emotion)) setEmotion(v.emotions[0]);
  };
  const charCount = text.length;
  const estSeconds = Math.max(2, Math.round(charCount / 14 / speed));

  // Nudge the GPU worker to start booting as soon as the user is on this
  // page, instead of waiting until they click Generate.
  React.useEffect(() => {
    warmupTTS(token);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const generate = async () => {
    const content = text.trim() || SAMPLE;
    if (!text.trim()) setText(SAMPLE);
    setState("generating");
    setErrorMsg(null);
    setElapsedS(0);
    setGenPhase("queue");
    setPartProgress({ i: 0, n: 1 });
    // Not revoking the outgoing audioUrl here: it now lives on as a
    // "Recent generations" entry below and needs to stay playable.
    setAudioUrl(null);

    const startedAt = Date.now();
    const timer = setInterval(() => setElapsedS(Math.round((Date.now() - startedAt) / 1000)), 1000);

    const parts = splitTextForTts(content);
    const partUrls = new Array(parts.length);
    try {
      // Parts run up to PART_CONCURRENCY at a time (matching max workers), each
      // result kept at its original index so the join stays in order.
      const PART_CONCURRENCY = 2;
      let format = "wav";
      let nextIdx = 0;
      let doneCount = 0;
      const runWorker = async () => {
        while (nextIdx < parts.length) {
          const idx = nextIdx++;
          const result = await generateSpeech({ text: parts[idx], speed, emotion, voice: voice.backend }, token, (status) => {
            setGenPhase(status === "IN_PROGRESS" ? "progress" : "queue");
          });
          partUrls[idx] = audioBase64ToUrl(result.audioBase64, result.format);
          format = result.format;
          doneCount++;
          setPartProgress({ i: doneCount, n: parts.length });
        }
      };
      await Promise.all(Array.from({ length: Math.min(PART_CONCURRENCY, parts.length) }, runWorker));
      let url = partUrls[0];
      if (partUrls.length > 1) {
        const joined = await combineAudioClips(partUrls, { gapMs: 200 });
        partUrls.forEach((u) => URL.revokeObjectURL(u));
        url = joined.url;
        format = joined.format;
      }
      setAudioUrl(url);
      setAudioFormat(format);
      setState("ready");
      setHistory((h) => {
        const emotionLabel = EMOTIONS.find(([id]) => id === emotion)?.[2];
        const next = [
          { text: content, voice: t(voice.nameKey), emotionLabel, when: new Date(), audioUrl: url, format },
          ...h,
        ];
        const dropped = next.slice(10);
        dropped.forEach((d) => URL.revokeObjectURL(d.audioUrl));
        return next.slice(0, 10);
      });
    } catch (e) {
      partUrls.forEach((u) => u && URL.revokeObjectURL(u));
      setErrorMsg(e.message || "Speech generation failed.");
      setState("error");
    } finally {
      clearInterval(timer);
    }
  };

  const updateSelection = () => {
    const el = textareaRef.current;
    if (!el) return;
    setSelRange({ start: el.selectionStart, end: el.selectionEnd });
  };
  const selectedText = text.slice(selRange.start, selRange.end).trim();

  const addSegment = async () => {
    if (!selectedText) return;
    setSegState("generating");
    setSegError(null);
    setSegElapsedS(0);
    const startedAt = Date.now();
    const timer = setInterval(() => setSegElapsedS(Math.round((Date.now() - startedAt) / 1000)), 1000);
    try {
      const result = await generateSpeech({ text: selectedText, speed, emotion, voice: voice.backend }, token);
      const url = audioBase64ToUrl(result.audioBase64, result.format);
      const emotionLabel = EMOTIONS.find(([id]) => id === emotion)?.[2];
      setSegments((s) => [...s, { id: Date.now(), text: selectedText, emotionLabel, audioUrl: url, format: result.format }]);
      setSegState("idle");
    } catch (e) {
      setSegError(e.message || "Segment generation failed.");
      setSegState("error");
    } finally {
      clearInterval(timer);
    }
  };

  const removeSegment = (id) => {
    setSegments((s) => {
      const seg = s.find((x) => x.id === id);
      if (seg) URL.revokeObjectURL(seg.audioUrl);
      return s.filter((x) => x.id !== id);
    });
    if (combinedUrl) {
      URL.revokeObjectURL(combinedUrl);
      setCombinedUrl(null);
    }
  };

  const clearSegments = () => {
    segments.forEach((s) => URL.revokeObjectURL(s.audioUrl));
    setSegments([]);
    if (combinedUrl) {
      URL.revokeObjectURL(combinedUrl);
      setCombinedUrl(null);
    }
    setPlayingSegment(null);
  };

  const moveSegment = (id, dir) => {
    setSegments((s) => {
      const idx = s.findIndex((x) => x.id === id);
      const swapWith = idx + dir;
      if (swapWith < 0 || swapWith >= s.length) return s;
      const copy = [...s];
      [copy[idx], copy[swapWith]] = [copy[swapWith], copy[idx]];
      return copy;
    });
  };

  const toggleSegmentPlay = (i) => {
    const el = segmentAudioRefs.current[i];
    if (!el) return;
    if (playingSegment === i) {
      el.pause();
      setPlayingSegment(null);
      return;
    }
    if (playingSegment !== null) segmentAudioRefs.current[playingSegment]?.pause();
    el.play();
    setPlayingSegment(i);
  };

  const combineSegments = async () => {
    if (segments.length === 0) return;
    setCombining(true);
    setErrorMsg(null);
    try {
      const result = await combineAudioClips(segments.map((s) => s.audioUrl));
      if (combinedUrl) URL.revokeObjectURL(combinedUrl);
      setCombinedUrl(result.url);
    } catch (e) {
      setErrorMsg(e.message || "Combining segments failed.");
      setState("error");
    } finally {
      setCombining(false);
    }
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.play();
      setPlaying(true);
    }
  };

  const toggleHistoryPlay = (i) => {
    const el = historyAudioRefs.current[i];
    if (!el) return;
    if (playingHistory === i) {
      el.pause();
      setPlayingHistory(null);
      return;
    }
    if (playingHistory !== null) historyAudioRefs.current[playingHistory]?.pause();
    el.play();
    setPlayingHistory(i);
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">
            {t("Text to")} <em>{t("speech")}</em>
          </h1>
          <div className="page-sub">{t("tts.sub")}</div>
        </div>
      </div>

      <div className="tts-grid">
        <div className="card tts-input-card">
          <div className="card-head">
            <h3>{t("tts.yourText")}</h3>
            <button
              className="btn btn-ghost"
              style={{ marginInlineStart: "auto", padding: "5px 10px", fontSize: 12 }}
              onClick={() => document.getElementById("ttsFile").click()}
            >
              <Icon name="upload" size={13} />
              {t("tts.uploadTxt")}
            </button>
            <input
              id="ttsFile"
              type="file"
              accept=".txt,text/plain"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files[0];
                if (!f) return;
                const r = new FileReader();
                r.onload = () => {
                  setText(String(r.result).slice(0, MAX_CHARS));
                  setState("idle");
                };
                r.readAsText(f);
                e.target.value = "";
              }}
            />
          </div>
          <div className="tts-submeta">
            <span className={charCount > MAX_CHARS * 0.9 ? "warn-text" : ""}>{charCount.toLocaleString()}</span>
            <span> / {MAX_CHARS.toLocaleString()}</span>
            <span style={{ margin: "0 8px", color: "var(--ink-4)" }}>·</span>
            <span>≈ {estSeconds}s</span>
          </div>
          <div style={{ padding: "0 20px 6px", fontSize: 12, color: "var(--ink-3)" }}>{t("tts.limitNote")}</div>
          <textarea
            ref={textareaRef}
            className="tts-textarea"
            placeholder={SAMPLE}
            maxLength={MAX_CHARS}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              updateSelection();
            }}
            onSelect={updateSelection}
            onMouseUp={updateSelection}
            onKeyUp={updateSelection}
            dir="rtl"
            style={{ fontSize, fontFamily: "var(--font-kr)" }}
          />
          <div className="tts-input-foot">
            <button className="btn btn-quiet" onClick={() => setText(SAMPLE)}>
              <Icon name="sparkle" size={13} />
              {t("Insert sample")}
            </button>
            <button
              className="btn btn-quiet"
              onClick={() => {
                setText("");
                setState("idle");
              }}
              disabled={!text}
            >
              <Icon name="trash" size={13} />
              {t("Clear")}
            </button>
            {selectedText && (
              <button className="btn btn-quiet" onClick={addSegment} disabled={segState === "generating"}>
                {segState === "generating" ? <span className="spinner" /> : <Icon name="plus" size={13} />}
                {segState === "generating" ? `${t("tts.generatingAudio")} · ${segElapsedS}s` : t("tts.addSelection")}
              </button>
            )}
            <button className="btn btn-primary" style={{ marginInlineStart: "auto" }} onClick={generate} disabled={state === "generating"}>
              {state === "generating" ? <span className="spinner" /> : <Icon name="sparkle" size={14} />}
              {state === "generating"
                ? `${partProgress.n > 1 ? `${t("tts.part")} ${partProgress.i}/${partProgress.n} · ` : ""}${t(genPhase === "progress" ? "tts.generatingAudio" : "stt.preparing")} · ${elapsedS}s`
                : t("tts.generate")}
            </button>
          </div>
          {segError && segState === "error" && (
            <div style={{ padding: "0 20px 16px" }}>
              <BackendNotice tone="error" message={segError} />
            </div>
          )}

          {(state === "generating" || segState === "generating") && (state === "generating" ? elapsedS : segElapsedS) >= 15 && (
            <div style={{ padding: "0 20px 14px", fontSize: 12.5, color: "var(--ink-3)" }}>{t("tts.loadingHint")}</div>
          )}

          {state === "error" && (
            <div style={{ padding: "0 20px 16px" }}>
              <BackendNotice tone="error" message={errorMsg} />
            </div>
          )}

          {state === "ready" && audioUrl && (
            <div className="tts-player">
              <button className="play-btn" onClick={togglePlay}>
                <Icon name={playing ? "pause" : "play"} size={18} />
              </button>
              <div className="tts-wave">
                {Array.from({ length: 56 }).map((_, i) => {
                  const h = 6 + Math.abs(Math.sin(i * 0.7)) * 10 + ((i * 7) % 9);
                  return <span key={i} className={"wave-bar" + (playing ? " active" : "")} style={{ height: h + "px" }} />;
                })}
              </div>
              <audio ref={audioRef} src={audioUrl} onEnded={() => setPlaying(false)} style={{ display: "none" }} />
              <span className="tts-time">{estSeconds}s</span>
              <a className="icon-btn" title={t("tts.download")} href={audioUrl} download={`badiniai-tts.${audioFormat}`}>
                <Icon name="download" size={14} />
              </a>
            </div>
          )}
          {state === "idle" && <div className="tts-ready-note">{t("stt.ready")}</div>}
        </div>

        <div className="card tts-side">
          <div className="card-head">
            <h3>{t("Voice")}</h3>
          </div>
          <div className="voice-list">
            {TTS_VOICES.map((v) => (
              <button key={v.id} className={"voice-row" + (voiceId === v.id ? " active" : "")} onClick={() => pickVoice(v)}>
                <div className="voice-avatar" data-gender={v.gender}>
                  <Icon name="mic" size={14} />
                </div>
                <div className="voice-meta">
                  <div className="voice-name">{t(v.nameKey)}</div>
                  <div className="voice-style">{t(v.descKey)}</div>
                </div>
              </button>
            ))}
          </div>

          {EMOTIONS.length > 0 && (
            <>
              <div className="card-head" style={{ marginTop: 8 }}>
                <h3>{t("tts.emotion")}</h3>
              </div>
              <div className="emotion-grid">
                {voiceEmotions.map(([id, emoji, key]) => (
                  <button key={id} className={"emotion-cell" + (emotion === id ? " active" : "")} onClick={() => setEmotion(id)}>
                    <span className="emotion-emoji">{emoji}</span>
                    <span className="emotion-label">{t(key)}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          <div className="card-head" style={{ marginTop: 8 }}>
            <h3>{t("Controls")}</h3>
          </div>
          <div style={{ padding: "4px 20px 20px" }}>
            <div className="tts-control">
              <div className="tts-control-row">
                <label>{t("Speed")}</label>
                <span className="tts-control-val">{speed.toFixed(2)}×</span>
              </div>
              <input type="range" min="0.5" max="2" step="0.05" value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} />
              <div className="tts-control-marks">
                <span>0.5×</span>
                <span>1×</span>
                <span>2×</span>
              </div>
            </div>
            <div className="tts-control">
              <div className="tts-control-row">
                <label>{t("tts.fontSize")}</label>
                <span className="tts-control-val">{fontSize}px</span>
              </div>
              <input type="range" min="16" max="34" step="1" value={fontSize} onChange={(e) => setFontSize(parseInt(e.target.value))} />
              <div className="tts-control-marks">
                <span>16</span>
                <span>24</span>
                <span>34</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <div className="card-head">
          <h3>{t("tts.segments")}</h3>
        </div>
        <div style={{ padding: "0 20px 12px", fontSize: 12.5, color: "var(--ink-3)" }}>{t("tts.segmentsSub")}</div>
        <div className="recent-tts">
          {segments.length === 0 && <div className="tts-empty">{t("tts.noSegments")}</div>}
          {segments.map((s, i) => (
            <div key={s.id} className="recent-tts-row">
              <button className="recent-play" onClick={() => toggleSegmentPlay(i)} style={{ border: "none", cursor: "pointer" }}>
                <Icon name={playingSegment === i ? "pause" : "play"} size={13} />
              </button>
              <div className="recent-meta">
                <div className="recent-text" dir="rtl" style={{ fontFamily: "var(--font-kr)" }}>
                  {s.text}
                </div>
                <div className="recent-sub">
                  <span>{t(s.emotionLabel)}</span>
                </div>
              </div>
              <audio
                ref={(el) => (segmentAudioRefs.current[i] = el)}
                src={s.audioUrl}
                onEnded={() => setPlayingSegment(null)}
                style={{ display: "none" }}
              />
              <button className="icon-btn" onClick={() => moveSegment(s.id, -1)} disabled={i === 0} title="↑">
                <Icon name="arrowUp" size={13} />
              </button>
              <button className="icon-btn" onClick={() => moveSegment(s.id, 1)} disabled={i === segments.length - 1} title="↓">
                <Icon name="arrowDown" size={13} />
              </button>
              <button className="icon-btn" title={t("tts.removeSegment")} onClick={() => removeSegment(s.id)}>
                <Icon name="trash" size={13} />
              </button>
            </div>
          ))}
        </div>
        {segments.length > 0 && (
          <div style={{ padding: "12px 20px 20px", display: "flex", alignItems: "center", gap: 12 }}>
            <button className="btn btn-primary" onClick={combineSegments} disabled={combining}>
              {combining ? <span className="spinner" /> : <Icon name="sparkle" size={14} />}
              {combining ? t("tts.combining") : t("tts.combineDownload")}
            </button>
            <button className="btn btn-quiet" onClick={clearSegments} disabled={combining}>
              <Icon name="trash" size={13} />
              {t("tts.clearSegments")}
            </button>
            {combinedUrl && (
              <>
                <audio controls src={combinedUrl} style={{ height: 36 }} />
                <a className="icon-btn" title={t("tts.download")} href={combinedUrl} download="badiniai-tts-combined.wav">
                  <Icon name="download" size={14} />
                </a>
              </>
            )}
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <div className="card-head">
          <h3>{t("Recent generations")}</h3>
          <span className="meta">{t("Last 7 days")}</span>
        </div>
        <div className="recent-tts">
          {history.length === 0 && <div className="tts-empty">{t("No generations yet — your history will appear here.")}</div>}
          {history.map((h, i) => (
            <div key={i} className="recent-tts-row">
              {h.audioUrl && (
                <button className="recent-play" onClick={() => toggleHistoryPlay(i)} style={{ border: "none", cursor: "pointer" }}>
                  <Icon name={playingHistory === i ? "pause" : "play"} size={13} />
                </button>
              )}
              <div className="recent-meta">
                <div className="recent-text" dir="rtl" style={{ fontFamily: "var(--font-kr)" }}>
                  {h.text}
                </div>
                <div className="recent-sub">
                  <span>{h.voice}</span>
                  {h.emotionLabel && (
                    <>
                      <span>·</span>
                      <span>{t(h.emotionLabel)}</span>
                    </>
                  )}
                  <span>·</span>
                  <span>{h.when.toLocaleTimeString()}</span>
                </div>
              </div>
              {h.audioUrl && (
                <>
                  <audio
                    ref={(el) => (historyAudioRefs.current[i] = el)}
                    src={h.audioUrl}
                    onEnded={() => setPlayingHistory(null)}
                    style={{ display: "none" }}
                  />
                  <a
                    className="icon-btn"
                    title={t("tts.download")}
                    href={h.audioUrl}
                    download={`badiniai-tts-${i}.${h.format || "wav"}`}
                  >
                    <Icon name="download" size={14} />
                  </a>
                </>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
