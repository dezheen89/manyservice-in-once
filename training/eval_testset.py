#!/usr/bin/env python
"""Standalone CosyVoice3 Badini test-set synthesis + evaluation.

Run with the cosyvoice env python (no notebook needed):
  /workspace/envs/cosyvoice/bin/python eval_testset.py
Options: --model_dir DIR  --work DIR  --limit N (per speaker)  --asr_dir DIR  --device cpu  --dry_run
Resumable: wavs already synthesized are skipped.
"""
import os, sys, re, json, csv, argparse, time, warnings
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/workspace/cosyvoice3_badini_all")
ap.add_argument("--data", default="/workspace/Data24_09_2026")
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--asr_dir", default="")
ap.add_argument("--device", default="auto")
ap.add_argument("--model_dir", default=None, help="folder with llm.pt/flow.pt/hift.pt/cosyvoice3.yaml (default WORK/model_ft)")
ap.add_argument("--repo", default="/workspace/CosyVoice")
ap.add_argument("--dry_run", action="store_true", help="only parse data, pick prompts, show converted text; no models")
A = ap.parse_args()
SERVE = A.model_dir or (A.work + "/model_ft")
OUT = A.out or (A.work + "/standalone_eval")
os.makedirs(OUT, exist_ok=True)
sys.path[:0] = [A.repo, A.repo + "/third_party/Matcha-TTS"]
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import numpy as np, torch, torchaudio, soundfile as sf
DEV = ("cuda" if torch.cuda.is_available() else "cpu") if A.device == "auto" else A.device
if A.device == "cuda" and not torch.cuda.is_available(): sys.exit("no GPU visible to torch; use --device cpu or wait for a free GPU")
print("device:", DEV, "| model:", SERVE)
if DEV == "cpu":
    print("WARNING: CPU only -> synthesis of 303 sentences will be VERY slow (hours). Use --limit 5 to test.")

# ---- digits -> Badini words (spelling taken from the user's corrected Excel sheet) ----
ONES  = {1:"ئێک",2:"دو",3:"سێ",4:"چار",5:"پێنج",6:"شەش",7:"حەفت",8:"هەشت",9:"نەه",
         10:"دەه",11:"یازدە",12:"دوازدە",13:"سیزدە",14:"چاردە",15:"پازدە",
         16:"شازدە",17:"حەڤدە",18:"هەژدە",19:"نوزدە"}
TENS  = {2:"بیست",3:"سیه",4:"چل",5:"پێنجی",6:"شێست",7:"حەفتێ",8:"هەشتا",9:"نوت"}
HUND, THOU, ZERO, AND = "سەد", "هزار", "سفر", " و "
POINT, HOUR, MINUTE, SECOND = "پوینت", "دەمژمێر", "خولەک", "چرکە"   # HOUR is said BEFORE the hour number
PLUS, MINUS = "کومکرنی", "ژێبرن"
DAY, MONTH, YEAR = "رۆژا", "هەیڤا", "سالا"
CURR = {"$": ("دولار", "سەنت"), "€": ("یۆرۆ", "سەنت")}   # symbol -> (main unit, cents)
D = "0-9٠-٩۰-۹"

def _below100(r):
    if r < 20: return ONES[r]
    t, u = divmod(r, 10)
    return TENS[t] + (AND + ONES[u] if u else "")

def _below1000(n):
    h, r = divmod(n, 100)
    parts = []
    if h: parts.append(HUND if h == 1 else ONES[h] + " " + HUND)
    if r: parts.append(_below100(r))
    return AND.join(parts)

def num2words_badini(n):
    if n == 0: return ZERO
    th, r = divmod(n, 1000)
    parts = []
    if th: parts.append(THOU if th == 1 else _below1000(th) + " " + THOU)
    if r:  parts.append(_below1000(r))
    return AND.join(parts)

_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def _int(s):                       # digit string -> words (0-999999; longer numbers / 007 codes stay as digits)
    s = s.translate(_ASCII)
    if len(s) > 1 and s[0] == "0": return s
    n = int(s)
    return num2words_badini(n) if n <= 999999 else s

def _frac(s):                      # fraction read as a number:  23 -> بیست و سێ ;  05 -> سفر پێنج
    s = s.translate(_ASCII); z = len(s) - len(s.lstrip("0")); rest = s.lstrip("0")
    return " ".join([ZERO] * z + ([_int(rest)] if rest else []))

def _num(s):                       # 34.567 / 45.6 / 12
    s = s.translate(_ASCII)
    if "." in s:
        i, f = s.split(".", 1)
        return f"{_int(i)} {POINT} {_frac(f)}"
    return _int(s)

def _money(s, sym):                # 45.6$ -> چل و پێنج دولار و شەش سەنت
    main, cent = CURR[sym]; s = s.translate(_ASCII)
    i, _, f = s.partition(".")
    out = f"{_int(i)} {main}"
    if f.strip("0"): out += f"{AND}{_int(f.lstrip('0'))} {cent}"
    return out

def _time(m):                      # 21:45:30 -> دەمژمێر بیست و ئێک و چل و پێنج خولەک و سیه چرکە
    h, mi, se = m.group(1), m.group(2), m.group(3)
    out = [f"{HOUR} {_int(h)}"]
    if int(mi.translate(_ASCII)): out.append(f"{_int(mi)} {MINUTE}")
    if se and int(se.translate(_ASCII)): out.append(f"{_int(se)} {SECOND}")
    return AND.join(out)

def _date(m):
    g = m.groups()
    if len(g[0].translate(_ASCII)) == 4: y, mo, d = g[0], g[1], g[2]     # 2026/11/23
    else:                                d, mo, y = g[0], g[1], g[2]     # 23/11/2026
    before = m.string[:m.start()].rstrip()
    day = "" if re.search(r"[ڕر]ۆژا$", before) else DAY + " "     # text already says "ڕۆژا" -> don't repeat it
    return f"{day}{_int(d)}، {MONTH} {_int(mo)}، {YEAR} {_int(y)}"

# ---- abbreviations (from your Excel table) -> full words, applied before the digit conversion ----
KURDISH_LETTERS = True    # write the expanded words with Kurdish ک / ی instead of Arabic ك / ي (check with the data check cell)
ABBR_TITLES = {            # only at the start of a word, written with a dot
    "پ.د.": "پرۆفیسۆر دكتۆر", "م.د.": "مامۆستا دكتۆر", "م.ب.": "مۆبایل", "ص.ب.": "سندوقا پۆستێ",
    "د.": "دكتۆر", "پ.": "پرۆفیسۆر", "م.": "مامۆستا", "ژ.": "ژمارە", "ت.": "تەلەفۆن"}
ABBR_UNITS = {             # only directly after a number:  5 كم -> پێنج كیلۆمەتر
    "كگم": "كیلۆگرام", "ملم": "ملیمەتر", "كم": "كیلۆمەتر", "سم": "سانتیمەتر", "گم": "گرام", "م": "مەتر"}
ABBR_RELIGIOUS = {"(ص)": "(سەڵا الله عەلەیهی وە سەلەم)", "(ع.س.)": "(عەلەیهی ئەلسلام)"}

def _kd(s):
    return s.replace("ك", "ک").replace("ي", "ی") if KURDISH_LETTERS else s
def _kaf(s):                                   # match Arabic ك and Kurdish ک in the input
    return s.replace("ك", "[كک]")

def expand_abbreviations(text):
    for k, v in ABBR_RELIGIOUS.items():
        text = text.replace(k, _kd(v))
    text = re.sub(r"(?<![\w.])پ\.\s*ز\.?(?![\w])", "پێش زایین", text)   # پ. ز = BC (before the Christian era), not "professor"
    t = "|".join(re.escape(_kaf(k)).replace(r"\[كک\]", "[كک]") for k in sorted(ABBR_TITLES, key=len, reverse=True))
    def title(m):
        key = m.group(0).replace("ک", "ك")
        nxt = m.string[m.end():m.end()+1]
        return _kd(ABBR_TITLES[key]) + ("" if (nxt == "" or nxt.isspace()) else " ")   # keep a space in "د.محمد"
    text = re.sub(rf"(?<![\w.])(?:{t})(?=\s*\S)", title, text)
    u = "|".join(_kaf(k) for k in sorted(ABBR_UNITS, key=len, reverse=True))
    def unit(m):
        before = m.string[:m.start()]
        if re.search(r":[0-9٠-٩۰-۹]{2}\s*$", before):            # 10:30 م  -> a time of day, leave it
            return m.group(0)
        if m.group(1) == "م" and (re.search(r"(?<![0-9٠-٩۰-۹])[0-9٠-٩۰-۹]{4}\s*$", before)      # 2024 م  -> year marker
                                  or re.search(r"[0-9٠-٩۰-۹]+[/\\][0-9٠-٩۰-۹]+[/\\][0-9٠-٩۰-۹]+\s*$", before)):  # 23/12/2024 م -> date marker
            return m.group(0)
        return " " + _kd(ABBR_UNITS[m.group(1).replace("ک", "ك")])
    text = re.sub(rf"(?<=[0-9٠-٩۰-۹])\s*({u})(?![\w])", unit, text)
    return re.sub(r"[ \t]{2,}", " ", text)

def digits_to_words(text):
    text = expand_abbreviations(text)             # abbreviations -> words first
    # 0) Arabic decimal/thousands marks -> ASCII; minus; plus
    text = text.replace("٫", ".").replace("٬", ",")
    text = re.sub(rf"(?<!\w)[-−](?=[{D}])", f"{MINUS} ", text)          # -234 / -56.23 (not 2020-2024)
    text = re.sub(rf"(?<=[{D}])\s*\+\s*(?=[{D}])", f" {PLUS} ", text)       # 234+ 78
    # 1) dates  2026/11/23, 23/11/2026 or 10\7\2018 (backslash form used in your data)
    text = re.sub(rf"(?<![{D}])([{D}]{{4}}|[{D}]{{1,2}})[/\\]([{D}]{{1,2}})[/\\]([{D}]{{4}}|[{D}]{{1,2}})(?![{D}])", _date, text)
    # 2) time  21:45:30 / 21:45
    text = re.sub(rf"(?<![{D}])([{D}]{{1,2}}):([{D}]{{2}})(?::([{D}]{{2}}))?(?![{D}])", _time, text)
    # 3) thousands commas  1,250 -> 1250
    text = re.sub(rf"(?<=[{D}]),(?=[{D}]{{3}}(?![{D}]))", "", text)
    # 4) currency  45.6$  /  $45.6
    sym = "".join(re.escape(k) for k in CURR)
    text = re.sub(rf"([{D}]+(?:\.[{D}]+)?)\s*([{sym}])", lambda m: _money(m.group(1), m.group(2)), text)
    text = re.sub(rf"([{sym}])\s*([{D}]+(?:\.[{D}]+)?)", lambda m: _money(m.group(2), m.group(1)), text)
    # 5) decimals and plain numbers
    return re.sub(rf"[{D}]+(?:\.[{D}]+)?", lambda m: _num(m.group(0)), text)


# ---------------- data ----------------
CLIPS = A.data + "/clips"
def read_tsv(p):
    with open(p, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    return [dict(spk=r[0].strip(), path=r[1].strip(), text=r[3].strip()) for r in rows[1:] if len(r) > 3]
def wavpath(p):
    return p if os.path.isabs(p) else os.path.join(CLIPS, p)
train = read_tsv(A.data + "/train_all.tsv")
test = read_tsv(A.data + "/test_all.tsv")
# hygiene: file names claimed by >1 speaker (bafreen/abdullah "1_" collision) are unusable; train clips also in test are not prompts
_own = {}
for r in train + test: _own.setdefault(r["path"], set()).add(r["spk"])
_bad = {p for p, o in _own.items() if len(o) > 1}
test_paths = {r["path"] for r in test}
n0 = len(test)
train = [r for r in train if r["path"] not in _bad]
test = [r for r in test if r["path"] not in _bad and os.path.exists(wavpath(r["path"]))]
print(f"conflicting file names dropped: {len(_bad)} | test usable: {len(test)}/{n0}")
SPK = sorted({r["spk"] for r in test})
print("test:", len(test), "| speakers:", SPK)
norm = lambda t: re.sub(r"\s+", " ", t).strip()
train_texts = {norm(r["text"]) for r in train}

def dur(p):
    i = sf.info(p); return i.frames / i.samplerate
PROMPT = {}
for s in SPK:
    for r in train:
        if r["spk"] == s and r["text"] and r["path"] not in test_paths and os.path.exists(wavpath(r["path"])) and 3 <= dur(wavpath(r["path"])) <= 12:
            PROMPT[s] = (wavpath(r["path"]), r["text"]); break
    print("prompt", s, "->", PROMPT[s][0])

by = {s: [r for r in test if r["spk"] == s] for s in SPK}
if A.limit:
    by = {s: v[:A.limit] for s, v in by.items()}
items = [dict(r, id=f"{s}_{i:03d}") for s in SPK for i, r in enumerate(by[s])]
print("sentences per speaker:", {s: len(v) for s, v in by.items()}, "| unseen-text:", sum(norm(i["text"]) not in train_texts for i in items))
if A.dry_run:
    for it in items[:5]: print(" ", it["id"], "|", it["text"], "->", digits_to_words(it["text"]))
    sys.exit("dry run finished: data OK, no models loaded")

# ---------------- synthesis ----------------
from cosyvoice.cli.cosyvoice import AutoModel
cosy = AutoModel(model_dir=SERVE)
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
os.makedirs(OUT + "/wav", exist_ok=True)
TJ = OUT + "/timing.json"
times = json.load(open(TJ)) if os.path.exists(TJ) else {}
t0 = time.time(); failed = []
for k, it in enumerate(items):
    it["syn"] = f"{OUT}/wav/{it['id']}.wav"
    if os.path.exists(it["syn"]): continue
    pw, pt = PROMPT[it["spk"]]
    txt = digits_to_words(it["text"])
    try:
        t1 = time.time()
        chunks = [o["tts_speech"] for o in cosy.inference_zero_shot(
            txt, INSTRUCT + pt, pw, stream=False, text_frontend=False)]
        wav = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
        sf.write(it["syn"], wav, cosy.sample_rate)
        times[it["id"]] = dict(sec=time.time() - t1, audio=len(wav) / cosy.sample_rate)
    except Exception as e:
        failed.append(it["id"]); print("FAILED", it["id"], repr(e)[:150], flush=True)
    if k % 10 == 0:
        print(f"synth {k+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
        json.dump(times, open(TJ, "w"))
json.dump(times, open(TJ, "w"))
items = [it for it in items if os.path.exists(it["syn"])]
print(f"synthesized: {len(items)} | failed: {len(failed)} {failed[:10]}")
del cosy; torch.cuda.empty_cache() if DEV == "cuda" else None

# ---------------- scoring ----------------
def load16(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    w = torch.from_numpy(w)
    return torchaudio.functional.resample(w, sr, 16000) if sr != 16000 else w

from speechbrain.inference.speaker import EncoderClassifier
ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
        savedir="/workspace/tmp/ecapa", run_opts={"device": DEV})
def emb(w):
    with torch.no_grad():
        e = ecapa.encode_batch(w.unsqueeze(0).to(DEV)).squeeze()
    return torch.nn.functional.normalize(e, dim=0).cpu()
cos = lambda a, b: float(torch.dot(a, b))

mos = None
try:
    os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
except Exception as e:
    print("UTMOS unavailable:", repr(e)[:200])
def utmos(w):
    if mos is None: return float("nan")
    with torch.no_grad(): return float(mos(w.unsqueeze(0).to(DEV), 16000).squeeze())

pe = {s: emb(load16(PROMPT[s][0])) for s in SPK}
rows = []
for it in items:
    real = load16(wavpath(it["path"])); syn = load16(it["syn"])
    er, es = emb(real), emb(syn)
    rows.append(dict(id=it["id"], spk=it["spk"], text=it["text"],
        unseen=norm(it["text"]) not in train_texts,
        sim_prompt=cos(es, pe[it["spk"]]), sim_real=cos(es, er),
        ceil_real_prompt=cos(er, pe[it["spk"]]),
        mos_syn=utmos(syn), mos_real=utmos(real),
        gen_sec=times.get(it["id"], {}).get("sec", float("nan")),
        audio_sec=times.get(it["id"], {}).get("audio", float("nan"))))

# ---------------- optional ASR CER ----------------
if A.asr_dir:
    from transformers import AutoProcessor, AutoModelForCTC
    proc = AutoProcessor.from_pretrained(A.asr_dir); am = AutoModelForCTC.from_pretrained(A.asr_dir).to(DEV).eval()
    def asr(w):
        x = proc(w.numpy(), sampling_rate=16000, return_tensors="pt")
        x = {k: v.to(DEV) for k, v in x.items()}
        with torch.no_grad(): ids = am(**x).logits.argmax(-1)
        return proc.batch_decode(ids)[0]
    def lev(a, b):
        d = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            p, d[0] = d[0], i
            for j, cb in enumerate(b, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j-1] + 1, p + (ca != cb))
        return d[-1]
    def cer(ref, hyp, loose=False):
        ref = re.sub(r"[^\w]", "", ref); hyp = re.sub(r"[^\w]", "", hyp)
        if loose:   # merge letter variants that ASR/orthography often confuse
            m = str.maketrans("ۆڤڕڵێیەكي", "وفرلیيهکی")
            ref, hyp = ref.translate(m), hyp.translate(m)
        return lev(ref, hyp) / max(1, len(ref))
    for r, it in zip(rows, items):
        ref = digits_to_words(it["text"])
        hs, hr = asr(load16(it["syn"])), asr(load16(wavpath(it["path"])))
        r["cer_syn"], r["cer_real"] = cer(ref, hs), cer(ref, hr)
        r["cer_syn_loose"], r["cer_real_loose"] = cer(ref, hs, True), cer(ref, hr, True)
        r["asr_syn"] = hs

# ---------------- report ----------------
def mean(rs, k):
    v = [r[k] for r in rs if k in r and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
keys = ["sim_prompt", "sim_real", "ceil_real_prompt", "mos_syn", "mos_real", "gen_sec", "audio_sec"] + (["cer_syn", "cer_real", "cer_syn_loose", "cer_real_loose"] if A.asr_dir else [])
summary = {}
for name, f in [("ALL", lambda r: True), ("UNSEEN_TEXT", lambda r: r["unseen"])] + \
               [(s, (lambda s: lambda r: r["spk"] == s)(s)) for s in SPK]:
    rs = [r for r in rows if f(r)]
    summary[name] = dict(n=len(rs), **{k: mean(rs, k) for k in keys})
print("\n{:<12}{:>5}".format("subset", "n") + "".join(f"{k:>18}" for k in keys))
for n, d in summary.items():
    print("{:<12}{:>5}".format(n, d["n"]) + "".join(f"{d[k]:>18.3f}" for k in keys))
json.dump(dict(summary=summary, rows=rows), open(OUT + "/scores.json", "w"), ensure_ascii=False, indent=1)
with open(OUT + "/scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r["gen_sec"] == r["gen_sec"]]
if ok:
    print(f"\nspeed: RTF = {sum(r['gen_sec'] for r in ok)/sum(r['audio_sec'] for r in ok):.2f} (generation time / audio time, lower is faster) on {DEV}")
print("\nsaved:", OUT + "/scores.json", "and scores.csv")
