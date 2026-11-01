"""Badini Kurdish text frontend for the OmniVoice fine-tune (numbers, dates, times, %, currency and
abbreviations -> spoken Badini words, plus a light clean-up). Use prepare_text(text)."""
import html
import re

# ======================= Badini text preprocessing =======================
# Based on badini_tts_inference_3speakers.ipynb; number spellings changed to the training-text majority.
# ---- digits -> Badini words (spelling = most frequent form in Data24_09_2026 transcripts) ----
ONES  = {1:"ئێک",2:"دوو",3:"سێ",4:"چار",5:"پێنج",6:"شەش",7:"هەفت",8:"هەشت",9:"نەهـ",
         10:"دەهـ",11:"یازدە",12:"دوازدە",13:"سێزدە",14:"چاردە",15:"پازدە",
         16:"شازدە",17:"هەڤدە",18:"هەژدە",19:"نوزدە"}
TENS  = {2:"بیست",3:"سیهـ",4:"چل",5:"پێنجی",6:"شێست",7:"هەفتێ",8:"هەشتێ",9:"نوت"}
HUNDREDS = {1:"سەد",2:"دووسەد",3:"سێسەد",4:"چارسەد",5:"پێنجسەد",6:"شەشسەد",7:"هەفتسەد",8:"هەشتسەد",9:"نەهسەد"}
HUND, THOU, ZERO, AND = "سەد", "هزار", "سفر", " و "
MILLION, BILLION = "ملیون", "ملیار"
PERCENT, OVER = "ژ سەدێ", "ل سەر"
POINT, HOUR, MINUTE, SECOND = "پوینت", "دەمژمێر", "خولەک", "چرکە"   # HOUR is said BEFORE the hour number
PLUS, MINUS = "کومکرنی", "ژێبرن"
DAY, MONTH, YEAR = "ڕۆژا", "هەیڤا", "سالا"
CURR = {"$": ("دولار", "سەنت"), "€": ("یۆرۆ", "سەنت")}   # symbol -> (main unit, cents)
D = "0-9٠-٩۰-۹"

def _below100(r):
    if r < 20: return ONES[r]
    t, u = divmod(r, 10)
    return TENS[t] + (AND + ONES[u] if u else "")

def _below1000(n):
    h, r = divmod(n, 100)
    parts = []
    if h: parts.append(HUNDREDS[h])
    if r: parts.append(_below100(r))
    return AND.join(parts)

def num2words_badini(n):
    if n == 0: return ZERO
    parts = []
    for scale, name in ((10**9, BILLION), (10**6, MILLION), (1000, THOU)):
        q, n = divmod(n, scale)
        if q: parts.append(name if (q == 1 and scale == 1000) else num2words_badini(q) + " " + name)
    if n: parts.append(_below1000(n))
    return AND.join(parts)

_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def _int(s):                       # digit string -> words (up to 999,999,999,999; longer numbers / 007 codes: digit by digit)
    s = s.translate(_ASCII)
    if len(s) > 1 and s[0] == "0" or len(s) > 12:
        return " ".join(num2words_badini(int(c)) for c in s)
    return num2words_badini(int(s))

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
    before = m.string[:m.start()].rstrip()
    hour = "" if re.search(rf"(?:{HOUR}|ساعەت|سەعات)$", before) else HOUR + " "   # text already says "دەمژمێر" -> don't repeat it
    out = [f"{hour}{_val(h)}"]                                  # 08:00 -> دەمژمێر هەشت (leading zero not read)
    if int(mi.translate(_ASCII)): out.append(f"{_val(mi)} {MINUTE}")
    if se and int(se.translate(_ASCII)): out.append(f"{_val(se)} {SECOND}")
    return AND.join(out)

def _val(s):                       # value of a clock / calendar field: 08 -> هەشت, 01 -> ئێک
    return num2words_badini(int(s.translate(_ASCII)))

def _fraction(m):                  # 3/8 -> سێ ل سەر هەشتێ  (same reading as NumbersAvin)
    return f"{_val(m.group(1))} {OVER} {_val(m.group(2))}ێ"

def _date(m):
    g = m.groups()
    if len(g[0].translate(_ASCII)) == 4: y, mo, d = g[0], g[1], g[2]     # 2026/11/23
    else:                                d, mo, y = g[0], g[1], g[2]     # 23/11/2026
    before = m.string[:m.start()].rstrip()
    day = "" if re.search(r"[ڕر]ۆژا$", before) else DAY + " "     # text already says "ڕۆژا" -> don't repeat it
    return f"{day}{_val(d)}، {MONTH} {_val(mo)}، {YEAR} {_val(y)}"

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
    # 2b) simple fraction  3/8 (dates were handled above)
    text = re.sub(rf"(?<![{D}/])([{D}]{{1,3}})/([{D}]{{1,3}})(?![{D}/])", _fraction, text)
    # 3) thousands commas  1,250 -> 1250
    text = re.sub(rf"(?<=[{D}]),(?=[{D}]{{3}}(?![{D}]))", "", text)
    # 4) currency  45.6$  /  $45.6
    sym = "".join(re.escape(k) for k in CURR)
    text = re.sub(rf"([{D}]+(?:\.[{D}]+)?)\s*([{sym}])", lambda m: _money(m.group(1), m.group(2)), text)
    text = re.sub(rf"([{sym}])\s*([{D}]+(?:\.[{D}]+)?)", lambda m: _money(m.group(2), m.group(1)), text)
    # 5) percent  25% / ٢٪  -> بیست و پێنج ژ سەدێ
    text = re.sub(rf"([{D}]+(?:\.[{D}]+)?)\s*[%٪]", lambda m: f"{_num(m.group(1))} {PERCENT}", text)
    # 6) decimals and plain numbers
    return re.sub(rf"[{D}]+(?:\.[{D}]+)?", lambda m: _num(m.group(0)), text)


# ---- light clean-up: only fixes what never occurs in the training transcripts ----
_ARABIC_TO_KURDISH = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "ے": "ی", "ة": "ە", "ھ": "ه"})
_DIACRITICS = re.compile("[ً-ْٰ]")       # Arabic harakat (fatha, damma, kasra, shadda, sukun ...)
ZWNJ = "‌"

def clean_text(text):
    text = html.unescape(text)                                  # &quot; &amp; ... (web text)
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)        # links cannot be spoken
    text = text.translate(_ARABIC_TO_KURDISH)
    text = _DIACRITICS.sub("", text)
    text = text.replace("ه" + ZWNJ, "ە").replace(ZWNJ, "")     # old Sorani-style ه+ZWNJ = ە; drop other ZWNJ
    text = re.sub("ـ{2,}", "ـ", text)                           # خـــۆش -> one tatweel at most
    text = re.sub(r"(?<=[^ه\s])ـ(?=\w)", "", text)              # tatweel only stays after ه (training style: شاهـ, دەهـ)
    text = re.sub(rf"(?<=[^\W\d_])(?=[{D}])", " ", text)        # ساڵی1950 -> ساڵی 1950 (a suffix after a number stays: 2024ێ)
    text = text.replace("“", "«").replace("”", "»")
    text = re.sub(r"\s+([؟?!:،,.؛])", r"\1", text)              # no space before punctuation
    text = re.sub(r"([(«])\s+", r"\1", text)
    text = re.sub(r"\s+([)»])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def final_h_tatweel(text):
    """Write word-final ه as هـ, like the training transcripts (هـ 324x vs ه 41x)."""
    return re.sub(r"ه(?=[^\wـ]|$)", "هـ", text)

def prepare_text(text):
    """clean-up -> numbers/dates/abbreviations to words -> training-style final هـ"""
    return final_h_tatweel(digits_to_words(clean_text(text)))
