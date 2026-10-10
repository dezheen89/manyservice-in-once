"""
Badini Kurdish text normalizer — companion module for computeram/badini-w2v-bert-normalized.

Applies the same normalization used to evaluate this model (punctuation
handling, Badini number-word expansion, character/word-variant unification)
to raw CTC decoder output, so predictions match the quality reported for the
"normalized" evaluation numbers rather than the raw greedy-decode numbers.

Usage:
    from badini_normalize import normalize_transcript
    clean_text = normalize_transcript(raw_model_output)
"""
import re
import unicodedata as ud

USE_BADINI_FORMS = True

def number2word(text):
    """Convert numerals in `text` into spoken (Sorani) Kurdish words."""
    unify = [
        ("٠|۰", "0"), ("١|۱", "1"), ("٢|۲", "2"), ("٣|۳", "3"), ("٤|۴", "4"),
        ("٥|۵", "5"), ("٦|۶", "6"), ("٧|۷", "7"), ("٨|۸", "8"), ("٩|۹", "9"),
    ]
    for pat, rep in unify:
        text = re.sub(pat, rep, text)
    text = re.sub(r"([0-9]{1,3})[,،](?=[0-9]{3})", r"\1", text)
    text = re.sub(r"(?<![0-9])-([0-9]+)", r"ناقس \1", text)
    text = re.sub(r"(?<![0-9])% ?([0-9]+)", r"لە سەددا \1", text)
    text = re.sub(r"([0-9]+) ?%", r"\1 لە سەد", text)
    text = re.sub(r"\$ ?([0-9]+(\.[0-9]+)?)", r"\1 دۆلار", text)
    text = re.sub(r"£ ?([0-9]+(\.[0-9]+)?)", r"\1 پاوەن", text)
    text = re.sub(r"€ ?([0-9]+(\.[0-9]+)?)", r"\1 یۆرۆ", text)
    text = re.sub(r"([0-9]+)\.([0-9]+)",
                  lambda m: _float_name(m.group(1), m.group(2)), text)
    text = re.sub(r"([0-9]+)", lambda m: _integer_name(m.group(1)), text)
    return text

def _float_name(integer_part, decimal_part):
    leading_zeros = len(decimal_part) - len(decimal_part.lstrip("0"))
    point = " پۆینت " + " سفر " * leading_zeros
    return _integer_name(integer_part) + point + _integer_name(decimal_part)

def _integer_name(input_integer):
    ones = ["", "یەک", "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ"]
    teens = ["دە", "یازدە", "دوازدە", "سێزدە", "چواردە", "پازدە", "شازدە",
             "حەڤدە", "هەژدە", "نۆزدە"]
    tens = ["", "", "بیست", "سی", "چل", "پەنجا", "شەست", "هەفتا", "هەشتا", "نەوەد"]
    hundreds = ["", "سەد", "دووسەد", "سێسەد", "چوارسەد", "پێنسەد", "شەشسەد",
                "حەوتسەد", "هەشتسەد", "نۆسەد"]
    thousands = ["", " هەزار", " ملیۆن", " ملیار", " بلیۆن", " بلیار",
                 " تریلیۆن", " تریلیار", " کوادرلیۆن"]
    if input_integer == "0":
        return "سفر"
    output = ""
    temp = input_integer
    for i in range(0, len(input_integer), 3):
        m = re.search(r"([0-9]{1,3})$", temp)
        current_three = m.group(1) if m else ""
        temp = temp[:len(temp) - len(current_three)]
        current_three = current_three.rjust(3, "0")
        C = int(current_three[0]); X = int(current_three[1]); I = int(current_three[2])
        conjunction1 = " و " if (C != 0 and (X != 0 or I != 0)) else ""
        conjunction2 = " و " if (X != 0 and I != 0) else ""
        if X == 1:
            current_three = hundreds[C] + conjunction1 + teens[I]
        else:
            current_three = hundreds[C] + conjunction1 + tens[X] + conjunction2 + ones[I]
        magnitude = "" if current_three == "" else thousands[i // 3]
        current_three += magnitude
        conjunction3 = "" if output == "" else " و "
        if current_three != "":
            output = current_three + conjunction3 + output
    output = output.replace("یەک هەزار", "هەزار")
    return output

_SORANI_TO_BADINI = {
    "یەک": "ئێک", "چوار": "چار", "حەوت": "هەفت", "نۆ": "نەه", "دە": "دەه",
    "چواردە": "چاردە", "حەڤدە": "هەڤدە",
    "سی": "سیه", "پەنجا": "پێنجی", "شەست": "شێست",
    "هەفتا": "هەفتێ", "هەشتا": "هەشتێ", "نەوەد": "نۆت",
    "چوارسەد": "چارسەد", "پێنسەد": "پێنجسەد",
    "حەوتسەد": "هەفتسەد", "نۆسەد": "نەهسەد",
    "هەزار": "هزار", "ملیۆن": "ملیون",
}

def to_badini_numbers(text):
    out = []
    for w in text.split():
        if   w in _SORANI_TO_BADINI:
            out.append(_SORANI_TO_BADINI[w])
        elif w[:-1] in _SORANI_TO_BADINI and w[-1:] in ('ێ', 'ی'):
            out.append(_SORANI_TO_BADINI[w[:-1]] + w[-1])
        elif w[:-2] in _SORANI_TO_BADINI and w[-2:] in ('یێ',):
            out.append(_SORANI_TO_BADINI[w[:-2]] + w[-2:])
        else:
            out.append(w)
    return " ".join(out)

def expand_numbers(text):
    text = re.sub(r'(?<=\d)\s*/\s*(?=\d)', ' ', text)
    text = number2word(text)
    if USE_BADINI_FORMS:
        text = re.sub(r'([،,؟!\.\-\;\:\"\'\»\«\(\)\[\]٪%؛/])', r' \1 ', text)
        text = to_badini_numbers(text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_chars(text):
    text=text.replace('ك','ک').replace('ى','ی').replace('ي','ی').replace('ة','ە')
    text=text.replace('ݒ','پ').replace('ݘ','چ').replace('ݵ','ێ')
    text=text.replace('ﻩ','ه').replace('ھ','ه')
    return text

def fix_ku_ko(text):
    return re.sub(r'\bکو\b', 'کۆ', text)

def normalize_shared(text):
    text=re.sub(r'(?<![ا-ی\w])دکەن\b','د کەن',text)
    text=re.sub(r'(?<![ا-ی\w])دکەت\b','د کەت',text)
    text=re.sub(r'(?<![ا-ی\w])دکەم\b','د کەم',text)
    text=re.sub(r'(?<![ا-ی\w])دکەین\b','د کەین',text)
    for a,b in [('دگەل','د گەل'),('لسەر','ل سەر'),('لپێش','ل پێش'),('لگەل','ل گەل'),
                ('بخۆڤە','ب خۆڤە'),('دناڤ','د ناڤ'),('لدویڤدا','ل دویڤدا'),('لدور','ل دور')]:
        text=text.replace(a,b)
    text=re.sub(r'\bب ڕێکا\b','برێکا',text)
    for a,b in [('نەخۆشییا','نەخۆشیا'),('هاتییە','هاتیە'),('بووییە','بوویە'),
                ('گرێداییی','گرێدایی'),('دەریاییی','دەریایی'),('ئاساییی','ئاسای'),
                ('ئاسایی','ئاسای'),('پترییا','پتریا')]:
        text=text.replace(a,b)
    for a,b in [('دور','دۆر'),('توز','تۆز'),('تسونامی','تسۆنامی'),('گورەی','گۆرەی'),
                ('ڕوژێ','ڕۆژێ'),('تومەتبارکرن','تۆمەتبارکرن')]:
        text=text.replace(a,b)
    for a,b in [('ڤەگوهازتن','ڤەگوهاستن'),('پێشکەڤتی','پێشکەفتی'),('بەرەڤ','بەرەف'),
                ('چاڤدێڕیکڕنا','چاڤدێریکرنا'),('بەرامبەر','بەرانبەر')]:
        text=text.replace(a,b)
    for a,b in [('بچویک','بچیک'),('نموونە','نمونە'),('دووبارە','دوبارە'),
                ('کۆمپانییێن','کۆمپانیێن'),('نڤێسین','نڤیسین'),('نێزیکی','نیزیکی'),
                ('زارۆکێن','زاروکێن'),('چاوا','چەوا'),('تەمووزێ','تەموزێ'),
                ('شێوەیەکێ','شێوەکێ'),('گووت','گوت'),('قۆتابیان','قوتابیان'),
                ('نیڤدەولەتی','نێڤدەولەتی'),('ئیتاڵی','ئیتالی'),('ئاگاهداریێن','ئاگەهداریێن'),
                ('هاریکاریا','هاریکارییا'),('گەشتیێن','گەشتییێن'),('ساڵ','سال')]:
        text=text.replace(a,b)
    return re.sub(r'\s+', ' ', text).strip()

WORD_VARIANTS = {
    'خوە':'خۆ', 'دەستپێکێ':'دەسپێکێ', 'دوماهیک':'دویماهیک',
    'دویڤ':'دیف', 'گەهە':'گهە',
    'بو':'بۆ','خو':'خۆ','کو':'کۆ','کوێ':'کۆێ','کویێ':'کۆیێ','کوژمێ':'کۆژمێ',
    'زور':'زۆر','تومارکرن':'تۆمارکرن','نەخوشیا':'نەخۆشیا','مروڤ':'مرۆڤ','مروڤان':'مرۆڤان',
    'کومەلێن':'کۆمەلێن','سەروک':'سەرۆک','سەروکێ':'سەرۆکێ','سەروکا':'سەرۆکا',
    'فروکە':'فرۆکە','فروکەیەک':'فرۆکەیەک','ڕوژا':'ڕۆژا','ڕوژان':'ڕۆژان',
    'توشی':'تووشی','ساڵا':'سالا','ساڵ':'سال','داکو':'دا کو',
    'هەروەسا':'هەر وەسا','هەروەسان':'هەر وەسان','هەرچەندە':'هەر چەندە','هەردوو':'هەر دوو',
    'بتنێ':'ب تنێ','ژبو':'ژ بۆ','ژبۆ':'ژ بۆ','شورەشا':'شۆڕەشا','پشتراست':'پشتڕاست',
    'کەلتور':'کەلتوور','کەلتورێ':'کەلتوورێ','کەلتوری':'کەلتووری','برێكا':'برێکا',
    'دەریای':'دەریایی','ساال':'سال','شێوەیەکێ':'شێوەکێ',
}

def strip_invisibles(text):
    return re.sub(r'[\u200c\u200d\ufeff\u0640]', '', text)

def collapse_doubles(text):
    return text.replace('وو','و').replace('یی','ی')

_DIGIT_MAP = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

def _front(text):
    text = strip_invisibles(text)
    text = ud.normalize('NFKC', text)
    text = text.translate(_DIGIT_MAP)
    text = normalize_chars(text)
    text = expand_numbers(text)
    return text

def _back(text):
    text = ' '.join(WORD_VARIANTS.get(w, w) for w in text.split())
    text = re.sub(r'\bر', 'ڕ', text)
    text = fix_ku_ko(text)
    text = normalize_shared(text)
    text = collapse_doubles(text)
    return text

def normalize(text):
    text = _front(text)
    text = re.sub(r'[،,؟!\.\-\;\:\"\'»«\(\)\[\]٪%؛/]', ' ', text)
    return _back(text)

def normalize_punct_ext(text):
    text = _front(text)
    text = text.replace('?', '؟').replace(',', '،')
    text = re.sub(r'([؟،\.\!\(\)«»":؛])', r' \1 ', text)
    return _back(text)

def normalize_transcript(text):
    """Recommended: punctuation stripped, word-level normalization.
    This is the configuration the model's reported WER/CER are measured on."""
    return normalize(text)

def normalize_transcript_keep_punct(text):
    """Alternative: keeps the 11 punctuation marks the model was trained on."""
    return normalize_punct_ext(text)
