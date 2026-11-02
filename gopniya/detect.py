"""Rule-based detection of personal identifiers in Indian clinical text (English + Hindi).

Everything here runs locally with the Python standard library only, so the same code
runs in a CLI, a server, or in the browser through Pyodide.

Design:
- Devanagari digits (U+0966..U+096F) are mapped to ASCII digits one-to-one before the
  numeric patterns run, so offsets stay valid for the original text.
- Numeric identifiers use format patterns plus check digits (Aadhaar Verhoeff, PAN
  structure). A label next to a number ("Aadhaar:", "आधार:") raises confidence and lets a
  number through even when its check digit is off (typos happen in hand-typed notes).
- Names and addresses use context cues: honorifics (Mr., Smt., श्री, श्रीमती), field
  labels ("Patient Name:", "रोगी का नाम:"), relation markers (S/o, W/o, पुत्र, पत्नी).
  There is deliberately no name dictionary: it would inflate scores on synthetic data.
- Every name found is then propagated: other mentions of the same name or of its parts
  are redacted too ("Smt. Sunita Devi ... Sunita was advised ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import validators as v

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_digits(text: str) -> str:
    """Map Devanagari digits to ASCII. Length-preserving, so offsets do not move."""
    return text.translate(_DEV_DIGITS)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    text: str
    score: float
    source: str = "rules"

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


ENTITY_TYPES = (
    "PERSON",
    "ADDRESS",
    "AADHAAR",
    "ABHA_NUMBER",
    "ABHA_ADDRESS",
    "PAN",
    "VOTER_ID",
    "PHONE",
    "EMAIL",
    "MRN",
    "DOB",
    "PINCODE",
    "ID_NUMBER",
)

# ---------------------------------------------------------------- numeric identifiers

_SEP = r"[ \-]?"
_AADHAAR_RE = re.compile(r"(?<![\dA-Za-z])[2-9]\d{3}" + _SEP + r"\d{4}" + _SEP + r"\d{4}(?![\dA-Za-z])")
_AADHAAR_LABEL_RE = re.compile(
    r"(?:Aadhaa?r|Aadhar|UIDAI?|UID|आधार)(?:\s*(?:card|no\.?|number|nos?|कार्ड|संख्या|नंबर|नं\.?))?"
    r"\s*[:\-#]?\s*((?:[\dXx*]{4}[ \-]?){2}\d{4})",
    re.IGNORECASE,
)
_ABHA_FMT_RE = re.compile(r"(?<![\dA-Za-z])\d{2}-\d{4}-\d{4}-\d{4}(?![\dA-Za-z])")
_ABHA_LABEL_RE = re.compile(
    r"(?:ABHA|Health\s*ID|आभा)(?:\s*(?:no\.?|number|ID|संख्या|नंबर))?\s*[:\-#]?\s*"
    r"(\d{2}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4})",
    re.IGNORECASE,
)
_ABHA_ADDR_RE = re.compile(r"(?<![\w.])[A-Za-z0-9][A-Za-z0-9._]{2,}@(?:abdm|sbx|ndhm)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"(?<![\w.])[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
_PAN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Za-z0-9])")
_VOTER_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}\d{7}(?![A-Za-z0-9])")
_PHONE_RE = re.compile(
    r"(?<![\dA-Za-z+])(?:(?:\+91|0091|91)[ \-]?|0)?(?:[6-9]\d{4}[ \-]?\d{5}|[6-9]\d{2}[ \-]\d{3}[ \-]\d{4})(?!\d)"
)
_PHONE_LABEL_RE = re.compile(
    r"(?:Ph(?:one)?|Tel|Mob(?:ile)?|Contact(?:\s*no\.?)?|Cell|फ़?फोन|मोबाइल|मो\.|संपर्क)\.?\s*(?:no\.?|नं\.?)?\s*[:\-]?\s*"
    r"((?:\+91[ \-]?)?\d[\d \-]{6,13}\d)",
    re.IGNORECASE,
)
_MRN_LABEL_RE = re.compile(
    r"(?:UHID|MRN|MR\s*No|CR\s*No|IPD\s*No|OPD\s*No|IP\s*No|Reg(?:istration|n)?\.?\s*No|Hospital\s*No|"
    r"Patient\s*ID|Sample\s*ID|Lab\s*ID|Bed\s*Ticket\s*No|पंजीकरण\s*(?:संख्या|सं\.?)|रजि\.?\s*नं\.?|"
    r"ओपीडी\s*(?:नं\.?|संख्या)|आईपीडी\s*(?:नं\.?|संख्या))\.?\s*[:\-#]?\s*([A-Za-z0-9][A-Za-z0-9/\-]{3,19})",
    re.IGNORECASE,
)
_DATE = (
    r"(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4}|"
    r"\d{1,2}\s+[ऀ-ॣ॰-ॿ]+\s+\d{4}|\d{4}-\d{2}-\d{2})"
)
_DOB_RE = re.compile(
    r"(?:DOB|D\.O\.B\.?|Date\s+of\s+Birth|Born\s+on|जन्म\s*तिथि|जन्मतिथि|जन्म\s*दिनांक)\s*[:\-]?\s*(" + _DATE + ")",
    re.IGNORECASE,
)
_PIN_LABEL_RE = re.compile(r"(?:PIN|Pin\s*code|Pincode|पिन(?:\s*कोड)?)\s*[:\-]?\s*(\d{6})(?!\d)", re.IGNORECASE)

# ---------------------------------------------------------------- names

_DEV = r"[ऀ-ॣ॰-ॿ‌‍]"
_DEV_WORD = _DEV + r"+"
_LAT_WORD = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"

# Capitalised words that follow honorifics/labels but are not names.
_EN_STOP = {
    "patient", "hospital", "age", "aged", "sex", "male", "female", "ward", "diagnosis", "admitted",
    "discharged", "on", "with", "was", "is", "has", "the", "and", "of", "in", "at", "for", "to",
    "years", "yrs", "year", "old", "from", "village", "district", "dist", "tehsil", "block",
    "address", "phone", "mobile", "contact", "aadhaar", "abha", "pan", "dob", "uhid", "mrn",
    "doctor", "consultant", "department", "dept", "medicine", "surgery", "obstetrics", "gynaecology",
    "pediatrics", "paediatrics", "chc", "phc", "district", "general", "civil", "government", "govt",
    "unit", "opd", "ipd", "icu", "na", "nil", "none", "unknown", "date", "time", "reg", "no",
    "chief", "complaint", "complaints", "history", "examination", "advice", "advised", "follow",
    "review", "medical", "officer", "staff", "nurse", "anm", "s", "d", "w", "c", "h",
    "son", "daughter", "wife", "husband", "relation", "relative", "attendant", "guardian",
    "presented", "presents", "came", "brought", "reports", "reported", "known", "case",
}
_HI_STOP = {
    "को", "का", "की", "के", "ने", "से", "में", "पर", "और", "उम्र", "आयु", "वर्ष", "साल", "पुत्र", "पुत्री",
    "पत्नी", "पति", "निवासी", "ग्राम", "गांव", "गाँव", "जिला", "तहसील", "मरीज", "मरीज़", "रोगी", "भर्ती",
    "है", "हैं", "था", "थी", "थे", "तथा", "एवं", "द्वारा", "लिए", "हेतु", "पता", "फोन", "मोबाइल", "आधार",
    "आभा", "डॉक्टर", "चिकित्सक", "अस्पताल", "वार्ड", "दिनांक", "जन्म", "तिथि", "नाम", "महिला", "पुरुष",
    "बच्चा", "बच्ची", "शिशु", "माता", "पिता", "परिचारक", "आशा", "कार्यकर्ता", "एएनएम", "सलाह", "दी", "गई",
    "गया", "लाया", "लाई", "लाए", "आई", "आए", "आया", "बताया", "जो", "जी",
}

_EN_HONORIFIC = r"(?:Mr|Mrs|Ms|Miss|Dr|Shri|Sri|Smt|Kumari|Km|Master|Baby\s+of|B/o)\.?"
_HI_HONORIFIC = r"(?:श्रीमती|श्री|सुश्री|कुमारी|कु\.|डॉ\.?|डा\.|स्व\.)(?![\u0900-\u0963\u0970-\u097F])"
_EN_LABEL = (
    r"(?:Patient(?:'s)?\s+Name|Name\s+of\s+(?:the\s+)?Patient|Pt\.?\s+Name|Name|Attendant(?:'s)?\s+Name|Attendant|"
    r"Guardian|Father(?:'s)?\s+Name|Husband(?:'s)?\s+Name|Mother(?:'s)?\s+Name|Informant|Next\s+of\s+Kin|"
    r"Emergency\s+Contact|Referred\s+by|ASHA(?:\s+worker)?|ANM|Consultant|Treating\s+Doctor|Signed\s+by)"
)
_HI_LABEL = (
    r"(?:मरीज़?\s+का\s+नाम|रोगी\s+का\s+नाम|पिता\s+का\s+नाम|पति\s+का\s+नाम|माता\s+का\s+नाम|परिचारक(?:\s+का\s+नाम)?|"
    r"आशा(?:\s+कार्यकर्ता)?|एएनएम|चिकित्सक|नाम)"
)
_EN_RELATION = r"(?:S/o|D/o|W/o|C/o|H/o|son\s+of|daughter\s+of|wife\s+of|husband\s+of|mother\s+of|father\s+of)"
_HI_RELATION = r"(?:पुत्र|पुत्री|पत्नी|पति|बेटा|बेटे|बेटी|भाई|बहन|बहू|सास|ससुर|मां|माँ|देवर|भाभी)"
# kinship nouns written before a name ("her husband Ramesh", "beti Mary", "accompanied by Iqbal")
_EN_KIN = (
    r"(?:husband|wife|son|daughter|brother|sister|mother|father|grandson|granddaughter|grandmother|grandfather|"
    r"nephew|niece|uncle|aunt|neighbour|neighbor|attendant|guardian|accompanied\s+by|brought\s+by|"
    r"beta|beti|bete|bhai|behen|bahan|bahu|patni|pati|maa|saas|sasur|devar|bhabhi)"
)

_EN_HONORIFIC_RE = re.compile(r"\b" + _EN_HONORIFIC + r"\s+")
_EN_LABEL_RE = re.compile(r"\b" + _EN_LABEL + r"\s*[:\-]\s*(?:" + _EN_HONORIFIC + r"\s+)?", re.IGNORECASE)
_EN_RELATION_RE = re.compile(r"\b" + _EN_RELATION + r"\s+(?:" + _EN_HONORIFIC + r"\s+)?", re.IGNORECASE)
_EN_KIN_RE = re.compile(r"\b" + _EN_KIN + r",?\s+(?:" + _EN_HONORIFIC + r"\s+)?", re.IGNORECASE)
_HI_HONORIFIC_RE = re.compile(r"(?<!" + _DEV + r")" + _HI_HONORIFIC + r"\s*")
_HI_LABEL_RE = re.compile(r"(?<!" + _DEV + r")" + _HI_LABEL + r"\s*[:\-]\s*(?:" + _HI_HONORIFIC + r"\s*)?")
_HI_RELATION_RE = re.compile(r"(?<!" + _DEV + r")" + _HI_RELATION + r"\s+(?:" + _HI_HONORIFIC + r"\s*)?")

_LAT_WORD_RE = re.compile(_LAT_WORD)
_DEV_TOKEN_RE = re.compile(_DEV_WORD)

# ---------------------------------------------------------------- addresses

_ADDR_STOP = (
    r"(?=\n|;|।|\.\s+[A-Zऀ-ॿ]|\s*(?:Ph(?:one)?|Mob(?:ile)?|Contact|Tel|Aadhaa?r|ABHA|PAN|DOB|"
    r"UHID|MRN|Age|फोन|मोबाइल|आधार|आभा|उम्र|आयु)\b|$)"
)
_ADDR_LABEL_RE = re.compile(
    r"(?:\b(?:Residential\s+address|Permanent\s+address|Present\s+address|Address|Addr\.|R/o|Resident\s+of|"
    r"Residing\s+at)|(?<!" + _DEV + r")(?:स्थायी\s+पता|पता|निवासी))\s*[:\-]?\s*",
    re.IGNORECASE,
)
_ADDR_START_RE = re.compile(
    r"(?:\b(?:Vill(?:age)?|Gram|Mohalla|H\.\s?No|House\s+No|Plot\s+No|Flat\s+No|Ward\s+No)\b\.?|"
    r"(?<!" + _DEV + r")(?:ग्राम|गांव|गाँव|मोहल्ला|मकान\s+नं\.?|वार्ड\s+नं\.?))\s*[:\-]?\s*",
    re.IGNORECASE,
)
_ADDR_BODY_RE = re.compile(r"(.{3,180}?)" + _ADDR_STOP, re.DOTALL)


def _take_name(text: str, pos: int, script: str) -> tuple[int, int] | None:
    """Consume 1-3 name tokens starting at ``pos``. Returns (start, end) or None."""
    tokens: list[tuple[int, int]] = []
    i = pos
    n = len(text)
    while len(tokens) < 3 and i < n:
        if script == "lat":
            m = _LAT_WORD_RE.match(text, i)
            if not m or m.group(0).lower() in _EN_STOP:
                break
        else:
            m = _DEV_TOKEN_RE.match(text, i)
            if not m or m.group(0) in _HI_STOP:
                break
        tokens.append((m.start(), m.end()))
        j = m.end()
        # allow one space (or ". " after an initial) between tokens on the same line
        k = j
        if k < n and text[k] == "." and script == "lat" and m.end() - m.start() == 1:
            k += 1
        if k < n and text[k] == " ":
            i = k + 1
        else:
            break
    if not tokens:
        return None
    return tokens[0][0], tokens[-1][1]


def _name_spans(text: str) -> list[Span]:
    out: list[Span] = []
    lat_cues = (
        (_EN_HONORIFIC_RE, 0.85),
        (_EN_LABEL_RE, 0.9),
        (_EN_RELATION_RE, 0.85),
        (_EN_KIN_RE, 0.7),
    )
    for rx, score in lat_cues:
        for m in rx.finditer(text):
            got = _take_name(text, m.end(), "lat")
            if got is None:
                # label followed by a Devanagari name ("Name: सुनीता देवी")
                got = _take_name(text, m.end(), "dev")
            if got:
                out.append(Span(got[0], got[1], "PERSON", text[got[0]:got[1]], score))
    dev_cues = ((_HI_HONORIFIC_RE, 0.85), (_HI_LABEL_RE, 0.9), (_HI_RELATION_RE, 0.8))
    for rx, score in dev_cues:
        for m in rx.finditer(text):
            got = _take_name(text, m.end(), "dev") or _take_name(text, m.end(), "lat")
            if got:
                out.append(Span(got[0], got[1], "PERSON", text[got[0]:got[1]], score))
    return out


def _address_spans(text: str) -> list[Span]:
    out: list[Span] = []
    for rx, score in ((_ADDR_LABEL_RE, 0.85), (_ADDR_START_RE, 0.75)):
        for m in rx.finditer(text):
            start = m.end() if rx is _ADDR_LABEL_RE else m.start()
            body = _ADDR_BODY_RE.match(text, m.end())
            if not body:
                continue
            end = body.end(1)
            # trim trailing punctuation/space
            while end > start and text[end - 1] in " ,.-:":
                end -= 1
            if end - start >= 4:
                out.append(Span(start, end, "ADDRESS", text[start:end], score))
    return out


def _group_or_all(m: re.Match[str]) -> tuple[int, int]:
    if m.lastindex:
        return m.start(1), m.end(1)
    return m.start(), m.end()


def _numeric_spans(text: str) -> list[Span]:
    t = normalize_digits(text)
    out: list[Span] = []

    def add(s: int, e: int, typ: str, score: float) -> None:
        out.append(Span(s, e, typ, text[s:e], score))

    for m in _AADHAAR_RE.finditer(t):
        if v.aadhaar_valid(m.group(0)):
            add(m.start(), m.end(), "AADHAAR", 0.95)
    for m in _AADHAAR_LABEL_RE.finditer(t):
        s, e = _group_or_all(m)
        add(s, e, "AADHAAR", 0.97)
    for m in _ABHA_FMT_RE.finditer(t):
        add(m.start(), m.end(), "ABHA_NUMBER", 0.9)
    for m in _ABHA_LABEL_RE.finditer(t):
        s, e = _group_or_all(m)
        add(s, e, "ABHA_NUMBER", 0.97)
    for m in _ABHA_ADDR_RE.finditer(t):
        add(m.start(), m.end(), "ABHA_ADDRESS", 0.97)
    for m in _EMAIL_RE.finditer(t):
        if not re.search(r"@(?:abdm|sbx|ndhm)\b", m.group(0), re.IGNORECASE):
            add(m.start(), m.end(), "EMAIL", 0.95)
    for m in _PAN_RE.finditer(t):
        if v.pan_valid(m.group(0)):
            add(m.start(), m.end(), "PAN", 0.93)
    for m in _VOTER_RE.finditer(t):
        add(m.start(), m.end(), "VOTER_ID", 0.8)
    for m in _PHONE_RE.finditer(t):
        if v.indian_mobile_valid(m.group(0)):
            add(m.start(), m.end(), "PHONE", 0.9)
    for m in _PHONE_LABEL_RE.finditer(t):
        s, e = _group_or_all(m)
        if sum(c.isdigit() for c in t[s:e]) >= 8:
            add(s, e, "PHONE", 0.92)
    for m in _MRN_LABEL_RE.finditer(t):
        s, e = _group_or_all(m)
        if any(c.isdigit() for c in t[s:e]):
            add(s, e, "MRN", 0.9)
    for m in _DOB_RE.finditer(t):
        s, e = _group_or_all(m)
        add(s, e, "DOB", 0.92)
    for m in _PIN_LABEL_RE.finditer(t):
        s, e = _group_or_all(m)
        add(s, e, "PINCODE", 0.9)
    return out


# ---------------------------------------------------------------- merge + propagation

_TYPE_PRIORITY = {
    # structured identifiers beat free-text spans when they overlap
    "AADHAAR": 9, "ABHA_NUMBER": 9, "ABHA_ADDRESS": 9, "PAN": 8, "VOTER_ID": 7, "EMAIL": 8,
    "PHONE": 7, "MRN": 6, "ID_NUMBER": 5, "DOB": 6, "PINCODE": 5, "PERSON": 4, "ADDRESS": 3,
}


def merge(spans: list[Span]) -> list[Span]:
    """Resolve overlaps: identifiers first, then score, then length."""
    ranked = sorted(spans, key=lambda s: (-_TYPE_PRIORITY.get(s.type, 0), -s.score, -s.length, s.start))
    kept: list[Span] = []
    for s in ranked:
        clash = [k for k in kept if k.overlaps(s)]
        if not clash:
            kept.append(s)
            continue
        # An address that contains an identifier: keep both by trimming nothing; the
        # identifier wins its characters and the address keeps the rest via split.
        if s.type == "ADDRESS" and all(k.type != "ADDRESS" for k in clash):
            pieces = []
            cur = s.start
            for k in sorted(clash, key=lambda x: x.start):
                if k.start > cur:
                    pieces.append((cur, k.start))
                cur = max(cur, k.end)
            if cur < s.end:
                pieces.append((cur, s.end))
            for a, b in pieces:
                seg = s.text[a - s.start:b - s.start]
                strip_l = len(seg) - len(seg.lstrip(" ,.-:"))
                strip_r = len(seg) - len(seg.rstrip(" ,.-:"))
                a2, b2 = a + strip_l, b - strip_r
                if b2 - a2 >= 3:
                    kept.append(Span(a2, b2, "ADDRESS", s.text[a2 - s.start:b2 - s.start], s.score, s.source))
    return sorted(kept, key=lambda s: s.start)


def _name_parts(name: str) -> list[str]:
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    out = [name.strip()]
    for p in parts:
        p = p.strip(".")
        if len(p) >= 3 and p.lower() not in _EN_STOP and p not in _HI_STOP:
            out.append(p)
    return out


def propagate_names(text: str, spans: list[Span]) -> list[Span]:
    """Add PERSON spans for every other mention of a detected name or its parts."""
    extra: list[Span] = []
    seen: set[str] = set()
    for s in spans:
        if s.type != "PERSON":
            continue
        for part in _name_parts(s.text):
            if part in seen:
                continue
            seen.add(part)
            if re.search(_DEV, part):
                rx = re.compile(r"(?<!" + _DEV + r")" + re.escape(part) + r"(?!" + _DEV + r")")
            else:
                rx = re.compile(r"(?<![A-Za-z])" + re.escape(part) + r"(?![A-Za-z])")
            for m in rx.finditer(text):
                extra.append(Span(m.start(), m.end(), "PERSON", m.group(0), 0.7, "propagation"))
    return spans + extra


def detect(text: str, extra_spans: list[Span] | None = None) -> list[Span]:
    """Run every rule pass, optional external spans (e.g. local LLM-NER), merge, propagate."""
    spans = _numeric_spans(text) + _name_spans(text) + _address_spans(text)
    if extra_spans:
        spans += extra_spans
    spans = merge(spans)
    spans = merge(propagate_names(text, spans))
    return spans
