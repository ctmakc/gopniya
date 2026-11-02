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
# label separator: ":", ":-", "-", "=", "|", "–" with optional spaces/tabs ("DOB:- 12/03/1992", "Name = X")
LSEP = r"[ \t]*(?:[:=|\-–.][ \t]*){1,3}"
_AADHAAR_RE = re.compile(r"(?<![\dA-Za-z])[2-9]\d{3}" + _SEP + r"\d{4}" + _SEP + r"\d{4}(?![\dA-Za-z])")
_AADHAAR_LABEL_RE = re.compile(
    r"(?:Aadhaa?r|Aadhar|UIDAI?|UID|आधार)(?:\s*(?:card|no\.?|number|nos?|कार्ड|संख्या|नंबर|नं\.?))?"
    r"(?:[ \t]*(?:[:=|\-–.#][ \t]*){1,3}|\s+)((?:[\dXx*]{4}[ \-]?){2}\d{4})",
    re.IGNORECASE,
)
_ABHA_FMT_RE = re.compile(r"(?<![\dA-Za-z])\d{2}-\d{4}-\d{4}-\d{4}(?![\dA-Za-z])")
_ABHA_LABEL_RE = re.compile(
    r"(?:ABHA|Health\s*ID|आभा)(?:\s*(?:no\.?|number|ID|संख्या|नंबर))?(?:[ \t]*(?:[:=|\-–.#][ \t]*){1,3}|\s+)"
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
    r"(?:Ph(?:one)?|Tel|Mob(?:ile|il)?|Contact(?:\s*no\.?)?|Cell|WhatsApp|फ़?फोन|मोबाइल|मो\.|संपर्क(?:\s*नं\.?)?)\.?\s*(?:no\.?|नं\.?)?(?:[ \t]*(?:[:=|\-–.][ \t]*){1,3}|\s+)"
    r"((?:\+91[ \-]?)?\d[\d \-]{6,13}\d)",
    re.IGNORECASE,
)
_MRN_LABEL_RE = re.compile(
    r"(?:UHID|MRN|MRD\s*No|MR\s*No|CR\s*No|IPD\s*No|OPD\s*No|IP\s*No|Reg(?:istration|n)?\.?\s*No|Hospital\s*No|"
    r"Patient\s*ID|Sample\s*(?:ID|No)|Lab\s*(?:ID|No)|Bed\s*Ticket\s*No|Case\s*No|Adm(?:ission)?\.?\s*No|"
    r"Ni-?kshay\s*ID|TB\s*No|RCH\s*ID|U-?WIN\s*ID|(?:AB-?)?PM-?JAY\s*ID|Beneficiary\s*ID|Pre-?auth(?:orisation|orization)?\s*No|"
    r"Consultation\s*ID|Tele-?consult(?:ation)?\s*ID|MLC\s*No|Donor\s*ID|"
    r"पंजीकरण\s*(?:संख्या|सं\.?)|रजि\.?\s*नं\.?|ओपीडी\s*(?:नं\.?|संख्या)|आईपीडी\s*(?:नं\.?|संख्या)|"
    r"निक्षय\s*आईडी|यूएचआईडी|भर्ती\s*(?:क्रमांक|संख्या|सं\.?)|पीएम-?जय\s*आईडी)\.?"
    r"(?:\s*\([^)\n]{0,30}\))?(?:[ \t]*(?:[:=|\-–.#][ \t]*){1,3}|\s+)([A-Za-z0-9][A-Za-z0-9/\-]{2,24})",
    re.IGNORECASE,
)
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE = (
    r"(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?[\s\-]+" + _MON + r",?[\s\-]+\d{2,4}|"
    r"" + _MON + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|"
    r"\d{1,2}\s+[\u0900-\u0963\u0970-\u097F]+\s+\d{4}|\d{4}-\d{2}-\d{2})"
)
_DOB_RE = re.compile(
    r"(?:DOB|D\.O\.B\.?|Date\s+of\s+Birth(?:\s+of\s+(?:child|baby))?|Birth\s*date|Born\s+on|Janm\s*tithi|"
    r"जन्म\s*तिथि|जन्मतिथि|जन्म\s*दिनांक|जन्म\s*की\s*तारीख़?)(?:\s*\((?:if\s+known|approx\.?)\))?"
    r"(?:[ \t]*(?:[:=|\-–.][ \t]*){1,3}|\s+)(" + _DATE + ")",
    re.IGNORECASE,
)
_PIN_LABEL_RE = re.compile(r"(?:PIN|Pin\s*code|Pincode|पिन(?:\s*कोड)?)(?:[ \t]*(?:[:=|\-–.][ \t]*){1,3}|\s+)(\d{6})(?!\d)", re.IGNORECASE)

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

_EN_HONORIFIC = r"(?:Mr|Mrs|Ms|Miss|Dr|Shri|Sri|Smt|Kumari|Km|Mst|Master|Baby\s+of|B/o|Late|Sister|Sr|SN)\.?"
_HI_HONORIFIC = r"(?:श्रीमती|श्री|सुश्री|कुमारी|कु\.|डॉ[\.०]?|डा\.|स्व\.|मो\.|बेबी\s+ऑफ़?)(?![\u0900-\u0963\u0970-\u097F])"
# Field labels after which a name follows (English, Hinglish). Matched case-insensitively.
_EN_LABEL = (
    r"(?:Patient(?:'s)?\s+Name|Name\s+of\s+(?:the\s+)?(?:Patient|child|baby|mother|father|husband|wife|deceased|donor|informant)|"
    r"Pt\.?\s+Name|Pateint|Patient|Pt|Name|Attendant(?:'s)?\s+Name|Attendant|Guardian|"
    r"(?:Father|Husband|Mother|Wife|Spouse|Son|Daughter|Child|Baby|Guardian)(?:'s)?(?:\s+Name)?|"
    r"F/N|H/N|M/N|Informant|Caller|Donor|Next\s+of\s+Kin|Emergency\s+Contact|Ref(?:erred)?\.?\s*By|Referred\s+to\s+Dr|"
    r"ASHA(?:\s+worker|\s+name)?|ANM|CHO|MO|Medical\s+Officer|Consultant|Treating\s+Doctor|Doctor|Hub\s+doctor|"
    r"Signed\s+by|Examined\s+by|Seen\s+by|Reported\s+by|Verified\s+by|Lab\s+Tech(?:nician)?|Technician|Pathologist|"
    r"Radiologist|Nurse|Staff\s+Nurse|Aarogya\s+Mitra|Treatment\s+supporter|Delivery\s+conducted\s+by|"
    r"Name\s+of\s+PW|Vaccinator|Escorted\s+by|Copy\s+to|CC|Naam|Mareez|Pita|Mata|Pati|Patni)"
)
_HI_LABEL = (
    r"(?:मरीज़?\s+का\s+नाम|रोगी\s+का\s+नाम|पिता\s+का\s+नाम|पति\s+का\s+नाम|माता\s+का\s+नाम|पिता/पति\s+का\s+नाम|"
    r"शिशु\s+का\s+नाम|बच्चे\s+का\s+नाम|मृतक\s+का\s+नाम|परिचारक(?:\s+का\s+नाम)?|उपचार\s+सहयोगी|सूचनादाता|"
    r"आरोग्य\s+मित्र|इलाज\s+करने\s+वाले\s+डॉक्टर|प्रसव\s+कराने\s+वाली|हस्ताक्षरकर्ता|"
    r"आशा(?:\s+कार्यकर्ता)?|एएनएम|चिकित्सक|डॉक्टर|नर्स|माता|पिता|पति|पत्नी|मरीज़?|रोगी|नाम)"
)
_EN_RELATION = r"(?:S/o|D/o|W/o|C/o|H/o|F/o|M/o|son\s+of|daughter\s+of|wife\s+of|husband\s+of|mother\s+of|father\s+of)"
_HI_RELATION = r"(?:धर्मपत्नी|पुत्र|पुत्री|पत्नी|पति|बेटा|बेटे|बेटी|भाई|बहन|बहू|सास|ससुर|मां|माँ|देवर|भाभी|दामाद)"
# kinship nouns written before a name ("her husband Ramesh", "beti Mary", "accompanied by Iqbal")
_EN_KIN = (
    r"(?:husband|wife|son|daughter|brother|sister|mother|father|grandson|granddaughter|grandmother|grandfather|"
    r"nephew|niece|uncle|aunt|neighbour|neighbor|attendant|guardian|spouse|son-in-law|daughter-in-law|"
    r"accompanied\s+by|brought\s+by|escorted\s+by|"
    r"beta|beti|bete|bhai|behen|bahan|bahu|patni|pati|maa|saas|sasur|devar|bhabhi|damaad|damad)"
)
_HINGLISH_KIN_RE = re.compile(r"\b(?:beta|beti|bete|bhai|behen|bahan|bahu|patni|pati|saas|sasur|devar|bhabhi|damaad|damad)[ \t]+")
# roles written right before a name without a colon: "ANM Rajwant Kaur", "CHO Vaishali", "स्टाफ़ नर्स लीलावती"
_ROLE_RE = re.compile(
    r"\b(?:ANM|ASHA|AWW|CHO|MPW|Staff\s+Nurse|Nurse|Vaccinator|Treatment\s+supporter|Aarogya\s+Mitra|Counsellor|"
    r"Lab\s+Tech(?:nician)?|Pharmacist)[ \t]+",
    re.IGNORECASE,
)
_HI_ROLE_RE = re.compile(r"(?:स्टाफ़?\s+नर्स|नर्स|आशा|एएनएम|आरोग्य\s+मित्र|सीएचओ|उपचार\s+सहयोगी)[ \t]+")
# "Rakesh Prajapati, 27 M", "Kamla, 65 yrs", "Sunita (34/F)" and "खुशबू साहू, आयु 63 वर्ष"
_AGE_SEX_RE = re.compile(
    r"([A-Z][a-z]+(?:[ \t][A-Z][a-z]+){0,2}),?[ \t]*\(?(?:aged[ \t]+)?\d{1,3}[ \t]*(?:y|yr|yrs|years|Y|Yrs)?[ \t]*"
    r"(?:[/,][ \t]*)?(?:M|F|male|female|yrs|years)\b"
)
_HI_AGE_RE = re.compile(r"(" + _DEV + r"+(?:[ \t]" + _DEV + r"+){0,2}),[ \t]*(?:आयु|उम्र)[ \t]*[\d०-९]{1,3}")

_EN_HONORIFIC_RE = re.compile(r"\b" + _EN_HONORIFIC + r"(?:[ \t]+|(?<=\.))", re.IGNORECASE)
_EN_LABEL_RE = re.compile(r"\b" + _EN_LABEL + LSEP + r"(?:" + _EN_HONORIFIC + r"\s*)?", re.IGNORECASE)
_EN_RELATION_RE = re.compile(r"\b" + _EN_RELATION + r"(?:" + LSEP + r"|\s+)(?:" + _EN_HONORIFIC + r"\s*)?", re.IGNORECASE)
_EN_KIN_RE = re.compile(r"\b" + _EN_KIN + r",?\s+(?:" + _EN_HONORIFIC + r"\s+)?", re.IGNORECASE)
_HI_HONORIFIC_RE = re.compile(r"(?<!" + _DEV + r")" + _HI_HONORIFIC + r"\s*")
_HI_LABEL_RE = re.compile(r"(?<!" + _DEV + r")" + _HI_LABEL + LSEP + r"(?:" + _HI_HONORIFIC + r"\s*)?")
_HI_RELATION_RE = re.compile(r"(?<!" + _DEV + r")" + _HI_RELATION + r"\s+(?:" + _HI_HONORIFIC + r"\s*)?")

_LAT_WORD_RE = re.compile(_LAT_WORD)
_DEV_TOKEN_RE = re.compile(_DEV_WORD)

# ---------------------------------------------------------------- addresses

_ADDR_LABEL_RE = re.compile(
    r"(?:\b(?:Residential\s+address|Permanent\s+address|Present\s+address|Household\s+address|Usual\s+residence|"
    r"Residence(?:\s+of\s+(?:mother|patient|deceased))?|Address|Adress|Addr|Add|R/o|Resident\s+of|Residing\s+at|"
    r"Pata)|(?<!" + _DEV + r")(?:स्थायी\s+पता|वर्तमान\s+पता|पता|निवासी|निवास)(?!" + _DEV + r"))"
    r"(?:[ \t]*(?:[:=|\-–.][ \t]*){1,3}|\s+(?=[A-Za-z0-9\u0900-\u097F]))",
    re.IGNORECASE,
)
_ADDR_START_RE = re.compile(
    r"(?:\b(?:Vill(?:age)?|Gram(?=\s+[A-Z])(?!\s+(?:stain|positive|negative))|VPO|At/Po|At\s*&\s*Po|Mohalla|H\.\s?No|House\s+No|Plot\s+No|Flat\s+No|Ward\s+No|"
    r"D\.\s?No|Door\s+No|Qtr\.?\s+No|Makan\s+No)\b\.?|"
    r"(?<![\d०-९.]\s)(?<![\d०-९.])(?<!" + _DEV + r")(?:ग्राम(?![/" + _DEV[1:-1] + r"])(?!\s+(?:दिन|प्रति|रोज़?))|गांव|गाँव|मोहल्ला|मकान\s+नं\.?|वार्ड\s+नं\.?))",
    re.IGNORECASE,
)
# words that end an address when they start the next field
_ADDR_FIELD_RE = re.compile(
    r"(?:Ph(?:one)?|Mob(?:ile|il)?|Contact|Tel|Cell|WhatsApp|E-?mail|Aadhaa?r|ABHA|PAN|DOB|D\.O\.B|UHID|MRN|Age|Sex|"
    r"Gender|Reg(?:n|d)?\.?\s*No|Chief|Diagnosis|Occupation|Religion|Caste|Mother|Father|Husband|Wife|"
    r"who|which|was|is|has|on|presented|came|admitted|with|ke|ki|ne|mein|se|"
    r"फोन|मोबाइल|संपर्क|आधार|आभा|उम्र|आयु|लिंग|निदान|व्यवसाय|को|ने|में|पर|दिनांक|से)(?![A-Za-z\u0900-\u097F])",
    re.IGNORECASE,
)
_ADDR_ABBR = {"vill", "distt", "dist", "dt", "teh", "tal", "tk", "tq", "po", "p.o", "ps", "p.s", "no", "h", "h.no", "st",
              "rd", "nr", "opp", "qtr", "mkt", "sec", "ext", "d", "d.no", "ward", "post", "u.p", "m.p", "raj", "ap", "up",
              "mp", "hp", "jk", "wb", "uk", "kt", "dist.", "blk", "colony", "sq", "apt", "flr", "a.p", "w.b", "j&k"}


def _address_end(text: str, start: int, max_len: int = 170) -> int:
    """End of an address that starts at ``start``: stop at a line break, tab, '|', ';', '।', two spaces,
    the next field label, or a sentence end (full stop after a non-abbreviation)."""
    n = min(len(text), start + max_len)
    i = start
    while i < n:
        ch = text[i]
        if ch in "\n\t|;।" or text.startswith("  ", i):
            break
        if ch in " ,/(" and _ADDR_FIELD_RE.match(text, i + 1) and i + 1 > start:
            break
        pin = re.match(r"[1-9]\d{2}\s?\d{3}(?!\d)", text[i:i + 8]) if (i == start or not text[i - 1].isdigit()) else None
        if pin:  # an address ends with its PIN code
            i += pin.end()
            break
        if ch == "." and i + 1 < len(text) and text[i + 1] in " \n":
            j = i
            while j > start and (text[j - 1].isalnum() or text[j - 1] in ".&"):
                j -= 1
            word = text[j:i].lower()
            if not (word in _ADDR_ABBR or len(word) <= 2 or word.isdigit() is False and len(word) <= 3 and word.isupper()):
                break
        i += 1
    end = i
    while end > start and text[end - 1] in " ,.-:(/":
        end -= 1
    # keep a closing bracket if its opener is inside the address: "(PIN 125222)"
    if end < len(text) and text[end:end + 1] == ")" and "(" in text[start:end]:
        end += 1
    return end


_HON_TOKEN_RE = re.compile(r"(?:[A-Za-z/]+|" + _DEV + r"+)[.०]?[ \t]*")


def _skip_honorifics(text: str, i: int) -> int:
    """Skip honorifics a cue regex did not consume ("Late Smt. Kamla", "नाम: श्री")."""
    while i < len(text):
        m = _HON_TOKEN_RE.match(text, i)
        if not m or m.group(0).strip().rstrip(".०").lower() not in _HONORIFIC_WORDS:
            break
        i = m.end()
    return i


def _take_name(text: str, pos: int, script: str) -> tuple[int, int] | None:
    """Consume 1-3 name tokens starting at ``pos``. Returns (start, end) or None."""
    tokens: list[tuple[int, int]] = []
    i = _skip_honorifics(text, pos)
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


_ANY_LAT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*\.?|(?:[A-Za-z]\.){1,3}")
_FACILITY_AFTER_RE = re.compile(
    r"[ \t]*(?:[A-Za-z.]+[ \t]+){0,3}?(?:Hospital|Institute|PGIMS|PGI|Medical\s+College|Memorial|Sansthan|Nursing\s+Home|"
    r"Trust|Clinic|Centre|Center|Foundation|Marg|Road|Nagar|Chowk|Park|Stadium|Colony|University|College|School)\b"
    r"|[ \t]*(?:[\u0900-\u0963\u0970-\u097F]+[ \t]+){0,3}?(?:संस्थान|अस्पताल|चिकित्सालय|मेडिकल\s+कॉलेज|स्मारक|मार्ग|नगर)",
    re.IGNORECASE,
)
_NAME_STOP_LOWER = {w.lower() for w in _EN_STOP} | {
    "m", "f", "y", "yrs", "yr", "mo", "mos", "months", "days", "wks", "weeks", "kg", "cm", "dob", "tel", "mob", "ph",
    "aged", "age/sex", "gender", "reg", "regd", "ipd", "opd", "bed", "no.", "ref", "by", "dt", "dated", "on", "at",
    "ke", "ki", "ka", "ko", "ne", "se", "me", "aur", "hai", "tha", "thi", "saal", "umar", "umr", "ghar", "visit",
    "kiya", "gaya", "gayi", "diya", "di", "ji", "sahab", "md", "mbbs", "ms", "dnb", "frcs", "dgo", "dch", "medicine",
    "unknown", "not", "known", "na", "n/a", "nil", "self", "same", "above", "below", "mother", "father", "husband",
    "wife", "son", "daughter", "patient", "relation", "relative", "baby", "male", "female", "child", "boy", "girl",
    "follow", "follow-up", "fever", "since", "loose", "stools", "htn", "dm", "id", "voter", "card", "and", "protection",
    "worker", "visited", "visit", "today", "came", "called", "said", "advised", "screened", "referred", "for", "the",
    "mcp", "pw", "rch", "hwc", "phc", "chc", "sc", "aww", "anm", "cho", "dr", "mr", "mrs", "ms", "smt", "shri",
    "ward", "in-charge", "incharge", "officer", "copy", "hospital", "district", "block",
    "निदान", "हस्ताक्षर", "दिनांक", "उपचार", "जाँच", "जांच", "रिपोर्ट", "गृह", "भ्रमण", "report", "home", "visit",
    "kal", "raat", "aaj", "subah", "shaam", "bhi", "nahi", "abhi", "phir",
}
_HONORIFIC_WORDS = {"mr", "mrs", "ms", "miss", "dr", "shri", "sri", "smt", "kumari", "km", "mst", "late", "master",
                    "sister", "sr", "sn", "श्री", "श्रीमती", "सुश्री", "कुमारी", "डॉ", "डॉ०", "स्व", "मो", "baby", "b/o"}


def _take_label_name(text: str, pos: int, max_tokens: int = 4) -> tuple[int, int] | None:
    """After a field label: take 1-4 name tokens of any case (ALL CAPS and lower case included)
    on the same line, stopping at punctuation, digits, double spaces or a stop word."""
    tokens: list[tuple[int, int]] = []
    i, n = pos, len(text)
    i = _skip_honorifics(text, i)
    while len(tokens) < max_tokens and i < n:
        m = _DEV_TOKEN_RE.match(text, i)
        if m and m.group(0) not in _HI_STOP:
            pass
        else:
            m = _ANY_LAT_TOKEN_RE.match(text, i)
            if not m:
                break
            raw = m.group(0).rstrip(".")
            word = raw.lower()
            initial = len(word) == 1 and m.group(0).endswith(".")
            if (word in _NAME_STOP_LOWER and not initial) or (len(word) <= 1 and not initial):
                break
            # "Asha Das visited": a lower-case word after a capitalised name ends it
            if tokens and raw.islower() and not text[tokens[0][0]:tokens[0][1]].islower():
                break
        tokens.append((m.start(), m.end()))
        k = m.end()
        tok = text[m.start():m.end()]
        if tok.endswith(".") and len(tok.rstrip(".")) > 1 and tok.rstrip(".").lower() not in {"mohd", "md", "kr", "mo"}:
            break  # "Khan. Later ..." — a full stop after a word ends the name
        if k < n and text[k] == " " and (k + 1 >= n or text[k + 1] not in " \t"):
            i = k + 1
        elif m.group(0).endswith(".") and k < n and text[k] != " ":
            i = k  # initials glued to the next token: "M.S.Mirza"
        else:
            break
    if not tokens:
        return None
    a, b = tokens[0][0], tokens[-1][1]
    while b > a and text[b - 1] in ".-'":
        b -= 1
    return (a, b) if b - a >= 2 else None


def _facility_before(text: str, start: int) -> bool:
    """'Sri Krishna Hospital': the mention sits inside a facility name that started a word or two earlier."""
    return bool(re.search(r"(?:Sri|Shri|Pt\.|Dr\.?|St\.?)\s*$", text[max(0, start - 6):start])) and _is_facility(
        text, start, start + len(re.match(r"\S*", text[start:]).group(0)))


def _is_facility(text: str, start: int, end: int) -> bool:
    return bool(_FACILITY_AFTER_RE.match(text, end)) or bool(
        re.search(r"(?:Hospital|Institute|Sansthan|संस्थान|अस्पताल)\W*$", text[start:end], re.IGNORECASE)
    )


def _name_spans(text: str) -> list[Span]:
    out: list[Span] = []

    def add(got: tuple[int, int] | None, score: float) -> None:
        if got and not _is_facility(text, got[0], got[1]):
            out.append(Span(got[0], got[1], "PERSON", text[got[0]:got[1]], score))

    for m in _EN_HONORIFIC_RE.finditer(text):
        lower_hon = m.group(0)[:1].islower()
        got = _take_label_name(text, m.end(), 3) if lower_hon else (
            _take_name(text, m.end(), "lat") or _take_upper_or_initials(text, m.end()) or _take_name(text, m.end(), "dev"))
        add(got, 0.85)
    for rx, score in ((_EN_LABEL_RE, 0.9), (_HI_LABEL_RE, 0.9)):
        for m in rx.finditer(text):
            add(_take_label_name(text, m.end()), score)
    for rx, score in ((_EN_RELATION_RE, 0.85), (_EN_KIN_RE, 0.7)):
        for m in rx.finditer(text):
            got = _take_name(text, m.end(), "lat") or _take_upper_or_initials(text, m.end()) or _take_name(text, m.end(), "dev")
            if got is None and rx is _EN_RELATION_RE:
                got = _take_label_name(text, m.end(), 3)
            add(got, score)
    for m in _ROLE_RE.finditer(text):
        lower = m.group(0)[:1].islower()
        add(_take_label_name(text, m.end(), 3) if lower else (_take_name(text, m.end(), "lat") or _take_upper_or_initials(text, m.end())), 0.75)
    for m in _HINGLISH_KIN_RE.finditer(text):
        add(_take_label_name(text, m.end(), 3), 0.7)
    for rx in (_AGE_SEX_RE, _HI_AGE_RE):
        for m in rx.finditer(text):
            a, b = m.start(1), m.end(1)
            words = text[a:b].split()
            while words and (words[0].lower() in _NAME_STOP_LOWER or words[0] in _HI_STOP):
                a += len(words[0]) + 1
                words = words[1:]
            if words:
                add((a, b), 0.75)
    for m in _HI_ROLE_RE.finditer(text):
        add(_take_name(text, m.end(), "dev") or _take_name(text, m.end(), "lat"), 0.75)
    for rx, score in ((_HI_HONORIFIC_RE, 0.85), (_HI_RELATION_RE, 0.8)):
        for m in rx.finditer(text):
            add(_take_name(text, m.end(), "dev") or _take_name(text, m.end(), "lat"), score)
    return out


_UPPER_OR_INITIAL_RE = re.compile(r"(?:[A-Z]\.\s?){1,3}[A-Z][a-z]+|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,2}|(?:[A-Z]\s){1,2}[A-Z][a-z]+")


def _take_upper_or_initials(text: str, pos: int) -> tuple[int, int] | None:
    """'DR. RAGHAVENDRA SHETTY', 'Dr S. Karthik', 'Dr. M.S. Mirza', 'Dr. H Baruah'."""
    m = _UPPER_OR_INITIAL_RE.match(text, pos)
    if not m:
        return None
    words = m.group(0).split()
    keep = 0
    for w in words:
        if w.lower().strip(".") in _NAME_STOP_LOWER and not re.fullmatch(r"[A-Z]\.?", w):
            break
        keep += 1
    if keep == 0:
        return None
    if keep < len(words):
        cut = m.start() + len(" ".join(words[:keep])) if " " in m.group(0) else m.end()
        return m.start(), cut
    # extend initials with following capitalised surname tokens
    end = m.end()
    nxt = _take_name(text, end + 1, "lat") if end < len(text) and text[end] == " " else None
    if nxt and nxt[0] == end + 1:
        end = nxt[1]
    return m.start(), end


def _address_spans(text: str) -> list[Span]:
    out: list[Span] = []
    for rx, score in ((_ADDR_LABEL_RE, 0.85), (_ADDR_START_RE, 0.75)):
        for m in rx.finditer(text):
            start = m.end() if rx is _ADDR_LABEL_RE else m.start()
            end = _address_end(text, start)
            if end - start >= 4 and re.search(r"[A-Za-z\u0900-\u097F]{3}", text[start:end]):
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
        if len(p) >= 3 and p.lower() not in _NAME_STOP_LOWER and p not in _HI_STOP:
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
                rx = re.compile(r"(?<![A-Za-z])" + re.escape(part) + r"(?![A-Za-z])", re.IGNORECASE)
            for m in rx.finditer(text):
                if _is_facility(text, m.start(), m.end()) or _facility_before(text, m.start()):
                    continue
                if m.group(0).isupper() and m.group(0) != part and m.group(0) in {"ASHA", "ANM", "CHO", "AWW", "RAM"}:
                    continue
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
