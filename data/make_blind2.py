#!/usr/bin/env python3
"""make_blind2.py - second blind synthetic PII evaluation set for Indian clinical notes.

Stdlib only.  Deterministic: fixed seed 7071947.
Writes blind2.jsonl next to this file: 150 notes (85 en, 40 hi, 25 mixed).
Schema per line: {"id", "lang", "doc_type", "text", "entities": [{"start","end","type","text"}]}

Labelling follows data/HELDOUT.md.  All people, numbers and records are invented.
The templates here were written from scratch; the author saw neither the detector
code nor the earlier dev/heldout sets.
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

SEED = 7071947
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "blind2.jsonl")
TODAY_Y = 2026

# --------------------------------------------------------------------------
# Verhoeff
# --------------------------------------------------------------------------
_VD = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VP = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 8, 7, 0, 6], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_VINV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def verhoeff_digit(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VD[c][_VP[(i + 1) % 8][int(ch)]]
    return _VINV[c]


assert verhoeff_digit("236") == 3 and verhoeff_valid("2363") and not verhoeff_valid("2364")

# --------------------------------------------------------------------------
# Span markup: E(type, text) embeds a marked span; finalize() strips the
# markers and records offsets.  Nesting is impossible by construction.
# --------------------------------------------------------------------------
TYPES = ("PERSON", "ADDRESS", "PHONE", "MRN", "PINCODE", "DOB", "AADHAAR",
         "ABHA_NUMBER", "ABHA_ADDRESS", "PAN", "VOTER_ID", "EMAIL")
_MK = re.compile("\x01([A-Z_]+)\x02([^\x01\x02\x03]*)\x03")


def E(t, s):
    assert t in TYPES, t
    s = str(s)
    assert s and s == s.strip(), (t, repr(s))
    assert not any(c in s for c in "\x01\x02\x03"), (t, s)
    return "\x01%s\x02%s\x03" % (t, s)


def PER(s): return E("PERSON", s)
def ADR(s): return E("ADDRESS", s)
def PH(s): return E("PHONE", s)
def MR(s): return E("MRN", s)
def PIN(s): return E("PINCODE", s)
def DB(s): return E("DOB", s)
def AAD(s): return E("AADHAAR", s)
def ABN(s): return E("ABHA_NUMBER", s)
def ABA(s): return E("ABHA_ADDRESS", s)
def PANE(s): return E("PAN", s)
def VOT(s): return E("VOTER_ID", s)
def EM(s): return E("EMAIL", s)


def finalize(raw):
    out, ents, pos, n = [], [], 0, 0
    for m in _MK.finditer(raw):
        chunk = raw[pos:m.start()]
        out.append(chunk)
        n += len(chunk)
        t, s = m.group(1), m.group(2)
        ents.append({"start": n, "end": n + len(s), "type": t, "text": s})
        out.append(s)
        n += len(s)
        pos = m.end()
    out.append(raw[pos:])
    text = "".join(out)
    assert not any(c in text for c in "\x01\x02\x03"), "stray marker"
    return text, ents


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
DEVTR = str.maketrans("0123456789", "०१२३४५६७८९")
ASCTR = str.maketrans("०१२३४५६७८९", "0123456789")


def dev(s): return s.translate(DEVTR)


def digits(r, n): return "".join(r.choice("0123456789") for _ in range(n))


def nz_digits(r, n): return r.choice("123456789") + digits(r, n - 1)


def letters(r, n): return "".join(r.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(n))


def case(s, mode):
    if mode == "upper": return s.upper()
    if mode == "lower": return s.lower()
    if mode == "title": return s.title()
    return s


class Style:
    SEPS = [": ", ":- ", " - ", " = ", "\t", " | ", ":", " : ", " :- ", ": ", "- ", " :  "]

    def __init__(self, r, caps=True, lowers=True):
        self.r = r
        modes = ["as", "as", "as", "as", "title"]
        if caps: modes += ["upper", "upper"]
        if lowers: modes += ["lower"]
        self.case = r.choice(modes)
        self.sep = r.choice(self.SEPS)
        self.namecase = self.case if self.case in ("upper", "lower") and r.random() < 0.7 else "as"

    def L(self, *v):
        return case(self.r.choice(v), self.case)

    def S(self):
        return self.sep if self.r.random() < 0.8 else self.r.choice([": ", " - ", ":-", "\t", " "])

    def F(self, *v):
        return self.L(*v) + self.S()

    def N(self, s):
        return case(s, self.namecase)


def pick(r, seq): return r.choice(seq)


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त",
             "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


def ev_date(r, m0=1, m1=10):
    return (2026, r.randint(m0, m1), r.randint(1, 28))


def dob_for(r, age):
    return (TODAY_Y - age - (1 if r.random() < 0.5 else 0), r.randint(1, 12), r.randint(1, 28))


def fdate(r, ymd, style=None, lang="en"):
    y, m, d = ymd
    if lang == "hi":
        styles = ["dmy/", "dmy-", "dmy.", "dMy_hi", "dmy/"]
    else:
        styles = ["dmy/", "dmy-", "dmy.", "dmy2", "dMonY", "MonthdY", "d-Mon-Y", "iso", "dthMonthY", "dmy/"]
    st = style or r.choice(styles)
    if st == "dmy/": return "%02d/%02d/%d" % (d, m, y)
    if st == "dmy-": return "%02d-%02d-%d" % (d, m, y)
    if st == "dmy.": return "%02d.%02d.%d" % (d, m, y)
    if st == "dmy2": return "%d/%d/%02d" % (d, m, y % 100)
    if st == "dMonY": return "%d %s %d" % (d, MONTHS[m - 1][:3], y)
    if st == "MonthdY": return "%s %d, %d" % (MONTHS[m - 1], d, y)
    if st == "d-Mon-Y": return "%02d-%s-%d" % (d, MONTHS[m - 1][:3], y)
    if st == "iso": return "%d-%02d-%02d" % (y, m, d)
    if st == "dthMonthY":
        suf = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        return "%d%s %s %d" % (d, suf, MONTHS[m - 1], y)
    if st == "dMy_hi": return "%d %s %d" % (d, HI_MONTHS[m - 1], y)
    raise ValueError(st)


def hdate(r, ymd, devn=False):
    s = fdate(r, ymd, lang="hi")
    return dev(s) if devn else s


def tstamp(r):
    h = r.randint(0, 23)
    return "%02d:%02d" % (h, r.randint(0, 59))


def ampm(r):
    h = r.randint(1, 12)
    return "%d:%02d %s" % (h, r.randint(0, 59), r.choice(["am", "pm", "AM", "PM"]))


# --------------------------------------------------------------------------
# Identifier generators
# --------------------------------------------------------------------------
def gen_aadhaar(r):
    while True:
        body = r.choice("23456789") + digits(r, 10)
        n = body + str(verhoeff_digit(body))
        if not re.search(r"(\d)\1{4}", n):
            return n


def fmt_aadhaar(r, n=None, mask=True, devn=False):
    n = n or gen_aadhaar(r)
    kinds = ["sp", "sp", "sp", "plain", "dash"] + (["m1", "m2", "m3"] if mask else [])
    k = r.choice(kinds)
    s = {"sp": "%s %s %s" % (n[:4], n[4:8], n[8:]), "plain": n,
         "dash": "%s-%s-%s" % (n[:4], n[4:8], n[8:]),
         "m1": "XXXX XXXX %s" % n[8:], "m2": "xxxx-xxxx-%s" % n[8:],
         "m3": "XXXXXXXX%s" % n[8:]}[k]
    return dev(s) if devn else s


def gen_fail12(r, lead="123456789"):
    """12 digits that FAIL Verhoeff (cannot be an Aadhaar)."""
    body = r.choice(lead) + digits(r, 10)
    c = verhoeff_digit(body)
    d = r.choice([x for x in range(10) if x != c])
    s = body + str(d)
    assert not verhoeff_valid(s)
    return s


def gen_mobile(r): return r.choice("6789") + digits(r, 9)


def fmt_mobile(r, devn=False):
    while True:
        n = gen_mobile(r)
        k = r.randrange(9)
        s = [n, "+91 %s %s" % (n[:5], n[5:]), "+91-" + n, "0" + n, "%s-%s" % (n[:5], n[5:]),
             "+91" + n, "%s %s %s" % (n[:3], n[3:6], n[6:]), "(+91) %s %s" % (n[:5], n[5:]),
             "91-%s-%s" % (n[:5], n[5:])][k]
        if k == 5 and verhoeff_valid("91" + n):
            continue
        return dev(s) if devn else s


def fmt_landline(r, std, devn=False):
    ln = 10 - len(std) + 1  # local digits so that std(with 0)+local = 11 chars
    loc = r.choice("2345") + digits(r, ln - 1)
    k = r.randrange(4)
    if k == 0: s = "%s-%s" % (std, loc)
    elif k == 1: s = "(%s) %s" % (std, loc)
    elif k == 2: s = "%s %s %s" % (std, loc[:4], loc[4:])
    else: s = "%s %s" % (std, loc)
    return dev(s) if devn else s


def gen_abha(r):
    n = nz_digits(r, 14)
    k = r.randrange(4)
    if k in (0, 1): return "%s-%s-%s-%s" % (n[:2], n[2:6], n[6:10], n[10:])
    if k == 2: return "%s %s %s %s" % (n[:2], n[2:6], n[6:10], n[10:])
    return n


def ascii_slug(s):
    s = s.lower().replace("'", "").replace(".", "")
    return re.sub(r"[^a-z]", "", s)


def gen_abha_addr(r, p):
    f = ascii_slug(p.first_ascii)
    l = ascii_slug(p.last_ascii or "")
    base = r.choice([f + (("." + l) if l else ""), f + l, f + digits(r, 2), f + "_" + (l or "h") + digits(r, 2),
                     (l or f) + f[:1] + nz_digits(r, 4)])
    return base + r.choice(["@abdm", "@abdm", "@sbx"])


def gen_email(r, p, doctor=False):
    f = ascii_slug(p.first_ascii)
    l = ascii_slug(p.last_ascii or "") or "x"
    dom = r.choice(["gmail.com", "gmail.com", "gmail.com", "yahoo.co.in", "rediffmail.com", "outlook.com", "hotmail.com"])
    if doctor:
        loc = r.choice(["dr." + f + "." + l, "dr" + l + digits(r, 2), f + l + ".md", l + "." + f + "ortho"[:0] + digits(r, 2)])
    else:
        loc = r.choice([f + "." + l, f + l + digits(r, 2), f + "_" + digits(r, 4), l + f[:3] + digits(r, 2), f + "." + l + "19" + digits(r, 2)])
    return loc + "@" + dom


def gen_pan(r, p):
    ini = (ascii_slug(p.last_ascii or p.first_ascii) or "k")[0].upper()
    return letters(r, 3) + "P" + ini + digits(r, 4) + letters(r, 1)


def gen_epic(r): return letters(r, 3) + nz_digits(r, 7)


def gen_mrn(r, kind):
    yy = "26"
    if kind == "uhid":
        return r.choice(["UH" + yy + digits(r, 6), "%s/%s" % (digits(r, 2), nz_digits(r, 7)),
                         "GMC/2026/" + nz_digits(r, 5), "U-" + nz_digits(r, 8), "%s%s" % (letters(r, 2), nz_digits(r, 7))])
    if kind == "ipd":
        return r.choice(["IP/2026/" + nz_digits(r, 5), "%s/26" % nz_digits(r, 5), "IPD-" + nz_digits(r, 6),
                         "26-%s" % nz_digits(r, 5), "IP" + nz_digits(r, 7)])
    if kind == "opd":
        return r.choice(["OPD/%s" % nz_digits(r, 6), "%s-%s" % (nz_digits(r, 4), digits(r, 3)), "O26%s" % nz_digits(r, 6)])
    if kind in ("cr12", "rch12"):
        return gen_fail12(r)
    if kind == "nikshay":
        return nz_digits(r, 8)
    if kind == "lab":
        return r.choice(["LB" + nz_digits(r, 8), "S-26-" + nz_digits(r, 5), "%s/%s" % (letters(r, 2), nz_digits(r, 6))])
    if kind == "dialysis":
        return r.choice(["HD/%s" % nz_digits(r, 3), "DU-%s" % nz_digits(r, 4), "NEP/HD/%s" % nz_digits(r, 4), "PMNDP-%s" % nz_digits(r, 5)])
    if kind == "donor":
        return r.choice(["BC/26/%s" % nz_digits(r, 5), "D%s" % nz_digits(r, 7), "DNR-%s-%s" % (yy, nz_digits(r, 5))])
    if kind == "mlc":
        return r.choice(["%s/26" % nz_digits(r, 4), "MLC/%s/2026" % nz_digits(r, 3), "%s/2026" % nz_digits(r, 3)])
    if kind == "call":
        return r.choice(["104-%s-%s" % (letters(r, 2), nz_digits(r, 8)), "HL%s%s" % (yy, nz_digits(r, 7)), "C-%s-%s" % (nz_digits(r, 4), nz_digits(r, 5))])
    if kind == "pmjay":
        return r.choice("PNHT") + "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    if kind == "mh":
        return r.choice(["DMHP/%s" % nz_digits(r, 4), "PSY-%s" % nz_digits(r, 5), "MH/26/%s" % nz_digits(r, 4)])
    if kind == "cghs":
        return nz_digits(r, 7)
    if kind == "rbsk":
        return "RBSK/%s/%s" % (letters(r, 3), nz_digits(r, 5))
    raise ValueError(kind)


def fmt_pin(r, pin, devn=False):
    s = r.choice([pin, pin, pin, "%s %s" % (pin[:3], pin[3:])])
    return dev(s) if devn else s


# hard-negative number makers (unlabelled)
def bill_no(r): return r.choice(["BL/26/%s" % nz_digits(r, 5), "INV-2611-%s" % digits(r, 4), "%s/%s" % (nz_digits(r, 4), nz_digits(r, 4)), "CB%s" % nz_digits(r, 6)])
def batch_no(r): return r.choice(["%s%s" % (letters(r, 2), nz_digits(r, 5)), "B%s" % nz_digits(r, 6), "%s-%s" % (letters(r, 3), nz_digits(r, 4))])
def exp_date(r): return "%02d/%s" % (r.randint(1, 12), r.choice(["27", "28", "2027", "2028"]))
def bag_no(r): return r.choice(["%s-%s" % (nz_digits(r, 4), nz_digits(r, 5)), "BB%s" % nz_digits(r, 7), "%s/%s" % (nz_digits(r, 2), nz_digits(r, 5))])


# --------------------------------------------------------------------------
# People
# --------------------------------------------------------------------------
GROUPS = {
    "up": dict(reg=["up"], m=["Ramesh", "Suresh Chandra", "Anil", "Rakesh", "Shivam", "Rajendra Prasad", "Brijesh", "Dinesh", "Harish"],
               f=["Sunita", "Geeta", "Pushpa", "Ramvati", "Kusum", "Neelam", "Phoolmati", "Saroj", "Anjali"],
               s=["Yadav", "Verma", "Maurya", "Kushwaha", "Tiwari", "Mishra", "Shukla", "Pandey", "Rajbhar", "Nishad", "Pal", "Srivastava"]),
    "br": dict(reg=["br"], m=["Rambabu", "Shambhu", "Vikash", "Lalan", "Manoj", "Pappu", "Dhananjay"],
               f=["Rinku", "Sunaina", "Babita", "Chanda", "Kiran", "Sanju", "Lalita"],
               s=["Paswan", "Yadav", "Mandal", "Sah", "Thakur", "Mahto", "Ram", "Jha"]),
    "rj": dict(reg=["rj"], m=["Bhanwar Lal", "Mangi Lal", "Om Prakash", "Rameshwar", "Kailash", "Hanuman Sahay"],
               f=["Bhanwari", "Kamla", "Santosh", "Lali", "Guddi", "Mohini"],
               s=["Meena", "Gurjar", "Bairwa", "Saini", "Choudhary", "Rathore", "Kumawat"]),
    "hr": dict(reg=["hr"], m=["Rajbir", "Satbir", "Dharambir", "Naveen", "Sandeep", "Jagdish"],
               f=["Santosh", "Kavita", "Sunita", "Monika", "Bimla", "Rekha"],
               s=["Malik", "Dahiya", "Hooda", "Sangwan", "Ahlawat", "Rathee", "Kadian"]),
    "pb": dict(reg=["pb"], kind="sikh", m=["Gurpreet", "Harjinder", "Kulwinder", "Jaswant", "Manpreet", "Baljit", "Sukhdev", "Iqbal"],
               f=["Harpreet", "Jasvir", "Manjit", "Sukhwinder", "Amandeep", "Rajwinder", "Navneet", "Paramjit"],
               s=["Sandhu", "Gill", "Dhillon", "Brar", "Sidhu", "Grewal", "Bajwa", "", "", ""]),
    "pbh": dict(reg=["dl", "pb"], m=["Rajesh", "Vinod", "Ashwani", "Sanjeev", "Deepak"], f=["Neeru", "Anju", "Ritu", "Pooja", "Kamlesh"],
                s=["Arora", "Khanna", "Malhotra", "Kapoor", "Bhatia", "Sachdeva", "Chawla"]),
    "mus": dict(reg=["up", "br", "dl"], m=["Mohd. Arif", "Md. Shakeel", "Irfan", "Salim", "Mohd. Zubair", "Abdul Rashid", "Naseem", "Mohd. Imran", "Shahid"],
                f=["Shabana", "Rukhsar", "Nazia", "Farzana", "Shaheen", "Gulnaz", "Nasreen", "Afsana"],
                s=["Ansari", "Qureshi", "Siddiqui", "Khan", "Saifi", "Malik", "Alvi", "Mansuri"]),
    "bn": dict(reg=["wb"], m=["Sourav", "Debashis", "Partha", "Arindam", "Tapas", "Subrata", "Anirban"],
               f=["Moumita", "Sharmila", "Rituparna", "Sanchita", "Paramita", "Jhuma", "Tanushree"],
               s=["Chakraborty", "Ghosh", "Mukherjee", "Mondal", "Sarkar", "Bhattacharya", "Das", "Halder", "Bag"]),
    "bnm": dict(reg=["wb", "as"], m=["Md. Rafiqul", "Sahidul", "Jahangir", "Nurul", "Ainul"], f=["Rehana", "Sabina", "Mumtaz", "Asma", "Rubina"],
                s=["Islam", "Sheikh", "Mollah", "Laskar", "Hoque"]),
    "od": dict(reg=["od"], m=["Bijay", "Prasanta", "Sushant", "Debasis", "Hrushikesh", "Sudam"],
               f=["Sasmita", "Pramila", "Sujata", "Lipsa", "Bandana", "Kuni"],
               s=["Panda", "Mohanty", "Behera", "Sahoo", "Nayak", "Pradhan", "Swain", "Jena"]),
    "as": dict(reg=["as"], m=["Bhaskar", "Pranjal", "Dhrubajyoti", "Hemanta", "Jitu"], f=["Rimjhim", "Pallavi", "Jonali", "Mousumi", "Bornali"],
               s=["Gogoi", "Bora", "Saikia", "Kalita", "Baruah", "Hazarika", "Deka"]),
    "mz": dict(reg=["mz"], m=["Lalthanpuia", "Lalnunmawia", "Vanlalruata", "Zothansanga", "Lalbiakzuala"],
               f=["Lalremruati", "Zothanpuii", "Vanlalhriatpuii", "Lalrinpuii", "Malsawmtluangi"],
               s=["Ralte", "Sailo", "Chhangte", "Hnamte", "Khiangte", "", ""]),
    "mn": dict(reg=["mn"], kind="meitei", m=["Ibomcha", "Tomba", "Rajen", "Premchand", "Ibungo"], f=["Bembem", "Sanatombi", "Memcha", "Linthoingambi", "Thoibi"],
               s=["Thangjam", "Ningthoujam", "Laishram", "Khumanthem", "Moirangthem", "Konsam"]),
    "nl": dict(reg=["nl"], m=["Vikhozo", "Kekhrie", "Temsu", "Neilazo", "Imkong"], f=["Temjenmenla", "Akhrielie", "Vitono", "Imtisenla", "Atoli"],
               s=["Angami", "Ao", "Sema", "Lotha", "Sekhose", "Rengma", "Jamir"]),
    "kh": dict(reg=["ml"], m=["Banteilang", "Wanphrang", "Kyrshan", "Iohborlang", "Shngain"], f=["Iarisha", "Daiahun", "Ibashisha", "Wandashisha", "Badaplin"],
               s=["Lyngdoh", "Kharkongor", "Marbaniang", "Syiem", "Nongrum", "Kharbuli", "Dkhar"]),
    "tr": dict(reg=["tr"], m=["Bikash", "Sukanta", "Ratan", "Jayanta"], f=["Sumitra", "Purnima", "Champa", "Rina"],
               s=["Debbarma", "Jamatia", "Reang", "Noatia"]),
    "sk": dict(reg=["sk"], m=["Pema", "Karma", "Dawa", "Sonam", "Ugen"], f=["Dechen", "Pema", "Diki", "Yangchen", "Choden"],
               s=["Bhutia", "Lepcha", "Sherpa", "Tamang", "Rai"]),
    "la": dict(reg=["la"], m=["Tsewang", "Stanzin", "Rigzin", "Jigmet", "Phuntsog"], f=["Dolkar", "Tsering", "Padma", "Chorol", "Skarma"],
               s=["", "Namgyal", "Dorjay", "Tundup", ""]),
    "jk": dict(reg=["jk"], m=["Mushtaq Ahmad", "Bilal Ahmad", "Showkat Ali", "Ghulam Nabi", "Tariq Hussain", "Javid Iqbal", "Ab. Rashid"],
               f=["Rifat Ara", "Shazia", "Nighat", "Shaista", "Rubeena", "Saba", "Mehjabeen", "Haleema"],
               s=["Bhat", "Wani", "Mir", "Dar", "Lone", "Rather", "Shah", "Ganai", "Sheikh"]),
    "kp": dict(reg=["jk", "dl"], m=["Ashok", "Rakesh", "Sanjay", "Moti Lal"], f=["Rajni", "Anita", "Kiran", "Sheela"],
               s=["Koul", "Raina", "Pandita", "Bhan", "Kachroo", "Dhar", "Tickoo"]),
    "gc": dict(reg=["goa"], m=["Joaquim", "Agnelo", "Savio", "Francis", "Caetano", "Rosario", "Peter"],
               f=["Lavina", "Fatima", "Sweeny", "Annie", "Maria", "Joyce", "Conceicao"],
               s=["D'Souza", "Fernandes", "Rodrigues", "Pereira", "Colaco", "D'Costa", "Menezes", "Gomes", "Furtado"]),
    "gk": dict(reg=["goa"], m=["Prakash", "Dattaram", "Sagar", "Ulhas", "Shripad"], f=["Sneha", "Pooja", "Deepali", "Sushma", "Vibha"],
               s=["Naik", "Gaonkar", "Desai", "Kamat", "Parab", "Shirodkar", "Velip"]),
    "klc": dict(reg=["kl"], m=["Jijo", "Thomas", "Sebastian", "Biju", "Shaji", "Mathew", "Tomy"], f=["Anitha", "Mercy", "Jincy", "Sherin", "Leena", "Reshma", "Ancy"],
                s=["Kurian", "Varghese", "Thomas", "Joseph", "Mathew", "Chacko", "George", "Abraham"]),
    "klh": dict(reg=["kl"], m=["Sreejith", "Anil Kumar", "Rajeev", "Unnikrishnan", "Vinod"], f=["Sreelekha", "Divya", "Ambili", "Sindhu", "Lakshmi"],
                s=["Nair", "Menon", "Pillai", "Kurup", "Nambiar", "Warrier", "Panicker"]),
    "klm": dict(reg=["kl"], m=["Shihab", "Muhammed Ashraf", "Faisal", "Abdul Nazar", "Rasheed"], f=["Fathima", "Shameena", "Raseena", "Jaseela", "Sajna"],
                s=["Rawther", "Kutty", "Koya", "Puthiyakath", "Kunju"]),
    "tn": dict(reg=["tn"], kind="tamil", m=["Karthikeyan", "Senthil Kumar", "Murugan", "Saravanan", "Arul", "Velmurugan", "Prabhu"],
               f=["Selvi", "Muthulakshmi", "Kavitha", "Revathi", "Tamilselvi", "Priya", "Malathi"],
               s=["Ramasamy", "Subramanian", "Palanisamy", "Natarajan", "Krishnan", "Pandian"]),
    "tnc": dict(reg=["tn"], m=["John Peter", "Arokiasamy", "Antony", "Xavier"], f=["Mary Stella", "Jesintha", "Arockia Mary", "Sahaya Rani"],
                s=["Arokiaraj", "Savarimuthu", "Lourdusamy", "Rayappan"]),
    "te": dict(reg=["ap", "tg"], kind="telugu", m=["Venkata Ramana", "Srinivas", "Naga Raju", "Ravi Teja", "Suresh Babu", "Koteswara Rao"],
               f=["Lakshmi Prasanna", "Padmavathi", "Sravani", "Swathi", "Anusha", "Nagamani"],
               s=["Reddy", "Naidu", "Kandula", "Gunda", "Bandaru", "Chowdary", "Peddireddy"]),
    "ka": dict(reg=["ka"], m=["Manjunath", "Basavaraj", "Prakash", "Shivakumar", "Raghavendra", "Nagaraj"],
               f=["Shwetha", "Kavya", "Lakshmamma", "Shilpa", "Roopa", "Sumangala"],
               s=["Gowda", "Hegde", "Shetty", "Patil", "Naik", "Kulkarni", "Hiremath"]),
    "mh": dict(reg=["mh"], m=["Sachin", "Ganesh", "Vitthal", "Sandip", "Mahesh", "Dnyaneshwar"], f=["Ashwini", "Sangita", "Manisha", "Pallavi", "Varsha", "Shubhangi"],
               s=["Patil", "Jadhav", "Kulkarni", "Shinde", "More", "Pawar", "Deshmukh", "Gaikwad"]),
    "parsi": dict(reg=["bom"], m=["Rustom", "Kersi", "Cyrus", "Jehangir", "Farokh", "Behram", "Darius"],
                  f=["Freny", "Perizad", "Shirin", "Roshan", "Dilnavaz", "Khorshed", "Zarine"],
                  s=["Daruwalla", "Mistry", "Irani", "Bharucha", "Contractor", "Engineer", "Batliwala", "Sethna", "Dastur", "Pithawalla"]),
    "gj": dict(reg=["gj"], m=["Jignesh", "Hitesh", "Bhavesh", "Ketan", "Nilesh"], f=["Hetal", "Nayana", "Bhavna", "Komal", "Daxa"],
               s=["Patel", "Shah", "Desai", "Parmar", "Solanki", "Chauhan", "Vaghela"]),
    "jh": dict(reg=["jh"], m=["Somra", "Etwa", "Budhram", "Sanjay", "Anand"], f=["Salomi", "Sushila", "Sukhmani", "Phulmani", "Jasinta", "Asrita"],
               s=["Oraon", "Tirkey", "Kujur", "Hembrom", "Soren", "Minz", "Ekka", "Toppo", "Lakra", "Munda"]),
    "gd": dict(reg=["cg", "mp"], m=["Ramlal", "Sukhram", "Budhu", "Manohar"], f=["Sukhiya", "Phoolbai", "Somari", "Durga"],
               s=["Maravi", "Uikey", "Netam", "Dhurve", "Markam"]),
    "uk": dict(reg=["uk"], m=["Bhupendra", "Diwan Singh", "Kundan", "Govind"], f=["Hema", "Parvati", "Kamla", "Basanti"],
               s=["Bisht", "Negi", "Rawat", "Bhandari", "Joshi", "Pant"]),
}
EN_GROUP_W = {k: 3 for k in GROUPS}
for k in ("up", "br", "mus", "pb", "bn", "tn", "te", "ka", "mh", "klc", "klh", "jk"):
    EN_GROUP_W[k] = 5
NORTH_GROUPS = ["up", "up", "br", "br", "rj", "hr", "pb", "pbh", "mus", "mus", "uk", "jh", "gd"]
MUMBAI_GROUPS = ["mh", "parsi", "gj", "mus"]

HI_GROUPS = {
    "up": dict(reg=["up"], m=["रमेश", "सुरेश", "राजेंद्र प्रसाद", "शिवम", "अनिल", "रामसागर", "हरिओम", "बृजेश"],
               f=["सुनीता", "रामवती", "गीता", "कुसुम", "पुष्पा", "नीलम", "फूलमती", "सरोज"],
               s=["यादव", "वर्मा", "मौर्य", "कुशवाहा", "तिवारी", "मिश्रा", "शुक्ला", "पाण्डेय", "राजभर", "निषाद", "पाल"]),
    "br": dict(reg=["br"], m=["रामबाबू", "शम्भू", "विकास", "ललन", "मनोज", "धनंजय"], f=["रिंकू", "सुनैना", "बबीता", "चंदा", "किरण", "ललिता"],
               s=["पासवान", "यादव", "मंडल", "साह", "ठाकुर", "राम", "महतो"]),
    "rj": dict(reg=["rj"], m=["भंवर लाल", "मांगी लाल", "ओम प्रकाश", "रामेश्वर", "कैलाश"], f=["भंवरी", "कमला", "संतोष", "लाली", "गुड्डी"],
               s=["मीणा", "गुर्जर", "बैरवा", "सैनी", "चौधरी", "कुमावत"]),
    "hr": dict(reg=["hr"], m=["राजबीर", "सतबीर", "धर्मबीर", "नवीन", "संदीप"], f=["संतोष", "कविता", "सुनीता", "मोनिका", "बिमला"],
               s=["मलिक", "दहिया", "हुड्डा", "सांगवान", "अहलावत"]),
    "pb": dict(reg=["pb"], kind="sikh", m=["गुरप्रीत", "हरजिंदर", "कुलविंदर", "जसवंत", "सुखदेव"], f=["हरप्रीत", "जसवीर", "मनजीत", "सुखविंदर", "अमनदीप"],
               s=["संधू", "गिल", "ढिल्लों", "बराड़", "", ""]),
    "mus": dict(reg=["up", "br"], m=["मो. इरफ़ान", "मोहम्मद शाकिर", "अब्दुल रशीद", "सलीम", "नसीम", "मो. ज़ुबैर"],
                f=["शबाना", "रुख़सार", "नाज़िया", "फ़रज़ाना", "शाहीन", "नसरीन"], s=["अंसारी", "कुरैशी", "सिद्दीकी", "ख़ान", "सैफ़ी", "मलिक"]),
    "gd": dict(reg=["mp", "cg"], m=["रामलाल", "सुखराम", "बुधराम", "मनोहर"], f=["सुखिया", "फूलबाई", "सोमारी", "दुर्गा"],
               s=["मरावी", "उइके", "नेताम", "धुर्वे", "साहू", "बघेल"]),
    "jh": dict(reg=["jh"], m=["सोमरा", "एतवा", "संजय", "बुधराम"], f=["सलोमी", "फुलमनी", "सुशीला", "जसिंता"],
               s=["उरांव", "तिर्की", "कुजूर", "हेम्ब्रम", "सोरेन", "मिंज", "टोप्पो"]),
    "uk": dict(reg=["uk"], m=["भूपेंद्र", "दीवान सिंह", "कुंदन", "गोविंद"], f=["हेमा", "पार्वती", "कमला", "बसंती"],
               s=["बिष्ट", "नेगी", "रावत", "भंडारी", "जोशी"]),
}
HI_GROUP_KEYS = ["up", "up", "up", "br", "br", "rj", "hr", "pb", "mus", "mus", "gd", "jh", "uk"]

HI_LAT = {
    "रमेश": "ramesh", "सुरेश": "suresh", "राजेंद्र प्रसाद": "rajendra", "शिवम": "shivam", "अनिल": "anil", "रामसागर": "ramsagar", "हरिओम": "hariom", "बृजेश": "brijesh",
    "सुनीता": "sunita", "रामवती": "ramvati", "गीता": "geeta", "कुसुम": "kusum", "पुष्पा": "pushpa", "नीलम": "neelam", "फूलमती": "phoolmati", "सरोज": "saroj",
    "यादव": "yadav", "वर्मा": "verma", "मौर्य": "maurya", "कुशवाहा": "kushwaha", "तिवारी": "tiwari", "मिश्रा": "mishra", "शुक्ला": "shukla", "पाण्डेय": "pandey",
    "राजभर": "rajbhar", "निषाद": "nishad", "पाल": "pal", "रामबाबू": "rambabu", "शम्भू": "shambhu", "विकास": "vikas", "ललन": "lalan", "मनोज": "manoj",
    "धनंजय": "dhananjay", "रिंकू": "rinku", "सुनैना": "sunaina", "बबीता": "babita", "चंदा": "chanda", "किरण": "kiran", "ललिता": "lalita", "पासवान": "paswan",
    "मंडल": "mandal", "साह": "sah", "ठाकुर": "thakur", "राम": "ram", "महतो": "mahto", "भंवर लाल": "bhanwarlal", "मांगी लाल": "mangilal", "ओम प्रकाश": "omprakash",
    "रामेश्वर": "rameshwar", "कैलाश": "kailash", "भंवरी": "bhanwari", "कमला": "kamla", "संतोष": "santosh", "लाली": "lali", "गुड्डी": "guddi", "मीणा": "meena",
    "गुर्जर": "gurjar", "बैरवा": "bairwa", "सैनी": "saini", "चौधरी": "choudhary", "कुमावत": "kumawat", "राजबीर": "rajbir", "सतबीर": "satbir", "धर्मबीर": "dharambir",
    "नवीन": "naveen", "संदीप": "sandeep", "कविता": "kavita", "मोनिका": "monika", "बिमला": "bimla", "मलिक": "malik", "दहिया": "dahiya", "हुड्डा": "hooda",
    "सांगवान": "sangwan", "अहलावत": "ahlawat", "गुरप्रीत": "gurpreet", "हरजिंदर": "harjinder", "कुलविंदर": "kulwinder", "जसवंत": "jaswant", "सुखदेव": "sukhdev",
    "हरप्रीत": "harpreet", "जसवीर": "jasvir", "मनजीत": "manjit", "सुखविंदर": "sukhwinder", "अमनदीप": "amandeep", "संधू": "sandhu", "गिल": "gill", "ढिल्लों": "dhillon",
    "बराड़": "brar", "मो. इरफ़ान": "irfan", "मोहम्मद शाकिर": "shakir", "अब्दुल रशीद": "rashid", "सलीम": "salim", "नसीम": "naseem", "मो. ज़ुबैर": "zubair",
    "शबाना": "shabana", "रुख़सार": "rukhsar", "नाज़िया": "nazia", "फ़रज़ाना": "farzana", "शाहीन": "shaheen", "नसरीन": "nasreen", "अंसारी": "ansari", "कुरैशी": "qureshi",
    "सिद्दीकी": "siddiqui", "ख़ान": "khan", "सैफ़ी": "saifi", "रामलाल": "ramlal", "सुखराम": "sukhram", "बुधराम": "budhram", "मनोहर": "manohar", "सुखिया": "sukhiya",
    "फूलबाई": "phoolbai", "सोमारी": "somari", "दुर्गा": "durga", "मरावी": "maravi", "उइके": "uikey", "नेताम": "netam", "धुर्वे": "dhurve", "साहू": "sahu", "बघेल": "baghel",
    "सोमरा": "somra", "एतवा": "etwa", "संजय": "sanjay", "सलोमी": "salomi", "फुलमनी": "phulmani", "सुशीला": "sushila", "जसिंता": "jasinta", "उरांव": "oraon",
    "तिर्की": "tirkey", "कुजूर": "kujur", "हेम्ब्रम": "hembrom", "सोरेन": "soren", "मिंज": "minz", "टोप्पो": "toppo", "भूपेंद्र": "bhupendra", "दीवान सिंह": "diwan",
    "कुंदन": "kundan", "गोविंद": "govind", "हेमा": "hema", "पार्वती": "parvati", "बसंती": "basanti", "बिष्ट": "bisht", "नेगी": "negi", "रावत": "rawat",
    "भंडारी": "bhandari", "जोशी": "joshi",
}
for _g in HI_GROUPS.values():
    for _n in _g["m"] + _g["f"] + [x for x in _g["s"] if x]:
        assert _n in HI_LAT, _n

TAMIL_INIT = ["S.", "R.", "K.", "M.", "P.", "A.", "V.", "T.", "N."]


class Person:
    def __init__(self, r, group, gender, given, surname, hindi=False):
        G = (HI_GROUPS if hindi else GROUPS)[group]
        self.group, self.g, self.given, self.surname, self.hindi = group, gender, given, surname, hindi
        self.kind = G.get("kind", "std")
        self.region = r.choice(G["reg"])
        k = self.kind
        if k == "sikh":
            mid = ("सिंह" if gender == "m" else "कौर") if hindi else ("Singh" if gender == "m" else "Kaur")
            self.full = " ".join(x for x in (given, mid, surname) if x)
            self.first, self.last = given, (surname or mid)
        elif k == "meitei":
            suf = "Singh" if gender == "m" else r.choice(["Devi", "Chanu"])
            self.full = "%s %s %s" % (surname, given, suf)
            self.first, self.last = given, surname
        elif k == "tamil":
            self.init = r.choice(TAMIL_INIT)
            st = r.random()
            if st < 0.45:
                self.full, self.last = "%s %s" % (self.init, given), None
            elif st < 0.6:
                self.full, self.last = "%s %s" % (given, self.init.rstrip(".")), None
            else:
                self.full, self.last = "%s %s" % (given, surname), surname
            self.first = given
        elif k == "telugu" and r.random() < 0.4:
            self.full = "%s %s" % (surname, given)
            self.first, self.last = given, surname
        else:
            self.full = " ".join(x for x in (given, surname) if x)
            self.first, self.last = given, (surname or None)
        # ASCII-ish forms for e-mail / ABHA address
        if hindi:
            self.first_ascii = HI_LAT[given]
            self.last_ascii = HI_LAT.get(surname) if surname else None
        else:
            self.first_ascii = self.first.split()[-1] if self.first.startswith(("Mohd.", "Md.", "Ab.")) else self.first.split()[0]
            self.last_ascii = self.last

    def later(self, r):
        """A later short mention: first name or surname."""
        opts = [self.first]
        if self.last and self.kind not in ("meitei",):
            opts.append(self.last)
        return r.choice(opts)


USED = set()  # given names already used in the current note


def _fresh(r, names):
    for _ in range(40):
        g = r.choice(names)
        if g not in USED:
            break
    USED.add(g)
    return g


def mk(r, group=None, gender=None, surname=None, region=None, pool=None):
    if group is None:
        if pool:
            group = r.choice(pool)
        else:
            keys = list(EN_GROUP_W)
            group = r.choices(keys, weights=[EN_GROUP_W[k] for k in keys])[0]
    G = GROUPS[group]
    gender = gender or r.choice("mf")
    given = _fresh(r, G[gender])
    if surname is None:
        surname = r.choice(G["s"])
    p = Person(r, group, gender, given, surname)
    if region:
        p.region = region
    return p


def mkh(r, group=None, gender=None, surname=None, region=None):
    group = group or r.choice(HI_GROUP_KEYS)
    G = HI_GROUPS[group]
    gender = gender or r.choice("mf")
    given = _fresh(r, G[gender])
    if surname is None:
        surname = r.choice(G["s"])
    p = Person(r, group, gender, given, surname, hindi=True)
    if region:
        p.region = region
    return p


def relative(r, p, word_gender, same_surname=True):
    if p.hindi:
        return mkh(r, p.group, word_gender, p.surname if same_surname else None, region=p.region)
    sur = p.surname if same_surname else None
    q = mk(r, p.group, word_gender, sur, region=p.region)
    if p.kind == "tamil" and same_surname:
        pass
    return q


KIN_EN = [("husband", "m", True, "f"), ("wife", "f", True, "m"), ("son", "m", True, None), ("daughter", "f", True, None),
          ("father", "m", True, None), ("mother", "f", True, None), ("brother", "m", True, None), ("sister", "f", True, None),
          ("daughter-in-law", "f", True, None), ("son-in-law", "m", False, None), ("nephew", "m", True, None), ("grandson", "m", True, None)]
KIN_HI = [("पति", "m", True, "f"), ("पत्नी", "f", True, "m"), ("बेटा", "m", True, None), ("बेटी", "f", True, None), ("पिता", "m", True, None),
          ("माँ", "f", True, None), ("भाई", "m", True, None), ("बहन", "f", True, None), ("बहू", "f", True, None), ("दामाद", "m", False, None),
          ("भतीजा", "m", True, None), ("पोता", "m", True, None)]
KIN_MX = [("pati", "m", True, "f"), ("husband", "m", True, "f"), ("wife", "f", True, "m"), ("beta", "m", True, None), ("beti", "f", True, None),
          ("papa", "m", True, None), ("mummy", "f", True, None), ("bhai", "m", True, None), ("behen", "f", True, None), ("bahu", "f", True, None),
          ("damad", "m", False, None), ("bhatija", "m", True, None), ("saas", "f", True, None)]


OLDER_ONLY = {"son", "daughter", "daughter-in-law", "son-in-law", "grandson", "बेटा", "बेटी", "बहू", "दामाद", "पोता", "beta", "beti", "bahu", "damad"}
YOUNGER_ONLY = {"father", "mother", "पिता", "माँ", "papa", "mummy"}


def kin(r, p, table=KIN_EN, age=None):
    opts = [k for k in table if k[3] is None or k[3] == p.g]
    if age is not None:
        opts = [k for k in opts if not (age < 34 and k[0] in OLDER_ONLY) and not (age > 62 and k[0] in YOUNGER_ONLY)]
        if age < 18:
            opts = [k for k in opts if k[0] not in ("husband", "wife", "पति", "पत्नी", "pati", "saas")]
    w, g, same, _ = r.choice(opts)
    return w, relative(r, p, g, same)


def pron(p):
    return ("he", "his", "him", "He", "His") if p.g == "m" else ("she", "her", "her", "She", "Her")


def hon_en(r, p, p_none=0.35):
    if r.random() < p_none:
        return ""
    if p.group in ("tn", "tnc") and r.random() < 0.3:
        return "Thiru " if p.g == "m" else "Tmt. "
    if p.g == "m":
        return r.choice(["Mr. ", "Mr ", "Shri ", "Sh. ", "Mr. "])
    return r.choice(["Mrs. ", "Ms. ", "Smt. ", "Smt ", "Ms "])


def hon_hi(r, p, p_none=0.3):
    if r.random() < p_none:
        return ""
    return r.choice(["श्री ", "श्री "]) if p.g == "m" else r.choice(["श्रीमती ", "श्रीमती ", "सुश्री "])


def doctor(r, pool=None):
    return mk(r, pool=pool) if pool else mk(r)


def DR(r):
    return r.choice(["Dr. ", "Dr ", "Dr.", "DR. "])


# --------------------------------------------------------------------------
# Places / addresses
# --------------------------------------------------------------------------
REG = {
    "up": dict(state=["U.P.", "Uttar Pradesh", "UP"], std=["05862", "0522", "05252"], style="north",
               rural=[("Sitapur", "261001", ["Sidhauli", "Laharpur", "Mahmudabad", "Biswan"]), ("Bahraich", "271801", ["Nanpara", "Kaiserganj", "Mihinpurwa"]),
                      ("Ballia", "277001", ["Rasra", "Bansdih", "Sikandarpur"])],
               urban=[("Lucknow", "226016", ["Indira Nagar", "Aliganj", "Rajajipuram"]), ("Meerut", "250004", ["Shastri Nagar", "Ganga Nagar"])]),
    "br": dict(state=["Bihar"], std=["06272", "0612"], style="north",
               rural=[("Darbhanga", "846004", ["Benipur", "Singhwara", "Bahadurpur"]), ("Gaya", "823001", ["Sherghati", "Tikari", "Belaganj"]),
                      ("Muzaffarpur", "842001", ["Kanti", "Musahari", "Sakra"])],
               urban=[("Patna", "800020", ["Kankarbagh", "Rajendra Nagar", "Boring Road"])]),
    "rj": dict(state=["Rajasthan", "Raj."], std=["0144", "0141"], style="north",
               rural=[("Alwar", "301001", ["Thanagazi", "Rajgarh", "Laxmangarh"]), ("Bhilwara", "311001", ["Mandal", "Asind", "Shahpura"])],
               urban=[("Jaipur", "302017", ["Malviya Nagar", "Pratap Nagar"])]),
    "hr": dict(state=["Haryana", "HR"], std=["01262", "01662"], style="north",
               rural=[("Rohtak", "124001", ["Meham", "Kalanaur", "Sampla"]), ("Jhajjar", "124103", ["Beri", "Badli", "Matanhail"])],
               urban=[("Hisar", "125001", ["Sector 15", "Urban Estate II"])]),
    "pb": dict(state=["Punjab", "Pb."], std=["0164", "0161"], style="north",
               rural=[("Bathinda", "151001", ["Rampura Phul", "Bhucho Mandi", "Maur"]), ("Moga", "142001", ["Bagha Purana", "Nihal Singh Wala", "Dharamkot"])],
               urban=[("Ludhiana", "141001", ["Model Town", "Sarabha Nagar"])]),
    "dl": dict(state=["Delhi", "New Delhi"], std=["011"], style="north", rural=[],
               urban=[("Delhi", "110092", ["Laxmi Nagar", "Shakarpur", "Mayur Vihar Phase 2"]), ("New Delhi", "110018", ["Tilak Nagar", "Vishnu Garden"])]),
    "mp": dict(state=["M.P.", "Madhya Pradesh"], std=["07662", "0755"], style="north",
               rural=[("Rewa", "486001", ["Mauganj", "Sirmour", "Hanumana"]), ("Mandla", "481661", ["Nainpur", "Bichhiya"])],
               urban=[("Bhopal", "462003", ["Shivaji Nagar", "Kolar Road"])]),
    "cg": dict(state=["Chhattisgarh", "C.G."], std=["07752", "0771"], style="north",
               rural=[("Bilaspur", "495001", ["Takhatpur", "Kota", "Masturi"]), ("Dantewada", "494449", ["Geedam", "Katekalyan"])],
               urban=[("Raipur", "492001", ["Shankar Nagar", "Tikrapara"])]),
    "jh": dict(state=["Jharkhand"], std=["0651"], style="north",
               rural=[("Ranchi", "834001", ["Kanke", "Ormanjhi", "Bero"]), ("Gumla", "835207", ["Sisai", "Ghaghra", "Basia"])],
               urban=[("Ranchi", "834002", ["Harmu Colony", "Bariatu"])]),
    "uk": dict(state=["Uttarakhand", "UK"], std=["05962", "0135"], style="north",
               rural=[("Almora", "263601", ["Dwarahat", "Chaukhutia", "Salt"]), ("Chamoli", "246401", ["Gairsain", "Tharali"])],
               urban=[("Dehradun", "248001", ["Rajpur Road", "Dalanwala"])]),
    "wb": dict(state=["West Bengal", "W.B."], std=["033", "03174"], style="east",
               rural=[("South 24 Parganas", "743331", ["Kulpi", "Mathurapur", "Magrahat"]), ("Murshidabad", "742101", ["Beldanga", "Domkal", "Jangipur"])],
               urban=[("Kolkata", "700034", ["Behala", "Sarsuna"]), ("Kolkata", "700032", ["Jadavpur", "Bijoygarh"]), ("Kolkata", "700084", ["Garia"])]),
    "od": dict(state=["Odisha"], std=["0674", "0680"], style="odisha",
               rural=[("Ganjam", "761001", ["Hinjilicut", "Aska", "Chhatrapur"]), ("Kendrapara", "754211", ["Pattamundai", "Marshaghai"])],
               urban=[("Bhubaneswar", "751007", ["Saheed Nagar", "Nayapalli"]), ("Cuttack", "753001", ["Buxi Bazar", "Chandni Chowk"])]),
    "as": dict(state=["Assam"], std=["0361", "0373"], style="east",
               rural=[("Dibrugarh", "786001", ["Lahowal", "Chabua", "Naharkatia"]), ("Nalbari", "781335", ["Belsor", "Tihu"])],
               urban=[("Guwahati", "781003", ["Chandmari", "Ganeshguri", "Beltola"])]),
    "mz": dict(state=["Mizoram"], std=["0389"], style="ne", rural=[("Lunglei", "796701", ["Serkawn", "Hnahthial"])],
               urban=[("Aizawl", "796001", ["Chaltlang", "Zarkawt", "Dawrpui", "Ramhlun North", "Bawngkawn"])]),
    "mn": dict(state=["Manipur"], std=["0385"], style="ne", rural=[("Thoubal", "795138", ["Wangjing", "Lilong"])],
               urban=[("Imphal West", "795001", ["Singjamei Chingamakha", "Uripok Tourangbam Leikai", "Keishampat Thiyam Leikai", "Sagolband"])]),
    "nl": dict(state=["Nagaland"], std=["0370"], style="ne", rural=[("Mokokchung", "798601", ["Ungma", "Longkhum"])],
               urban=[("Kohima", "797001", ["Kitsubozou Ward", "Midland Colony", "PR Hill", "Daklane"])]),
    "ml": dict(state=["Meghalaya"], std=["0364"], style="ne", rural=[("East Khasi Hills", "793108", ["Sohra", "Mawsynram"])],
               urban=[("Shillong", "793003", ["Laitumkhrah", "Mawlai Mawroh", "Nongthymmai", "Jaiaw Langsning"])]),
    "tr": dict(state=["Tripura"], std=["0381"], style="ne", rural=[("Gomati", "799120", ["Matabari", "Killa"])],
               urban=[("Agartala", "799001", ["Krishnanagar", "Ramnagar Road No. 4", "Jogendranagar"])]),
    "sk": dict(state=["Sikkim"], std=["03592"], style="ne", rural=[("Namchi", "737126", ["Temi", "Ravangla"])],
               urban=[("Gangtok", "737101", ["Tadong", "Development Area", "Upper Sichey"])]),
    "la": dict(state=["Ladakh", "UT Ladakh"], std=["01982"], style="ne", rural=[("Kargil", "194103", ["Drass", "Sankoo"])],
               urban=[("Leh", "194101", ["Skara", "Chuchot Yokma", "Choglamsar", "Changspa"])]),
    "jk": dict(state=["J&K", "Jammu and Kashmir"], std=["0194", "01932"], style="kashmir",
               rural=[("Anantnag", "192101", ["Dooru", "Bijbehara", "Kokernag"]), ("Baramulla", "193101", ["Pattan", "Sopore"])],
               urban=[("Srinagar", "190002", ["Rainawari", "Habba Kadal", "Soura", "Bemina", "Hazratbal"])]),
    "goa": dict(state=["Goa"], std=["0832"], style="goa",
                rural=[("Bardez", "403517", ["Siolim", "Assagao", "Anjuna"]), ("Salcete", "403709", ["Curtorim", "Chinchinim", "Benaulim"]),
                       ("Pernem", "403512", ["Mandrem", "Morjim"])], urban=[]),
    "kl": dict(state=["Kerala"], std=["0481", "0487", "0495"], style="kerala",
               rural=[("Kottayam", "686001", ["Kumarakom", "Ettumanoor", "Pala", "Changanassery"]), ("Thrissur", "680001", ["Ollur", "Guruvayur", "Irinjalakuda"]),
                      ("Kozhikode", "673001", ["Feroke", "Koyilandy", "Vadakara"])], urban=[]),
    "tn": dict(state=["Tamil Nadu", "TN"], std=["0452", "044", "0462"], style="tn",
               rural=[("Tirunelveli", "627002", ["Kalakkad", "Nanguneri"]), ("Madurai", "625020", ["Melur", "Vadipatti"])],
               urban=[("Madurai", "625020", ["K.K. Nagar", "Anna Nagar"]), ("Chennai", "600045", ["Tambaram West", "Selaiyur"]), ("Tirunelveli", "627002", ["Palayamkottai"])]),
    "ap": dict(state=["Andhra Pradesh", "A.P."], std=["0866", "0863"], style="telugu",
               rural=[("Guntur", "522201", ["Tenali", "Mangalagiri", "Ponnur"])],
               urban=[("Vijayawada", "520010", ["Patamata", "Governorpet", "Benz Circle"]), ("Guntur", "522002", ["Brodipet", "Arundelpet"])]),
    "tg": dict(state=["Telangana", "TS"], std=["040", "0870"], style="telugu",
               rural=[("Nalgonda", "508001", ["Miryalaguda", "Devarakonda"])],
               urban=[("Hyderabad", "500060", ["Dilsukhnagar", "Kothapet"]), ("Warangal", "506002", ["Hanamkonda", "Kazipet"])]),
    "ka": dict(state=["Karnataka"], std=["0821", "0836", "0824"], style="ka",
               rural=[("Mandya", "571401", ["Maddur", "Malavalli", "Pandavapura"])],
               urban=[("Mysuru", "570023", ["Kuvempu Nagar", "Vidyaranyapuram", "Gokulam"]), ("Hubballi", "580020", ["Vidyanagar", "Keshwapur"]),
                      ("Mangaluru", "575001", ["Kadri", "Bejai"])]),
    "mh": dict(state=["Maharashtra"], std=["020", "0231", "0253"], style="mh",
               rural=[("Satara", "415001", ["Wai", "Koregaon", "Karad"]), ("Latur", "413512", ["Ausa", "Nilanga"])],
               urban=[("Pune", "411038", ["Kothrud", "Karve Nagar"]), ("Kolhapur", "416008", ["Rajarampuri", "Shahupuri"]), ("Nashik", "422002", ["Gangapur Road", "Panchavati"])]),
    "bom": dict(state=["Maharashtra"], std=["022"], style="bom", rural=[],
                urban=[("Mumbai", "400014", ["Dadar Parsi Colony", "Dadar East"]), ("Mumbai", "400027", ["Byculla"]), ("Mumbai", "400007", ["Tardeo", "Grant Road"])]),
    "gj": dict(state=["Gujarat"], std=["0261", "0281"], style="mh",
               rural=[("Anand", "388001", ["Petlad", "Borsad"]), ("Banaskantha", "385001", ["Deesa", "Tharad"])],
               urban=[("Surat", "395009", ["Adajan", "Pal"]), ("Rajkot", "360005", ["Kalawad Road", "Raiya Road"])]),
}

KL_HOUSES = ["Puthenveettil", "Kizhakkethil", "Thekkumkattil", "Valiyaparambil", "Chirakkal", "Moozhayil", "Kalathil", "Padinjarekara"]
GOA_WARDS = ["Fondvem", "Gaunsawaddo", "Bairo Alto", "Dando", "Vaddy", "Madlo Vaddo", "Tembi Waddo"]
TN_STREETS = ["Kamarajar Salai", "Periyar Street", "VOC Nagar 2nd Street", "Bharathiyar Street", "Gandhiji Road", "Thiru Vi Ka Street", "Main Road", "North Car Street"]
BOM_BLDG = ["Rustom Baug", "Cusrow Baug", "Khareghat Colony", "Jer Mahal", "Navroze Baug", "Shapur Baug"]
MH_BLDG = ["Sai Krupa Apts", "Shivneri CHS", "Gokul Residency", "Shantiniketan Society", "Sardar Patel Nagar Society", "Om Sai Complex"]
KOL_ROADS = ["Rashbehari Avenue", "Netaji Subhas Road", "Bidhan Pally", "Deshbandhu Nagar", "Raja S.C. Mullick Road", "Diamond Harbour Road"]


def address(r, p_or_region, pinmode="in"):
    """Return (address_text, pin).  pinmode: 'in' -> PIN inside text; 'sep'/'none' -> not in text."""
    region = p_or_region if isinstance(p_or_region, str) else p_or_region.region
    R = REG[region]
    st = R["style"]
    rural = R["rural"] and (not R["urban"] or r.random() < 0.55)
    dist, pin, locs = r.choice(R["rural"] if rural else R["urban"])
    loc = r.choice(locs)
    state = r.choice(R["state"])
    n = str(r.randint(2, 999))
    if st == "north":
        if rural:
            po = r.choice(locs)
            a = r.choice([
                "Vill. %s, P.O. %s, Distt. %s" % (loc, po, dist),
                "Village- %s, Post- %s, Tehsil %s, District %s (%s)" % (loc, po, po, dist, state),
                "vill %s po %s dist %s" % (loc.lower(), po.lower(), dist.lower()),
                "Gram %s, Post %s, Thana %s, Jila %s" % (loc, po, po, dist),
                "H.No. %s, Ward No. %d, Vill. %s, %s" % (n, r.randint(1, 15), loc, dist),
            ])
        else:
            a = r.choice([
                "H.No. %s, %s, %s" % (n, loc, dist),
                "House No. %s/%d, Gali No. %d, %s, %s" % (n, r.randint(1, 30), r.randint(1, 12), loc, dist),
                "%s-%s, %s, %s" % (r.choice("ABCDEFG"), n, loc, dist),
                "Flat %d, %s Apartments, %s, %s" % (r.randint(101, 904), r.choice(["Shanti", "Ganga", "Surya", "Vasundhara"]), loc, dist),
                "%s, Block %s, %s, %s" % (n, r.choice("ABCDEJK"), loc, dist),
            ])
    elif st in ("east", "odisha"):
        if rural:
            po = r.choice(locs)
            if st == "odisha" and r.random() < 0.6:
                a = "At/Po %s, Via %s, Dist. %s" % (loc, po, dist)
            else:
                a = "Vill. %s, P.O. %s, P.S. %s, Dist. %s" % (loc, po, po, dist)
        else:
            a = r.choice([
                "%s/%d %s, %s, %s" % (n, r.randint(1, 40), r.choice(KOL_ROADS) if region == "wb" else "Lane %d" % r.randint(1, 9), loc, dist),
                "House No. %s, Bye Lane %d, %s, %s" % (n, r.randint(1, 8), loc, dist),
                "Plot No. %s, %s, %s" % (n, loc, dist),
            ])
    elif st == "ne":
        a = r.choice([
            "House No. %s-%s, %s, %s" % (r.choice("ABCDEFGHJK"), r.randint(1, 120), loc, dist),
            "%s, %s, %s" % (loc, dist, state),
            "Qtr No. %d, %s, %s" % (r.randint(1, 60), loc, dist),
            "%s Veng, %s" % (loc, dist) if region == "mz" else "%s Locality, %s" % (loc, dist),
        ])
    elif st == "kashmir":
        if rural:
            a = "Village %s, Tehsil %s, District %s" % (loc, r.choice(locs), dist)
        else:
            a = r.choice(["Mohalla %s, %s, %s" % (r.choice(["Hakim Bagh", "Khanqah", "Malik Sahib", "Dalgate Mohalla"]), loc, dist),
                          "H.No. %s, %s, %s" % (n, loc, dist),
                          "Near Jamia Masjid, %s, %s" % (loc, dist)])
    elif st == "goa":
        a = r.choice(["H.No. %s, %s, %s, %s, Goa" % (n, r.choice(GOA_WARDS), loc, dist),
                      "House No. %s/%s, %s, %s, %s" % (n, r.choice("ABC"), r.choice(GOA_WARDS), loc, dist),
                      "%s, near St. Anthony's Chapel, %s, %s" % (r.choice(GOA_WARDS), loc, dist)])
    elif st == "kerala":
        a = r.choice(["%s House, %s P.O., %s" % (r.choice(KL_HOUSES), loc, dist),
                      "%s (H), %s, %s Dist." % (r.choice(KL_HOUSES), loc, dist),
                      "%s Veedu, Ward %d, %s, %s" % (r.choice(KL_HOUSES), r.randint(1, 20), loc, dist)])
    elif st == "tn":
        if rural:
            a = "Door No. %d/%d, %s, %s, %s Tk., %s Dt." % (r.randint(1, 9), r.randint(10, 400), r.choice(TN_STREETS), loc, loc, dist)
        else:
            a = "No. %s, %s, %s, %s" % (n, r.choice(TN_STREETS), loc, dist)
    elif st == "telugu":
        a = r.choice(["D.No. %d-%d-%d, %s, %s" % (r.randint(1, 40), r.randint(1, 20), r.randint(1, 200), loc, dist),
                      "H.No. %d-%d-%d/%s, %s, %s" % (r.randint(1, 20), r.randint(1, 9), r.randint(1, 300), r.choice("ABC"), loc, dist)])
    elif st == "ka":
        if rural:
            a = "#%s, %s village, %s Tq, %s Dist" % (n, loc, loc, dist)
        else:
            a = "#%s, %d%s Cross, %s, %s" % (n, r.randint(1, 12), r.choice(["th", "th", "nd", "rd"]), loc, dist)
    elif st == "mh":
        if rural:
            a = "At Post %s, Tal. %s, Dist. %s" % (loc, loc, dist)
        else:
            a = "Flat %d, %s, %s, %s" % (r.randint(1, 40), r.choice(MH_BLDG), loc, dist)
    elif st == "bom":
        a = "%s, %s, %s, %s" % (r.choice(["Flat %d" % r.randint(1, 40), "Block %d/%d" % (r.randint(1, 9), r.randint(1, 30))]), r.choice(BOM_BLDG), loc, dist)
    else:
        raise ValueError(st)
    if pinmode == "in":
        a += r.choice([" - %s" % pin, " %s" % pin, "-%s" % pin, ", %s %s" % (state, pin), " (%s) - %s" % (state, pin), ", PIN %s" % pin, " – %s" % pin])
    return a, pin


def addr_city(region):
    R = REG[region]
    return (R["urban"] or R["rural"])[0][0]


def std_of(r, region):
    return r.choice(REG[region]["std"])


# --------------------------------------------------------------------------
# Hindi places
# --------------------------------------------------------------------------
HI_REG = {
    "up": dict(state=["उ.प्र.", "उत्तर प्रदेश"], rural=[("सीतापुर", "261001", ["सिधौली", "लहरपुर", "महमूदाबाद", "बिसवां"]), ("बहराइच", "271801", ["नानपारा", "कैसरगंज", "मिहींपुरवा"]),
                                                     ("बलिया", "277001", ["रसड़ा", "बांसडीह", "सिकंदरपुर"])],
               urban=[("लखनऊ", "226016", ["इंदिरा नगर", "अलीगंज", "राजाजीपुरम"]), ("मेरठ", "250004", ["शास्त्री नगर", "गंगा नगर"])]),
    "br": dict(state=["बिहार"], rural=[("दरभंगा", "846004", ["बेनीपुर", "सिंहवाड़ा", "बहादुरपुर"]), ("गया", "823001", ["शेरघाटी", "टिकारी"]), ("मुजफ्फरपुर", "842001", ["कांटी", "मुशहरी", "सकरा"])],
               urban=[("पटना", "800020", ["कंकड़बाग", "राजेंद्र नगर"])]),
    "rj": dict(state=["राजस्थान"], rural=[("अलवर", "301001", ["थानागाजी", "राजगढ़", "लक्ष्मणगढ़"]), ("भीलवाड़ा", "311001", ["मांडल", "आसींद", "शाहपुरा"])],
               urban=[("जयपुर", "302017", ["मालवीय नगर", "प्रताप नगर"])]),
    "hr": dict(state=["हरियाणा"], rural=[("रोहतक", "124001", ["महम", "कलानौर", "सांपला"]), ("झज्जर", "124103", ["बेरी", "बादली"])],
               urban=[("हिसार", "125001", ["सेक्टर 15", "अर्बन एस्टेट"])]),
    "pb": dict(state=["पंजाब"], rural=[("बठिंडा", "151001", ["रामपुरा फूल", "भुच्चो मंडी", "मौड़"]), ("मोगा", "142001", ["बाघापुराना", "निहाल सिंह वाला", "धर्मकोट"])],
               urban=[("लुधियाना", "141001", ["मॉडल टाउन", "सराभा नगर"])]),
    "mp": dict(state=["म.प्र."], rural=[("रीवा", "486001", ["मऊगंज", "सिरमौर", "हनुमना"]), ("मंडला", "481661", ["नैनपुर", "बिछिया"])],
               urban=[("भोपाल", "462003", ["शिवाजी नगर", "कोलार रोड"])]),
    "cg": dict(state=["छत्तीसगढ़", "छ.ग."], rural=[("बिलासपुर", "495001", ["तखतपुर", "कोटा", "मस्तूरी"]), ("दंतेवाड़ा", "494449", ["गीदम", "कटेकल्याण"])],
               urban=[("रायपुर", "492001", ["शंकर नगर", "टिकरापारा"])]),
    "jh": dict(state=["झारखंड"], rural=[("रांची", "834001", ["कांके", "ओरमांझी", "बेड़ो"]), ("गुमला", "835207", ["सिसई", "घाघरा", "बसिया"])],
               urban=[("रांची", "834002", ["हरमू कॉलोनी", "बरियातू"])]),
    "uk": dict(state=["उत्तराखंड"], rural=[("अल्मोड़ा", "263601", ["द्वाराहाट", "चौखुटिया", "सल्ट"]), ("चमोली", "246401", ["गैरसैंण", "थराली"])],
               urban=[("देहरादून", "248001", ["राजपुर रोड", "डालनवाला"])]),
}


def address_hi(r, p_or_region, pinmode="in", devn=False):
    region = p_or_region if isinstance(p_or_region, str) else p_or_region.region
    R = HI_REG[region]
    rural = r.random() < 0.65
    dist, pin, locs = r.choice(R["rural"] if rural else R["urban"])
    loc = r.choice(locs)
    state = r.choice(R["state"])
    num = lambda s: dev(s) if devn else s
    if rural:
        po = r.choice(locs)
        a = r.choice([
            "ग्राम %s, पोस्ट %s, थाना %s, जिला %s" % (loc, po, po, dist),
            "गाँव- %s, डाकघर- %s, तहसील- %s, जनपद %s (%s)" % (loc, po, po, dist, state),
            "ग्राम व पोस्ट %s, जिला %s" % (loc, dist),
            "मौजा %s, पो. %s, जि. %s" % (loc, po, dist),
            "ग्राम %s, वार्ड नं. %s, जिला %s" % (loc, num(str(r.randint(1, 15))), dist),
        ])
    else:
        a = r.choice([
            "मकान नं. %s, %s, %s" % (num(str(r.randint(2, 900))), loc, dist),
            "म.नं. %s/%s, गली नं. %s, %s, %s" % (num(str(r.randint(2, 300))), num(str(r.randint(1, 20))), num(str(r.randint(1, 12))), loc, dist),
            "वार्ड नं. %s, %s, %s" % (num(str(r.randint(1, 40))), loc, dist),
            "%s-%s, %s, %s" % (r.choice(["ए", "बी", "सी", "डी"]), num(str(r.randint(2, 400))), loc, dist),
        ])
    if pinmode == "in":
        a += r.choice([" - %s" % num(pin), " %s" % num(pin), ", %s %s" % (state, num(pin)), " (%s) %s" % (state, num(pin))])
    return a, pin


# --------------------------------------------------------------------------
# Facilities (hard negatives: many contain person names)
# --------------------------------------------------------------------------
FAC_EN = [
    "Guru Teg Bahadur Hospital, Shahdara", "Lok Nayak Jai Prakash Narayan Hospital", "Netaji Subhas Chandra Bose Medical College, Jabalpur",
    "Rajiv Gandhi Government General Hospital, Chennai", "Indira Gandhi Medical College, Shimla", "Sardar Vallabhbhai Patel Hospital, Patel Nagar",
    "Acharya Vinoba Bhave Rural Hospital, Sawangi", "Sher-i-Kashmir Institute of Medical Sciences, Soura", "Lady Hardinge Medical College",
    "Maulana Azad Medical College", "St. John's Medical College Hospital, Bengaluru", "Dr. Rajendra Prasad Government Medical College, Tanda",
    "Baba Raghav Das Medical College, Gorakhpur", "Guru Gobind Singh Medical College, Faridkot", "Kalpana Chawla Government Medical College, Karnal",
    "Lokmanya Tilak Municipal General Hospital, Sion", "Nil Ratan Sircar Medical College, Kolkata", "Pandit Raghunath Murmu Medical College, Baripada",
    "Sardar Patel Medical College, Bikaner", "Mahatma Gandhi Memorial Medical College, Indore", "Jawaharlal Nehru Medical College, Belagavi",
    "Sir Thutob Namgyal Memorial Hospital, Gangtok", "Sonam Norboo Memorial Hospital, Leh", "Smt. Kashibai Navale Medical College, Pune",
    "Bai Jerbai Wadia Hospital for Children, Parel", "Sir J.J. Group of Hospitals, Byculla", "Christian Medical College, Vellore",
    "Goa Medical College, Bambolim", "Regional Institute of Medical Sciences, Imphal", "Zoram Medical College, Falkawn", "Civil Hospital Aizawl",
    "NEIGRIHMS, Shillong", "Agartala Government Medical College", "District Hospital Bahraich", "Taluk Headquarters Hospital, Pala",
    "AIIMS Bhubaneswar", "Government Medical College, Kozhikode", "Civil Hospital Bathinda", "Assam Medical College, Dibrugarh",
    "Shri Vasantrao Naik Government Medical College, Yavatmal", "Kasturba Hospital, Manipal", "Sri Venkateswara Ramnarain Ruia Government General Hospital, Tirupati",
]
FAC_SMALL = ["CHC %s", "PHC %s", "Sub-District Hospital %s", "UPHC %s", "Block PHC %s", "Health & Wellness Centre %s", "Sub-centre %s"]
FAC_HI = ["लोकनायक जयप्रकाश नारायण अस्पताल", "महारानी लक्ष्मीबाई मेडिकल कॉलेज, झाँसी", "गुरु गोबिंद सिंह मेडिकल कॉलेज, फरीदकोट",
          "डॉ. श्यामा प्रसाद मुखर्जी सिविल अस्पताल, लखनऊ", "पं. दीनदयाल उपाध्याय जिला चिकित्सालय", "बाबा राघव दास मेडिकल कॉलेज, गोरखपुर",
          "सरदार वल्लभभाई पटेल अस्पताल", "राजेंद्र आयुर्विज्ञान संस्थान, रांची", "अटल बिहारी वाजपेयी आयुर्विज्ञान संस्थान", "इंदिरा गांधी आयुर्विज्ञान संस्थान, पटना",
          "जिला अस्पताल बहराइच", "महर्षि वाल्मीकि मेडिकल कॉलेज", "छत्रपति शिवाजी महाराज जिला चिकित्सालय", "संत कबीर जिला अस्पताल"]
FAC_SMALL_HI = ["सामुदायिक स्वास्थ्य केंद्र %s", "प्राथमिक स्वास्थ्य केंद्र %s", "सीएचसी %s", "पीएचसी %s", "उपकेंद्र %s"]


def small_fac(r, p_region, hindi=False):
    if hindi:
        R = HI_REG[p_region]
        place = r.choice(r.choice(R["rural"])[2])
        return r.choice(FAC_SMALL_HI) % place
    R = REG[p_region]
    place = r.choice(r.choice(R["rural"] or R["urban"])[2])
    return r.choice(FAC_SMALL) % place


# --------------------------------------------------------------------------
# Clinical filler (hard negatives)
# --------------------------------------------------------------------------
COMPLAINTS = [
    ("a slowly worsening tremor of the right hand", "early Parkinson's disease"),
    ("weakness that started in both feet and climbed to the thighs over four days", "Guillain-Barré syndrome"),
    ("a diffuse neck swelling with weight gain and lethargy", "Hashimoto's thyroiditis"),
    ("palpitations, heat intolerance and a fine tremor", "Graves' disease"),
    ("chronic diarrhoea with weight loss and perianal tags", "Crohn's disease"),
    ("round face, purple striae and new-onset diabetes", "Cushing's syndrome"),
    ("fatigue, salt craving and darkening of the gums", "Addison's disease"),
    ("back pain at night with evening rise of temperature", "Pott's spine"),
    ("progressive forgetfulness noticed by the family", "Alzheimer's dementia"),
    ("painless cervical nodes and drenching night sweats", "Hodgkin lymphoma"),
    ("episodes of vertigo with ringing in the left ear", "Meniere's disease"),
    ("fingers turning white then blue in cold water", "Raynaud's phenomenon"),
    ("hypopigmented anaesthetic patches over the back", "Hansen's disease"),
    ("pain in the calves on walking with a blackened toe", "Buerger's disease"),
    ("heartburn for years and recent difficulty swallowing", "Barrett's oesophagus"),
    ("absent pulses in the left arm with arm claudication", "Takayasu arteritis"),
    ("confusion and unsteady gait after heavy drinking", "Wernicke's encephalopathy"),
]
SIGNS = ["Murphy's sign negative", "Kernig's sign negative", "plantars bilaterally flexor (Babinski absent)", "Romberg's test positive",
         "tenderness at McBurney's point", "Trendelenburg test negative", "GCS E4V5M6", "Mallampati grade II", "Snellen 6/9 both eyes",
         "Mantoux 14 mm induration", "Lasègue's sign positive at 40 degrees", "Chvostek's sign negative", "Phalen's test positive on the right"]
DEVICES = ["Foley's catheter", "Ryle's tube", "Hickman line", "Tenckhoff catheter", "Thomas splint", "Austin Moore prosthesis",
           "Steinmann pin", "Brescia-Cimino fistula", "Hartmann's solution", "Jackson-Rees circuit", "Sengstaken tube", "Bain circuit"]
DRUGS = [("Tab. Metformin", "500 mg BD"), ("Tab. Telmisartan", "40 mg OD"), ("Tab. Amlodipine", "5 mg OD"), ("Tab. Ecosprin", "75 mg OD"),
         ("Tab. Atorvastatin", "20 mg HS"), ("Tab. Thyronorm", "50 mcg empty stomach"), ("Inj. Ceftriaxone", "1 g IV BD"), ("Tab. Dolo", "650 mg SOS"),
         ("Cap. Pantop", "40 mg OD"), ("Tab. Glimepiride", "1 mg OD"), ("Tab. Shelcal", "500 mg OD"), ("Tab. Livogen", "1 OD"),
         ("Syp. Ascoril", "5 ml TDS"), ("Tab. Montair LC", "1 HS"), ("Inj. Monocef", "1 g IV BD"), ("Tab. Arjuna", "1 BD"), ("Liv 52", "2 tab BD"),
         ("Tab. Sevelamer", "800 mg TDS"), ("Tab. Meftal Spas", "1 SOS"), ("Tab. Ferium XT", "1 OD"), ("Tab. Becosules", "1 OD"),
         ("Tab. Sitagliptin", "100 mg OD"), ("Tab. Rosuvas", "10 mg HS"), ("Tab. Ondem", "4 mg SOS"), ("Tab. Zincovit", "1 OD"),
         ("Tab. Azithral", "500 mg OD x 3 days"), ("Tab. Augmentin", "625 mg BD"), ("Inj. Insulin Glargine", "12 units HS")]
PSY_DRUGS = [("Tab. Escitalopram", "10 mg HS"), ("Tab. Sertraline", "50 mg OD"), ("Tab. Olanzapine", "5 mg HS"), ("Tab. Lithosun SR", "400 mg BD"),
             ("Tab. Clonazepam", "0.25 mg HS"), ("Tab. Nexito", "10 mg OD"), ("Tab. Fluoxetine", "20 mg OD"), ("Tab. Mirtazapine", "7.5 mg HS"),
             ("Tab. Quetiapine", "25 mg HS"), ("Tab. Divalproex", "500 mg BD"), ("Tab. Naltrexone", "50 mg OD"), ("Tab. Thiamine", "100 mg TDS")]


def labs(r):
    items = ["Hb %.1f g/dL" % r.uniform(6.5, 14.5), "TLC %d" % r.randrange(4000, 18000, 100), "platelets %.2f lakh" % r.uniform(0.6, 3.9),
             "creatinine %.1f mg/dL" % r.uniform(0.6, 8.5), "urea %d" % r.randint(18, 190), "Na %d" % r.randint(126, 145),
             "K %.1f" % r.uniform(3.1, 6.4), "HbA1c %.1f%%" % r.uniform(5.4, 11.9), "TSH %.2f mIU/L" % r.uniform(0.1, 48),
             "SGPT %d IU/L" % r.randint(14, 240), "ESR %d mm/hr" % r.randint(5, 110), "RBS %d mg/dL" % r.randint(70, 420)]
    return ", ".join(r.sample(items, r.randint(2, 5)))


def inr(n):
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


assert inr(210237) == "2,10,237" and inr(4500) == "4,500"


def bp(r):
    sy = r.randrange(90, 190, 2)
    return "%d/%d" % (sy, r.randrange(56, min(120, sy - 24), 2))


def night_ts(r):
    return "%02d:%02d" % (r.choice([22, 23, 0, 1, 2, 3, 4, 5]), r.randint(0, 59))


def occupation(r, p):
    return r.choice(["farmer", "school teacher", "auto-rickshaw driver", "homemaker", "retired clerk", "tailor", "construction worker",
                     "shopkeeper", "fisherman", "nurse", "software engineer", "anganwadi helper", "tea-garden worker", "weaver", "bank cashier"])


# ==========================================================================
# ENGLISH GENERATORS
# ==========================================================================
def en_doctor_narrative(r):
    pat = mk(r)
    age = r.randint(19, 83)
    he, his, him, He, His = pron(pat)
    doc = mk(r)
    fac = r.choice(FAC_EN)
    v = r.choice(["referral", "discharge", "certificate", "progress"])
    kw, rel = kin(r, pat, age=age)
    cx, dx = r.choice(COMPLAINTS)
    out = []
    if v == "referral":
        ref = mk(r)
        a, pin = address(r, pat, r.choice(["in", "none"]))
        out.append(fdate(r, ev_date(r)) + "\n\n")
        out.append("Dear %s%s,\n\n" % (DR(r), PER(ref.later(r))))
        reline = "Re: %s%s" % (hon_en(r, pat, 0.6), PER(pat.full))
        if r.random() < 0.55:
            reline += r.choice([", born %s", " (DOB %s)", ", d.o.b. %s"]) % DB(fdate(r, dob_for(r, age)))
        reline += r.choice([" – UHID %s", ", hospital no. %s", " [CR %s]", "; IP no %s"]) % MR(gen_mrn(r, r.choice(["uhid", "cr12", "ipd"])))
        out.append(reline + "\n\n")
        out.append("I would be grateful if you could see %s, a %d-year-old %s who lives at %s. " % (PER(pat.later(r)), age, occupation(r, pat), ADR(a)))
        out.append("%s has had %s for about %d weeks, and I am concerned about %s. " % (He, cx, r.randint(2, 20), dx))
        out.append("On examination %s; BP %s. Recent bloods: %s.\n\n" % (r.choice(SIGNS), bp(r), labs(r)))
        d1, d2 = r.sample(DRUGS, 2)
        out.append("%s is on %s %s and %s %s. " % (He, d1[0], d1[1], d2[0], d2[1]))
        out.append("%s %s %s usually accompanies %s and can be reached on %s. " % (r.choice(["The", His]), kw, PER(rel.first), him, PH(fmt_mobile(r))))
        out.append("I have explained to %s that the wait at %s may be long." % (PER(pat.first), r.choice(["your OPD", "the tertiary centre", "the referral clinic"])))
        if r.random() < 0.55:
            out.append(" %s ABHA number is %s." % (His, ABN(gen_abha(r))))
        out.append("\n\n")
        out.append(r.choice(["With regards,", "Yours sincerely,", "Thanks and regards,", "Warm regards,"]) + "\n\n")
        out.append("%s%s\n%s, %s\n" % (DR(r), PER(doc.full), r.choice(["MD (Medicine)", "MBBS, DNB (Family Medicine)", "MS (Gen Surg)", "MD (Paed)"]), small_fac(r, pat.region)))
        if r.random() < 0.5:
            out.append("%s %s\n" % (r.choice(["Email:", "e-mail -", "mail:"]), EM(gen_email(r, doc, True))))
    elif v == "discharge":
        adm = ev_date(r, 1, 9)
        dis = (adm[0], adm[1] + 1, r.randint(1, 28))
        out.append(r.choice(["DISCHARGE NOTE", "Discharge narrative", "Summary of stay"]) + " – " + fac + "\n\n")
        sx = "M" if pat.g == "m" else "F"
        out.append("%s%s (%d/%s), IPD No. %s, was admitted on %s with %s and discharged on %s. " % (
            hon_en(r, pat), PER(pat.full), age, sx, MR(gen_mrn(r, "ipd")), fdate(r, adm), cx, fdate(r, dis)))
        dg = r.choice(DRUGS)
        out.append("Working diagnosis was %s. %s was managed with %s %s and a %s; %s at discharge. " % (
            dx, He, dg[0], dg[1], r.choice(DEVICES), r.choice(SIGNS)))
        out.append("Labs on the day of discharge: %s.\n\n" % labs(r))
        out.append("Throughout the stay %s was looked after by %s %s %s, who was taught %s. " % (
            PER(pat.later(r)), his, kw, PER(rel.full if r.random() < 0.5 else rel.first), r.choice(["insulin technique", "catheter care", "pressure-area care", "the diet chart"])))
        if r.random() < 0.6:
            out.append("Contact on file: %s (%s). " % (PH(fmt_mobile(r)), kw))
        if r.random() < 0.6:
            out.append("ABHA %s linked. " % ABN(gen_abha(r)))
        if r.random() < 0.6:
            out.append("Home address: %s. " % ADR(address(r, pat, r.choice(["in", "in", "none"]))[0]))
        if r.random() < 0.6:
            out.append("Final bill ₹%s was settled in cash by %s %s; PAN %s recorded at the billing desk. " % (
                inr(r.randint(205000, 480000)), kw, PER(rel.first), PANE(gen_pan(r, rel))))
        out.append("Review in Medicine OPD on %s with %s%s.\n\n" % (fdate(r, (2026, min(dis[1] + 1, 12), r.randint(1, 28))), DR(r), PER(doc.later(r))))
        out.append("— %s%s, %s\n" % (DR(r), PER(doc.full), r.choice(["Senior Resident", "Consultant Physician", "Unit II"])))
    elif v == "certificate":
        rel2 = relative(r, pat, "m", True)
        a, pin = address(r, pat, "in")
        out.append(r.choice(["TO WHOM IT MAY CONCERN", "MEDICAL CERTIFICATE", "Medical Fitness / Leave Certificate"]) + "\n\n")
        so = r.choice(["S/o", "son of"]) if pat.g == "m" else r.choice(["D/o", "W/o", "daughter of"])
        out.append("This is to certify that %s%s, %s %s%s, aged %d years, resident of %s" % (
            hon_en(r, pat, 0.2), PER(pat.full), so, hon_en(r, rel2, 0.5), PER(rel2.full), age, ADR(a)))
        if r.random() < 0.85:
            out.append(", Aadhaar No. %s" % AAD(fmt_aadhaar(r)))
        out.append(", was under my treatment at %s from %s to %s for %s. " % (fac, fdate(r, ev_date(r, 1, 5)), fdate(r, ev_date(r, 6, 9)), dx))
        out.append("%s is advised rest for %d days and may resume duty from %s." % (He, r.randint(5, 30), fdate(r, ev_date(r, 9, 10))))
        if r.random() < 0.35:
            out.append(" Issued on request for an insurance claim (PAN of patient: %s)." % PANE(gen_pan(r, pat)))
        out.append("\n\n")
        out.append("Signature of patient: ____________\n\n%s%s\n%s\n(Seal)\n" % (DR(r), PER(doc.full), r.choice(["Medical Officer", "CMO", "Consultant, Dept. of Orthopaedics"])))
    else:
        out.append("Ward round notes – %s\n\n" % fac)
        nm = pat.later(r)
        for day in range(1, r.randint(3, 5)):
            if day == 1:
                out.append("Day 1: %s%s, %d, %s, admitted with %s (? %s). %s.\n" % (
                    hon_en(r, pat, 0.3), PER(pat.full), age, r.choice(["UHID", "MRD no", "Hosp. No."]) + " " + MR(gen_mrn(r, "uhid")), cx, dx, r.choice(SIGNS)))
            else:
                out.append("Day %d: %s%s %s. %s. Plan: %s %s.\n" % (
                    day, hon_en(r, pat, 0.6) if pat.last and nm == pat.last else "", PER(nm),
                    r.choice(["comfortable overnight", "spiked 38.6 at 2 am", "tolerating orally", "more alert today", "still breathless on minimal exertion"]),
                    labs(r), *r.choice(DRUGS)))
        if r.random() < 0.5:
            out.append("Address on admission sheet: %s.\n" % ADR(address(r, pat, r.choice(["in", "none"]))[0]))
        out.append("Family (%s %s) updated at bedside by %s%s; %s asked about transfer to %s.\n" % (
            kw, PER(rel.first), DR(r), PER(doc.later(r)), PER(rel.first), r.choice(FAC_EN)))
    return "".join(out)


def en_asha_whatsapp(r):
    asha = mk(r, gender="f", pool=NORTH_GROUPS + ["od", "bn", "ka", "mh", "te", "as", "kh"])
    mo = mk(r)
    pat = mk(r, group=asha.group, gender="f", region=asha.region)
    hus = relative(r, pat, "m", True)
    d = ev_date(r)
    android = r.random() < 0.5
    t0h = r.randint(7, 20)
    mins = [r.randint(0, 35)]

    def hdr(who):
        mins[0] = min(59, mins[0] + r.randint(0, 4))
        if android:
            return "%02d/%02d/%d, %02d:%02d - %s: " % (d[2], d[1], d[0], t0h, mins[0], who)
        return "[%d/%d/%02d, %d:%02d %s] %s: " % (d[2], d[1], d[0] % 100, (t0h - 1) % 12 + 1, mins[0], "am" if t0h < 12 else "pm", who)
    asha_tag = r.choice(["%s ASHA" % PER(asha.first), "ASHA %s" % PER(asha.first), "%s (Asha)" % PER(asha.full), PER(asha.full), "Asha didi %s" % PER(asha.first)])
    mo_tag = r.choice(["%s%s MO" % (DR(r), PER(mo.first)), "MO %s%s" % (DR(r), PER(mo.later(r))), "%s%s" % (DR(r), PER(mo.full)), "MO PHC"])
    scen = r.choice(["hrp", "newborn", "tb", "fever", "elder"])
    L = []
    L.append(hdr(asha_tag) + r.choice(["Good morning sir 🙏", "Namaste sir", "Sir urgent", "gd mrng sir"]))
    if scen == "hrp":
        age = r.randint(18, 36)
        L.append(hdr(asha_tag) + "Sir one HRP case in my area. Name %s w/o %s, age %d, %d month pregnant" % (PER(pat.full), PER(hus.full), age, r.randint(6, 9)))
        L.append(hdr(asha_tag) + "BP %s today, Hb %.1f. Swelling on both feet and headache from 2 days" % (bp(r), r.uniform(6.5, 10.5)))
        L.append(hdr(mo_tag) + r.choice(["Send her to CHC today itself. Call 108.", "Refer to FRU. Did she get Inj. MgSO4 loading?", "Whats her RCH ID?"]))
        L.append(hdr(asha_tag) + "RCH ID %s sir. Her mobile %s" % (MR(gen_mrn(r, "rch12")), PH(fmt_mobile(r))))
        if r.random() < 0.6:
            L.append(hdr(asha_tag) + r.choice(["adress ", "Address: ", "address- "]) + ADR(address(r, pat, r.choice(["in", "none"]))[0]))
        if r.random() < 0.6:
            L.append(hdr(mo_tag) + "Is her aadhaar seeded for JSY?")
            L.append(hdr(asha_tag) + "yes sir %s" % AAD(fmt_aadhaar(r)))
        if r.random() < 0.6:
            mil = relative(r, pat, "f", True)
            L.append(hdr(asha_tag) + "sir family not ready, mother in law %s saying go after Navratri" % PER(mil.first))
            L.append(hdr(mo_tag) + "Give me %s's number, I will talk to him" % PER(hus.first))
            L.append(hdr(asha_tag) + PH(fmt_mobile(r)))
    elif scen == "newborn":
        L.append(hdr(asha_tag) + "Sir home delivery last night in %s family. Baby of %s, weight %.1f kg on my scale" % (PER(hus.last or hus.first), PER(pat.first), r.uniform(1.4, 2.4)))
        L.append(hdr(asha_tag) + "baby DOB %s, not sucking well. Mother %s w/o %s" % (DB(fdate(r, d, "dmy/")), PER(pat.full), PER(hus.first)))
        L.append(hdr(mo_tag) + "Keep baby skin to skin, KMC. Bring to SNCU %s. Share address" % small_fac(r, pat.region))
        L.append(hdr(asha_tag) + ADR(address(r, pat, r.choice(["in", "none"]))[0]))
        L.append(hdr(asha_tag) + "<Media omitted>")
    elif scen == "tb":
        pt = mk(r, group=asha.group, gender="m", region=asha.region)
        L.append(hdr(asha_tag) + "Sir %s s/o %s cough 3 weeks, weight loss, sputum given at %s" % (PER(pt.full), PER(relative(r, pt, "m").full), small_fac(r, pt.region)))
        L.append(hdr(mo_tag) + "CBNAAT result came - MTB detected, Rif sensitive. Ni-kshay ID %s. Start him on FDC today" % MR(gen_mrn(r, "nikshay")))
        L.append(hdr(asha_tag) + "ok sir. His wife %s also coughing, I will bring her tomorrow" % PER(relative(r, pt, "f").first))
        L.append(hdr(asha_tag) + "Nikshay Poshan Yojana account details pending, aadhaar %s" % AAD(fmt_aadhaar(r)))
        if r.random() < 0.5:
            L.append(hdr(asha_tag) + "his address %s" % ADR(address(r, pt, r.choice(["in", "none"]))[0]))
    elif scen == "fever":
        kids = [mk(r, group=asha.group, region=asha.region) for _ in range(r.randint(2, 3))]
        L.append(hdr(asha_tag) + "sir fever cases in ward %d, %d houses" % (r.randint(1, 12), r.randint(3, 9)))
        L.append(hdr(asha_tag) + " ,".join("%s %d yr" % (PER(k.first if r.random() < 0.5 else k.full), r.randint(3, 60)) for k in kids))
        L.append(hdr(mo_tag) + "RDT done? Send names to MPW %s" % PER(mk(r, group=asha.group, gender="m").first))
        L.append(hdr(asha_tag) + "RDT for %s +ve Pv sir. Others negative" % PER(kids[0].first))
    else:
        old = mk(r, group=asha.group, gender=r.choice("mf"), region=asha.region)
        kw, rel = kin(r, old, age=70)
        L.append(hdr(asha_tag) + "Sir %s%s age %d fell in bathroom, cant move right side, mouth deviated" % (hon_en(r, old, 0.7), PER(old.full), r.randint(60, 88)))
        L.append(hdr(mo_tag) + "Time of onset? Take to %s now, stroke window" % r.choice(FAC_EN))
        L.append(hdr(asha_tag) + "1 hour back. %s %s is taking in auto. no %s" % (kw, PER(rel.first), PH(fmt_mobile(r))))
        if r.random() < 0.5:
            L.append(hdr(mo_tag) + "Send exact address, I will inform 108")
            L.append(hdr(asha_tag) + ADR(address(r, old, r.choice(["in", "none"]))[0]))
    if r.random() < 0.4:
        L.append(hdr(PH(fmt_mobile(r))) + r.choice(["Sir i am ANM of subcenter, I will reach", "This message was deleted", "ok"]))
    L.append(hdr(asha_tag) + r.choice(["ok sir", "thank u sir 🙏", "Done sir", "👍"]))
    return "\n".join(L)


def en_nurse_handover(r):
    St = Style(r)
    ward = r.choice(["Female Medical Ward", "MICU", "Ward 6 (Surgery)", "Post-op Ward", "PNC Ward", "SNCU", "Ortho Ward B", "Male Medical Ward"])
    wg = {"Female Medical Ward": "f", "PNC Ward": "f", "Male Medical Ward": "m"}.get(ward)
    d = ev_date(r)
    hdrs = ["HANDOVER — %s — Night to Morning — %s" % (ward, fdate(r, d)), "Shift handover %s 08:00 | %s" % (fdate(r, d), ward),
            "%s  ::  N→M handover  ::  %s" % (ward, fdate(r, d)), "NURSING HANDOVER (SBAR) %s, %s" % (ward.upper(), fdate(r, d))]
    out = [r.choice(hdrs), ""]
    fmt = r.choice(["sbar", "line", "prose"])
    region_pool = r.choice([None, NORTH_GROUPS, ["klc", "klh", "klm", "tn", "tnc"], ["mz", "mn", "nl", "kh", "as", "tr", "sk"], ["jk", "kp", "pb", "la"], ["gc", "gk", "mh", "parsi"]])
    for _ in range(r.randint(2, 4)):
        bed = r.randint(1, 42)
        if ward == "SNCU":
            mother = mk(r, gender="f", pool=region_pool)
            who = "Baby of %s" % PER(St.N(mother.full if r.random() < 0.6 else mother.first))
            p = mother
            age_s = "%d days, %d g" % (r.randint(1, 14), r.randrange(1100, 2600, 10))
            dxs = r.choice(["LBW on OG feeds, Apgar 6/8", "NNJ under phototherapy, SBR 14.2", "RDS on bubble CPAP, FiO2 30%", "sepsis screen positive, on Inj. Cefotaxime"])
        else:
            p = mk(r, pool=region_pool, gender=wg)
            who = hon_en(r, p, 0.5) + PER(St.N(p.full))
            page = r.randint(18, 88)
            age_s = "%d%s" % (page, "M" if p.g == "m" else "F")
            dxs = r.choice(["CKD-5 on MHD, K+ %.1f" % r.uniform(4.5, 6.6), "post-op day %d lap chole, drain 40 ml" % r.randint(1, 3),
                            "DKA, on insulin infusion 4 U/hr, RBS %d" % r.randint(140, 380), "?Pott's spine, on ATT, %s in situ" % r.choice(DEVICES),
                            "CAP, on %s %s, SpO2 94%% on 2 L" % r.choice(DRUGS), "#NOF L, %s planned" % r.choice(["Austin Moore prosthesis", "DHS fixation"]),
                            "acute GE, I/O 2100/1650", "AKI, %s, urine output 20 ml/hr" % r.choice(["Foley's catheter", "condom catheter"]),
                            "Bell's palsy, eye care, Tab. Prednisolone 60 mg", "CVA L hemiparesis, %s feeds 200 ml 2-hrly" % "Ryle's tube"])
        idl = r.choice(["IP No", "IPD", "Reg. no.", "UHID", "CR"])
        idv = MR(gen_mrn(r, "cr12" if idl == "CR" else r.choice(["ipd", "uhid"])))
        kw, rel = kin(r, p, age=None if ward == "SNCU" else page)
        att = r.random() < 0.7
        if fmt == "sbar":
            out.append("Bed %d – %s, %s – %s %s" % (bed, who, age_s, idl, idv))
            out.append("S: %s" % dxs)
            out.append("B: %s" % r.choice(["k/c/o HTN, T2DM", "G2P1L1, LSCS", "no known comorbidities", "COPD, ex-smoker", "Down syndrome, lives with family"]))
            out.append("A: BP %s, PR %d, T %.1f, %s" % (bp(r), r.randint(58, 124), r.uniform(36.4, 39.2), r.choice(["Braden 14", "GCS 14/15", "pain 3/10", "Morse fall score 45"])))
            out.append("R: %s" % r.choice(["repeat KFT at 6 am", "watch for bleeding PV", "inform Dr on call if SpO2 < 92", "strict I/O", "turn 2-hrly"]))
            if att:
                out.append("Attendant: %s %s%s" % (kw, PER(St.N(rel.first)), (" (%s)" % PH(fmt_mobile(r))) if r.random() < 0.6 else ""))
            out.append("")
        elif fmt == "line":
            seg = ["B%d" % bed, who, age_s, "%s %s" % (idl, idv), dxs]
            if att:
                seg.append("%s %s %s" % (kw, PER(St.N(rel.later(r))), PH(fmt_mobile(r)) if r.random() < 0.5 else "at bedside"))
            out.append(r.choice([" | ", " ; ", " / "]).join(seg))
        else:
            s = "Bed %d: %s (%s, %s %s) – %s. " % (bed, who, age_s, idl, idv, dxs)
            if att:
                s += "%s %s is staying overnight" % (kw.capitalize(), PER(St.N(rel.first)))
                s += (", phone %s. " % PH(fmt_mobile(r))) if r.random() < 0.5 else ". "
            if p.last and r.random() < 0.5 and ward != "SNCU":
                s += "%s%s refused dinner, ate a banana at 11 pm." % (hon_en(r, p, 0), PER(St.N(p.last)))
            out.append(s)
    n1, n2 = mk(r, gender="f", pool=["klc", "klh", "tnc", "mz", "kh", "gc", "bn"]), mk(r)
    doc = mk(r)
    if fmt != "sbar":
        out.append("")
    out.append(r.choice(["%s%s informed about K+ at 03:10, advised repeat sample." % (DR(r), PER(doc.full)),
                         "Night: crash cart checked, O2 cylinder changed. %s%s (PG) reviewed bed %d." % (DR(r), PER(doc.later(r)), r.randint(1, 40)),
                         "Narcotic register tallied. Glucometer batch %s." % batch_no(r)]))
    out.append("%s %s%s   %s %s%s" % (St.F("Given by", "Handed over by", "H/O by"), r.choice(["Sr. ", "SN ", "NO ", ""]), PER(n1.full),
                                       St.F("Taken by", "Received by", "T/O"), r.choice(["Br. ", "Sr. ", "", "NO "]), PER(n2.first)))
    return "\n".join(out)


def en_pharmacy_slip(r):
    St = Style(r)
    pat = mk(r)
    doc = mk(r)
    fac = r.choice(["Pradhan Mantri Bhartiya Janaushadhi Kendra", "Hospital Pharmacy (OPD Counter 3)", "AMRIT Pharmacy", "Free Drug Distribution Counter", "%s Dispensary" % r.choice(FAC_EN).split(",")[0]])
    out = [fac.upper() if St.case == "upper" else fac, St.L("Dispensing Slip", "Drug Issue Slip", "Cash Memo / Dispensing Record", "DISPENSING SLIP")]
    out.append("%s%s    %s%s" % (St.F("Bill No", "Bill no.", "Invoice"), bill_no(r), St.F("Date", "Dt"), fdate(r, ev_date(r))))
    out.append("%s%s%s    %s%d/%s" % (St.F("Patient", "Pt. Name", "Name"), hon_en(r, pat, 0.6), PER(St.N(pat.full)), St.F("Age/Sex", "A/S"), r.randint(1, 90), "M" if pat.g == "m" else "F"))
    out.append("%s%s" % (St.F("UHID", "OPD No.", "Reg No", "CR No"), MR(gen_mrn(r, r.choice(["uhid", "opd", "cr12"])))))
    if r.random() < 0.6:
        out.append("%s%s" % (St.F("Mobile", "Mob.", "Ph", "Contact no"), PH(fmt_mobile(r))))
    out.append("%s%s%s" % (St.F("Prescriber", "Prescribed by", "Ref. Dr", "Doctor"), DR(r), PER(St.N(doc.full))))
    if r.random() < 0.35:
        out.append("%s%s" % (St.F("ABHA", "ABHA addr"), ABA(gen_abha_addr(r, pat))))
    elif r.random() < 0.45:
        out.append("%s%s" % (St.F("ABHA No", "ABHA number"), ABN(gen_abha(r))))
    if r.random() < 0.35:
        a, pin = address(r, pat, "in")
        out.append("%s%s" % (St.F("Home delivery to", "Deliver to", "Address"), ADR(St.N(a))))
    out.append("")
    cols = r.choice(["Sl | Item | Batch | Exp | Qty | Amt", "S.No\tDrug\tBatch\tExpiry\tQty", "#  Medicine                Batch     Exp    Qty"])
    out.append(cols)
    sep = " | " if "|" in cols else ("\t" if "\t" in cols else "  ")
    for i, (dn, dose) in enumerate(r.sample(DRUGS, r.randint(3, 6)), 1):
        m_ = re.match(r"\d+(\.\d+)? ?(mg|mcg|g|ml|IU|units)", dose)
        out.append(sep.join([str(i), ("%s %s" % (dn, m_.group(0))) if m_ else dn, batch_no(r), exp_date(r), str(r.choice([10, 14, 15, 28, 30, 60, 1]))] + (["%.2f" % r.uniform(8, 480)] if "Amt" in cols else [])))
    out.append("")
    if r.random() < 0.4:
        payer = mk(r, group=pat.group, region=pat.region)
        out.append("Oncology/biologic issue, bill ₹%s paid in cash – PAN of payer %s: %s" % (inr(r.randint(200500, 390000)), PER(St.N(payer.full)), PANE(gen_pan(r, payer))))
    if r.random() < 0.5:
        out.append(r.choice(["Counselled: take %s after food. Do not stop Ecosprin without asking the doctor." % r.choice(DRUGS)[0],
                             "Short supply: %s – substitute given with consent of %s%s." % (r.choice(DRUGS)[0], DR(r), PER(doc.later(r))),
                             "Cold chain item (insulin) – told to keep in fridge door."]))
    if r.random() < 0.5:
        kw, rel = kin(r, pat)
        out.append("Collected by %s %s%s" % (kw, PER(St.N(rel.first if r.random() < 0.5 else rel.full)), (", ID Voter card %s" % VOT(gen_epic(r))) if r.random() < 0.3 else ""))
    out.append("%s%s" % (St.F("Dispensed by", "Pharmacist", "Issued by"), PER(St.N(mk(r).first))))
    return "\n".join(out)


def en_mental_health_opd(r):
    pat = mk(r)
    age = r.randint(16, 70)
    he, his, him, He, His = pron(pat)
    kw, rel = kin(r, pat, age=age)
    psy = mk(r)
    out = []
    out.append(r.choice(["MENTAL HEALTH OPD – District Mental Health Programme", "Psychiatry OPD follow-up", "DMHP clinic, %s" % small_fac(r, pat.region), "Dept. of Psychiatry – New case"]))
    out.append("%s %s    %s" % (r.choice(["Reg. No.", "Psy OPD no", "File no:"]), MR(gen_mrn(r, "mh")), fdate(r, ev_date(r))))
    out.append("")
    marital = r.choice(["married", "unmarried", "widowed", "separated"])
    intro = "%d-year-old %s %s, %s, brought by %s %s %s" % (age, marital, "man" if pat.g == "m" else "woman", occupation(r, pat), his, kw, PER(rel.first if r.random() < 0.6 else rel.full))
    out.append("%s%s – %s." % (hon_en(r, pat, 0.5), PER(pat.full), intro))
    if r.random() < 0.3:
        out.append("DOB %s (as per Aadhaar)." % DB(fdate(r, dob_for(r, age))))
    if r.random() < 0.55:
        out.append("Resides at %s with %s." % (ADR(address(r, pat, r.choice(["in", "none"]))[0]), r.choice(["joint family", "spouse and two children", "elderly parents", "alone"])))
    cc = r.choice([
        ("low mood, crying spells and early-morning waking for %d months" % r.randint(2, 9), "PHQ-9 %d" % r.randint(10, 24), "Moderate depressive episode (F32.1)",
         "kempt, PMR present, affect restricted", ["Tab. Escitalopram", "Tab. Sertraline", "Tab. Mirtazapine"]),
        ("hearing voices commenting on %s actions, suspicious of neighbours" % his, "PANSS not done", "Schizophrenia, paranoid (F20.0)",
         "guarded, hallucinatory behaviour +", ["Tab. Olanzapine", "Tab. Quetiapine"]),
        ("decreased need for sleep, overspending, talking fast", "YMRS %d" % r.randint(14, 34), "Bipolar affective disorder, current episode manic (F31.1)",
         "elated, pressure of speech", ["Tab. Lithosun SR", "Tab. Divalproex"]),
        ("daily alcohol use since age 19, tremors in the morning", "AUDIT %d" % r.randint(16, 32), "Alcohol dependence syndrome (F10.2)",
         "anxious, tremulous, no hallucinations", ["Tab. Naltrexone", "Tab. Thiamine"]),
        ("palpitations and fear of dying in crowded buses", "GAD-7 %d" % r.randint(10, 19), "Panic disorder (F41.0)",
         "anxious, rapport established", ["Tab. Escitalopram", "Tab. Clonazepam"]),
        ("repeated hand-washing up to 60 times a day", "Y-BOCS %d" % r.randint(18, 32), "OCD (F42)",
         "kempt, obsessions about contamination", ["Tab. Fluoxetine", "Tab. Sertraline"])])
    out.append("C/o %s. %s." % (cc[0], cc[1]))
    out.append("No h/o head injury or seizures. %s" % r.choice(["Family h/o depression in maternal aunt.", "Father had alcohol use.", "No family history.", "Elder brother on treatment for schizophrenia at %s." % r.choice(FAC_EN)]))
    out.append("MSE: %s; insight grade %d. Risk: %s." % (cc[3], r.randint(1, 5), r.choice(["no current suicidal ideation", "passive death wishes, no plan", "DSH 2 months ago (OP poisoning), safety plan made"])))
    out.append("Impression: %s." % cc[2])
    _dn = r.choice(cc[4])
    d1 = next(d for d in PSY_DRUGS if d[0] == _dn)
    out.append("Plan: %s %s; psychoeducation given to %s and %s %s. Tele-MANAS 14416 explained. %s" % (
        d1[0], d1[1], PER(pat.first), kw, PER(rel.first), r.choice(["Review 2 weeks.", "Review after 1 month with TFT, LFT.", "Refer to de-addiction ward if willing.", "Hamilton scale next visit."])))
    if r.random() < 0.6:
        out.append("Contact: %s (%s)%s" % (PH(fmt_mobile(r)), kw, (", alt %s" % PH(fmt_mobile(r))) if r.random() < 0.3 else ""))
    if r.random() < 0.65:
        out.append("Disability certificate (UDID) process explained; Aadhaar %s noted for application." % AAD(fmt_aadhaar(r)))
    out.append("%s%s, %s" % (DR(r), PER(psy.full), r.choice(["Psychiatrist", "MD Psychiatry", "Clinical Psychologist (M.Phil)", "Medical Officer (Mental Health)"])))
    return "\n".join(out)


def en_dialysis_register(r):
    d = ev_date(r)
    out = [r.choice(["NEPHROLOGY – HAEMODIALYSIS UNIT DAILY REGISTER", "PMNDP Dialysis Centre – Session Register", "HD Unit Register", "DIALYSIS LOG"]) +
           "    Date: %s    Shift: %s" % (fdate(r, d), r.choice(["I", "II", "III", "Morning", "Evening"]))]
    sep = r.choice([" | ", "\t", " ; ", "  "])
    cols = ["S.No", "Reg No", "Patient name", "Age/Sex", "Access", "Pre wt", "Post wt", "UF(L)", "Machine", "Hb", "Tech"]
    out.append(sep.join(cols))
    region_pool = r.choice([None, NORTH_GROUPS, ["klc", "klh", "klm", "tn", "tnc", "te", "ka"], ["mz", "mn", "nl", "kh", "as", "sk"], ["bn", "bnm", "od", "tr"]])
    techs = [mk(r, pool=region_pool) for _ in range(2)]
    for i in range(r.randint(1, 5)):
        p = mk(r, pool=region_pool)
        pre = r.uniform(42, 88)
        uf = r.uniform(1.0, 3.8)
        nm = p.full if r.random() < 0.7 else (p.full.upper())
        row = [str(r.randint(1, 30)), MR(gen_mrn(r, "dialysis")), PER(nm), "%d/%s" % (r.randint(16, 80), "M" if p.g == "m" else "F"),
               r.choice(["AVF L", "AVF R (Brescia-Cimino)", "IJV cath", "Femoral", "AVG"]), "%.1f" % pre, "%.1f" % (pre - uf), "%.1f" % uf,
               "M-%d" % r.randint(1, 12), "%.1f" % r.uniform(6.8, 12.5), PER(r.choice(techs).first)]
        out.append(sep.join(row))
    if r.random() < 0.8:
        p = mk(r, pool=region_pool)
        kw, rel = kin(r, p)
        out.append("")
        out.append(r.choice([
            "Remarks: %s c/o cramps at 3rd hour, NS 200 ml bolus. Inj. EPO 4000 IU given. Next session Thu." % PER(p.full),
            "Remarks: new IJV catheter (Seldinger) for %s inserted by %s%s. HBsAg/HCV/HIV non-reactive. %s %s informed on %s." % (
                PER(p.full), DR(r), PER(mk(r).full), kw.capitalize(), PER(rel.first), PH(fmt_mobile(r))),
            "Remarks: %s absent – called %s %s on %s, said bus strike." % (PER(p.full), kw, PER(rel.first), PH(fmt_mobile(r))),
            "Kt/V for %s 1.2, URR 66%%. PM-JAY package HD-%d claimed, ID %s." % (PER(p.full), r.randint(1, 9), MR(gen_mrn(r, "pmjay"))),
            "PM-JAY e-KYC for %s done with Aadhaar %s; beneficiary ID %s. Lives at %s." % (PER(p.full), AAD(fmt_aadhaar(r)), MR(gen_mrn(r, "pmjay")), ADR(address(r, p, "in")[0])),
        ]))
    return "\n".join(out)


SCHOOLS_EN = ["Netaji Subhas Chandra Bose Govt. Girls Sr. Sec. School", "Kendriya Vidyalaya No. 2", "PM SHRI Govt. Middle School", "Sant Kabir Public School",
              "Rani Laxmi Bai Govt. Girls Inter College", "St. Aloysius High School", "Swami Vivekananda Vidya Mandir", "Maulana Abul Kalam Azad Urdu High School",
              "Dr. Zakir Husain Memorial School", "Savitribai Phule Kanya Shala", "Govt. Higher Secondary School", "Jawahar Navodaya Vidyalaya",
              "Don Bosco High School", "Mother Teresa Convent School", "Guru Nanak Khalsa Sr. Sec. School"]


def en_school_health(r):
    St = Style(r)
    region_pool = r.choice([NORTH_GROUPS, ["klc", "klh", "klm"], ["tn", "tnc"], ["mz", "kh", "nl"], ["mh", "gj"], ["gc", "gk"], ["jk", "kp"], ["od", "bn"], ["te", "ka"]])
    mo = mk(r)
    anm = mk(r, gender="f")
    school = r.choice(SCHOOLS_EN)
    d = ev_date(r, 7, 10)
    if r.random() < 0.55:
        child = mk(r, pool=region_pool)
        father = relative(r, child, "m", True)
        mother = relative(r, child, "f", True)
        age = r.randint(6, 16)
        out = [St.L("RASHTRIYA BAL SWASTHYA KARYAKRAM (RBSK) – SCHOOL SCREENING CARD", "RBSK Child Health Screening Card", "School Health Check-up")]
        out.append("%s%s Block, Team %s" % (St.F("Mobile Health Team"), addr_city(child.region), r.choice("ABC")))
        out.append("%s%s" % (St.F("School", "Name of school"), school))
        out.append("%s%s" % (St.F("Date of screening", "Screening date"), fdate(r, d)))
        if r.random() < 0.4:
            out.append("%s%s" % (St.F("Card No", "Screening ID"), MR(gen_mrn(r, "rbsk"))))
        out.append("%s%s    %s%d-%s    %s%d" % (St.F("Name of student", "Child's name", "Name"), PER(St.N(child.full)), St.F("Class/Sec"), r.randint(1, 10), r.choice("ABCD"), St.F("Roll No."), r.randint(1, 48)))
        out.append("%s%s    %s%d y    %s%s" % (St.F("DOB", "Date of birth", "D.O.B."), DB(fdate(r, dob_for(r, age))), St.F("Age"), age, St.F("Sex"), "M" if child.g == "m" else "F"))
        out.append("%s%s%s" % (St.F("Father's name", "Father", "F/N"), hon_en(r, father, 0.6), PER(St.N(father.full))))
        if r.random() < 0.7:
            out.append("%s%s" % (St.F("Mother's name", "Mother", "M/N"), PER(St.N(mother.full if r.random() < 0.6 else mother.first))))
        out.append("%s%s" % (St.F("Parent mobile", "Contact", "Mob"), PH(fmt_mobile(r))))
        if r.random() < 0.65:
            out.append("%s%s" % (St.F("Aadhaar (child)", "Child Aadhaar", "UID"), AAD(fmt_aadhaar(r))))
        if r.random() < 0.5:
            out.append("%s%s" % (St.F("ABHA created", "ABHA No."), ABN(gen_abha(r))))
        if r.random() < 0.75:
            a, pin = address(r, child, r.choice(["in", "sep", "sep", "none"]))
            out.append("%s%s" % (St.F("Address", "Adress", "Residence"), ADR(St.N(a))))
            if r.random() < 0.85:
                out.append("%s%s" % (St.F("PIN", "Pincode", "Pin code"), PIN(fmt_pin(r, pin))))
        ht = r.randint(110, 165)
        wt = r.uniform(18, 48)
        out.append("Findings: Ht %d cm, Wt %.1f kg, BMI %.1f. Snellen %s. %s. Hb (HemoCue) %.1f." % (ht, wt, wt / (ht / 100) ** 2, r.choice(["6/6 both", "6/12 R, 6/9 L", "6/18 both"]),
                                                                                              r.choice(["Dental caries (D2)", "Pediculosis", "Ichthyosis", "No defect", "Undescended testis R", "Squint L eye"]), r.uniform(8.2, 13.4)))
        out.append("4D: %s. Action: %s" % (r.choice(["Deficiency – anaemia", "Disease – dental caries", "Developmental delay – speech", "Defect – refractive error"]),
                                          r.choice(["WIFS IFA weekly", "referred to DEIC, %s" % small_fac(r, child.region), "spectacles under RBSK", "Albendazole 400 mg given"])))
        out.append("%s%s%s (MO, RBSK)   ANM: %s" % (St.F("Screened by"), DR(r), PER(mo.full), PER(anm.first)))
    else:
        n = r.randint(32, 64)
        out = ["%s – %s, Class %d" % (St.L("School health visit", "RBSK visit report", "SCHOOL HEALTH CAMP"), school, r.randint(1, 10)),
               "%s. %d children screened by %s%s and ANM %s. Referred:" % (fdate(r, d), n, DR(r), PER(mo.full), PER(anm.first))]
        for i in range(1, r.randint(3, 5)):
            c = mk(r, pool=region_pool)
            f = relative(r, c, "m", True)
            age = r.randint(5, 15)
            bits = ["%d) %s %s %s" % (i, PER(St.N(c.full if r.random() < 0.7 else c.first)), "S/o" if c.g == "m" else "D/o", PER(f.full))]
            if r.random() < 0.6:
                bits.append("DOB %s" % DB(fdate(r, dob_for(r, age))))
            else:
                bits.append("%d yrs" % age)
            bits.append("roll %d" % r.randint(1, 50))
            bits.append(r.choice(["squint", "CHD murmur – ECHO at %s" % r.choice(FAC_EN), "SAM, MUAC 11.2", "hearing loss L", "cleft lip repaired, speech therapy", "Down syndrome features, DEIC", "scabies"]))
            if r.random() < 0.4:
                bits.append("father mob %s" % PH(fmt_mobile(r)))
            out.append(", ".join(bits))
        out.append("Albendazole given to all; WIFS stock %s, batch %s." % (r.choice(["adequate", "short"]), batch_no(r)))
    return "\n".join(out)


def en_mlc_intimation(r):
    pat = mk(r, gender=r.choice("mmmf"))
    age = r.randint(14, 70)
    he, his, him, He, His = pron(pat)
    fac = r.choice(FAC_EN + [small_fac(r, pat.region)] * 8)
    city = addr_city(pat.region)
    father = relative(r, pat, "m", True)
    a, pin = address(r, pat, r.choice(["in", "none"]))
    broughtby_kw, bb = kin(r, pat, age=age)
    cmo = mk(r)
    d = ev_date(r)
    so = r.choice(["S/o", "s/o"]) if pat.g == "m" else r.choice(["D/o", "W/o"])
    mech = r.choice(["alleged road traffic accident (two-wheeler vs truck)", "alleged physical assault with a blunt object", "alleged consumption of an unknown poison (?OP compound)",
                     "alleged fall from height at a construction site", "alleged burns (kerosene) at home", "alleged dog bite while returning from school", "alleged hanging, found by family"])
    out = []
    out.append("To,\nThe SHO,\nPolice Station %s,\n%s\n" % (r.choice(["Sadar", "Kotwali", "City", "Civil Lines", "Sector 17", "Saddar Bazar"]), city))
    out.append("Sub: %s\n" % r.choice(["Intimation regarding MLC case", "MLC intimation – kindly depute IO", "Information of medico-legal case"]))
    out.append("Sir,\n")
    s = "This is to inform you that one %s%s, %s %s, aged about %d years, R/o %s, was brought to the Casualty of %s on %s at %s by %s" % (
        hon_en(r, pat, 0.7), PER(pat.full), so, PER(father.full), age, ADR(a), fac, fdate(r, d), ampm(r),
        r.choice(["%s %s" % (broughtby_kw, PER(bb.full)), "a passer-by, %s%s" % (hon_en(r, mk(r, gender='m'), 0.3), PER(mk(r).full)), "108 ambulance (EMT %s)" % PER(mk(r).first)]))
    s += " with history of %s. " % mech
    if r.random() < 0.3:
        s += "DOB as stated by attendant: %s. " % DB(fdate(r, dob_for(r, age)))
    out.append(s)
    out.append("MLC No. %s has been registered. %s is %s; %s. CR No. %s.\n" % (
        MR(gen_mrn(r, "mlc")), He, r.choice(["conscious and oriented", "drowsy, GCS 10/15", "unconscious, on ventilator", "stable"]),
        r.choice(["injuries are being documented", "stomach wash done and sample preserved", "X-ray shows Colles' fracture of the left wrist", "patient is fit for statement"]), MR(gen_mrn(r, "cr12"))))
    if r.random() < 0.85:
        out.append("ID produced by attendant: %s.\n" % r.choice(["Voter ID %s" % VOT(gen_epic(r)), "Aadhaar %s" % AAD(fmt_aadhaar(r)), "EPIC No. %s" % VOT(gen_epic(r))]))
    if r.random() < 0.5:
        out.append("Attendant contact: %s.\n" % PH(fmt_mobile(r)))
    out.append("Kindly depute an IO for further necessary action.\n")
    out.append("\n%s\n%s%s\n%s\n%s, %s\n" % (r.choice(["Yours faithfully,", "Regards,", ""]), DR(r), PER(cmo.full), r.choice(["CMO on duty", "Casualty Medical Officer", "EMO"]), fac.split(",")[0], fdate(r, d)))
    if r.random() < 0.6:
        cop = mk(r, gender=r.choice("mmf"), pool=NORTH_GROUPS if r.random() < 0.6 else None)
        out.append("Received by %s %s, No. %d/%s, at %s.\n" % (r.choice(["HC", "Constable", "ASI", "SI"]), PER(cop.full), r.randint(100, 2999), r.choice(["PTL", "SDR", "CP", "BTD"]), tstamp(r)))
    return "".join(out).rstrip() + "\n"


def en_blood_bank_slip(r):
    St = Style(r)
    donor = mk(r)
    age = r.randint(18, 60)
    kw = r.choice(["S/o", "D/o", "W/o"]) if donor.g == "f" else "S/o"
    if donor.g == "f" and kw == "S/o":
        kw = "D/o"
    rel = relative(r, donor, "m", True)
    out = [St.L("BLOOD CENTRE – DONOR REGISTRATION SLIP", "Blood Bank | Donor Questionnaire & Slip", "Voluntary Blood Donation Camp – Donor Card"),
           "%s%s" % (St.F("Licensed blood centre", "Centre"), r.choice(["Indian Red Cross Society Blood Centre", "Rotary Blood Bank", r.choice(FAC_EN).split(",")[0] + " Blood Centre"]))]
    if r.random() < 0.5:
        out.append("%s%s" % (St.F("Camp at", "Camp venue"), r.choice(["Guru Nanak Khalsa College", "St. Xavier's College", "Rajiv Gandhi Bhawan", "Maharaja Agrasen Dharamshala", "SBI Regional Office"])))
    out.append("%s%s    %s%s" % (St.F("Donor ID", "Donor No.", "Regn"), MR(gen_mrn(r, "donor")), St.F("Date"), fdate(r, ev_date(r))))
    out.append("%s%s%s    %s %s" % (St.F("Name", "Donor name"), hon_en(r, donor, 0.6), PER(St.N(donor.full)), kw, PER(St.N(rel.full))))
    out.append("%s%s    %s%d    %s%s" % (St.F("DOB", "Date of Birth"), DB(fdate(r, dob_for(r, age))), St.F("Age"), age, St.F("Sex"), "Male" if donor.g == "m" else "Female"))
    out.append("%s%s" % (St.F("Mobile", "Mob no", "Phone"), PH(fmt_mobile(r))))
    if r.random() < 0.3:
        out.append("%s%s" % (St.F("Phone (Res.)", "Landline"), PH(fmt_landline(r, std_of(r, donor.region)))))
    if r.random() < 0.5:
        out.append("%s%s" % (St.F("Email", "E-mail"), EM(gen_email(r, donor))))
    if r.random() < 0.2:
        out.append("%s%s" % (St.F("ABHA"), ABA(gen_abha_addr(r, donor))))
    a, pin = address(r, donor, r.choice(["in", "sep"]))
    out.append("%s%s" % (St.F("Address", "Addr", "Present address"), ADR(St.N(a))))
    if r.random() < 0.85 and pin not in a:
        out.append("%s%s" % (St.F("PIN", "Pin"), PIN(fmt_pin(r, pin))))
    if r.random() < 0.7:
        out.append("%s%s" % (St.F("ID proof", "Photo ID"), r.choice(["Voter ID " + VOT(gen_epic(r)), "Aadhaar " + AAD(fmt_aadhaar(r)), "EPIC " + VOT(gen_epic(r))])))
    if r.random() < 0.5:
        pt = mk(r)
        out.append("%s%s – for patient %s, CR No. %s, %s" % (St.F("Donation type"), "Replacement", PER(St.N(pt.full)), MR(gen_mrn(r, "cr12")), r.choice(FAC_EN)))
    else:
        out.append("%s%s (donation no. %d)" % (St.F("Donation type"), "Voluntary", r.randint(1, 40)))
    out.append("Hb %.1f g/dL | BP %s | Wt %d kg | Pulse %d | Temp %.1f F" % (r.uniform(12.5, 16.2), bp(r), r.randint(48, 92), r.randint(64, 96), r.uniform(97.2, 99.0)))
    out.append("%s%s  |  %s%s  |  %s" % (St.F("Bag No", "Bag no."), bag_no(r), St.F("Segment"), digits(r, 6), r.choice(["Triple 450 ml", "Double 350 ml", "SAGM quadruple"])))
    if r.random() < 0.3:
        out.append("Deferred: %s" % r.choice(["Hb below 12.5", "tattoo 4 months back", "on Tab. Isotretinoin"]))
    out.append("%s%s%s" % (St.F("Medical officer", "MO", "Phlebotomy by"), DR(r), PER(St.N(mk(r).full))))
    return "\n".join(out)


def en_jsy_payment(r):
    St = Style(r)
    mom = mk(r, gender="f", pool=NORTH_GROUPS + ["od", "bn", "as", "jh", "gd"])
    hus = relative(r, mom, "m", True)
    asha = mk(r, gender="f", group=mom.group, region=mom.region)
    fac = small_fac(r, mom.region)
    dd = ev_date(r, 4, 9)
    adm, dis = (dd[0], dd[1], max(1, dd[2] - r.randint(0, 1))), (dd[0], dd[1], min(28, dd[2] + r.randint(2, 4)))
    out = [St.L("JSY Payment Note", "JANANI SURAKSHA YOJANA – BENEFICIARY PAYMENT", "JSY/ JSSK incentive sheet") + " — " + fac]
    out.append("%s%s%s %s %s" % (St.F("Beneficiary", "Mother's name", "Name of beneficiary"), hon_en(r, mom, 0.4), PER(St.N(mom.full)), r.choice(["W/o", "w/o", "wife of"]), PER(St.N(hus.full))))
    out.append("%s%d    %s%s" % (St.F("Age"), r.randint(18, 38), St.F("Category"), r.choice(["Rural / BPL", "SC", "ST", "Rural, LPS state"])))
    out.append("%s%s" % (St.F("RCH ID", "MCTS/RCH No", "RCH No."), MR(gen_mrn(r, "rch12"))))
    a, pin = address(r, mom, r.choice(["in", "none", "sep", "sep"]))
    out.append("%s%s" % (St.F("Address", "Village/Address"), ADR(St.N(a))))
    if r.random() < 0.85:
        out.append("%s%s" % (St.F("PIN", "Pincode"), PIN(fmt_pin(r, pin))))
    out.append("%s%s, %s" % (St.F("Delivery"), r.choice(["Normal", "LSCS", "Assisted (outlet forceps)"]), r.choice(["live male baby", "live female baby", "twins, both live"])))
    if r.random() < 0.6:
        out.append("%s%s    %s%.2f kg" % (St.F("Baby DOB", "DOB of baby"), DB(fdate(r, dd)), St.F("Birth wt"), r.uniform(1.9, 3.6)))
    out.append("%s%s    %s%s" % (St.F("Admitted"), fdate(r, adm), St.F("Discharged"), fdate(r, dis)))
    out.append("%s%s" % (St.F("Mobile"), PH(fmt_mobile(r))))
    if r.random() < 0.3:
        out.append("%s%s" % (St.F("ID proof", "EPIC"), VOT(gen_epic(r))))
    if r.random() < 0.5:
        out.append("%s%s" % (St.F("ABHA (mother)", "ABHA"), ABN(gen_abha(r))))
    out.append("%s₹1400 (mother)  |  ₹600 ASHA incentive – %s %s" % (St.F("Amount"), r.choice(["ASHA", "Asha"]), PER(St.N(asha.full))))
    if r.random() < 0.85:
        out.append("%s%s" % (St.F("Bank seeding", "DBT"), "Aadhaar-seeded a/c, Aadhaar %s" % AAD(fmt_aadhaar(r))))
    else:
        out.append("%s%s" % (St.F("Bank seeding", "DBT"), "IFSC %s%s, a/c verified" % (r.choice(["SBIN", "PUNB", "BKID", "UBIN"]), "0" + digits(r, 6))))
    out.append("%s%s    %s%s" % (St.F("PFMS ref", "Txn ID"), gen_fail12(r), St.F("Date of payment"), fdate(r, ev_date(r, 10, 10))))
    out.append("%s%s%s (BCPM)   %s%s%s (MOIC)" % (St.F("Verified by"), "", PER(St.N(mk(r).full)), St.F("Approved"), DR(r), PER(St.N(mk(r).later(r)))))
    return "\n".join(out)


HELP_CASES = [
    ("child has loose motions 6 times since morning, no blood, drinking well", "ORS + zinc 20 mg x 14 days, danger signs explained", (1, 5)),
    ("pregnant 7 months, blurring of vision and headache", "advised immediate visit to %s; 108 ambulance number given", (19, 36)),
    ("fever 102 for 3 days with body ache, platelet report 88,000", "plenty of fluids, paracetamol only, no ibuprofen; repeat CBC tomorrow", (8, 60)),
    ("snake bite on the foot 40 minutes ago, swelling", "do not tie tourniquet, go to nearest CHC for ASV", (9, 65)),
    ("not passing urine since last night, lower abdomen swollen", "advised immediate visit to %s; 108 ambulance number given", (62, 85)),
    ("wants to know where to get the HPV vaccine", "nearest session site and eligibility (9-14 yrs) explained", (9, 14)),
    ("TB medicines finished, next box not received", "linked to Ni-kshay helpline, ID %s", (20, 70)),
    ("feeling hopeless after exam result, cannot sleep", "call transferred to Tele-MANAS 14416", (16, 22)),
]


def en_helpline_104(r):
    St = Style(r)
    caller = mk(r)
    cz = r.choice(HELP_CASES)
    page = r.randint(*cz[2])
    pat_self = page >= 18 and r.random() < 0.4
    if pat_self:
        pat, relw = caller, "self"
    elif page < 18:
        relw, g = r.choice([("son", "m"), ("daughter", "f")])
        pat = relative(r, caller, g, True)
    elif page > 55:
        relw, g = r.choice([("father", "m"), ("mother", "f")])
        pat = relative(r, caller, g, True)
    else:
        relw, pat = kin(r, caller, age=page)
    agent = mk(r)
    out = ["%s %s %s" % (St.L("104 Health Helpline – Call Record", "104 AROGYA HELPLINE CALL LOG", "Health Helpline 104 :: ticket"), r.choice(["|", "-", "::"]), fdate(r, ev_date(r)))]
    out.append("%s%s | %s%s | %s%02d:%02d" % (St.F("Call ID", "Ticket"), MR(gen_mrn(r, "call")), St.F("Time"), tstamp(r), St.F("Duration"), r.randint(1, 18), r.randint(0, 59)))
    out.append("%s%s (%s) | %s%s" % (St.F("Caller", "Caller name"), PER(St.N(caller.full)), relw if relw == "self" else "for " + relw, St.F("Caller no", "CLI", "Number"),
                                       PH(fmt_mobile(r) if r.random() < 0.8 else fmt_landline(r, std_of(r, caller.region)))))
    if not pat_self:
        out.append("%s%s, %d %s" % (St.F("Patient"), PER(St.N(pat.full if r.random() < 0.6 else pat.first)), page, "yrs"))
        if r.random() < 0.25:
            out.append("%s%s" % (St.F("DOB", "Patient DOB"), DB(fdate(r, dob_for(r, page)))))
    out.append("%s%s" % (St.F("District"), addr_city(caller.region)))
    if r.random() < 0.6:
        a, pin = address(r, caller, r.choice(["in", "sep"]))
        out.append("%s%s" % (St.F("Location", "Address", "Caller address"), ADR(a)))
        if r.random() < 0.85:
            out.append("%s%s" % (St.F("Pincode", "PIN"), PIN(fmt_pin(r, pin))))
    out.append("%s%s" % (St.F("Concern", "Complaint"), cz[0]))
    adv = cz[1]
    if "%s" in adv:
        adv = adv % (small_fac(r, caller.region) if "visit" in adv else MR(gen_mrn(r, "nikshay")))
    out.append("%s%s" % (St.F("Advice", "Action taken"), adv))
    if r.random() < 0.45:
        out.append("%s%s" % (St.F("ABHA", "ABHA address"), ABA(gen_abha_addr(r, pat))))
    if r.random() < 0.3:
        out.append("%s%s" % (St.F("Email for report"), EM(gen_email(r, caller))))
    if r.random() < 0.4:
        out.append("%s%s" % (St.F("Aadhaar (for ABHA creation)"), AAD(fmt_aadhaar(r))))
    if r.random() < 0.4:
        out.append("%s%s" % (St.F("Callback"), "requested on alternate number %s" % PH(fmt_mobile(r))))
    out.append("%s%s (HCP-%d)" % (St.F("Agent", "Counsellor", "Attended by"), PER(St.N(agent.full)), r.randint(100, 999)))
    return "\n".join(out)


def en_referral_email(r):
    doc = mk(r)
    to = mk(r)
    pat = mk(r)
    age = r.randint(4, 80)
    he, his, him, He, His = pron(pat)
    cx, dx = r.choice(COMPLAINTS)
    out = []
    out.append("From: %s%s <%s>" % (DR(r), PER(doc.full), EM(gen_email(r, doc, True))))
    out.append("To: %s" % EM(gen_email(r, to, True)))
    if r.random() < 0.4:
        cc = mk(r)
        out.append("Cc: %s" % EM(gen_email(r, cc)))
    out.append("Date: %s, %s" % (fdate(r, ev_date(r), "dMonY"), tstamp(r)))
    out.append("Subject: %s" % r.choice(["Referral – %s, %d%s, ? %s" % (PER(pat.full), age, "M" if pat.g == "m" else "F", dx), "Fwd: opinion please", "Re: patient transfer"]))
    out.append("")
    out.append("%s %s," % (r.choice(["Dear", "Hi", "Respected"]), "Dr. " + PER(to.later(r)) if r.random() < 0.7 else "Sir"))
    out.append("")
    out.append("Sending %s%s (%s%s) for your opinion. %s %d years old and has had %s. Our working diagnosis is %s." % (
        hon_en(r, pat, 0.5), PER(pat.full), r.choice(["UHID ", "our MRD ", "IP "]), MR(gen_mrn(r, r.choice(["uhid", "ipd"]))), He + " is", age, cx, dx))
    if r.random() < 0.7:
        out.append("DOB on the ABHA card is %s; ABHA number %s." % (DB(fdate(r, dob_for(r, age))), ABN(gen_abha(r))))
    else:
        out.append("ABHA: %s" % ABA(gen_abha_addr(r, pat)))
    out.append("Investigations: %s. %s." % (labs(r), r.choice(SIGNS)))
    kw, rel = kin(r, pat, age=age)
    out.append("%s %s %s will bring the films; %s number is %s." % (His, kw, PER(rel.full if r.random() < 0.5 else rel.first), "his" if rel.g == "m" else "her", PH(fmt_mobile(r))))
    if r.random() < 0.45:
        out.append("Family lives at %s." % ADR(address(r, pat, "in")[0]))
    out.append("Please let me know if %s should go to %s instead." % (PER(pat.later(r)), r.choice(FAC_EN)))
    out.append("")
    out.append(r.choice(["Regards,", "Thanks,", "Best,"]))
    out.append(PER(doc.first) if r.random() < 0.5 else "%s%s" % (DR(r), PER(doc.full)))
    out.append(small_fac(r, doc.region) + " | Mob " + PH(fmt_mobile(r)))
    return "\n".join(out)


def en_reimbursement_claim(r):
    St = Style(r)
    emp = mk(r)
    kw, pat = kin(r, emp)
    if r.random() < 0.3:
        kw, pat = "self", emp
    out = [St.L("MEDICAL REIMBURSEMENT CLAIM", "CGHS / Medical Reimbursement – Form MRC(S)", "Claim for reimbursement of medical expenses")]
    out.append("%s%s%s" % (St.F("1. Name of the Government servant", "1. Name of employee", "1) Claimant"), hon_en(r, emp, 0.5), PER(St.N(emp.full))))
    out.append("%s%s" % (St.F("2. Designation & office"), r.choice(["Assistant, Office of the Accountant General", "Lab Technician, ICAR-NDRI", "UDC, Dte. of Health Services", "Sr. Auditor, PAO"])))
    out.append("%s%s" % (St.F("3. PAN", "3. PAN No."), PANE(gen_pan(r, emp))))
    if kw in ("wife", "husband") and r.random() < 0.5:
        out.append("%s%s" % (St.F("   PAN of spouse (joint declaration)"), PANE(gen_pan(r, pat))))
    if r.random() < 0.6:
        out.append("%s%s" % (St.F("4. CGHS card / Beneficiary ID", "4. CGHS Ben. ID"), MR(gen_mrn(r, "cghs"))))
    a, pin = address(r, emp, r.choice(["in", "sep"]))
    out.append("%s%s" % (St.F("5. Residential address"), ADR(St.N(a))))
    if r.random() < 0.85:
        out.append("%s%s" % (St.F("   PIN"), PIN(fmt_pin(r, pin))))
    out.append("%s%s" % (St.F("6. Mobile / Email", "6. Contact"), PH(fmt_mobile(r)) + " / " + EM(gen_email(r, emp))))
    if r.random() < 0.4:
        out.append("%s%s" % (St.F("   Phone (R)"), PH(fmt_landline(r, std_of(r, emp.region)))))
    if r.random() < 0.8:
        out.append("%s%s" % (St.F("   Aadhaar (for e-KYC)"), AAD(fmt_aadhaar(r))))
    if kw == "self":
        out.append("%s%s" % (St.F("7. Name of patient & relationship"), "Self"))
    else:
        out.append("%s%s, %s" % (St.F("7. Name of patient & relationship"), PER(St.N(pat.full)), kw))
    if r.random() < 0.5:
        rng_age = {"father": (55, 85), "mother": (55, 85), "wife": (28, 60), "husband": (28, 60), "son": (1, 25), "daughter": (1, 25)}.get(kw, (20, 60))
        out.append("%s%s" % (St.F("   DOB of patient"), DB(fdate(r, dob_for(r, r.randint(*rng_age))))))
    out.append("%s%s, UHID %s" % (St.F("8. Hospital"), r.choice(FAC_EN), MR(gen_mrn(r, "uhid"))))
    out.append("%s%s" % (St.F("9. Treated by"), DR(r) + PER(St.N(mk(r).full))))
    out.append("%s%s to %s" % (St.F("10. Period of treatment"), fdate(r, ev_date(r, 1, 5)), fdate(r, ev_date(r, 6, 9))))
    out.append("%s%s" % (St.F("11. Diagnosis"), r.choice(COMPLAINTS)[1]))
    out.append("12. Bills: " + "; ".join("%s dt %s ₹%s" % (bill_no(r), fdate(r, ev_date(r), "dmy."), inr(r.randint(300, 98000))) for _ in range(r.randint(2, 4))))
    out.append("%s%s, IFSC %s" % (St.F("13. Bank"), r.choice(["SBI", "Canara Bank", "PNB", "Bank of Baroda"]), r.choice(["SBIN", "CNRB", "PUNB", "BARB"]) + "0" + digits(r, 6)))
    out.append("")
    out.append("I, %s, hereby declare that the statements above are true and the expenditure was actually incurred by me." % PER(St.N(emp.full)))
    out.append("Signature: ________   %s%s" % (St.F("Date"), fdate(r, ev_date(r, 9, 10))))
    return "\n".join(out)


# ==========================================================================
# HINDI GENERATORS (Devanagari)
# ==========================================================================
def hon_rel_hi(p):
    return ("पुत्र" if p.g == "m" else "पुत्री")


def hph(r, devn): return fmt_mobile(r, devn=devn)


def hi_asha_whatsapp(r):
    devn = r.random() < 0.35
    asha = mkh(r, gender="f")
    mo = mkh(r)
    pat = mkh(r, group=asha.group, gender="f", region=asha.region)
    hus = relative(r, pat, "m", True)
    d = ev_date(r)
    h = r.randint(7, 20)
    m = [r.randint(0, 40)]

    def hdr(who):
        m[0] = min(59, m[0] + r.randint(0, 5))
        return "[%d/%d/%02d, %d:%02d %s] %s: " % (d[2], d[1], d[0] % 100, (h - 1) % 12 + 1, m[0], "am" if h < 12 else "pm", who)
    at = r.choice(["%s आशा" % PER(asha.first), "आशा %s" % PER(asha.full), PER(asha.full), "%s (आशा बहू)" % PER(asha.first)])
    mt = r.choice(["डॉ. %s MO" % PER(mo.first), "डॉ. %s" % PER(mo.full), "MO साहब"])
    L = [hdr(at) + r.choice(["नमस्ते सर 🙏", "सर प्रणाम", "सर जरूरी है"])]
    scen = r.choice(["hrp", "newborn", "tb", "elder"])
    if scen == "hrp":
        L.append(hdr(at) + "सर एक हाई रिस्क केस है। %s पत्नी %s, उम्र %s, %s महीने का गर्भ" % (PER(pat.full), PER(hus.full), str(r.randint(18, 35)), str(r.randint(6, 9))))
        L.append(hdr(at) + "आज बीपी %s आया, Hb %.1f, पैरों में सूजन और सिर दर्द" % (bp(r), r.uniform(6.5, 10.5)))
        L.append(hdr(mt) + r.choice(["आज ही FRU भेजो, 108 बुला लो", "RCH आईडी भेजो", "MgSO4 लगा या नहीं?"]))
        L.append(hdr(at) + "RCH आईडी %s, मोबाइल %s" % (MR(dev(gen_mrn(r, "rch12")) if devn else gen_mrn(r, "rch12")), PH(hph(r, devn))))
        if r.random() < 0.6:
            L.append(hdr(at) + "पता: " + ADR(address_hi(r, pat, r.choice(["in", "none"]), devn)[0]))
        if r.random() < 0.55:
            L.append(hdr(at) + "JSY के लिए आधार %s" % AAD(fmt_aadhaar(r, devn=devn)))
        if r.random() < 0.6:
            sas = relative(r, pat, "f", True)
            L.append(hdr(at) + "सर सास %s मना कर रही हैं, कह रही हैं घर पर ही होगा" % PER(sas.first))
            L.append(hdr(mt) + "%s का नंबर दो, मैं बात करता हूँ" % PER(hus.first))
            L.append(hdr(at) + PH(hph(r, devn)))
    elif scen == "newborn":
        L.append(hdr(at) + "सर रात में घर पर डिलीवरी हुई, %s की बच्ची, वजन %.1f किलो" % (PER(pat.full), r.uniform(1.4, 2.4)))
        L.append(hdr(at) + "जन्म तिथि %s, दूध नहीं पी रही" % DB(hdate(r, d, devn)))
        L.append(hdr(mt) + "KMC कराओ, SNCU %s लेकर आओ। पता भेजो" % small_fac(r, pat.region, hindi=True))
        L.append(hdr(at) + ADR(address_hi(r, pat, r.choice(["in", "none"]), devn)[0]))
    elif scen == "tb":
        pt = mkh(r, group=asha.group, gender="m", region=asha.region)
        L.append(hdr(at) + "सर %s पुत्र %s को 3 हफ्ते से खांसी, वजन घटा, बलगम जांच %s में दिया" % (PER(pt.full), PER(relative(r, pt, "m").full), small_fac(r, pt.region, hindi=True)))
        L.append(hdr(mt) + "रिपोर्ट आ गई, MTB detected. निक्षय आईडी %s। आज से दवा शुरू" % MR(gen_mrn(r, "nikshay")))
        L.append(hdr(at) + "ठीक सर। निक्षय पोषण के लिए आधार %s" % AAD(fmt_aadhaar(r, devn=devn)))
    else:
        old = mkh(r, group=asha.group, region=asha.region)
        kw, rel = kin(r, old, KIN_HI, age=70)
        L.append(hdr(at) + "सर %s%s उम्र %s, बाथरूम में गिर गए, दायां हाथ-पैर नहीं चल रहा" % (hon_hi(r, old, 0.6), PER(old.full), str(r.randint(60, 88))))
        L.append(hdr(mt) + "कितने बजे हुआ? तुरंत %s ले जाओ" % r.choice(FAC_HI))
        L.append(hdr(at) + "एक घंटा पहले। %s %s ऑटो से ले जा रहे हैं, नंबर %s" % (kw, PER(rel.first), PH(hph(r, devn))))
    L.append(hdr(at) + r.choice(["जी सर", "धन्यवाद सर 🙏", "ठीक है सर", "<Media omitted>"]))
    return "\n".join(L)


def hi_jsy_payment(r):
    devn = r.random() < 0.3
    mom = mkh(r, gender="f")
    hus = relative(r, mom, "m", True)
    asha = mkh(r, gender="f", group=mom.group, region=mom.region)
    fac = small_fac(r, mom.region, hindi=True)
    sep = r.choice([": ", ":- ", " - ", " = "])
    n = lambda s: dev(s) if devn else s
    out = ["जननी सुरक्षा योजना – भुगतान विवरण — %s" % fac]
    out.append("लाभार्थी का नाम%s%s%s पत्नी %s%s" % (sep, hon_hi(r, mom, 0.4), PER(mom.full), hon_hi(r, hus, 0.5), PER(hus.full)))
    out.append("उम्र%s%s वर्ष    श्रेणी%s%s" % (sep, n(str(r.randint(18, 38))), sep, r.choice(["ग्रामीण / बीपीएल", "अनुसूचित जाति", "अनुसूचित जनजाति"])))
    out.append("आरसीएच आईडी%s%s" % (sep, MR(n(gen_mrn(r, "rch12")))))
    a, pin = address_hi(r, mom, r.choice(["in", "none", "sep"]), devn)
    out.append("पता%s%s" % (sep, ADR(a)))
    if r.random() < 0.85:
        out.append("पिन कोड%s%s" % (sep, PIN(n(pin))))
    dd = ev_date(r, 4, 9)
    adm, dis = (dd[0], dd[1], max(1, dd[2] - r.randint(0, 1))), (dd[0], dd[1], min(28, dd[2] + r.randint(2, 4)))
    out.append("प्रसव%s%s, %s" % (sep, r.choice(["सामान्य", "सिजेरियन"]), r.choice(["जीवित पुत्र", "जीवित पुत्री"])))
    if r.random() < 0.6:
        out.append("शिशु की जन्म तिथि%s%s    वजन%s%.1f किग्रा" % (sep, DB(hdate(r, dd, devn)), sep, r.uniform(1.9, 3.6)))
    out.append("भर्ती%s%s    छुट्टी%s%s" % (sep, hdate(r, adm, devn), sep, hdate(r, dis, devn)))
    out.append("मोबाइल%s%s" % (sep, PH(hph(r, devn))))
    if r.random() < 0.35:
        out.append("मतदाता पहचान पत्र%s%s" % (sep, VOT(gen_epic(r))))
    if r.random() < 0.5:
        out.append("आभा नंबर%s%s" % (sep, ABN(gen_abha(r))))
    out.append("राशि%s₹1400 (माँ) | ₹600 आशा प्रोत्साहन – आशा %s" % (sep, PER(asha.full)))
    if r.random() < 0.85:
        out.append("आधार (बैंक से जुड़ा)%s%s" % (sep, AAD(fmt_aadhaar(r, devn=devn))))
    out.append("पीएफएमएस संदर्भ%s%s    आईएफएससी%s%s" % (sep, gen_fail12(r), sep, r.choice(["SBIN", "PUNB", "BKID"]) + "0" + digits(r, 6)))
    out.append("सत्यापनकर्ता%s%s (बीसीपीएम)    स्वीकृत%sडॉ. %s (प्रभारी चिकित्सा अधिकारी)" % (sep, PER(mkh(r).full), sep, PER(mkh(r).full)))
    return "\n".join(out)


SCHOOLS_HI = ["नेताजी सुभाष चंद्र बोस राजकीय बालिका विद्यालय", "पं. दीनदयाल उपाध्याय राजकीय इंटर कॉलेज", "रानी लक्ष्मीबाई कन्या पाठशाला", "संत कबीर प्राथमिक विद्यालय",
              "सरस्वती शिशु मंदिर", "राजकीय उच्च प्राथमिक विद्यालय", "महात्मा गांधी राजकीय विद्यालय", "कस्तूरबा गांधी बालिका विद्यालय", "डॉ. भीमराव आंबेडकर छात्रावास विद्यालय"]


def hi_school_health(r):
    devn = r.random() < 0.3
    n = lambda s: dev(s) if devn else s
    mo, anm = mkh(r), mkh(r, gender="f")
    school = r.choice(SCHOOLS_HI)
    sep = r.choice([": ", ":- ", " - ", "\t"])
    if r.random() < 0.6:
        c = mkh(r)
        f, mtr = relative(r, c, "m", True), relative(r, c, "f", True)
        age = r.randint(6, 16)
        out = ["राष्ट्रीय बाल स्वास्थ्य कार्यक्रम (RBSK) – विद्यालय स्वास्थ्य जांच कार्ड", "विद्यालय%s%s" % (sep, school), "जांच तिथि%s%s" % (sep, hdate(r, ev_date(r, 7, 10), devn))]
        out.append("बच्चे का नाम%s%s    कक्षा%s%s    अनुक्रमांक%s%s" % (sep, PER(c.full), sep, n(str(r.randint(1, 10))), sep, n(str(r.randint(1, 45)))))
        out.append("जन्म तिथि%s%s    उम्र%s%s वर्ष    लिंग%s%s" % (sep, DB(hdate(r, dob_for(r, age), devn)), sep, n(str(age)), sep, "बालक" if c.g == "m" else "बालिका"))
        out.append("पिता का नाम%s%s%s" % (sep, hon_hi(r, f, 0.5), PER(f.full)))
        if r.random() < 0.7:
            out.append("माता का नाम%s%s%s" % (sep, hon_hi(r, mtr, 0.5), PER(mtr.full)))
        out.append("अभिभावक मोबाइल%s%s" % (sep, PH(hph(r, devn))))
        if r.random() < 0.55:
            out.append("बच्चे का आधार%s%s" % (sep, AAD(fmt_aadhaar(r, devn=devn))))
        if r.random() < 0.45:
            out.append("आभा नंबर%s%s" % (sep, ABN(gen_abha(r))))
        if r.random() < 0.75:
            a, pin = address_hi(r, c, r.choice(["in", "sep", "none"]), devn)
            out.append("पता%s%s" % (sep, ADR(a)))
            if r.random() < 0.85:
                out.append("पिन%s%s" % (sep, PIN(n(pin))))
        out.append("ऊंचाई %s सेमी, वजन %s किग्रा, Hb %.1f। %s" % (n(str(r.randint(110, 160))), n(str(r.randint(18, 45))), r.uniform(8.0, 13.0),
                                                       r.choice(["दांतों में कीड़ा (D2)", "नज़र कमज़ोर, Snellen 6/12", "जुएं", "कोई दोष नहीं", "भेंगापन बाईं आँख"])))
        out.append("कार्यवाही%s%s" % (sep, r.choice(["DEIC रेफर", "चश्मा हेतु रेफर", "IFA साप्ताहिक", "एल्बेंडाजोल दी गई"])))
        out.append("जांचकर्ता%sडॉ. %s (RBSK)    एएनएम%s%s" % (sep, PER(mo.full), sep, PER(anm.first)))
    else:
        out = ["%s – विद्यालय स्वास्थ्य शिविर, %s" % (school, hdate(r, ev_date(r, 7, 10), devn)),
               "डॉ. %s एवं एएनएम %s द्वारा %s बच्चों की जांच। रेफर किए गए बच्चे:" % (PER(mo.full), PER(anm.first), n(str(r.randint(30, 70))))]
        for i in range(1, r.randint(3, 5)):
            c = mkh(r)
            f = relative(r, c, "m", True)
            s = "%s) %s %s %s" % (n(str(i)), PER(c.full if r.random() < 0.7 else c.first), "पुत्र" if c.g == "m" else "पुत्री", PER(f.full))
            if r.random() < 0.5:
                s += ", जन्म तिथि %s" % DB(hdate(r, dob_for(r, r.randint(5, 15)), devn))
            else:
                s += ", उम्र %s वर्ष" % n(str(r.randint(5, 15)))
            s += ", " + r.choice(["दिल में मर्मर – ECHO हेतु %s" % r.choice(FAC_HI), "कुपोषण (SAM)", "कम सुनाई देना", "कटे होंठ", "खुजली (स्केबीज़)"])
            out.append(s)
    return "\n".join(out)


def hi_mlc_intimation(r):
    devn = r.random() < 0.25
    pat = mkh(r, gender=r.choice("mmmf"))
    f = relative(r, pat, "m", True)
    kw, bb = kin(r, pat, KIN_HI)
    a, pin = address_hi(r, pat, r.choice(["in", "none"]), devn)
    cmo = mkh(r)
    fac = r.choice(FAC_HI)
    d = ev_date(r)
    so = "पुत्र" if pat.g == "m" else r.choice(["पुत्री", "पत्नी"])
    mech = r.choice(["सड़क दुर्घटना (मोटरसाइकिल)", "मारपीट में सिर पर लाठी से चोट", "अज्ञात ज़हर (कीटनाशक) खाने", "छत से गिरने", "आग से जलने", "कुत्ते के काटने"])
    out = ["सेवा में,", "थानाध्यक्ष महोदय,", "थाना %s, %s" % (r.choice(["कोतवाली", "सदर", "सिविल लाइंस", "कैंट"]), HI_REG[pat.region]["rural"][0][0]), "",
           "विषय: एम.एल.सी. सूचना के संबंध में", "", "महोदय,"]
    s = "सूचित किया जाता है कि %s%s %s %s, उम्र लगभग %s वर्ष, निवासी %s, को दिनांक %s को समय %s बजे %s द्वारा %s की हालत में आपातकालीन विभाग, %s लाया गया। " % (
        hon_hi(r, pat, 0.6), PER(pat.full), so, PER(f.full), dev(str(r.randint(14, 70))) if devn else str(r.randint(14, 70)), ADR(a), hdate(r, d, devn), tstamp(r),
        r.choice(["%s %s" % (kw, PER(bb.full)), "राहगीर %s" % PER(mkh(r).full), "108 एम्बुलेंस"]), mech, fac)
    out.append(s)
    out.append("एम.एल.सी. संख्या %s दर्ज की गई है। मरीज़ %s। सीआर नं. %s।" % (MR(gen_mrn(r, "mlc")), r.choice(["होश में है", "बेहोश है, वेंटिलेटर पर", "बयान देने योग्य है"]), MR(gen_mrn(r, "cr12"))))
    if r.random() < 0.8:
        out.append("परिजन द्वारा दिया गया पहचान पत्र: %s" % r.choice(["मतदाता पहचान पत्र %s" % VOT(gen_epic(r)), "आधार %s" % AAD(fmt_aadhaar(r, devn=devn))]))
    if r.random() < 0.5:
        out.append("परिजन का मोबाइल: %s" % PH(hph(r, devn)))
    out.append("अतः आवश्यक कार्यवाही हेतु विवेचक भेजने का कष्ट करें।")
    out.append("")
    out.append("डॉ. %s\nआपातकालीन चिकित्सा अधिकारी" % PER(cmo.full))
    if r.random() < 0.6:
        out.append("प्राप्तकर्ता: %s %s, समय %s" % (r.choice(["हेड कांस्टेबल", "सिपाही", "उपनिरीक्षक"]), PER(mkh(r, gender="m").full), tstamp(r)))
    return "\n".join(out)


def hi_helpline_104(r):
    devn = r.random() < 0.3
    caller = mkh(r)
    self_ = r.random() < 0.4
    if self_:
        pat, kw = caller, "स्वयं"
    else:
        kw, pat = kin(r, caller, KIN_HI)
    agent = mkh(r)
    sep = r.choice([": ", " - ", " | ", ":- "])
    out = ["104 स्वास्थ्य हेल्पलाइन – कॉल विवरण  %s" % hdate(r, ev_date(r), devn)]
    out.append("कॉल आईडी%s%s  समय%s%s" % (sep, MR(gen_mrn(r, "call")), sep, tstamp(r)))
    out.append("कॉलर का नाम%s%s (%s)" % (sep, PER(caller.full), kw))
    out.append("कॉलर नंबर%s%s" % (sep, PH(hph(r, devn))))
    if r.random() < 0.5:
        a, pin = address_hi(r, caller, r.choice(["in", "sep"]), devn)
        out.append("पता%s%s" % (sep, ADR(a)))
        if r.random() < 0.85:
            out.append("पिन कोड%s%s" % (sep, PIN(dev(pin) if devn else pin)))
    if not self_:
        pa = r.randint(1, 85)
        out.append("मरीज़%s%s, उम्र %s" % (sep, PER(pat.full if r.random() < 0.6 else pat.first), dev(str(pa)) if devn else str(pa)))
        if r.random() < 0.25:
            out.append("जन्म तिथि%s%s" % (sep, DB(hdate(r, dob_for(r, pa), devn))))
    out.append("समस्या%s%s" % (sep, r.choice(["बच्चे को सुबह से 6 बार दस्त", "गर्भवती, 7 महीने, आँखों के आगे धुंधलापन", "3 दिन से तेज़ बुखार, प्लेटलेट 90,000", "साँप ने पैर पर काटा",
                                         "टीबी की दवा ख़त्म हो गई", "रात को नींद नहीं आती, मन उदास"])))
    out.append("सलाह%s%s" % (sep, r.choice(["ओआरएस और ज़िंक, ख़तरे के लक्षण बताए", "तुरंत %s जाने की सलाह, 108 नंबर दिया" % small_fac(r, caller.region, hindi=True), "Tele-MANAS 14416 पर ट्रांसफर",
                                      "नज़दीकी सीएचसी में एएसवी हेतु भेजा"])))
    if r.random() < 0.4:
        out.append("आभा पता%s%s" % (sep, ABA(gen_abha_addr(r, pat))))
    if r.random() < 0.5:
        out.append("आभा नंबर%s%s" % (sep, ABN(gen_abha(r))))
    if r.random() < 0.25:
        out.append("रिपोर्ट हेतु ईमेल%s%s" % (sep, EM(gen_email(r, caller))))
    out.append("काउंसलर%s%s" % (sep, PER(agent.full)))
    return "\n".join(out)


def hi_doctor_narrative(r):
    devn = r.random() < 0.25
    n = lambda s: dev(s) if devn else s
    pat = mkh(r)
    age = r.randint(18, 80)
    f = relative(r, pat, "m", True)
    kw, rel = kin(r, pat, KIN_HI, age=age)
    doc = mkh(r)
    fac = r.choice(FAC_HI)
    v = r.choice(["discharge", "referral", "certificate"])
    so = "पुत्र" if pat.g == "m" else r.choice(["पुत्री", "पत्नी"])
    a, pin = address_hi(r, pat, r.choice(["in", "none"]), devn)
    dx = r.choice(["पार्किंसन रोग", "गुलियन-बैरे सिंड्रोम", "हाशिमोटो थायरॉइडाइटिस", "पॉट्स स्पाइन (रीढ़ की टीबी)", "क्रोहन रोग", "बेल्स पाल्सी", "डेंगू बुखार", "टाइफाइड (विडाल पॉज़िटिव)"])
    out = []
    if v == "discharge":
        out.append("डिस्चार्ज सारांश – %s" % fac)
        out.append("")
        out.append("मरीज़ %s%s, उम्र %s वर्ष, %s %s, निवासी %s, आईपीडी नं. %s, को दिनांक %s को %s की शिकायत के साथ भर्ती किया गया तथा %s को छुट्टी दी गई।" % (
            hon_hi(r, pat, 0.6), PER(pat.full), n(str(age)), so, PER(f.full), ADR(a), MR(n(gen_mrn(r, "ipd"))), hdate(r, ev_date(r, 1, 8), devn),
            r.choice(["तेज़ बुखार और कमज़ोरी", "पैरों में कमज़ोरी", "पीठ दर्द", "दस्त और वजन घटने"]), hdate(r, ev_date(r, 9, 10), devn)))
        out.append("निदान: %s। जांच: Hb %.1f, प्लेटलेट %s, क्रिएटिनिन %.1f।" % (dx, r.uniform(7, 14), n(inr(r.randrange(40000, 300000, 1000))), r.uniform(0.6, 4.0)))
        out.append("भर्ती के दौरान %s %s%s साथ रहे/रहीं। %s को दवा समय पर देने की सलाह दी गई।" % (kw, PER(rel.full if r.random() < 0.5 else rel.first), "", PER(pat.first)))
        out.append("दवाएं: टैब. पैरासिटामोल 500 मि.ग्रा., टैब. पैन्टोप्राज़ोल 40, आयरन की गोली। %s दिन बाद ओपीडी में डॉ. %s को दिखाएं।" % (n(str(r.randint(7, 15))), PER(doc.later(r))))
        if r.random() < 0.5:
            out.append("संपर्क: %s" % PH(hph(r, devn)))
        if r.random() < 0.5:
            out.append("अंतिम बिल ₹%s नकद जमा, भुगतानकर्ता %s %s, पैन %s" % (n(inr(r.randint(201000, 450000))), kw, PER(rel.first), PANE(gen_pan(r, rel))))
        if r.random() < 0.3:
            out.append("आभा पता: %s" % ABA(gen_abha_addr(r, pat)))
        out.append("डॉ. %s, %s" % (PER(doc.full), r.choice(["एम.डी. (मेडिसिन)", "वरिष्ठ परामर्शदाता", "चिकित्सा अधिकारी"])))
    elif v == "referral":
        out.append("रेफरल पत्र")
        out.append("दिनांक: %s" % hdate(r, ev_date(r), devn))
        out.append("प्रति, चिकित्सा अधीक्षक, %s" % fac)
        out.append("")
        out.append("महोदय, मरीज़ %s%s (%s वर्ष), %s %s, यूएचआईडी %s, को %s होने की आशंका है। मरीज़ को उच्च चिकित्सा केंद्र पर भेजा जा रहा है।" % (
            hon_hi(r, pat, 0.5), PER(pat.full), n(str(age)), so, PER(f.full), MR(gen_mrn(r, "uhid")), dx))
        out.append("बीपी %s, नब्ज़ %s, SpO2 %s%%। मरीज़ के साथ %s %s हैं, मोबाइल %s।" % (n(bp(r)), n(str(r.randint(60, 130))), n(str(r.randint(86, 99))), kw, PER(rel.first), PH(hph(r, devn))))
        if r.random() < 0.6:
            out.append("आभा नंबर: %s" % ABN(gen_abha(r)))
        out.append("डॉ. %s\n%s" % (PER(doc.full), small_fac(r, pat.region, hindi=True)))
        if r.random() < 0.4:
            out.append("ईमेल: %s" % EM(gen_email(r, doc, True)))
    else:
        out.append("चिकित्सा प्रमाण पत्र")
        out.append("")
        out.append("प्रमाणित किया जाता है कि %s%s %s %s%s, उम्र %s वर्ष, निवासी %s" % (hon_hi(r, pat, 0.2), PER(pat.full), so, hon_hi(r, f, 0.4), PER(f.full), n(str(age)), ADR(a)))
        if r.random() < 0.8:
            out[-1] += ", आधार संख्या %s" % AAD(fmt_aadhaar(r, devn=devn))
        if r.random() < 0.4:
            out[-1] += ", जन्म तिथि %s" % DB(hdate(r, dob_for(r, age), devn))
        out[-1] += ", का इलाज मेरे द्वारा %s में दिनांक %s से %s तक %s के लिए किया गया। इन्हें %s दिन आराम की सलाह दी जाती है।" % (
            fac, hdate(r, ev_date(r, 1, 5), devn), hdate(r, ev_date(r, 6, 9), devn), dx, n(str(r.randint(5, 21))))
        out.append("")
        out.append("डॉ. %s\nचिकित्सा अधिकारी (मुहर)" % PER(doc.full))
    return "\n".join(out)


def hi_blood_bank_slip(r):
    devn = r.random() < 0.25
    n = lambda s: dev(s) if devn else s
    donor = mkh(r)
    age = r.randint(18, 60)
    rel = relative(r, donor, "m", True)
    sep = r.choice([": ", " - ", ":- ", " | "])
    out = ["रक्त केंद्र – रक्तदाता पंजीकरण पर्ची", "केंद्र%s%s रक्त केंद्र" % (sep, r.choice(FAC_HI).split(",")[0])]
    out.append("रक्तदाता क्रमांक%s%s    दिनांक%s%s" % (sep, MR(gen_mrn(r, "donor")), sep, hdate(r, ev_date(r), devn)))
    out.append("नाम%s%s%s    %s %s" % (sep, hon_hi(r, donor, 0.6), PER(donor.full), "पुत्र" if donor.g == "m" else r.choice(["पुत्री", "पत्नी"]), PER(rel.full)))
    out.append("जन्म तिथि%s%s    उम्र%s%s    लिंग%s%s" % (sep, DB(hdate(r, dob_for(r, age), devn)), sep, n(str(age)), sep, "पुरुष" if donor.g == "m" else "महिला"))
    out.append("मोबाइल%s%s" % (sep, PH(hph(r, devn))))
    if r.random() < 0.3:
        out.append("ईमेल%s%s" % (sep, EM(gen_email(r, donor))))
    a, pin = address_hi(r, donor, r.choice(["in", "sep"]), devn)
    out.append("पता%s%s" % (sep, ADR(a)))
    if r.random() < 0.85:
        out.append("पिन%s%s" % (sep, PIN(n(pin))))
    if r.random() < 0.7:
        out.append("पहचान पत्र%s%s" % (sep, r.choice(["वोटर आईडी " + VOT(gen_epic(r)), "आधार " + AAD(fmt_aadhaar(r, devn=devn))])))
    if r.random() < 0.5:
        pt = mkh(r)
        out.append("रक्तदान प्रकार%sप्रतिस्थापन – मरीज़ %s, सीआर नं. %s" % (sep, PER(pt.full), MR(gen_mrn(r, "cr12"))))
    else:
        out.append("रक्तदान प्रकार%sस्वैच्छिक" % sep)
    out.append("Hb %.1f | बीपी %s | वजन %s किग्रा | बैग नं. %s" % (r.uniform(12.5, 16), n(bp(r)), n(str(r.randint(48, 90))), bag_no(r)))
    out.append("चिकित्सा अधिकारी%sडॉ. %s" % (sep, PER(mkh(r).full)))
    return "\n".join(out)


def hi_nurse_handover(r):
    devn = r.random() < 0.2
    n = lambda s: dev(s) if devn else s
    ward = r.choice(["महिला वार्ड", "पुरुष मेडिसिन वार्ड", "प्रसूति वार्ड", "SNCU", "सर्जरी वार्ड"])
    wg = {"महिला वार्ड": "f", "प्रसूति वार्ड": "f", "पुरुष मेडिसिन वार्ड": "m"}.get(ward)
    out = ["शिफ्ट हैंडओवर – %s – %s (रात से सुबह)" % (ward, hdate(r, ev_date(r), devn)), ""]
    for _ in range(r.randint(2, 4)):
        bed = n(str(r.randint(1, 40)))
        if ward == "SNCU":
            mom = mkh(r, gender="f")
            who = r.choice(["%s की बच्ची" % PER(mom.full), "बेबी ऑफ़ %s" % PER(mom.full), "%s का शिशु" % PER(mom.first)])
            p = mom
            st = r.choice(["कम वजन %s ग्राम, ओजी ट्यूब फीड" % n(str(r.randrange(1100, 2400, 10))), "पीलिया, फोटोथेरेपी जारी", "बबल सीपैप पर"])
        else:
            p = mkh(r, gender=wg)
            who = hon_hi(r, p, 0.5) + PER(p.full)
            st = r.choice(["डायलिसिस मरीज़, K+ %.1f" % r.uniform(4.5, 6.5), "ऑपरेशन के बाद दूसरा दिन, ड्रेन 30 मि.ली.", "फोली कैथेटर लगा है, यूरिन 600 मि.ली.", "राइल्स ट्यूब से फीड",
                           "शुगर %s, इंसुलिन चालू" % n(str(r.randint(150, 380))), "डेंगू, प्लेटलेट %s" % n(inr(r.randrange(20000, 90000, 1000)))] +
                          (["एक्लेम्पसिया, MgSO4 चालू", "सिजेरियन के बाद पहला दिन"] if ward == "प्रसूति वार्ड" else []))
        kw, rel = kin(r, p, KIN_HI)
        s = "बेड %s – %s – आईपी नं. %s – %s।" % (bed, who, MR(gen_mrn(r, "ipd")), st)
        if r.random() < 0.7:
            s += " साथ में %s %s" % (kw, PER(rel.first))
            if r.random() < 0.5:
                s += " (%s)" % PH(hph(r, devn))
            s += "।"
        out.append(s)
    out.append("")
    out.append("रात %s बजे डॉ. %s को सूचित किया गया।" % (night_ts(r), PER(mkh(r).full)))
    out.append("हैंडओवर दिया: सिस्टर %s    लिया: %s" % (PER(mkh(r, gender="f").full), PER(mkh(r).first)))
    return "\n".join(out)


# ==========================================================================
# HINGLISH / CODE-SWITCHED GENERATORS
# ==========================================================================
MX_POOL = ["up", "up", "br", "br", "rj", "hr", "pb", "pbh", "mus", "mus", "uk", "jh", "gd", "mh", "gj", "bn", "kp"]


def mx_asha_whatsapp(r):
    asha = mk(r, gender="f", pool=MX_POOL)
    mo = mk(r)
    pat = mk(r, group=asha.group, gender="f", region=asha.region)
    hus = relative(r, pat, "m", True)
    d = ev_date(r)
    h = r.randint(7, 21)
    m = [r.randint(0, 40)]

    def hdr(who):
        m[0] = min(59, m[0] + r.randint(0, 5))
        return "%02d/%02d/%02d, %02d:%02d - %s: " % (d[2], d[1], d[0] % 100, h, m[0], who)
    at = r.choice(["%s Asha" % PER(asha.first), "ASHA %s" % PER(asha.full), PER(asha.first.lower()), "%s didi" % PER(asha.first)])
    mt = r.choice(["Dr %s" % PER(mo.first), "MO %s sir" % PER(mo.later(r)), "Dr. %s MOIC" % PER(mo.full)])
    L = [hdr(at) + r.choice(["Namaste sir", "sir ji pranam", "Sir urgent h"])]
    scen = r.choice(["hrp", "child", "tb", "dog"])
    if scen == "hrp":
        L.append(hdr(at) + "Sir mere area me ek HRP case h. %s w/o %s, umar %d, %d mahine ka pet h" % (PER(pat.first if r.random() < 0.5 else pat.full), PER(hus.full), r.randint(18, 36), r.randint(6, 9)))
        L.append(hdr(at) + "aaj BP %s aaya, sar dard aur pairo me sujan" % bp(r))
        L.append(hdr(mt) + r.choice(["Turant CHC bhejo, 108 call karo", "RCH id bhejo aur Hb kitna h?"]))
        L.append(hdr(at) + "Hb %.1f sir. RCH no %s, mob %s" % (r.uniform(6.5, 10.5), MR(gen_mrn(r, "rch12")), PH(fmt_mobile(r))))
        if r.random() < 0.6:
            L.append(hdr(at) + r.choice(["pata: ", "address ", "ghar: "]) + ADR(address(r, pat, r.choice(["in", "none"]))[0]))
        if r.random() < 0.5:
            L.append(hdr(at) + "uski saas %s bol rahi h ghar pe hi delivery karayenge" % PER(relative(r, pat, "f", True).first))
            L.append(hdr(mt) + "%s ko phone do, main samjhata hu" % PER(hus.first))
    elif scen == "child":
        kid = mk(r, group=asha.group, region=asha.region)
        L.append(hdr(at) + "sir %s ka %s %s, 2 saal, bahut patla ho gaya, MUAC 11.1 cm" % (PER(pat.full), "beta" if kid.g == "m" else "beti", PER(kid.first)))
        L.append(hdr(mt) + "SAM h. NRC %s me admit karao. DOB kya h?" % small_fac(r, pat.region))
        L.append(hdr(at) + "DOB %s sir, MCP card me likha h" % DB(fdate(r, dob_for(r, 2), r.choice(["dmy/", "dmy-", "dmy."]))))
        L.append(hdr(at) + "pata: %s" % ADR(address(r, pat, r.choice(["in", "none"]))[0]))
    elif scen == "tb":
        pt = mk(r, group=asha.group, gender="m", region=asha.region)
        L.append(hdr(at) + "Sir %s s/o %s ko 3 hafte se khansi, sputum %s me diya tha" % (PER(pt.full), PER(relative(r, pt, "m").full), small_fac(r, pt.region)))
        L.append(hdr(mt) + "MTB detected aaya h. Nikshay ID %s. Aaj se dawai shuru karwao, DOT tum karogi" % MR(gen_mrn(r, "nikshay")))
        L.append(hdr(at) + "ji sir. Poshan yojana ke liye aadhar %s" % AAD(fmt_aadhaar(r)))
        if r.random() < 0.5:
            L.append(hdr(at) + "ghar ka pata %s" % ADR(address(r, pt, r.choice(["in", "none"]))[0]))
    else:
        kid = mk(r, group=asha.group, region=asha.region)
        L.append(hdr(at) + "sir %s ko kutte ne kaata, 8 saal ka h, papa %s le ja rahe h" % (PER(kid.full), PER(relative(r, kid, "m").first)))
        L.append(hdr(mt) + "ghav ko sabun pani se 15 min dhulwao. ARV + RIG %s me milega" % r.choice(FAC_EN))
        L.append(hdr(at) + "unka no %s h sir" % PH(fmt_mobile(r)))
    L.append(hdr(at) + r.choice(["ok sir", "ji sir 🙏", "thik h sir", "<Media omitted>"]))
    return "\n".join(L)


MX_CASES = [
    ("bachche ko subah se 6 baar loose motion, ulti bhi", "ORS + zinc 14 din, khatre ke lakshan samjhaye", (1, 5)),
    ("pet me 7 mahine, aankh ke aage dhundhla aur sar dard", "turant %s jaane ko kaha, 108 ka number diya", (19, 36)),
    ("3 din se bukhar 102, platelet 85,000 aaya", "paani khoob, sirf paracetamol, kal CBC repeat", (8, 60)),
    ("kal raat se peshab nahi hua, pet phoola h", "turant %s jaane ko kaha, 108 ka number diya", (62, 85)),
    ("TB ki dawai khatam, nayi nahi mili", "DTO office ka number diya, Ni-kshay pe complaint darj", (20, 70)),
    ("neend nahi aati, ghabrahat hoti h, mann nahi lagta", "Tele-MANAS 14416 pe transfer kiya", (16, 40)),
    ("saanp ne pair pe kaata, sujan aa rahi", "nearest CHC me ASV ke liye bheja, tourniquet mat baandho", (9, 65)),
]


def mx_helpline_104(r):
    St = Style(r, caps=False)
    caller = mk(r, pool=MX_POOL)
    self_ = r.random() < 0.4
    cz = r.choice(MX_CASES)
    page = r.randint(*cz[2])
    self_ = self_ and page >= 18
    if self_:
        pat, kw = caller, "khud"
    elif page < 18:
        kw, g = r.choice([("beta", "m"), ("beti", "f")])
        pat = relative(r, caller, g, True)
    elif page > 55:
        kw, g = r.choice([("papa", "m"), ("mummy", "f"), ("saas", "f")])
        pat = relative(r, caller, g, kw != "saas")
    else:
        kw, pat = kin(r, caller, KIN_MX, age=page)
    agent = mk(r, pool=MX_POOL)
    out = ["104 Helpline call – %s %s" % (fdate(r, ev_date(r)), tstamp(r))]
    out.append("%s%s" % (St.F("Call ID", "Ticket no"), MR(gen_mrn(r, "call"))))
    out.append("%s%s (%s), number %s" % (St.F("Caller"), PER(caller.full), kw if kw == "khud" else kw + " ke liye", PH(fmt_mobile(r))))
    if not self_:
        out.append("%s%s, %d saal" % (St.F("Patient"), PER(pat.first if r.random() < 0.5 else pat.full), page))
        if r.random() < 0.25:
            out.append("%s%s" % (St.F("DOB"), DB(fdate(r, dob_for(r, page)))))
    if r.random() < 0.5:
        a, pin = address(r, caller, r.choice(["in", "sep"]))
        out.append("%s%s" % (St.F("Pata", "Address"), ADR(a)))
        if r.random() < 0.85:
            out.append("%s%s" % (St.F("Pincode", "PIN"), PIN(fmt_pin(r, pin))))
    out.append("%s%s" % (St.F("Samasya", "Problem"), cz[0]))
    adv = cz[1] % small_fac(r, caller.region) if "%s" in cz[1] else cz[1]
    out.append("%s%s" % (St.F("Salah", "Advice"), adv))
    if r.random() < 0.4:
        out.append("%s%s" % (St.F("ABHA"), ABA(gen_abha_addr(r, pat))))
    elif r.random() < 0.6:
        out.append("%s%s" % (St.F("ABHA no"), ABN(gen_abha(r))))
    if r.random() < 0.25:
        out.append("%s%s" % (St.F("Email"), EM(gen_email(r, caller))))
    if r.random() < 0.3:
        out.append("Caller ne bataya ki %s %s ka bhi BP high rehta h, unka no %s" % (r.choice(["unki", "unke"]), PER(relative(r, caller, r.choice("mf")).first), PH(fmt_mobile(r))))
    out.append("%s%s" % (St.F("Counsellor", "Agent"), PER(agent.full)))
    return "\n".join(out)


def mx_nurse_handover(r):
    ward = r.choice(["Female ward", "Medicine ward", "Labour room", "SNCU", "Surgery ward", "ICU"])
    wg = {"Female ward": "f", "Labour room": "f"}.get(ward)
    out = ["Handover %s – %s, night se morning" % (fdate(r, ev_date(r)), ward), ""]
    for _ in range(r.randint(2, 4)):
        bed = r.randint(1, 40)
        if ward == "SNCU":
            mom = mk(r, gender="f", pool=MX_POOL)
            who = "Baby of %s" % PER(mom.full if r.random() < 0.5 else mom.first)
            p = mom
            st = r.choice(["weight %d g, OG feed chal raha" % r.randrange(1100, 2400, 10), "jaundice, phototherapy pe h", "CPAP pe, FiO2 30"])
        else:
            p = mk(r, pool=MX_POOL, gender=wg)
            who = hon_en(r, p, 0.5) + PER(p.full)
            st = r.choice(["eclampsia, MgSO4 chal raha", "LSCS ke baad day 1, PV bleed normal", "PPH, 2 unit PRBC chadha"]) if ward == "Labour room" else r.choice(["dialysis wala patient, K+ %.1f aaya" % r.uniform(4.5, 6.5), "post-op day 2, drain me 30 ml", "Foley's laga h, urine 500 ml",
                           "Ryle's tube se feed de rahe", "sugar %d, insulin infusion chalu" % r.randint(150, 380), "eclampsia, MgSO4 chal raha", "dengue, platelet %d,000" % r.randint(18, 90)])
        kw, rel = kin(r, p, KIN_MX)
        s = "Bed %d – %s – IP %s – %s." % (bed, who, MR(gen_mrn(r, "ipd")), st)
        if r.random() < 0.7:
            s += " Saath me %s %s h" % (kw, PER(rel.first))
            if r.random() < 0.5:
                s += ", no. %s" % PH(fmt_mobile(r))
            s += "."
        if p.last and r.random() < 0.3 and ward != "SNCU":
            s += " %s ji ne raat ko khana nahi khaya." % PER(p.last)
        out.append(s)
    out.append("")
    out.append("Raat %s baje Dr %s ko inform kiya, repeat KFT bola." % (night_ts(r), PER(mk(r).full)))
    out.append("Given by: Sr. %s / Taken by: %s" % (PER(mk(r, gender="f").full), PER(mk(r).first)))
    return "\n".join(out)


def mx_mental_health_opd(r):
    pat = mk(r, pool=MX_POOL)
    age = r.randint(16, 65)
    kw, rel = kin(r, pat, KIN_MX, age=age)
    psy = mk(r)
    out = ["Psychiatry OPD – %s" % fdate(r, ev_date(r)), "Reg no %s" % MR(gen_mrn(r, "mh")), ""]
    out.append("%s%s, %d saal, %s, %s %s ke saath aaye/aayi." % (hon_en(r, pat, 0.5), PER(pat.full), age, r.choice(["married", "unmarried", "widow"]), kw, PER(rel.first if r.random() < 0.6 else rel.full)))
    c = r.choice([("4 mahine se mann udaas, neend nahi, rona aata h", "PHQ-9 %d" % r.randint(10, 24), "Depression, moderate (F32.1)", "affect restricted, PMR +", ["Tab. Escitalopram", "Tab. Sertraline"]),
                  ("kaan me awaazein aati h, padosiyon pe shak", "PANSS baad me", "Schizophrenia (F20.0)", "guarded, suspicious", ["Tab. Olanzapine", "Tab. Quetiapine"]),
                  ("roz sharab, subah haath kaanpte h", "AUDIT %d" % r.randint(16, 32), "Alcohol dependence (F10.2)", "anxious, tremulous", ["Tab. Naltrexone", "Tab. Thiamine"]),
                  ("bus me ghabrahat, dil tez dhadakta h, lagta h mar jaunga", "GAD-7 %d" % r.randint(10, 19), "Panic disorder (F41.0)", "anxious, rapport theek", ["Tab. Escitalopram", "Tab. Clonazepam"])])
    if r.random() < 0.4:
        out.append("Rehte h: %s" % ADR(address(r, pat, r.choice(["in", "none"]))[0]))
    if r.random() < 0.3:
        out.append("DOB (Aadhar ke hisaab se): %s" % DB(fdate(r, dob_for(r, age))))
    out.append("C/o %s. %s." % (c[0], c[1]))
    out.append("MSE: %s. Suicidal ideation: %s." % (c[3], r.choice(["nahi", "passive, no plan", "2 mahine pehle DSH"])))
    out.append("Impression: %s" % c[2])
    _dn = r.choice(c[4])
    d1 = next(d for d in PSY_DRUGS if d[0] == _dn)
    out.append("Plan: %s %s. %s ko aur %s %s ko psychoeducation diya. Tele-MANAS 14416 bataya. Review 2 hafte." % (d1[0], d1[1], PER(pat.first), kw, PER(rel.first)))
    if r.random() < 0.6:
        out.append("Contact: %s" % PH(fmt_mobile(r)))
    if r.random() < 0.5:
        out.append("UDID ke liye aadhar %s liya." % AAD(fmt_aadhaar(r)))
    out.append("%s%s (Psychiatrist)" % (DR(r), PER(psy.full)))
    return "\n".join(out)


def mx_pharmacy_slip(r):
    pat = mk(r, pool=MX_POOL)
    doc = mk(r)
    out = [r.choice(["Jan Aushadhi Kendra – dawai parchi", "Hospital pharmacy – dispensing slip", "Free dawa counter"]),
           "Bill no: %s   Date: %s" % (bill_no(r), fdate(r, ev_date(r))),
           "Mareez: %s%s   Umar: %d" % (hon_en(r, pat, 0.6), PER(pat.full), r.randint(1, 85)),
           "OPD no: %s" % MR(gen_mrn(r, r.choice(["opd", "uhid"]))),
           "Doctor: %s%s" % (DR(r), PER(doc.full))]
    if r.random() < 0.6:
        out.append("Mob: %s" % PH(fmt_mobile(r)))
    if r.random() < 0.3:
        out.append("ABHA: %s" % ABA(gen_abha_addr(r, pat)))
    out.append("")
    for i, (dn, dose) in enumerate(r.sample(DRUGS, r.randint(3, 5)), 1):
        unit = "1 bottle" if dn.startswith("Syp") else ("%d vial" % r.choice([1, 2, 5]) if dn.startswith("Inj") else "%d tab" % r.choice([10, 15, 30, 60]))
        out.append("%d. %s – batch %s, exp %s – %s – %s" % (i, dn, batch_no(r), exp_date(r), unit,
                                                           r.choice(["khane ke baad", "subah khali pet", "raat ko sote waqt", "din me 2 baar", "zarurat pe"])))
    out.append("")
    if r.random() < 0.6:
        kw, rel = kin(r, pat, KIN_MX)
        out.append(("Dawai %s %s ne li. %s" % (kw, PER(rel.first), r.choice(["Insulin fridge me rakhne ko bola.", "Ecosprin band na kare bina doctor se puche.", ""]))).rstrip())
    if r.random() < 0.3:
        out.append("Cash bill ₹%s se upar, isliye PAN liya: %s" % (inr(r.randint(200100, 300000)), PANE(gen_pan(r, pat))))
    if r.random() < 0.25:
        out.append("ID: voter card %s" % VOT(gen_epic(r)))
    out.append("Pharmacist: %s" % PER(mk(r, pool=MX_POOL).first))
    return "\n".join(out)


def mx_jsy_payment(r):
    mom = mk(r, gender="f", pool=MX_POOL)
    hus = relative(r, mom, "m", True)
    asha = mk(r, gender="f", group=mom.group, region=mom.region)
    out = ["JSY payment note – %s" % small_fac(r, mom.region)]
    out.append("Labharthi: %s w/o %s, umar %d" % (PER(mom.full), PER(hus.full), r.randint(18, 38)))
    out.append("RCH ID: %s" % MR(gen_mrn(r, "rch12")))
    a, pin = address(r, mom, r.choice(["in", "none", "sep"]))
    out.append("Pata: %s" % ADR(a))
    if r.random() < 0.85:
        out.append("PIN: %s" % PIN(pin))
    dd = ev_date(r, 4, 9)
    adm, dis = (dd[0], dd[1], max(1, dd[2] - r.randint(0, 1))), (dd[0], dd[1], min(28, dd[2] + r.randint(2, 4)))
    out.append("Delivery: %s, %s" % (r.choice(["normal", "LSCS"]), r.choice(["beta hua", "beti hui"])))
    if r.random() < 0.6:
        out.append("Bachche ki DOB: %s, wazan %.1f kg" % (DB(fdate(r, dd)), r.uniform(1.9, 3.6)))
    out.append("Admit: %s, chhutti: %s" % (fdate(r, adm), fdate(r, dis)))
    out.append("Mob: %s" % PH(fmt_mobile(r)))
    if r.random() < 0.3:
        out.append("Voter ID: %s" % VOT(gen_epic(r)))
    if r.random() < 0.5:
        out.append("ABHA no: %s" % ABN(gen_abha(r)))
    out.append("Paisa: ₹1400 maa ke khate me, ₹600 ASHA %s ko" % PER(asha.full))
    if r.random() < 0.8:
        out.append("Aadhar linked khata, aadhar %s" % AAD(fmt_aadhaar(r)))
    out.append("PFMS ref %s, IFSC %s" % (gen_fail12(r), r.choice(["SBIN", "PUNB", "BKID"]) + "0" + digits(r, 6)))
    out.append("Verify kiya: %s (BCPM), approve: Dr %s" % (PER(mk(r, pool=MX_POOL).full), PER(mk(r).later(r))))
    return "\n".join(out)


# ==========================================================================
# Plan, generation, validation
# ==========================================================================
PLAN = [
    ("en", "doctor_narrative", en_doctor_narrative, 8), ("en", "asha_whatsapp", en_asha_whatsapp, 6), ("en", "nurse_handover", en_nurse_handover, 7),
    ("en", "pharmacy_slip", en_pharmacy_slip, 7), ("en", "mental_health_opd", en_mental_health_opd, 7), ("en", "dialysis_register", en_dialysis_register, 6),
    ("en", "school_health", en_school_health, 6), ("en", "mlc_intimation", en_mlc_intimation, 7), ("en", "blood_bank_slip", en_blood_bank_slip, 7),
    ("en", "jsy_payment", en_jsy_payment, 5), ("en", "helpline_104", en_helpline_104, 7), ("en", "referral_email", en_referral_email, 6),
    ("en", "reimbursement_claim", en_reimbursement_claim, 6),
    ("hi", "asha_whatsapp", hi_asha_whatsapp, 6), ("hi", "jsy_payment", hi_jsy_payment, 6), ("hi", "school_health", hi_school_health, 5),
    ("hi", "mlc_intimation", hi_mlc_intimation, 5), ("hi", "helpline_104", hi_helpline_104, 5), ("hi", "doctor_narrative", hi_doctor_narrative, 5),
    ("hi", "blood_bank_slip", hi_blood_bank_slip, 4), ("hi", "nurse_handover", hi_nurse_handover, 4),
    ("mixed", "asha_whatsapp", mx_asha_whatsapp, 5), ("mixed", "helpline_104", mx_helpline_104, 4), ("mixed", "nurse_handover", mx_nurse_handover, 4),
    ("mixed", "mental_health_opd", mx_mental_health_opd, 4), ("mixed", "pharmacy_slip", mx_pharmacy_slip, 4), ("mixed", "jsy_payment", mx_jsy_payment, 4),
]

_TWELVE = re.compile(r"(?<![\dA-Za-z])(\d{4})[ -]?(\d{4})[ -]?(\d{4})(?![\dA-Za-z])")
_RUN12 = re.compile(r"(?<!\d)\d{12}(?!\d)")


def check_note(text, ents):
    """Raise AssertionError on any defect."""
    last_end = -1
    for e in sorted(ents, key=lambda x: x["start"]):
        assert text[e["start"]:e["end"]] == e["text"], e
        assert e["start"] >= last_end, ("overlap", e)
        last_end = e["end"]
        assert e["text"] == e["text"].strip()
    asc = text.translate(ASCTR)
    covered = [(e["start"], e["end"], e["type"]) for e in ents]

    def in_type(a, b, t):
        return any(s <= a and b <= e and tt == t for s, e, tt in covered)

    for e in ents:
        v = e["text"].translate(ASCTR)
        if e["type"] == "AADHAAR":
            dd = re.sub(r"[ -]", "", v)
            if re.fullmatch(r"\d{12}", dd):
                assert dd[0] in "23456789" and verhoeff_valid(dd), v
            else:
                assert re.fullmatch(r"[Xx*]{8}\d{4}", dd), v
        if e["type"] == "PAN":
            assert re.fullmatch(r"[A-Z]{3}P[A-Z]\d{4}[A-Z]", v), v
        if e["type"] == "VOTER_ID":
            assert re.fullmatch(r"[A-Z]{3}\d{7}", v), v
        if e["type"] == "ABHA_NUMBER":
            assert len(re.sub(r"\D", "", v)) == 14, v
        if e["type"] == "ABHA_ADDRESS":
            assert v.endswith(("@abdm", "@sbx")), v
        if e["type"] == "PINCODE":
            assert re.fullmatch(r"\d{3} ?\d{3}", v), v
        if e["type"] == "MRN" and re.fullmatch(r"\d{12}", v):
            assert not verhoeff_valid(v), v
    # every 12-digit candidate outside an AADHAAR span must fail Verhoeff
    for m in list(_TWELVE.finditer(asc)) + list(_RUN12.finditer(asc)):
        dd = re.sub(r"\D", "", m.group(0))
        if in_type(m.start(), m.end(), "AADHAAR"):
            continue
        assert not (dd[0] in "23456789" and verhoeff_valid(dd)), ("stray verhoeff-valid 12 digits", m.group(0))


def main():
    rng = random.Random(SEED)
    jobs = []
    for lang, dt, fn, n in PLAN:
        jobs += [(lang, dt, fn)] * n
    rng.shuffle(jobs)
    rows = []
    for i, (lang, dt, fn) in enumerate(jobs, 1):
        for attempt in range(50):
            USED.clear()
            raw = fn(rng)
            text, ents = finalize(raw)
            try:
                check_note(text, ents)
                break
            except AssertionError:
                continue
        else:
            raise SystemExit("could not build a valid note for %s/%s" % (lang, dt))
        rows.append({"id": "b2-%04d" % i, "lang": lang, "doc_type": dt, "text": text, "entities": ents})
    assert len(rows) == 150
    with open(OUT, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # summary
    by_lang = Counter(r_["lang"] for r_ in rows)
    by_type = defaultdict(Counter)
    for r_ in rows:
        for e in r_["entities"]:
            by_type[e["type"]][r_["lang"]] += 1
    print("wrote %s: %d notes  %s" % (OUT, len(rows), dict(by_lang)))
    print("doc types:", dict(Counter(r_["doc_type"] for r_ in rows)))
    print("%-13s %6s %5s %5s %5s" % ("type", "total", "en", "hi", "mixed"))
    tot = 0
    for t in TYPES:
        c = by_type[t]
        s = sum(c.values())
        tot += s
        print("%-13s %6d %5d %5d %5d" % (t, s, c["en"], c["hi"], c["mixed"]))
    print("%-13s %6d" % ("ALL", tot))


if __name__ == "__main__":
    main()
