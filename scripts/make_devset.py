"""Generate the development set (data/dev.jsonl).

These templates were written by the same person who wrote the detection rules and were
used while tuning them, so scores on this set are optimistic. The held-out set
(data/heldout.jsonl, made by data/make_heldout.py) was written separately by an author
who did not see the detector code; quote that one.

All values are synthetic. Aadhaar numbers carry a valid Verhoeff digit so that they
look structurally real; they were drawn at random and belong to nobody we know of.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gopniya.validators import verhoeff_check_digit  # noqa: E402

SEED = 1508
FIRST_F = ["Sunita", "Priya", "Lakshmi", "Fatima", "Kavita", "Meena", "Anjali", "Rekha", "Shabana", "Geeta", "Pooja", "Asha"]
FIRST_M = ["Ramesh", "Suresh", "Mohan", "Imran", "Rajesh", "Vikram", "Arjun", "Harpreet", "Sanjay", "Deepak", "Ravi", "Joseph"]
LAST = ["Devi", "Yadav", "Sharma", "Khan", "Patel", "Reddy", "Singh", "Gupta", "Nair", "Das", "Verma", "Kumar"]
HI_FIRST_F = ["सुनीता", "प्रिया", "गीता", "कविता", "रेखा", "पूजा"]
HI_FIRST_M = ["रमेश", "सुरेश", "मोहन", "राजेश", "संजय", "दीपक"]
HI_LAST = ["देवी", "यादव", "शर्मा", "सिंह", "गुप्ता", "वर्मा"]
VILLAGES = ["Rampur", "Sitapur", "Chandpur", "Belha", "Kothi", "Mahoba"]
DISTRICTS = [("Barabanki", "Uttar Pradesh", "225001"), ("Gaya", "Bihar", "823001"), ("Nashik", "Maharashtra", "422001"),
             ("Mysuru", "Karnataka", "570001"), ("Jodhpur", "Rajasthan", "342001")]
HI_VILLAGES = ["रामपुर", "सीतापुर", "चांदपुर"]
HI_DISTRICTS = ["बाराबंकी", "गया", "जोधपुर"]
DX = [("Severe anaemia in pregnancy", "Hb 6.8 g/dl", "Tab Ferrous sulphate 100 mg BD"),
      ("Pulmonary tuberculosis, new case", "Sputum CBNAAT MTB detected", "Tab HRZE 4 FDC 3 tabs OD"),
      ("Type 2 diabetes mellitus, uncontrolled", "FBS 212 mg/dl", "Tab Metformin 500 mg BD"),
      ("Dengue fever with thrombocytopenia", "Platelets 48,000 /cmm", "Tab Paracetamol 650 mg SOS")]


def aadhaar(rng: random.Random) -> str:
    base = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    num = base + verhoeff_check_digit(base)
    return rng.choice([f"{num[:4]} {num[4:8]} {num[8:]}", num, f"{num[:4]}-{num[4:8]}-{num[8:]}"])


def abha(rng: random.Random) -> str:
    d = "91" + "".join(str(rng.randint(0, 9)) for _ in range(12))
    return f"{d[:2]}-{d[2:6]}-{d[6:10]}-{d[10:]}"


def mobile(rng: random.Random) -> str:
    d = str(rng.choice([6, 7, 8, 9])) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    return rng.choice([d, f"{d[:5]} {d[5:]}", f"+91 {d[:5]} {d[5:]}", f"0{d}"])


def pan(rng: random.Random) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(L) for _ in range(3)) + "P" + rng.choice(L) + f"{rng.randint(1000, 9999)}" + rng.choice(L)


class Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.ents: list[dict] = []
        self.pos = 0

    def t(self, s: str) -> "Builder":
        self.parts.append(s)
        self.pos += len(s)
        return self

    def e(self, s: str, typ: str) -> "Builder":
        self.ents.append({"start": self.pos, "end": self.pos + len(s), "type": typ, "text": s})
        return self.t(s)

    def done(self) -> tuple[str, list[dict]]:
        text = "".join(self.parts)
        for en in self.ents:
            assert text[en["start"]:en["end"]] == en["text"]
        return text, self.ents


def en_note(rng: random.Random) -> tuple[str, list[dict]]:
    f, h, last = rng.choice(FIRST_F), rng.choice(FIRST_M), rng.choice(LAST)
    doc = rng.choice(FIRST_F + FIRST_M) + " " + rng.choice(LAST)
    vill = rng.choice(VILLAGES)
    dist, state, pin = rng.choice(DISTRICTS)
    dx, lab, med = rng.choice(DX)
    b = Builder()
    b.t(f"DISCHARGE SUMMARY - District Hospital, {dist}\nPatient Name: Smt. ").e(f"{f} {last}", "PERSON")
    b.t(f"   Age: {rng.randint(18, 70)} yrs   Sex: F\nW/o Shri ").e(f"{h} {last}", "PERSON")
    b.t("\nUHID: ").e(f"{dist[:3].upper()}/2026/{rng.randint(1000, 99999):05d}", "MRN")
    b.t("   IPD No: ").e(str(rng.randint(1000, 9999)), "MRN")
    b.t("\nAddress: ").e(f"Village {vill}, Tehsil Sadar, District {dist}, {state} - {pin}", "ADDRESS")
    b.t("\nMobile: ").e(mobile(rng), "PHONE").t("   Aadhaar: ").e(aadhaar(rng), "AADHAAR")
    b.t("\nABHA: ").e(abha(rng), "ABHA_NUMBER").t("   ABHA address: ").e(f"{f.lower()}.{rng.randint(1970, 2005)}@abdm", "ABHA_ADDRESS")
    if rng.random() < 0.5:
        b.t("\nPAN (for PM-JAY claim): ").e(pan(rng), "PAN")
    b.t("\nDOB: ").e(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1960, 2005)}", "DOB")
    b.t(f"\nDiagnosis: {dx} ({lab}).\n")
    b.e(f, "PERSON").t(f" was admitted on 0{rng.randint(1, 9)}/10/2026. Her husband ").e(h, "PERSON").t(" was counselled.\n")
    b.t(f"Advice: {med}. Review on 15/11/2026 with Dr. ").e(doc, "PERSON").t(".\n")
    return b.done()


def hi_note(rng: random.Random) -> tuple[str, list[dict]]:
    f, h, last = rng.choice(HI_FIRST_F), rng.choice(HI_FIRST_M), rng.choice(HI_LAST)
    vill, dist = rng.choice(HI_VILLAGES), rng.choice(HI_DISTRICTS)
    dev = str.maketrans("0123456789", "०१२३४५६७८९")
    b = Builder()
    b.t("रोगी का नाम: श्रीमती ").e(f"{f} {last}", "PERSON").t(", पत्नी श्री ").e(f"{h} {last}", "PERSON")
    b.t(", निवासी ").e(f"ग्राम {vill}, जिला {dist}", "ADDRESS").t("।\nमोबाइल: ")
    b.e(mobile(rng).translate(dev) if rng.random() < 0.5 else mobile(rng), "PHONE")
    b.t("\nआधार: ").e(aadhaar(rng).translate(dev) if rng.random() < 0.5 else aadhaar(rng), "AADHAAR")
    b.t("\nनिदान: गर्भावस्था में गंभीर खून की कमी (Hb 6.8 g/dl)। ").e(f, "PERSON")
    b.t(" को दो यूनिट रक्त चढ़ाया गया। आयरन की गोली दिन में दो बार।\n")
    return b.done()


def asha_note(rng: random.Random) -> tuple[str, list[dict]]:
    asha = rng.choice(FIRST_F) + " " + rng.choice(LAST)
    child_mother = rng.choice(FIRST_F)
    last = rng.choice(LAST)
    b = Builder()
    b.t("ASHA home visit report. ASHA: ").e(asha, "PERSON").t(", contact ").e(mobile(rng), "PHONE")
    b.t(".\nVisited Mrs. ").e(f"{child_mother} {last}", "PERSON").t(f" (D/o ").e(rng.choice(FIRST_M) + " " + last, "PERSON")
    hn =f"H. No. {rng.randint(1, 300)}, Ward No. {rng.randint(1, 20)}, {rng.choice(VILLAGES)}"
    b.t(") at ").e(hn, "ADDRESS")
    b.t(f". Newborn weight 2.1 kg, breastfeeding well. PIN ").e(rng.choice(DISTRICTS)[2], "PINCODE")
    b.t(". Next visit after 7 days.\n")
    return b.done()


def main() -> None:
    rng = random.Random(SEED)
    out = ROOT / "data" / "dev.jsonl"
    rows = []
    makers = [("en", en_note), ("hi", hi_note), ("en", asha_note)]
    for i in range(150):
        lang, fn = makers[i % 3]
        text, ents = fn(rng)
        rows.append({"id": f"dev-{i + 1:04d}", "lang": lang, "doc_type": fn.__name__, "text": text, "entities": ents})
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} notes to {out}")


if __name__ == "__main__":
    main()
