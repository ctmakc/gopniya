import asyncio

import pytest

from gopniya import RedactionAgent, Vault, detect
from gopniya.guard import scan
from gopniya.validators import aadhaar_valid, pan_valid, verhoeff_check_digit, verhoeff_valid

AADHAAR = "234567890124"  # synthetic, valid Verhoeff digit
NOTE = f"""Patient Name: Smt. Sunita Devi   Age: 34 yrs
W/o Shri Ramesh Kumar Yadav
UHID: BBK/2026/004512
Address: Village Rampur, Tehsil Sadar, District Barabanki, UP - 225001
Mobile: 98390 12345   Aadhaar: {AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}
ABHA: 91-4421-7783-2210   ABHA address: sunita.1990@abdm   PAN: ABCPY1234K
DOB: 12/03/1992
Hb 6.8 g/dl, platelets 1,50,000 /cmm. Tab Ferrous sulphate 100 mg BD. Batch No. 2309114.
Sunita was transfused 2 units PRBC. Review on 15/11/2026 with Dr. Anjali Mishra.
"""


def types(text):
    return {(s.type, s.text) for s in detect(text)}


# ------------------------------------------------------------------ validators
def test_verhoeff_reference_vector():
    # textbook example: 236 -> check digit 3
    assert verhoeff_check_digit("236") == "3"
    assert verhoeff_valid("2363")
    assert not verhoeff_valid("2364")


def test_aadhaar_rules():
    assert aadhaar_valid(AADHAAR)
    assert not aadhaar_valid("134567890124")  # cannot start with 0/1
    assert not aadhaar_valid("234567890125")  # bad check digit


def test_pan_structure():
    assert pan_valid("ABCPY1234K")
    assert not pan_valid("ABCXY1234K")  # 4th letter must be a holder-type code


# ------------------------------------------------------------------ detection
def test_detects_every_identifier_in_sample():
    found = types(NOTE)
    expected = {
        ("PERSON", "Sunita Devi"), ("PERSON", "Ramesh Kumar Yadav"), ("PERSON", "Anjali Mishra"),
        ("PERSON", "Sunita"), ("MRN", "BBK/2026/004512"), ("PHONE", "98390 12345"),
        ("AADHAAR", "2345 6789 0124"), ("ABHA_NUMBER", "91-4421-7783-2210"),
        ("ABHA_ADDRESS", "sunita.1990@abdm"), ("PAN", "ABCPY1234K"), ("DOB", "12/03/1992"),
    }
    assert expected <= found
    assert any(t == "ADDRESS" and "Rampur" in v for t, v in found)


def test_hard_negatives_untouched():
    redacted = Vault().redact(NOTE, detect(NOTE))
    for keep in ("6.8 g/dl", "1,50,000 /cmm", "100 mg BD", "2309114", "15/11/2026", "34 yrs"):
        assert keep in redacted


def test_unlabelled_12_digit_number_needs_valid_checksum():
    assert not any(s.type == "AADHAAR" for s in detect("Invoice 234567890125 paid"))
    assert any(s.type == "AADHAAR" for s in detect("ref 234567890124"))


def test_labelled_aadhaar_with_typo_still_redacted():
    assert any(s.type == "AADHAAR" for s in detect("Aadhaar no: 2345 6789 0125"))


def test_masked_aadhaar():
    assert any(s.type == "AADHAAR" for s in detect("Aadhaar: XXXX XXXX 0124"))


def test_hindi_note_with_devanagari_digits():
    text = "रोगी का नाम: श्रीमती सुनीता देवी, पत्नी श्री रमेश यादव, निवासी ग्राम रामपुर, जिला बाराबंकी। मोबाइल: ९८३९० १२३४५ आधार: २३४५ ६७८९ ०१२४"
    found = types(text)
    assert ("PERSON", "सुनीता देवी") in found
    assert ("PERSON", "रमेश यादव") in found
    assert ("PHONE", "९८३९० १२३४५") in found
    assert ("AADHAAR", "२३४५ ६७८९ ०१२४") in found
    assert any(t == "ADDRESS" and "रामपुर" in v for t, v in found)


def test_hindi_honorific_prefix_is_not_split():
    red = Vault().redact("श्रीमती सुनीता देवी आई", detect("श्रीमती सुनीता देवी आई"))
    assert red.startswith("श्रीमती [PERSON_1]")


def test_name_propagation():
    text = "Patient Name: Mr. Imran Khan. Later Imran complained of chest pain."
    spans = [s for s in detect(text) if s.type == "PERSON"]
    assert {"Imran Khan", "Imran"} <= {s.text for s in spans}


def test_asha_is_a_name_when_titlecase():
    assert ("PERSON", "Asha Das") in types("ASHA: Asha Das visited the house.")


# ------------------------------------------------------------------ vault
def test_vault_referential_consistency_and_restore():
    v = Vault()
    red = v.redact(NOTE, detect(NOTE))
    assert "Sunita" not in red and AADHAAR[:4] not in red
    assert red.count("[ABHA_NUMBER_1]") == 1
    restored, unknown = v.restore("Dear [[PERSON_1]], call PERSON_2 or [AADHAAR 1]. [PERSON_99]")
    assert "Sunita Devi" in restored and "Ramesh Kumar Yadav" in restored and "2345 6789 0124" in restored
    assert unknown == ["[PERSON_99]"]


# ------------------------------------------------------------------ guard
def test_guard_catches_unknown_long_number_and_missed_name():
    findings = scan("Card no 4417 2290 8812 3341 seen by Dr. Okonkwo", [])
    kinds = {f.kind for f in findings}
    assert {"digit_run", "honorific_name"} <= kinds


def test_guard_ignores_dates_and_lab_values():
    assert scan("Review 15/11/2026. Platelets 150000000 /cmm. [PERSON_1] ok", []) == []


# ------------------------------------------------------------------ agent loop
class FakeLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    async def __call__(self, messages):
        self.seen.append(messages)
        return self.answers.pop(0)


def test_agent_never_sends_raw_values_and_restores_answer():
    fake = FakeLLM(["[PERSON_1] should take iron tablets. Call [PHONE_1] if bleeding."])
    res = asyncio.run(RedactionAgent(llm=fake).run(NOTE))
    sent = "\n".join(m["content"] for m in fake.seen[0])
    for raw in ("Sunita", "Ramesh", "98390", AADHAAR[:4], "sunita.1990@abdm", "ABCPY1234K", "Rampur"):
        assert raw not in sent
    assert res.status == "ok"
    assert res.answer.startswith("Sunita Devi should take iron tablets. Call 98390 12345")
    assert [s.name for s in res.steps][:2] == ["detect", "guard"]


def test_agent_retries_when_model_invents_placeholder():
    fake = FakeLLM(["Hello [PERSON_42]", "Hello [PERSON_1]"])
    res = asyncio.run(RedactionAgent(llm=fake).run(NOTE))
    assert len(fake.seen) == 2
    assert res.answer == "Hello Sunita Devi"


def test_agent_escalates_guard_findings_before_sending():
    note = "Seen by Dr. Okonkwo. Insurance card 4417 2290 8812 3341."
    fake = FakeLLM(["ok"])
    res = asyncio.run(RedactionAgent(llm=fake).run(note))
    sent = fake.seen[0][1]["content"]
    assert "Okonkwo" not in sent and "4417" not in sent
    assert any(s.name == "guard" and "escalating" in s.detail for s in res.steps)


def test_remote_ner_is_refused():
    async def ner(text):
        return []
    with pytest.raises(ValueError):
        RedactionAgent(llm=None, ner=ner, ner_is_local=False)


def test_ner_spans_are_merged():
    from gopniya.detect import Span

    async def ner(text):
        i = text.index("Mahesh")
        return [Span(i, i + 6, "PERSON", "Mahesh", 0.65, "local_ner")]

    text = "Her brother Mahesh brought her to the PHC."
    red, spans, _, _ = asyncio.run(RedactionAgent(llm=None, ner=ner).deidentify(text))
    assert "Mahesh" not in red


# ------------------------------------------------------------------ v2 cues (added after held-out v1 error analysis)
def test_age_sex_cue_finds_bare_name():
    assert ("PERSON", "Rakesh Prajapati") in types("Ni-kshay ID 31114707. Rakesh Prajapati, 27 M, cough for 3 weeks.")


def test_role_cue_without_colon():
    assert ("PERSON", "Rajwant Kaur") in types("Escorted by ANM Rajwant Kaur to the FRU.")


def test_lowercase_hinglish_kinship():
    found = types("asha ne aaj fatima khan (pati imran khan) ke ghar visit kiya")
    assert ("PERSON", "imran khan") in found


def test_gram_unit_and_gram_stain_are_not_addresses():
    for text in ("Sputum gram stain negative.", "आयरन 100 मि.ग्रा., प्रोटीन 2 ग्राम दिन में दो बार", "Hb 9 ग्राम/डेसीलीटर"):
        assert not any(s.type == "ADDRESS" for s in detect(text)), text


def test_address_ends_at_pin_code():
    text = "resident of Vill. Bariyarpur, Post Lakhanpur, Distt. Hisar, Haryana 125222, who presented with fever"
    addr = [s for s in detect(text) if s.type == "ADDRESS"]
    assert addr and addr[0].text.endswith("125222") and "presented" not in addr[0].text


def test_facility_named_after_person_is_kept():
    text = "Referred to Dr. Ram Manohar Lohia Hospital, New Delhi."
    assert "Ram Manohar Lohia Hospital" in Vault().redact(text, detect(text))


def test_flexible_label_separators():
    found = types("DOB:- 10/07/1995\nNi-kshay ID - 16409322\nजन्म तिथि:- ०४/०७/१९९९")
    assert ("DOB", "10/07/1995") in found and ("MRN", "16409322") in found and ("DOB", "०४/०७/१९९९") in found


# ------------------------------------------------------------------ new identifier tests

def test_passport_labelled_english_and_hindi():
    assert any(s.type == "PASSPORT" for s in detect("Passport: K1234567"))
    assert any(s.type == "PASSPORT" for s in detect("पासपोर्ट: K1234567"))


def test_driving_licence_labelled_and_unlabelled():
    # labelled forms
    assert any(s.type == "DRIVING_LICENCE" for s in detect("DL: MH12 20110012345"))
    assert any(s.type == "DRIVING_LICENCE" for s in detect("Driving Licence: DL-0420110149646"))
    assert any(s.type == "DRIVING_LICENCE" for s in detect("ड्राइविंग लाइसेंस: DL-0420110149646"))
    # unlabelled when unambiguous
    assert any(s.type == "DRIVING_LICENCE" for s in detect("Seen DL MH12 20110012345"))
    assert any(s.type == "DRIVING_LICENCE" for s in detect("MH12 20110012345 was presented"))


def test_pmjay_labelled():
    assert any(s.type == "PMJAY" for s in detect("PM-JAY: ABCD123456"))
    assert any(s.type == "PMJAY" for s in detect("Ayushman ID: XYZ987654321"))
    assert any(s.type == "PMJAY" for s in detect("पीएम-जय: ABCD123456"))


def test_negative_similar_tokens_not_redacted():
    # 'Tab K1234567' — a medicine with code-like token should not be treated as passport
    assert not any(s.type == "PASSPORT" for s in detect("Patient given Tab K1234567 1 OD"))
    # lab/dosage numbers should not be caught as DL/PMJAY
    assert not any(s.type in {"DRIVING_LICENCE", "PMJAY"} for s in detect("Dose 100 mg BD. Hb 6.8 g/dl."))


def test_kit_lot_number_is_not_a_passport():
    text = "Td-1 given 16.05.2026 (Kit Lot B7894686), Td-2 after 4 weeks."
    assert not any(s.type == "PASSPORT" for s in detect(text))
    assert not any(f.span.type == "PASSPORT" for f in scan(text, []))


def test_driving_licence_needs_real_state_code():
    text = "Ref ZZ12 20110012345"
    assert not any(s.type == "DRIVING_LICENCE" for s in detect(text))
    assert not any(f.span.type == "DRIVING_LICENCE" for f in scan(text, []))


def test_ayushman_word_is_not_an_id():
    text = "Ayushman cardholder since 2021, admitted under PM-JAY package."
    assert not any(s.type == "PMJAY" for s in detect(text))
    assert not any(f.span.type == "PMJAY" for f in scan(text, []))
