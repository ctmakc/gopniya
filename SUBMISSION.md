# Submission draft — Code for a Billion 2026

Where: GitHub issue form **https://github.com/karlmehta/code-for-a-billion/issues/new?template=submission.yml**
(the form creates a public issue labelled `submission`; one is filed per project). Deadline: 15 November 2026.
Fields below follow `.github/ISSUE_TEMPLATE/submission.yml` in that repository, in order.

---

**Title:** `[Submission] Gopniya — keep patient identifiers out of cloud AI (Health)`

### Team / Project name
Gopniya (गोपनीय)

### Team members (names + GitHub handles)
Maksym Stepanenko (@ctmakc)

### Problem statement
Clinics, ASHA programmes and district hospitals in India want cloud AI to turn clinical notes into
discharge advice in Hindi, referral letters and visit summaries. Those notes carry Aadhaar and ABHA
numbers, phone numbers, names and home addresses, and pasting them into a cloud model hands that
data to a third party. India has issued 93.95 crore ABHA numbers and linked over 105 crore digital
health records (as of 10 July 2026), so more and more clinical text is tied to a national
identifier. Gopniya lets a health worker use any cloud model while the identifiers stay on the
clinic's own device.

### What you built
An open-source agent (Python, standard library only, MIT) that de-identifies Indian clinical notes
in English, Hindi and Hinglish before they reach a cloud model, and restores the values afterwards.

- **Detect** Aadhaar (Verhoeff check digit or "Aadhaar/आधार" label, masked forms, Devanagari
  digits), ABHA number and ABHA address, PAN, voter ID, mobiles, e-mail, UHID/MRN/IPD numbers, DOB,
  PIN codes, names (honorifics, field labels, S/o–W/o, पुत्र/पत्नी, kinship words like beti/भाई) and
  home addresses (Address, R/o, पता, निवासी, Village, ग्राम). Each name is propagated to its later mentions.
- **Guard**: an independent re-scan of the redacted text; anything that still looks like an
  identifier is mapped back and redacted, up to three rounds, and the call is blocked if a leak remains.
- **Cloud call** with placeholders only (`[PERSON_1]`, `[AADHAAR_1]`), then **verify**: invented
  placeholders or identifier-shaped numbers in the answer trigger one corrective retry.
- **Restore** the real values locally.
- Optional **local LLM-NER** pass (Ollama) for names with no cue; the agent refuses a non-local NER backend.
- **Live demo** on Cloudflare Pages: the redaction engine runs in the visitor's browser through
  Pyodide, so the raw note never leaves the tab. The server function re-checks every request and
  refuses text that still contains Aadhaar/ABHA/PAN/phone/e-mail patterns (HTTP 422).
- CLI with four backends (demo endpoint, Workers AI, Ollama for fully offline use, Gemini CLI),
  unit tests, and a reproducible evaluation on two synthetic sets.

How we used AgentFoundry (AF): the core agent was first written with Claude Code (commits up to
`1a69838`). We then opened the repository in the AgentFoundry desktop IDE (Linux, v0.1.80) and built
the next feature there on the branch `agentfoundry`: detection of passport numbers, driving licence
numbers and PM-JAY/Ayushman IDs. AF's SMART-Code agent read the codebase and wrote the detectors,
guard re-scan and tests (26 tool calls, each approved in Confirm mode); tests and the evaluation ran
in the AF terminal. The evaluation caught a regression in the agent's first version (207 false
alarms on vaccine kit lot numbers, precision 98.6% → 88.6%); SMART-Code fixed most of it, and after
the free Cloud quota ran out the last fix and three regression tests were written by hand.
Result: 34 tests pass, held-out precision back to 98.6%. Transcript, approval log and screenshots:
`docs/AGENTFOUNDRY.md` and `docs/agentfoundry-proof/`.

### GitHub repository URL
https://github.com/ctmakc/gopniya

### Demo URL (live app or video)
https://gopniya.pages.dev (live). Video (2:32): https://github.com/ctmakc/gopniya/raw/main/video/gopniya-demo.mp4

### Scale & severity of the problem
- Scale: 93.95 crore ABHA numbers issued and over 105 crore digital health records linked as of
  10 July 2026; 5.33 lakh health facilities and 9.85 lakh healthcare professionals registered on the
  national registries (GKToday report on the ABDM Mission Steering Group,
  https://www.gktoday.in/ayushman-bharat-digital-mission-crosses-93-95-crore-health-accounts/).
  Every one of those facilities that tries a cloud model on clinical text faces this problem.
- Severity for patients: a leaked Aadhaar plus phone number plus diagnosis exposes a person to
  identity fraud, and diagnoses such as TB, HIV or a pregnancy complication carry stigma.
- Severity for providers: under the Digital Personal Data Protection Act, 2023, failure to take
  reasonable security safeguards to prevent a personal data breach can be penalised up to
  ₹250 crore (Schedule, item 1).
- Health systems are already targets: the AIIMS Delhi e-hospital servers went down in a ransomware
  attack on 23 November 2022 and services ran manually for days
  (https://www.healthcareitnews.com/news/asia/aiims-delhi-turns-manual-following-ransomware-attack).
- We do not have a measured count of Indian clinicians who paste notes into chatbots; we did not
  find a published survey and do not claim a number.

### Deployment & impact data
No field deployment yet, and no real users. What we can show is measured detection quality on
synthetic notes (no real personal data anywhere):

- Blind test (150 synthetic notes, 1,499 identifiers, written by an author who never saw our code):
  rules only protected 69.9% of identifiers at 90.0% precision; rules plus a local LLM-NER pass
  protected 82.0% at 89.4% precision.
- Fixed-format IDs (ABHA number and address, PAN, voter ID, e-mail) were caught 100% on that blind
  set; Aadhaar 84.4%, phone 88.0%, names 62.4% rules-only (83.2% with local NER).
- On the first test set, after we studied its misses, rules reached 95.7% recall and 98.6% precision;
  that number is optimistic because the rules were tuned on it. All results with every leak listed:
  `eval/results/`.

Method: protection recall counts an identifier as protected only when every character is covered;
precision counts a redaction as correct when it overlaps a labelled identifier. The held-out set was
generated by a separate author who did not see the detection code, after the rules were committed.
Template-generated notes are cleaner than real ones, so these numbers are an upper bound. The next
step is a pilot with a clinic or health programme on consented notes, with a person labelling leaks.

### Track / category
Health

### Contact email
maksym@innovacg.com

### Confirmations
- [x] Built using AgentFoundry (AF), the official IDE (see `docs/AGENTFOUNDRY.md`)
- [x] Our repository is public and includes a README with setup steps
- [x] The demo link works and is accessible to judges (https://gopniya.pages.dev, checked 4 Nov 2026)

