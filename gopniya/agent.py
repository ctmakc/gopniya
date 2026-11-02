"""The redaction agent loop.

    note ──► detect (rules [+ optional local NER]) ──► redact with placeholders
         ──► guard scan ──(findings?)──► escalate + redact again (max 3 rounds) ──► block if still leaking
         ──► cloud LLM gets ONLY the redacted text
         ──► verify answer (invented placeholders? new identifiers?) ──► one corrective retry
         ──► restore original values locally ──► answer for the clinician

Raw text never reaches ``llm``. ``ner`` is meant for a model that runs on the same machine
(for example Ollama); the agent refuses an NER backend that declares itself remote.

Everything is async so the same loop runs in CPython and in the browser (Pyodide), where
the LLM call is a JavaScript ``fetch``.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from . import detect as det
from .guard import scan
from .vault import Vault

LLM = Callable[[list[dict]], Awaitable[str]]
NER = Callable[[str], Awaitable[list[det.Span]]]

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant for clinics and public-health programmes in India. "
    "The note you receive has been de-identified: private values were replaced with placeholders such as "
    "[PERSON_1], [AADHAAR_1], [PHONE_1], [ADDRESS_1]. Rules: (1) when you refer to a person, number or place "
    "that was replaced, copy its placeholder exactly, including the square brackets; (2) never guess what a "
    "placeholder stands for; (3) never create a placeholder that is not in the note; (4) keep medicines, "
    "doses, dates of admission and discharge, and test values exactly as written."
)

TASKS: dict[str, str] = {
    "summary_en": (
        "Write a short discharge summary for the patient's family in simple English: what happened, "
        "the medicines with dose and timing, warning signs that need a hospital visit, and the follow-up date."
    ),
    "summary_hi": (
        "मरीज़ के परिवार के लिए सरल हिंदी में छोटा डिस्चार्ज सारांश लिखें: क्या हुआ, दवाइयाँ (खुराक और समय), "
        "किन लक्षणों पर तुरंत अस्पताल आना है, और अगली जाँच की तारीख़।"
    ),
    "referral": (
        "Draft a brief referral letter from the treating doctor to the district hospital, "
        "with the clinical findings and the reason for referral."
    ),
    "extract": (
        "Return JSON only with keys: patient, diagnosis, medicines (list of {name, dose, frequency}), "
        "follow_up_date, red_flags (list)."
    ),
}

_ID_LIKE_RE = re.compile(r"(?<!\d)(?:[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}|\d{2}-\d{4}-\d{4}-\d{4}|[6-9]\d{9})(?!\d)")


def _locate(text: str, spans: list[det.Span], found: list[det.Span]) -> list[det.Span]:
    """Find guard findings (offsets in redacted text) in the original text."""
    out: list[det.Span] = []
    for f in found:
        needle = f.text.strip()
        if len(needle) < 2:
            continue
        start = 0
        while (i := text.find(needle, start)) != -1:
            j = i + len(needle)
            cand = det.Span(i, j, f.type, needle, f.score, "guard")
            if not any(cand.overlaps(s) and s.start <= i and j <= s.end for s in spans):
                out.append(cand)
            start = j
    return out


@dataclass
class Step:
    name: str
    detail: str
    ms: float
    data: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    status: str  # "ok" | "blocked" | "llm_error"
    redacted: str
    spans: list[det.Span]
    answer_redacted: str = ""
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    entity_counts: dict[str, int] = field(default_factory=dict)
    placeholders: list[str] = field(default_factory=list)  # aligned with spans

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "redacted": self.redacted,
            "answer_redacted": self.answer_redacted,
            "answer": self.answer,
            "entity_counts": self.entity_counts,
            "spans": [
                {"start": s.start, "end": s.end, "type": s.type, "source": s.source,
                 "placeholder": self.placeholders[i] if i < len(self.placeholders) else None}
                for i, s in enumerate(self.spans)
            ],
            "steps": [{"name": s.name, "detail": s.detail, "ms": round(s.ms, 1), **s.data} for s in self.steps],
        }


class RedactionAgent:
    def __init__(self, llm: LLM | None, ner: NER | None = None, ner_is_local: bool = True,
                 max_guard_rounds: int = 3, max_llm_retries: int = 1):
        if ner is not None and not ner_is_local:
            raise ValueError("NER backend sees raw text; it must run locally")
        self.llm = llm
        self.ner = ner
        self.max_guard_rounds = max_guard_rounds
        self.max_llm_retries = max_llm_retries

    # -------------------------------------------------------------- de-identification
    async def deidentify(self, text: str, steps: list[Step] | None = None) -> tuple[str, list[det.Span], Vault, bool]:
        steps = steps if steps is not None else []
        t0 = time.perf_counter()
        extra: list[det.Span] = []
        if self.ner is not None:
            try:
                extra = await self.ner(text)
            except Exception as exc:  # noqa: BLE001 - NER is a bonus pass, never fatal
                steps.append(Step("local_ner", f"skipped: {type(exc).__name__}", 0.0))
                extra = []
            else:
                steps.append(Step("local_ner", f"local model proposed {len(extra)} spans",
                                  (time.perf_counter() - t0) * 1000))
            t0 = time.perf_counter()
        spans = det.detect(text, extra)
        by_type: dict[str, int] = {}
        for s in spans:
            by_type[s.type] = by_type.get(s.type, 0) + 1
        steps.append(Step("detect", f"{len(spans)} spans from rules"
                          + (" + local NER" if self.ner else ""), (time.perf_counter() - t0) * 1000,
                          {"by_type": by_type}))

        clean = False
        for round_no in range(1, self.max_guard_rounds + 1):
            vault = Vault()
            redacted = vault.redact(text, spans)
            t0 = time.perf_counter()
            findings = scan(redacted, vault.known_values())
            if not findings:
                steps.append(Step("guard", f"round {round_no}: no residual identifiers", (time.perf_counter() - t0) * 1000))
                clean = True
                break
            kinds: dict[str, int] = {}
            for f in findings:
                kinds[f.kind] = kinds.get(f.kind, 0) + 1
            # map each finding back to the original text, so the redaction stays one clean pass
            added = _locate(text, spans, [f.span for f in findings])
            steps.append(Step("guard", f"round {round_no}: {len(findings)} residual hits, escalating",
                              (time.perf_counter() - t0) * 1000, {"kinds": kinds}))
            if not added:
                break
            spans = det.merge(spans + added)
        if not clean:
            steps.append(Step("block", "residual identifiers after max rounds; nothing sent", 0.0))
        return redacted, spans, vault, clean

    # -------------------------------------------------------------- full loop
    async def run(self, text: str, task: str = "summary_en") -> AgentResult:
        steps: list[Step] = []
        redacted, spans, vault, clean = await self.deidentify(text, steps)
        result = AgentResult("blocked", redacted, spans, steps=steps, entity_counts=vault.counts(),
                             placeholders=[vault.placeholder_for(s.type, s.text) for s in spans])
        if not clean or self.llm is None:
            if self.llm is None and clean:
                result.status = "ok"
            return result

        instruction = TASKS.get(task, task)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\n--- NOTE ---\n{redacted}"},
        ]
        answer_red = ""
        for attempt in range(self.max_llm_retries + 1):
            t0 = time.perf_counter()
            try:
                answer_red = await self.llm(messages)
            except Exception as exc:  # noqa: BLE001
                steps.append(Step("llm", f"error: {type(exc).__name__}: {str(exc)[:160]}", (time.perf_counter() - t0) * 1000))
                result.status = "llm_error"
                return result
            steps.append(Step("llm", f"cloud model answered ({len(answer_red)} chars) from redacted text only",
                              (time.perf_counter() - t0) * 1000, {"attempt": attempt + 1}))
            restored, unknown = vault.restore(answer_red)
            new_ids = [m.group(0) for m in _ID_LIKE_RE.finditer(answer_red)]
            if not unknown and not new_ids:
                steps.append(Step("verify", "answer uses only known placeholders", 0.0))
                break
            problem = []
            if unknown:
                problem.append(f"placeholders not in the note: {', '.join(sorted(set(unknown)))}")
            if new_ids:
                problem.append("identifier-like numbers that are not in the note")
            steps.append(Step("verify", "; ".join(problem) + ("; asking for a fix" if attempt < self.max_llm_retries else ""), 0.0))
            if attempt < self.max_llm_retries:
                messages = messages + [
                    {"role": "assistant", "content": answer_red},
                    {"role": "user", "content": "Your answer used " + "; ".join(problem)
                     + ". Rewrite it using only the placeholders that appear in the note and no other identifiers."},
                ]
        t0 = time.perf_counter()
        restored, unknown = vault.restore(answer_red)
        for bad in set(unknown):
            restored = restored.replace(bad, "[unknown]")
        steps.append(Step("restore", f"{len(vault.placeholders)} placeholders mapped back on the trusted side",
                          (time.perf_counter() - t0) * 1000))
        result.status = "ok"
        result.answer_redacted = answer_red
        result.answer = restored
        return result

    def run_sync(self, text: str, task: str = "summary_en") -> AgentResult:
        return asyncio.run(self.run(text, task))
