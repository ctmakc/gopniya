"""Leak scan: an independent second look at text that is about to leave the trusted side.

The detector decides what to redact. The guard re-reads the already-redacted text with
looser, recall-first heuristics and reports anything that still looks personal:

- a value the vault already knows appears verbatim (a boundary slip somewhere)
- a run of 8+ digits outside a placeholder (an identifier no rule recognised)
- an honorific (Mr., Smt., श्री, ...) followed by a word instead of a placeholder
- an e-mail-like string

Each finding carries the span to redact, so the agent can escalate and re-check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .detect import Span, normalize_digits
from .vault import PLACEHOLDER_RE

_DIGIT_RUN_RE = re.compile(r"(?<![\dA-Za-z_])\d(?:[\d \-/]{6,22})\d(?![\d_])")
_HONORIFIC_WORD_RE = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Smt|Shri|Sri|Kumari|Dr)\.?\s+|(?:श्रीमती|श्री|सुश्री|कुमारी|डॉ\.?)(?![\u0900-\u0963\u0970-\u097F])\s*)"
    r"(?!\[)([A-Z][A-Za-z'\-]+|[ऀ-ॣ॰-ॿ]+)"
)
_EMAILISH_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*")
# Clinical numbers that look like long digit runs but are not identifiers.
_BENIGN_CONTEXT_RE = re.compile(r"(?:mg|ml|mmHg|g/dl|/cmm|cells|lakh|IU|%|x\s*10)", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    kind: str
    span: Span


def _inside_placeholder(text: str, start: int, end: int) -> bool:
    for m in PLACEHOLDER_RE.finditer(text):
        if m.start() <= start and end <= m.end():
            return True
    return False


def scan(redacted: str, known_values: list[tuple[str, str]]) -> list[Finding]:
    found: list[Finding] = []
    low = redacted.casefold()
    for typ, val in known_values:
        val = val.strip()
        if len(val) < 3:
            continue
        idx = low.find(val.casefold())
        while idx != -1:
            end = idx + len(val)
            # only a leak when it is a whole word, not a substring of another word
            before = redacted[idx - 1] if idx > 0 else " "
            after = redacted[end] if end < len(redacted) else " "
            if not before.isalnum() and not after.isalnum():
                found.append(Finding("known_value", Span(idx, end, typ, redacted[idx:end], 0.6, "guard")))
            idx = low.find(val.casefold(), end)
    norm = normalize_digits(redacted)
    for m in _DIGIT_RUN_RE.finditer(norm):
        digits = sum(c.isdigit() for c in m.group(0))
        if digits < 8 or _inside_placeholder(redacted, m.start(), m.end()):
            continue
        tail = norm[m.end():m.end() + 8]
        if _BENIGN_CONTEXT_RE.match(tail.strip()):
            continue
        # dates such as 12/03/2026 are 8 digits but are not identifiers
        if re.fullmatch(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}", m.group(0).strip()):
            continue
        found.append(Finding("digit_run", Span(m.start(), m.end(), "ID_NUMBER", redacted[m.start():m.end()], 0.6, "guard")))
    for m in _HONORIFIC_WORD_RE.finditer(redacted):
        found.append(Finding("honorific_name", Span(m.start(1), m.end(1), "PERSON", m.group(1), 0.6, "guard")))
    for m in _EMAILISH_RE.finditer(redacted):
        if not _inside_placeholder(redacted, m.start(), m.end()):
            found.append(Finding("email_like", Span(m.start(), m.end(), "EMAIL", m.group(0), 0.6, "guard")))
    return found
