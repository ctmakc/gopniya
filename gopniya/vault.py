"""Reversible placeholders.

``Vault.redact`` swaps every detected span for a typed placeholder such as
``[PERSON_1]`` or ``[AADHAAR_1]``. The same value always gets the same placeholder inside
one vault (referential consistency), so a model can still reason that "[PERSON_1]" in the
history section and in the advice section is one person.

``Vault.restore`` puts the original values back into a model answer. It tolerates the
small format slips models make (``[[PERSON_1]]``, ``PERSON_1``, ``[PERSON 1]``).

The vault lives in memory on the trusted side only. It is never serialised or sent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .detect import ENTITY_TYPES, Span

_TYPES_ALT = "|".join(sorted(ENTITY_TYPES, key=len, reverse=True))
PLACEHOLDER_RE = re.compile(r"\[(" + _TYPES_ALT + r")_(\d+)\]")
# tolerant form used on model output
LOOSE_PLACEHOLDER_RE = re.compile(r"\[{0,2}\s*(" + _TYPES_ALT + r")[ _](\d+)\s*\]{0,2}")


def _norm(span_type: str, value: str) -> str:
    val = unicodedata.normalize("NFC", value)
    if span_type in {"AADHAAR", "ABHA_NUMBER", "PHONE", "PINCODE"}:
        return re.sub(r"\D", "", val.translate(str.maketrans("०१२३४५६७८९", "0123456789")))[-10:]
    return re.sub(r"\s+", " ", val).strip().casefold()


@dataclass
class Vault:
    _by_key: dict[tuple[str, str], str] = field(default_factory=dict)
    _values: dict[str, str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)

    def placeholder_for(self, span_type: str, value: str) -> str:
        key = (span_type, _norm(span_type, value))
        if key not in self._by_key:
            n = self._counters.get(span_type, 0) + 1
            self._counters[span_type] = n
            ph = f"[{span_type}_{n}]"
            self._by_key[key] = ph
            self._values[ph] = value
        return self._by_key[key]

    def redact(self, text: str, spans: list[Span]) -> str:
        out: list[str] = []
        cur = 0
        for s in sorted(spans, key=lambda x: x.start):
            if s.start < cur:
                continue
            out.append(text[cur:s.start])
            out.append(self.placeholder_for(s.type, text[s.start:s.end]))
            cur = s.end
        out.append(text[cur:])
        return "".join(out)

    @property
    def placeholders(self) -> set[str]:
        return set(self._values)

    def value(self, placeholder: str) -> str | None:
        return self._values.get(placeholder)

    def known_values(self) -> list[tuple[str, str]]:
        """(entity type, original value) pairs."""
        return [(ph[1:].rsplit("_", 1)[0], val) for ph, val in self._values.items()]

    def counts(self) -> dict[str, int]:
        return dict(self._counters)

    def restore(self, answer: str) -> tuple[str, list[str]]:
        """Return (restored text, list of placeholders the model invented)."""
        unknown: list[str] = []

        def sub(m: re.Match[str]) -> str:
            ph = f"[{m.group(1)}_{m.group(2)}]"
            if ph in self._values:
                return self._values[ph]
            unknown.append(ph)
            return m.group(0)

        return LOOSE_PLACEHOLDER_RE.sub(sub, answer), unknown
