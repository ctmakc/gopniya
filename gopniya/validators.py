"""Check-digit and format validators for Indian identifiers.

Pure functions, no I/O. A validator receives the raw matched text and strips its own
separators. Validators never log or store their input.

Sources:
- Aadhaar: 12 digits, first digit 2-9, last digit is a Verhoeff check digit
  (UIDAI, "Aadhaar Numbering" / Verhoeff scheme, widely documented).
- PAN: 5 letters + 4 digits + 1 letter; the 4th letter encodes the holder type
  (Income Tax Department PAN structure).
"""

from __future__ import annotations

import re

_SEP_RE = re.compile(r"[\s\-.]")

# Verhoeff tables (dihedral group D5).
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def digits_only(s: str) -> str:
    return _SEP_RE.sub("", s)


def verhoeff_valid(number: str) -> bool:
    """True when ``number`` (digits only) passes the Verhoeff check."""
    if not number.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> str:
    """Return the Verhoeff check digit to append to ``number`` (digits only)."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


def aadhaar_valid(s: str) -> bool:
    d = digits_only(s)
    return len(d) == 12 and d[0] in "23456789" and verhoeff_valid(d)


_PAN_RE = re.compile(r"^[A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]$")


def pan_valid(s: str) -> bool:
    return bool(_PAN_RE.match(digits_only(s).upper()))


def indian_mobile_valid(s: str) -> bool:
    d = digits_only(s).lstrip("+")
    if d.startswith("91") and len(d) == 12:
        d = d[2:]
    elif d.startswith("0") and len(d) == 11:
        d = d[1:]
    return len(d) == 10 and d[0] in "6789" and len(set(d)) > 2


def abha_number_valid(s: str) -> bool:
    """ABHA number: 14 digits (displayed as XX-XXXX-XXXX-XXXX). No public checksum used."""
    d = digits_only(s)
    return len(d) == 14 and d.isdigit()


def pincode_valid(s: str) -> bool:
    d = digits_only(s)
    return len(d) == 6 and d.isdigit() and d[0] != "0"
