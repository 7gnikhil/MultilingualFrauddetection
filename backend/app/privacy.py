"""Server-side privacy backstop.

The extension anonymizes text client-side *before* anything is sent
(phones -> [PHONE], OTPs -> [OTP], emails -> [EMAIL], UPI ids -> [UPI],
URLs -> [URL], long digit runs -> [ACCOUNT]). This module is the second
line of defence: if a value that still looks like raw personal data
slips through, the event is rejected instead of stored.
"""

from __future__ import annotations

import re

# Placeholders produced by the client-side anonymizer. They are removed
# before scanning so legitimate anonymized text is never flagged.
PLACEHOLDERS = ("[PHONE]", "[OTP]", "[EMAIL]", "[UPI]", "[URL]", "[ACCOUNT]")

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("web URL", re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)),
    (
        "UPI id",
        re.compile(r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z][a-zA-Z0-9]{1,}\b"),
    ),
    (
        "card / account number",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    ),
    (
        "phone-like digit run",
        re.compile(r"\+?\d[\d -]{7,}\d"),
    ),
    ("long digit run", re.compile(r"\d{6,}")),
    (
        "digit code after an OTP keyword",
        re.compile(r"\botp\b[^0-9a-zA-Z]{0,12}\d{4,8}\b", re.IGNORECASE),
    ),
]


def find_raw_pii(text: str) -> str | None:
    """Return a short description of the first raw-PII-looking match, or None."""
    scrubbed = text
    for placeholder in PLACEHOLDERS:
        scrubbed = scrubbed.replace(placeholder, " ")
    for label, pattern in _PATTERNS:
        if pattern.search(scrubbed):
            return label
    return None


def contains_raw_pii(text: str) -> bool:
    return find_raw_pii(text) is not None
