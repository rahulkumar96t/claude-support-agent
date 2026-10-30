"""PII redaction for logs and traces.

Redaction is applied to what we *write down* (JSONL traces, log records), never to the
conversation history sent to the model: rewriting history would change what the model saw
and break the append-only contract the Messages API relies on.

The detectors are regular expressions plus a Luhn check for card numbers, and an optional
list of known values (for example the signed-in customer's own name and address). They are
deliberately conservative about order ids, SKUs, tracking numbers, dates and prices.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD_CANDIDATE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
PHONE_CANDIDATE = re.compile(r"(?<![\w-])\+?\(?\d{1,4}\)?(?:[ .-]?\(?\d{2,4}\)?){2,4}(?![\w-])")
STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Place|Pl)\b"
)
DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "[CARD]" if 13 <= len(digits) <= 19 and luhn_valid(digits) else match.group()


def _phone(match: re.Match[str]) -> str:
    raw = match.group()
    digits = re.sub(r"\D", "", raw)
    has_separator = bool(re.search(r"[ .()+-]", raw))
    if 7 <= len(digits) <= 15 and has_separator and not DATE_LIKE.match(raw):
        return "[PHONE]"
    return raw


class Redactor:
    def __init__(self, known_values: Iterable[str] = ()) -> None:
        values = sorted({v.strip() for v in known_values if v and len(v.strip()) >= 3}, key=len)
        self._known = [re.compile(re.escape(v), re.IGNORECASE) for v in reversed(values)]

    def text(self, value: str) -> str:
        for pattern in self._known:
            value = pattern.sub("[REDACTED]", value)
        value = EMAIL.sub("[EMAIL]", value)
        value = CARD_CANDIDATE.sub(_card, value)
        value = STREET.sub("[ADDRESS]", value)
        value = PHONE_CANDIDATE.sub(_phone, value)
        return value

    def value(self, obj: Any) -> Any:
        """Redact strings inside nested dict/list/tuple structures."""
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            return {k: self.value(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [self.value(v) for v in obj]
        return obj


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the fully formatted message of every record."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        super().__init__()
        self.redactor = redactor or Redactor()

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redactor.text(record.getMessage())
        record.args = None
        return True
