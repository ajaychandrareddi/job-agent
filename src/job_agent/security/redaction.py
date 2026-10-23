"""PII redaction for logs and error messages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Keys whose values must never be logged.
SENSITIVE_KEYS = frozenset(
    {
        "resume_text",
        "extracted_text",
        "text",
        "content",
        "profile",
        "answer",
        "answers",
        "authorization",
        "token",
        "api_key",
        "llm_api_key",
        "password",
        "cookie",
        "secret",
        "email",
        "phone",
        "full_name",
    }
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"(?i)\b(bearer|token|api[_-]?key)\s*[:=]?\s*[A-Za-z0-9._\-]{8,}"), r"\1 [SECRET]"),
]


def redact_text(value: str) -> str:
    for pattern, repl in _PATTERNS:
        value = pattern.sub(repl, value)
    return value


def redact(value: Any, *, _depth: int = 0) -> Any:
    """Recursively redact a log payload."""
    if _depth > 6:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            k: ("[REDACTED]" if str(k).lower() in SENSITIVE_KEYS else redact(v, _depth=_depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v, _depth=_depth + 1) for v in value]
    return value
