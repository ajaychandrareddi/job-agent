"""Shared helpers for domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator


def utcnow() -> datetime:
    return datetime.now(UTC)


def _require_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


# A timezone-aware datetime normalised to UTC. Naive datetimes are rejected.
AwareDatetime = Annotated[datetime, AfterValidator(_require_aware_utc)]


def normalize_token(value: str) -> str:
    """Lower-case, collapse whitespace. Used for skill / title comparisons."""
    return " ".join(value.lower().split())
