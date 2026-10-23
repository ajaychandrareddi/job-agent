"""Portable column types (PostgreSQL in production, SQLite allowed in unit tests)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect

# JSONB on PostgreSQL, generic JSON elsewhere.
JSONType = sa.JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(sa.TypeDecorator[datetime]):
    """Stores timezone-aware datetimes as UTC and always returns aware UTC values.

    Naive datetimes are rejected on write so local-time bugs cannot slip in.
    """

    impl = sa.DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime passed to UTCDateTime column")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:  # SQLite drops tzinfo
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def enum_column(enum_cls: type[StrEnum], name: str) -> sa.Enum:
    """Non-native enum stored as VARCHAR with a CHECK constraint (easy to migrate)."""
    return sa.Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


def json_default_list() -> list[Any]:
    return []
