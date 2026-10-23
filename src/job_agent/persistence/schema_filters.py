"""Tables in the application database that Alembic must never manage.

LangGraph's PostgreSQL checkpointer creates and migrates its own tables in the same
database. Without this filter, ``alembic revision --autogenerate`` would emit DROP
TABLE statements for them and destroy in-flight workflow state.
"""

from __future__ import annotations

from typing import Any

LANGGRAPH_CHECKPOINT_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)


def include_name(name: str | None, type_: str, parent_names: Any) -> bool:
    """Alembic ``include_name`` hook: skip externally managed tables."""
    return not (type_ == "table" and name in LANGGRAPH_CHECKPOINT_TABLES)
