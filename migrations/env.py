"""Alembic environment. Database URL comes from application settings
(or ``-x db_url=...`` / an explicitly passed connection in tests)."""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any

import sqlalchemy as sa
from alembic import context

from job_agent.config import get_settings
from job_agent.persistence.orm import Base
from job_agent.persistence.schema_filters import include_name
from job_agent.persistence.types import UTCDateTime

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    return (
        x_args.get("db_url")
        or config.get_main_option("sqlalchemy.url")
        or (get_settings().database_url)
    )


def _render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    # Keep migrations independent of app code: UTCDateTime is a plain timestamptz in the DB.
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    return False


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_item=_render_item,
        include_name=include_name,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        engine = sa.create_engine(_url(), poolclass=sa.pool.NullPool)
        with engine.connect() as conn:
            _run(conn)
    else:
        _run(connection)


def _run(connection: sa.Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_item=_render_item,
        include_name=include_name,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
