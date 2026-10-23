"""Migrations must produce exactly the ORM schema, on SQLite and (if available) PostgreSQL."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext

from job_agent.persistence.orm import Base
from job_agent.persistence.schema_filters import LANGGRAPH_CHECKPOINT_TABLES, include_name

ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(connection: sa.Connection) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["connection"] = connection
    cfg.attributes["configure_logger"] = False
    return cfg


def _upgrade_compare_downgrade(engine: sa.Engine) -> None:
    with engine.begin() as conn:
        command.upgrade(_alembic_config(conn), "head")
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"include_name": include_name})
        diff = compare_metadata(ctx, Base.metadata)
        assert diff == [], f"migration drift: {diff}"
        tables = set(sa.inspect(conn).get_table_names())
        assert set(Base.metadata.tables) <= tables
    with engine.begin() as conn:
        command.downgrade(_alembic_config(conn), "base")
    with engine.connect() as conn:
        remaining = set(sa.inspect(conn).get_table_names()) - LANGGRAPH_CHECKPOINT_TABLES
        assert remaining <= {"alembic_version"}


def test_migrations_sqlite(tmp_path: Path) -> None:
    _upgrade_compare_downgrade(sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}"))


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_migrations_postgres() -> None:
    """Runs in a throwaway database so leftovers from other tests can't mask drift."""
    base = sa.engine.make_url(os.environ["TEST_DATABASE_URL"])
    name = f"migtest_{uuid.uuid4().hex[:12]}"
    admin = sa.create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    engine = sa.create_engine(base.set(database=name))
    try:
        _upgrade_compare_downgrade(engine)
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


def test_autogenerate_ignores_langgraph_checkpoint_tables(tmp_path: Path) -> None:
    """A checkpoint table in the database must never produce a DROP in autogenerate."""
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'lg.db'}")
    with engine.begin() as conn:
        command.upgrade(_alembic_config(conn), "head")
        conn.execute(sa.text("CREATE TABLE checkpoints (thread_id TEXT PRIMARY KEY)"))
        conn.execute(sa.text("CREATE TABLE checkpoint_writes (thread_id TEXT PRIMARY KEY)"))
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"include_name": include_name})
        assert compare_metadata(ctx, Base.metadata) == []
        unfiltered = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        assert any(d[0] == "remove_table" for d in unfiltered)  # the hazard the filter blocks
