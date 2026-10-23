"""Safety guard for destructive test fixtures (create_all/drop_all)."""

from __future__ import annotations

import pytest
import sqlalchemy as sa


def assert_test_database(url: str) -> None:
    """Fail loudly instead of dropping tables in a non-test PostgreSQL database."""
    parsed = sa.engine.make_url(url)
    if parsed.get_backend_name() == "sqlite":
        return
    if "test" not in (parsed.database or "").lower():
        pytest.fail(
            f"refusing to drop tables in database '{parsed.database}': "
            "use a dedicated database whose name contains 'test'",
            pytrace=False,
        )
