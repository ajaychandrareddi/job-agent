"""Create the PostgreSQL test database if it does not exist (used by validate.sh).

Refuses database names without "test" so the destructive test fixtures can never be
pointed at an application database.
"""

from __future__ import annotations

import sys

import sqlalchemy as sa


def main(url: str) -> int:
    target = sa.engine.make_url(url)
    name = target.database or ""
    if "test" not in name.lower():
        print(f"refusing: TEST_DATABASE_URL database '{name}' must contain 'test'")
        return 1
    admin = sa.create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.scalar(sa.text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name})
        if not exists:
            conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
            print(f"created test database {name}")
    admin.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
