#!/usr/bin/env bash
# Mandatory phase-completion checks. Exit code is non-zero if any check fails.
# Usage: scripts/validate.sh            (uses .venv)
#        PY=/path/to/python scripts/validate.sh
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${VENV_BIN:-.venv/bin}"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
failures=()

run() {
  local name="$1"; shift
  echo "==> $name"
  if "$@"; then echo "    PASS: $name"; else echo "    FAIL: $name"; failures+=("$name"); fi
}

if [[ -n "${TEST_DATABASE_URL:-}" ]]; then
  # Tests create and drop tables: only ever against a dedicated *test* database.
  # Runs first because the main test step also runs the PostgreSQL-marked tests.
  run "PostgreSQL test database ready" "$BIN/python" scripts/ensure_test_db.py "$TEST_DATABASE_URL"
fi

run "lint (ruff check)"        "$BIN/ruff" check src tests migrations
run "format (ruff format)"     "$BIN/ruff" format --check src tests migrations
run "types (mypy --strict)"    "$BIN/mypy" src
run "tests + coverage >= 85%"  "$BIN/pytest" -q -p no:cacheprovider --cov --cov-fail-under=85 \
    --cov-report=term --cov-report="json:$SCRATCH/coverage.json"
run "per-module coverage >= 85% (logic + workflows)" \
    "$BIN/python" scripts/check_module_coverage.py "$SCRATCH/coverage.json"
run "migration renders for PostgreSQL" bash -c \
    "'$BIN/alembic' -x db_url=postgresql+psycopg://u@localhost/x upgrade head --sql \
     >'$SCRATCH/0001.sql' 2>'$SCRATCH/render.log' && grep -q 'CREATE TABLE applications' '$SCRATCH/0001.sql'"
run "migration up/down on SQLite" bash -c \
    "'$BIN/alembic' -x db_url=sqlite:///$SCRATCH/m.db upgrade head 2>/dev/null && \
     '$BIN/alembic' -x db_url=sqlite:///$SCRATCH/m.db downgrade base 2>/dev/null"

if [[ -n "${TEST_DATABASE_URL:-}" ]]; then
  run "PostgreSQL migration test" "$BIN/pytest" -q -p no:cacheprovider -m postgres
  run "full test suite on PostgreSQL" env TEST_DATABASE_URL_UNIT="$TEST_DATABASE_URL" \
      "$BIN/pytest" -q -p no:cacheprovider -m "not postgres"
else
  echo "==> PostgreSQL checks: NOT RUN (TEST_DATABASE_URL unset) -> record as BLOCKED"
fi

echo
if ((${#failures[@]})); then
  echo "VALIDATION FAILED: ${failures[*]}"
  exit 1
fi
echo "ALL CHECKS PASSED"
