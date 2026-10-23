"""Fail if any business-logic or workflow module is below the per-module floor.

Spec §12: at least 85% coverage for business logic and workflow nodes. The overall
total can hide a weak module, so each one is checked individually.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FLOOR = 85.0
SCOPES = (
    "services/",
    "workflows/",
    "repositories/",
    "job_sources/",
    "models/",
    "scheduler/",
    "integrations/",
    "security/",
)


def main(path: str = "coverage.json") -> int:
    data = json.loads(Path(path).read_text())
    failures = []
    for name, info in sorted(data["files"].items()):
        rel = name.split("src/job_agent/", 1)[-1]
        if not rel.startswith(SCOPES) or rel.endswith("__init__.py"):
            continue
        pct = info["summary"]["percent_covered"]
        if pct < FLOOR:
            failures.append(f"{rel}: {pct:.1f}%")
    if failures:
        print(f"modules below {FLOOR:.0f}%:\n  " + "\n  ".join(failures))
        return 1
    print(f"all business-logic and workflow modules >= {FLOOR:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
