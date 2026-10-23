"""Deterministic in-memory client for development and tests.

``StaticFeedClient`` stands in for an approved platform client. Plug it into any real
adapter (``LinkedInAdapter(client=StaticFeedClient(...))``) to exercise the full
retry, normalisation and filtering path without network access.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime

from job_agent.models.job import RawPosting, SearchCriteria


class StaticFeedClient:
    def __init__(
        self,
        postings: Sequence[RawPosting] = (),
        *,
        errors: Sequence[BaseException] = (),
        delay_seconds: float = 0.0,
    ) -> None:
        self.postings = list(postings)
        self._errors = list(errors)  # raised in order, one per call, before succeeding
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[SearchCriteria, datetime, int]] = []

    async def search(
        self, criteria: SearchCriteria, posted_after: datetime, *, max_results: int
    ) -> list[RawPosting]:
        self.calls.append((criteria, posted_after, max_results))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self._errors:
            raise self._errors.pop(0)
        return self.postings[:max_results]
