"""Search all sources for one resume, then deduplicate and date-filter the results.

This is the unit the Phase 4 LangGraph nodes (search_* -> normalize -> deduplicate ->
filter_by_publication_date) call. Sources run concurrently under a semaphore
(``MAX_CONCURRENT_JOB_SEARCHES``); a failing source never fails the whole search.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from job_agent.config import Settings
from job_agent.job_sources.base import JobSourceAdapter
from job_agent.models.job import JobSource, SearchCriteria, SourceSearchResult, SourceStatus
from job_agent.models.resume import JobSearchPreferences
from job_agent.observability.logging import get_logger
from job_agent.services.date_filter import (
    DateFilterResult,
    SearchWindow,
    filter_by_publication_date,
)
from job_agent.services.deduplication import DedupedJob, deduplicate

log = get_logger(__name__)


class NoSearchCriteria(ValueError):
    pass


def build_search_criteria(prefs: JobSearchPreferences) -> SearchCriteria:
    if not prefs.preferred_titles:
        raise NoSearchCriteria("at least one approved preferred title is required to search")
    return SearchCriteria(
        titles=prefs.preferred_titles,
        keywords=prefs.keywords,
        locations=prefs.locations,
        work_arrangements=prefs.work_arrangements,
        employment_types=prefs.employment_types,
    )


@dataclass
class SourceLimitation:
    source: JobSource
    reason: str
    manual_search_url: str | None


@dataclass
class DiscoveryReport:
    window: SearchWindow
    source_results: list[SourceSearchResult]
    jobs: list[DedupedJob]
    date_filter: DateFilterResult
    duration_seconds: float = 0.0
    limitations: list[SourceLimitation] = field(default_factory=list)

    @property
    def discovered_by_source(self) -> dict[str, int]:
        return {r.source.value: len(r.postings) for r in self.source_results}

    @property
    def total_discovered(self) -> int:
        return sum(len(r.postings) for r in self.source_results)

    @property
    def unique_count(self) -> int:
        return len(self.jobs)

    @property
    def failed_sources(self) -> list[JobSource]:
        return [r.source for r in self.source_results if r.status is SourceStatus.FAILED]

    @property
    def all_sources_failed(self) -> bool:
        searched = [r for r in self.source_results if r.status is not SourceStatus.UNAVAILABLE]
        return bool(searched) and all(r.status is SourceStatus.FAILED for r in searched)

    def summary(self) -> dict[str, object]:
        return {
            "window_start": self.window.start.isoformat(),
            "window_end": self.window.end.isoformat(),
            "trigger": self.window.trigger.value,
            "discovered_by_source": self.discovered_by_source,
            "total_discovered": self.total_discovered,
            "unique_after_dedup": self.unique_count,
            "date_filter": self.date_filter.counts(),
            "failed_sources": {
                r.source.value: r.error
                for r in self.source_results
                if r.status is SourceStatus.FAILED
            },
            "unavailable_sources": {
                lim.source.value: {"reason": lim.reason, "manual_search_url": lim.manual_search_url}
                for lim in self.limitations
            },
            "rejected_records": {
                r.source.value: r.rejected_count for r in self.source_results if r.rejected_count
            },
        }


class JobDiscoveryService:
    def __init__(
        self,
        adapters: Sequence[JobSourceAdapter],
        settings: Settings,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if len({a.source for a in adapters}) != len(adapters):
            raise ValueError("at most one adapter per source")
        self._adapters = list(adapters)
        self._settings = settings
        self._clock = clock

    async def discover(self, criteria: SearchCriteria, window: SearchWindow) -> DiscoveryReport:
        started = time.monotonic()
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_job_searches)

        async def run(adapter: JobSourceAdapter) -> SourceSearchResult:
            async with semaphore:
                try:
                    return await adapter.search_jobs(criteria, window.start)
                except Exception as exc:  # adapter bug: isolate it to this source
                    log.error(
                        "job_source_adapter_crashed",
                        source=adapter.source.value,
                        error=type(exc).__name__,
                    )
                    return SourceSearchResult(
                        source=adapter.source,
                        status=SourceStatus.FAILED,
                        error=f"adapter error: {type(exc).__name__}",
                    )

        results = list(await asyncio.gather(*(run(a) for a in self._adapters)))
        postings = [p for r in results for p in r.postings]
        jobs = deduplicate(postings)
        date_result = filter_by_publication_date(
            jobs,
            window,
            now=self._clock(),
            missing_date_policy=self._settings.missing_publication_date_policy,
            future_tolerance=timedelta(seconds=self._settings.future_date_tolerance_seconds),
        )
        limitations = [
            SourceLimitation(
                r.source,
                r.limitation or "unavailable",
                str(r.manual_search_url) if r.manual_search_url else None,
            )
            for r in results
            if r.status is SourceStatus.UNAVAILABLE
        ]
        report = DiscoveryReport(
            window=window,
            source_results=results,
            jobs=jobs,
            date_filter=date_result,
            duration_seconds=time.monotonic() - started,
            limitations=limitations,
        )
        log.info(
            "job_discovery_completed",
            **{
                k: v
                for k, v in report.summary().items()
                if k in ("discovered_by_source", "unique_after_dedup", "date_filter", "trigger")
            },
        )
        return report
