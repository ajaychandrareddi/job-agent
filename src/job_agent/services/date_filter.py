"""Search-window selection and publication-date filtering.

Window rules:
* first run for a resume (``initial_search_completed`` is false): previous 30 days;
* recurring run: previous 24 hours;
* after missed runs: by default still only the previous 24 hours, so an outage never
  produces a backlog of stale auto-applications. ``MissedRunPolicy.COVER_GAP`` extends
  the window back to the last covered window end, capped at ``max_catch_up_hours``.

Date rules (publication time only; modification time is never used):
* the whole possible publication interval inside the window  -> eligible;
* interval entirely outside the window                       -> excluded;
* interval straddling a window edge (date-only / "3 days ago") -> review;
* missing or unverifiable date -> review (default) or skip, per configuration;
* published in the future beyond clock-skew tolerance        -> review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from job_agent.config import MissedRunPolicy, MissingPublicationDatePolicy, Settings
from job_agent.services.deduplication import DedupedJob


class WindowTrigger(StrEnum):
    INITIAL = "INITIAL"
    SCHEDULED = "SCHEDULED"
    CATCH_UP = "CATCH_UP"


@dataclass(frozen=True)
class SearchWindow:
    start: datetime
    end: datetime
    trigger: WindowTrigger

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("search window bounds must be timezone-aware")
        if self.start >= self.end:
            raise ValueError("search window start must precede end")

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600


def determine_search_window(
    *,
    initial_search_completed: bool,
    last_window_end: datetime | None,
    now: datetime,
    settings: Settings,
) -> SearchWindow:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not initial_search_completed:
        return SearchWindow(
            now - timedelta(days=settings.initial_search_days), now, WindowTrigger.INITIAL
        )
    start = now - timedelta(hours=settings.recurring_search_hours)
    if (
        settings.missed_run_policy is MissedRunPolicy.COVER_GAP
        and last_window_end is not None
        and last_window_end < start
    ):
        cap = now - timedelta(hours=settings.max_catch_up_hours)
        return SearchWindow(max(last_window_end, cap), now, WindowTrigger.CATCH_UP)
    return SearchWindow(start, now, WindowTrigger.SCHEDULED)


class DateFilterReason(StrEnum):
    MISSING_DATE = "missing_publication_date"
    UNVERIFIED_DATE = "unverified_publication_date"
    AMBIGUOUS_PRECISION = "publication_date_straddles_window"
    FUTURE_DATE = "publication_date_in_future"
    BEFORE_WINDOW = "published_before_window"
    AFTER_WINDOW = "published_after_window"


@dataclass
class DateFilterResult:
    eligible: list[DedupedJob] = field(default_factory=list)
    needs_review: list[tuple[DedupedJob, DateFilterReason]] = field(default_factory=list)
    excluded: list[tuple[DedupedJob, DateFilterReason]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"eligible": len(self.eligible)}
        for bucket, items in (("review", self.needs_review), ("excluded", self.excluded)):
            for _, reason in items:
                key = f"{bucket}:{reason.value}"
                out[key] = out.get(key, 0) + 1
        return out


def filter_by_publication_date(
    jobs: list[DedupedJob],
    window: SearchWindow,
    *,
    now: datetime,
    missing_date_policy: MissingPublicationDatePolicy,
    future_tolerance: timedelta = timedelta(minutes=5),
) -> DateFilterResult:
    result = DateFilterResult()
    for job in jobs:
        p = job.posting
        if p.published_at is None or p.published_at_latest is None:
            reason = (
                DateFilterReason.UNVERIFIED_DATE
                if "unverified_publication_value" in p.raw_metadata
                else DateFilterReason.MISSING_DATE
            )
            if missing_date_policy is MissingPublicationDatePolicy.SKIP:
                result.excluded.append((job, reason))
            else:
                result.needs_review.append((job, reason))
            continue
        earliest, latest = p.published_at, p.published_at_latest
        if earliest > now + future_tolerance:
            result.needs_review.append((job, DateFilterReason.FUTURE_DATE))
        elif latest < window.start:
            result.excluded.append((job, DateFilterReason.BEFORE_WINDOW))
        elif earliest > window.end + future_tolerance:
            result.excluded.append((job, DateFilterReason.AFTER_WINDOW))
        elif earliest >= window.start and latest <= window.end + future_tolerance:
            result.eligible.append(job)
        else:
            result.needs_review.append((job, DateFilterReason.AMBIGUOUS_PRECISION))
    return result
