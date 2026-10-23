from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from job_agent.config import MissedRunPolicy, MissingPublicationDatePolicy, Settings
from job_agent.models.job import JobSource
from job_agent.services.date_filter import (
    DateFilterReason,
    SearchWindow,
    WindowTrigger,
    determine_search_window,
    filter_by_publication_date,
)
from job_agent.services.deduplication import DedupedJob, deduplicate
from job_agent.services.normalization import normalize_raw_posting
from tests.conftest import RUN_AT, make_raw


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def job(external_id: str, published: object, **kw: object) -> DedupedJob:
    p = normalize_raw_posting(
        make_raw(external_id, published_at=published, **kw), JobSource.INDEED, now=RUN_AT
    )
    return deduplicate([p])[0]


class TestSearchWindow:
    def test_first_run_searches_previous_30_days(self) -> None:
        w = determine_search_window(
            initial_search_completed=False, last_window_end=None, now=RUN_AT, settings=settings()
        )
        assert (w.start, w.end, w.trigger) == (
            RUN_AT - timedelta(days=30),
            RUN_AT,
            WindowTrigger.INITIAL,
        )

    def test_subsequent_run_searches_previous_24_hours(self) -> None:
        w = determine_search_window(
            initial_search_completed=True,
            last_window_end=RUN_AT - timedelta(hours=24),
            now=RUN_AT,
            settings=settings(),
        )
        assert (w.start, w.trigger) == (RUN_AT - timedelta(hours=24), WindowTrigger.SCHEDULED)
        assert w.hours == 24

    def test_restart_with_completed_flag_never_repeats_initial_search(self) -> None:
        # The flag is persisted; a restarted process reads it back as True.
        w = determine_search_window(
            initial_search_completed=True, last_window_end=None, now=RUN_AT, settings=settings()
        )
        assert w.trigger is WindowTrigger.SCHEDULED and w.hours == 24

    def test_missed_runs_default_policy_does_not_backfill(self) -> None:
        w = determine_search_window(
            initial_search_completed=True,
            last_window_end=RUN_AT - timedelta(days=5),
            now=RUN_AT,
            settings=settings(),
        )
        assert w.hours == 24 and w.trigger is WindowTrigger.SCHEDULED

    def test_cover_gap_policy_is_capped(self) -> None:
        s = settings(missed_run_policy=MissedRunPolicy.COVER_GAP, max_catch_up_hours=72)
        short = determine_search_window(
            initial_search_completed=True,
            last_window_end=RUN_AT - timedelta(hours=40),
            now=RUN_AT,
            settings=s,
        )
        assert short.hours == 40 and short.trigger is WindowTrigger.CATCH_UP
        long = determine_search_window(
            initial_search_completed=True,
            last_window_end=RUN_AT - timedelta(days=10),
            now=RUN_AT,
            settings=s,
        )
        assert long.hours == 72

    def test_configurable_lengths(self) -> None:
        s = settings(initial_search_days=14, recurring_search_hours=12)
        assert (
            determine_search_window(
                initial_search_completed=False, last_window_end=None, now=RUN_AT, settings=s
            ).hours
            == 14 * 24
        )
        assert (
            determine_search_window(
                initial_search_completed=True, last_window_end=None, now=RUN_AT, settings=s
            ).hours
            == 12
        )

    def test_invalid_windows(self) -> None:
        with pytest.raises(ValueError):
            SearchWindow(RUN_AT, RUN_AT, WindowTrigger.SCHEDULED)
        with pytest.raises(ValueError):
            SearchWindow(datetime(2026, 1, 1), RUN_AT, WindowTrigger.SCHEDULED)  # noqa: DTZ001
        with pytest.raises(ValueError):
            determine_search_window(
                initial_search_completed=True,
                last_window_end=None,
                now=datetime(2026, 1, 1),
                settings=settings(),
            )  # noqa: DTZ001


class TestPublicationDateFilter:
    window = SearchWindow(RUN_AT - timedelta(hours=24), RUN_AT, WindowTrigger.SCHEDULED)

    def _run(
        self,
        *jobs: DedupedJob,
        policy: MissingPublicationDatePolicy = MissingPublicationDatePolicy.EXCLUDE_FROM_AUTO_APPLY,
    ):  # type: ignore[no-untyped-def]  # noqa: E501
        return filter_by_publication_date(
            list(jobs), self.window, now=RUN_AT, missing_date_policy=policy
        )

    def test_boundaries_are_inclusive(self) -> None:
        r = self._run(
            job("start", self.window.start.isoformat()), job("end", self.window.end.isoformat())
        )
        assert len(r.eligible) == 2

    def test_outside_window_excluded(self) -> None:
        r = self._run(
            job("old", (self.window.start - timedelta(seconds=1)).isoformat()),
            job("future", (RUN_AT + timedelta(hours=2)).isoformat()),
        )
        assert [(j.posting.external_id, reason) for j, reason in r.excluded] == [
            ("old", DateFilterReason.BEFORE_WINDOW)
        ]
        assert [(j.posting.external_id, reason) for j, reason in r.needs_review] == [
            ("future", DateFilterReason.FUTURE_DATE)
        ]

    def test_small_clock_skew_tolerated(self) -> None:
        r = self._run(job("skew", (RUN_AT + timedelta(minutes=2)).isoformat()))
        assert len(r.eligible) == 1

    def test_missing_date_never_eligible(self) -> None:
        r = self._run(job("none", None), job("naive", "2026-10-01T10:00:00"))
        assert r.eligible == []
        assert {reason for _, reason in r.needs_review} == {
            DateFilterReason.MISSING_DATE,
            DateFilterReason.UNVERIFIED_DATE,
        }

    def test_missing_date_skip_policy(self) -> None:
        r = self._run(job("none", None), policy=MissingPublicationDatePolicy.SKIP)
        assert r.needs_review == [] and r.excluded[0][1] is DateFilterReason.MISSING_DATE

    def test_imprecise_dates(self) -> None:
        r = self._run(
            job("inside", "5 hours ago"),  # [6h, 5h] ago: fully inside
            job("straddle", "1 day ago"),  # [48h, 24h] ago: touches start edge
            job("stale", "3 days ago"),  # fully before
            job("day", "2026-09-30"),  # ±14h padding: straddles start
        )
        assert [j.posting.external_id for j in r.eligible] == ["inside"]
        assert {j.posting.external_id for j, _ in r.needs_review} == {"straddle", "day"}
        assert all(reason is DateFilterReason.AMBIGUOUS_PRECISION for _, reason in r.needs_review)
        assert [j.posting.external_id for j, _ in r.excluded] == ["stale"]

    def test_counts(self) -> None:
        r = self._run(job("a", (RUN_AT - timedelta(hours=1)).isoformat()), job("b", None))
        assert r.counts() == {"eligible": 1, "review:missing_publication_date": 1}

    def test_initial_window_includes_month_old_posts(self) -> None:
        initial = SearchWindow(RUN_AT - timedelta(days=30), RUN_AT, WindowTrigger.INITIAL)
        r = filter_by_publication_date(
            [
                job("m", (RUN_AT - timedelta(days=29)).isoformat()),
                job("x", (RUN_AT - timedelta(days=31)).isoformat()),
            ],
            initial,
            now=RUN_AT,
            missing_date_policy=MissingPublicationDatePolicy.EXCLUDE_FROM_AUTO_APPLY,
        )
        assert [j.posting.external_id for j in r.eligible] == ["m"]


def test_window_rejects_naive_now() -> None:
    with pytest.raises(ValueError):
        SearchWindow(RUN_AT - timedelta(hours=1), datetime.now(), WindowTrigger.SCHEDULED)  # noqa: DTZ005
    assert RUN_AT.tzinfo is UTC
