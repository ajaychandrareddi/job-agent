from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest

from job_agent.config import Settings
from job_agent.job_sources.base import (
    PartnerFeedAdapter,
    RateLimited,
    SourceAuthError,
    TransientSourceError,
)
from job_agent.job_sources.indeed import IndeedAdapter
from job_agent.job_sources.linkedin import LinkedInAdapter
from job_agent.job_sources.mock import StaticFeedClient
from job_agent.job_sources.registry import build_adapters
from job_agent.job_sources.ziprecruiter import ZipRecruiterAdapter
from job_agent.models.job import JobSource, RawPosting, SearchCriteria, SourceStatus
from job_agent.models.resume import WorkArrangement
from tests.conftest import RUN_AT, make_raw

CRITERIA = SearchCriteria(
    titles=["Senior Software Engineer"],
    locations=["New York, NY"],
    work_arrangements=[WorkArrangement.REMOTE],
)
POSTED_AFTER = RUN_AT - timedelta(hours=24)


class RecordingSleep:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def adapter(
    cls: type[PartnerFeedAdapter], client: StaticFeedClient | None, **kw: object
) -> tuple[PartnerFeedAdapter, RecordingSleep]:
    sleep = RecordingSleep()
    return cls(client, sleep=sleep, clock=lambda: RUN_AT, **kw), sleep  # type: ignore[arg-type]


def run(a: PartnerFeedAdapter, posted_after: datetime = POSTED_AFTER):  # type: ignore[no-untyped-def]
    return asyncio.run(a.search_jobs(CRITERIA, posted_after))


@pytest.mark.parametrize("cls", [LinkedInAdapter, IndeedAdapter, ZipRecruiterAdapter])
def test_unconfigured_sources_report_limitation_not_results(cls: type[PartnerFeedAdapter]) -> None:
    a, _ = adapter(cls, None)
    result = run(a)
    assert result.status is SourceStatus.UNAVAILABLE and not result.succeeded
    assert result.postings == [] and result.limitation
    assert result.manual_search_url is not None
    assert not a.supports_automated_search and not a.supports_automated_application


def test_manual_search_links() -> None:
    li = parse_qs(urlsplit(str(run(adapter(LinkedInAdapter, None)[0]).manual_search_url)).query)
    assert li["keywords"] == ["Senior Software Engineer"] and li["f_TPR"] == ["r86400"]
    assert li["location"] == ["New York, NY"] and li["f_WT"] == ["2"]
    indeed = run(adapter(IndeedAdapter, None)[0], RUN_AT - timedelta(days=30))
    assert parse_qs(urlsplit(str(indeed.manual_search_url)).query)["fromage"] == ["30"]
    zr = run(adapter(ZipRecruiterAdapter, None)[0])
    assert parse_qs(urlsplit(str(zr.manual_search_url)).query)["days"] == ["1"]


def test_successful_search_normalizes_and_counts_rejects() -> None:
    client = StaticFeedClient([make_raw("1"), make_raw("2"), make_raw("bad", url=None)])
    a, _ = adapter(IndeedAdapter, client)
    result = run(a)
    assert result.status is SourceStatus.OK and result.succeeded
    assert [p.external_id for p in result.postings] == ["1", "2"]
    assert (result.raw_count, result.rejected_count) == (3, 1)
    assert result.rejection_reasons == ["missing or invalid job URL"]
    assert client.calls[0][1] == POSTED_AFTER


def test_transient_errors_retried_with_backoff() -> None:
    client = StaticFeedClient([make_raw("1")], errors=[TransientSourceError(), TimeoutError()])
    a, sleep = adapter(IndeedAdapter, client)
    result = run(a)
    assert result.succeeded and result.attempts == 3 and sleep.waits == [1.0, 2.0]


def test_retries_exhausted_reports_failure() -> None:
    client = StaticFeedClient([make_raw("1")], errors=[TransientSourceError()] * 3)
    result = run(adapter(IndeedAdapter, client, max_retries=2)[0])
    assert result.status is SourceStatus.FAILED and result.attempts == 3
    assert result.postings == [] and "temporary failure" in (result.error or "")


def test_rate_limit_retry_after_is_honoured() -> None:
    client = StaticFeedClient([make_raw("1")], errors=[RateLimited(retry_after=30)])
    a, sleep = adapter(LinkedInAdapter, client)
    assert run(a).succeeded and sleep.waits == [30]


def test_excessive_retry_after_gives_up_instead_of_hammering() -> None:
    client = StaticFeedClient([make_raw("1")], errors=[RateLimited(retry_after=3600)])
    a, sleep = adapter(LinkedInAdapter, client, max_retry_after_seconds=120)
    result = run(a)
    assert result.status is SourceStatus.FAILED and result.error == "rate limited by source"
    assert sleep.waits == [] and len(client.calls) == 1


def test_auth_errors_not_retried() -> None:
    client = StaticFeedClient(errors=[SourceAuthError()])
    result = run(adapter(ZipRecruiterAdapter, client)[0])
    assert result.status is SourceStatus.FAILED and len(client.calls) == 1
    assert "authentication" in (result.error or "")


def test_unexpected_client_error_is_contained() -> None:
    client = StaticFeedClient(errors=[KeyError("boom")])
    result = run(adapter(IndeedAdapter, client)[0])
    assert result.status is SourceStatus.FAILED and result.error == "unexpected error: KeyError"


def test_slow_source_times_out() -> None:
    client = StaticFeedClient([make_raw("1")], delay_seconds=1.0)
    a = IndeedAdapter(client, timeout_seconds=0.05, max_retries=0, clock=lambda: RUN_AT)
    result = asyncio.run(a.search_jobs(CRITERIA, POSTED_AFTER))
    assert result.status is SourceStatus.FAILED and "TimeoutError" in (result.error or "")


def test_max_results_enforced() -> None:
    client = StaticFeedClient([make_raw(str(i)) for i in range(10)])
    result = run(adapter(IndeedAdapter, client, max_results=3)[0])
    assert len(result.postings) == 3 and client.calls[0][2] == 3


def test_naive_posted_after_rejected() -> None:
    a, _ = adapter(IndeedAdapter, StaticFeedClient())
    with pytest.raises(ValueError):
        asyncio.run(a.search_jobs(CRITERIA, datetime(2026, 1, 1)))  # noqa: DTZ001


def test_registry_respects_enabled_flags_and_clients() -> None:
    s = Settings(_env_file=None, ziprecruiter_enabled=False)  # type: ignore[call-arg]
    client = StaticFeedClient([RawPosting(external_id="x")])
    adapters = build_adapters(s, {JobSource.INDEED: client})
    assert [a.source for a in adapters] == [JobSource.LINKEDIN, JobSource.INDEED]
    assert [a.supports_automated_search for a in adapters] == [False, True]
