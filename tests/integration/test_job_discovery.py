from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from job_agent.config import Settings
from job_agent.job_sources.base import JobSourceAdapter, TransientSourceError
from job_agent.job_sources.indeed import IndeedAdapter
from job_agent.job_sources.linkedin import LinkedInAdapter
from job_agent.job_sources.mock import StaticFeedClient
from job_agent.job_sources.ziprecruiter import ZipRecruiterAdapter
from job_agent.models.application import ApplicationStatus
from job_agent.models.job import JobSource, SearchCriteria, SourceSearchResult, SourceStatus
from job_agent.models.resume import JobSearchPreferences
from job_agent.persistence.orm import Application, JobPostingRecord, User
from job_agent.repositories.jobs import JobPostingRepository
from job_agent.services.date_filter import SearchWindow, WindowTrigger
from job_agent.services.deduplication import application_key, deduplicate
from job_agent.services.job_discovery import (
    JobDiscoveryService,
    NoSearchCriteria,
    build_search_criteria,
)
from job_agent.services.normalization import normalize_raw_posting
from job_agent.services.resume_service import ResumeService
from tests.conftest import RUN_AT, make_raw

WINDOW = SearchWindow(RUN_AT - timedelta(hours=24), RUN_AT, WindowTrigger.SCHEDULED)
CRITERIA = SearchCriteria(titles=["Senior Software Engineer"])
SHARED_ATS = "https://boards.greenhouse.io/acme/jobs/42"


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def feed(source_cls: type, raws: list, **client_kw: object) -> JobSourceAdapter:  # type: ignore[type-arg]
    return source_cls(StaticFeedClient(raws, **client_kw), clock=lambda: RUN_AT, max_retries=0)  # type: ignore[no-any-return]


def discover(adapters: list[JobSourceAdapter], **kw: object):  # type: ignore[no-untyped-def]
    service = JobDiscoveryService(adapters, settings(**kw), clock=lambda: RUN_AT)
    return asyncio.run(service.discover(CRITERIA, WINDOW))


class TestDiscovery:
    def test_cross_source_duplicates_are_merged_and_dates_filtered(self) -> None:
        report = discover(
            [
                feed(
                    LinkedInAdapter,
                    [
                        make_raw("li-1", apply_url=SHARED_ATS),
                        make_raw(
                            "li-old",
                            title="Data Engineer",
                            published_at=(RUN_AT - timedelta(days=3)).isoformat(),
                        ),
                    ],
                ),
                feed(
                    IndeedAdapter,
                    [make_raw("in-1", apply_url=SHARED_ATS, company_name="ACME HEALTH LLC")],
                ),
                feed(
                    ZipRecruiterAdapter,
                    [
                        make_raw("zr-1", apply_url=SHARED_ATS + "?utm_source=zr"),
                        make_raw("zr-nodate", title="QA Lead", published_at=None),
                    ],
                ),
            ]
        )
        assert report.discovered_by_source == {"linkedin": 2, "indeed": 1, "ziprecruiter": 2}
        assert report.total_discovered == 5 and report.unique_count == 3
        assert [j.posting.external_id for j in report.date_filter.eligible] == ["li-1"]
        assert len(report.date_filter.eligible[0].duplicates) == 2
        assert [j.posting.external_id for j, _ in report.date_filter.needs_review] == ["zr-nodate"]
        assert [j.posting.external_id for j, _ in report.date_filter.excluded] == ["li-old"]
        summary = report.summary()
        assert summary["unique_after_dedup"] == 3 and summary["trigger"] == "SCHEDULED"

    def test_failed_source_does_not_fail_search(self) -> None:
        report = discover(
            [
                feed(LinkedInAdapter, [make_raw("li-1")]),
                feed(IndeedAdapter, [], errors=[TransientSourceError()]),
                ZipRecruiterAdapter(None, clock=lambda: RUN_AT),
            ]
        )
        assert report.failed_sources == [JobSource.INDEED]
        assert [lim.source for lim in report.limitations] == [JobSource.ZIPRECRUITER]
        assert report.limitations[0].manual_search_url
        assert len(report.date_filter.eligible) == 1 and not report.all_sources_failed
        assert report.summary()["failed_sources"] == {
            "indeed": "temporary failure: TransientSourceError"
        }

    def test_all_sources_failed_is_detectable(self) -> None:
        report = discover(
            [
                feed(IndeedAdapter, [], errors=[TransientSourceError()]),
                LinkedInAdapter(None, clock=lambda: RUN_AT),
            ]
        )
        assert report.all_sources_failed and report.unique_count == 0

    def test_unavailable_sources_never_produce_postings(self) -> None:
        report = discover(
            [LinkedInAdapter(None, clock=lambda: RUN_AT), IndeedAdapter(None, clock=lambda: RUN_AT)]
        )
        assert report.total_discovered == 0 and len(report.limitations) == 2
        assert not report.all_sources_failed

    def test_adapter_crash_is_isolated(self) -> None:
        class Broken(JobSourceAdapter):
            source = JobSource.INDEED

            async def search_jobs(
                self, search_criteria: SearchCriteria, posted_after: datetime
            ) -> SourceSearchResult:
                raise RuntimeError("bug")

        report = discover([Broken(), feed(LinkedInAdapter, [make_raw("ok")])])
        statuses = {r.source: r.status for r in report.source_results}
        assert statuses == {
            JobSource.INDEED: SourceStatus.FAILED,
            JobSource.LINKEDIN: SourceStatus.OK,
        }
        assert report.unique_count == 1

    def test_concurrency_limit_respected(self) -> None:
        active = 0
        peak = 0

        class Slow(JobSourceAdapter):
            def __init__(self, source: JobSource) -> None:
                self.source = source

            async def search_jobs(
                self, search_criteria: SearchCriteria, posted_after: datetime
            ) -> SourceSearchResult:
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.02)
                active -= 1
                return SourceSearchResult(source=self.source)

        discover([Slow(s) for s in JobSource], max_concurrent_job_searches=1)
        assert peak == 1
        peak = 0
        discover([Slow(s) for s in JobSource], max_concurrent_job_searches=3)
        assert peak == 3

    def test_duplicate_adapters_rejected(self) -> None:
        with pytest.raises(ValueError):
            JobDiscoveryService([LinkedInAdapter(), LinkedInAdapter()], settings())


def test_search_criteria_require_approved_titles() -> None:
    with pytest.raises(NoSearchCriteria):
        build_search_criteria(JobSearchPreferences())
    crit = build_search_criteria(JobSearchPreferences(preferred_titles=["SRE"], locations=["NYC"]))
    assert crit.titles == ["SRE"] and crit.locations == ["NYC"]


# --------------------------------------------------------------------------- persistence


def _group(*raws_by_source: tuple[JobSource, dict[str, object]]):  # type: ignore[no-untyped-def]
    postings = [
        normalize_raw_posting(make_raw(**kw), src, now=RUN_AT)  # type: ignore[arg-type]
        for src, kw in raws_by_source
    ]
    groups = deduplicate(postings)
    assert len(groups) == 1
    return groups[0]


class TestJobPostingRepository:
    def test_persist_group_upserts_all_members(self, session: Session) -> None:
        repo = JobPostingRepository(session)
        g = _group(
            (JobSource.LINKEDIN, {"external_id": "li"}), (JobSource.INDEED, {"external_id": "in"})
        )
        rep = repo.persist_group(g, now=RUN_AT)
        records = session.query(JobPostingRecord).all()
        assert len(records) == 2 and {r.canonical_key for r in records} == {g.canonical_key}
        assert rep.source is JobSource.LINKEDIN and rep.first_discovered_at == RUN_AT

        later = RUN_AT + timedelta(days=1)
        repo.persist_group(g, now=later)
        assert session.query(JobPostingRecord).count() == 2
        assert rep.last_seen_at == later and rep.first_discovered_at == RUN_AT

    def test_publication_date_is_immutable(self, session: Session) -> None:
        repo = JobPostingRepository(session)
        original = RUN_AT - timedelta(days=10)
        repo.persist_group(
            _group((JobSource.INDEED, {"external_id": "x", "published_at": original.isoformat()})),
            now=RUN_AT,
        )
        # Platform "refreshes" the posting: a newer publication value must not make it new.
        rec = repo.persist_group(
            _group(
                (
                    JobSource.INDEED,
                    {
                        "external_id": "x",
                        "published_at": RUN_AT.isoformat(),
                        "updated_at": RUN_AT.isoformat(),
                    },
                )
            ),
            now=RUN_AT,
        )
        assert rec.published_at == original and rec.source_updated_at == RUN_AT

    def test_missing_date_filled_once_known(self, session: Session) -> None:
        repo = JobPostingRepository(session)
        repo.persist_group(
            _group((JobSource.INDEED, {"external_id": "y", "published_at": None})), now=RUN_AT
        )
        rec = repo.persist_group(_group((JobSource.INDEED, {"external_id": "y"})), now=RUN_AT)
        assert rec.published_at == RUN_AT - timedelta(hours=2)
        assert rec.publication_date_unverified is False

    def test_canonical_key_stable_across_runs(self, session: Session) -> None:
        repo = JobPostingRepository(session)
        first = _group(
            (
                JobSource.INDEED,
                {"external_id": "in", "title": "Platform Engineer", "apply_url": SHARED_ATS},
            )
        )
        repo.persist_group(first, now=RUN_AT)
        # Next run also finds the LinkedIn copy with a differently-normalising title.
        second = _group(
            (
                JobSource.LINKEDIN,
                {"external_id": "li", "title": "Engineer, Platform", "apply_url": SHARED_ATS},
            ),
            (
                JobSource.INDEED,
                {"external_id": "in", "title": "Platform Engineer", "apply_url": SHARED_ATS},
            ),
        )
        repo.persist_group(second, now=RUN_AT)
        assert second.canonical_key == first.canonical_key
        assert {r.canonical_key for r in session.query(JobPostingRecord)} == {first.canonical_key}

    def test_previously_applied_jobs_filtered_per_resume(
        self,
        session: Session,
        user: User,
        resume_service: ResumeService,
        sample_docx: bytes,
    ) -> None:
        from tests.integration.test_resume_service import _second_resume

        r1 = resume_service.create_resume(user.id, "a.docx", sample_docx).resume.id
        r2 = resume_service.create_resume(user.id, "b.docx", _second_resume()).resume.id
        repo = JobPostingRepository(session)

        applied = _group((JobSource.LINKEDIN, {"external_id": "li-9", "apply_url": None}))
        rec = repo.persist_group(applied, now=RUN_AT)
        session.add(
            Application(
                application_key=application_key(user.id, r1, applied.canonical_key),
                user_id=user.id,
                resume_id=r1,
                resume_version=1,
                job_posting_id=rec.id,
                status=ApplicationStatus.APPLIED,
            )
        )
        session.flush()

        # Same job resurfaces on another board the next day, plus a genuinely new job.
        resurfaced = _group((JobSource.ZIPRECRUITER, {"external_id": "zr-9", "apply_url": None}))
        fresh = _group(
            (JobSource.INDEED, {"external_id": "new", "title": "Staff Engineer", "apply_url": None})
        )
        new, seen = repo.partition_untracked(r1, [resurfaced, fresh])
        assert [j.posting.external_id for j in new] == ["new"]
        assert [j.posting.external_id for j in seen] == ["zr-9"]
        # A different resume has not applied, so it may still evaluate the same job.
        new2, seen2 = repo.partition_untracked(r2, [resurfaced, fresh])
        assert len(new2) == 2 and seen2 == []
        assert repo.tracked_canonical_keys(uuid.uuid4()) == set()
