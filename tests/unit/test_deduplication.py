from __future__ import annotations

import uuid
from datetime import timedelta

from job_agent.models.job import JobPosting, JobSource
from job_agent.models.resume import WorkArrangement
from job_agent.services.deduplication import (
    application_key,
    canonical_job_key,
    deduplicate,
    normalize_company,
    normalize_location,
    normalize_title,
)
from job_agent.services.normalization import normalize_raw_posting
from tests.conftest import RUN_AT, make_raw


def posting(source: JobSource, external_id: str, **kw: object) -> JobPosting:
    return normalize_raw_posting(make_raw(external_id, **kw), source, now=RUN_AT)


class TestNormalizers:
    def test_company(self) -> None:
        assert normalize_company("Acme Health, Inc.") == normalize_company("ACME Health LLC")
        assert normalize_company("The Home Depot") == "home depot"
        assert normalize_company("AT&T") == "at and t"

    def test_title(self) -> None:
        assert normalize_title("Sr. Software Eng (Remote)") == "senior software engineer"
        assert normalize_title("Software Engineer II") == normalize_title("software engineer 2")
        assert normalize_title("Backend Engineer") != normalize_title("Frontend Engineer")

    def test_location(self) -> None:
        assert normalize_location("New York, New York 10001", WorkArrangement.UNKNOWN) == (
            normalize_location("New York, NY", WorkArrangement.HYBRID)
        )
        assert normalize_location("Anywhere", WorkArrangement.REMOTE) == "remote"
        assert normalize_location(None, WorkArrangement.UNKNOWN) == ""


class TestDeduplicate:
    def test_same_job_on_three_sources_is_merged(self) -> None:
        jobs = deduplicate(
            [
                posting(
                    JobSource.LINKEDIN,
                    "li-1",
                    company_name="Acme Health Inc",
                    title="Sr. Software Engineer",
                    apply_url=None,
                ),
                posting(
                    JobSource.INDEED,
                    "in-1",
                    company_name="Acme Health, Inc.",
                    location="New York, New York",
                    apply_url=None,
                ),
                posting(
                    JobSource.ZIPRECRUITER,
                    "zr-1",
                    company_name="ACME HEALTH LLC",
                    title="Senior Software Engineer",
                    apply_url=None,
                ),
            ]
        )
        assert len(jobs) == 1
        assert jobs[0].sources == [JobSource.LINKEDIN, JobSource.INDEED, JobSource.ZIPRECRUITER]
        assert len(jobs[0].duplicates) == 2

    def test_shared_application_url_merges_differently_titled_postings(self) -> None:
        same_ats = "https://boards.greenhouse.io/acme/jobs/777"
        jobs = deduplicate(
            [
                posting(
                    JobSource.LINKEDIN,
                    "li-2",
                    title="Software Engineer, Platform",
                    apply_url=same_ats,
                ),
                posting(
                    JobSource.INDEED,
                    "in-2",
                    title="Platform Software Engineer",
                    apply_url=same_ats + "?utm_source=indeed",
                ),
            ]
        )
        assert len(jobs) == 1 and len(jobs[0].member_keys) == 2

    def test_union_is_transitive(self) -> None:
        a = posting(JobSource.LINKEDIN, "a", title="Data Engineer", apply_url="https://x.io/1")
        b = posting(JobSource.INDEED, "b", title="Data Engineer", apply_url="https://x.io/2")
        c = posting(JobSource.ZIPRECRUITER, "c", title="Data Eng II", apply_url="https://x.io/2")
        assert len(deduplicate([a, b, c])) == 1

    def test_different_roles_or_locations_are_kept_apart(self) -> None:
        jobs = deduplicate(
            [
                posting(JobSource.LINKEDIN, "1", apply_url=None),
                posting(JobSource.INDEED, "2", location="Boston, MA", apply_url=None),
                posting(JobSource.INDEED, "3", title="Staff Software Engineer", apply_url=None),
                posting(JobSource.INDEED, "4", company_name="Other Co", apply_url=None),
            ]
        )
        assert len(jobs) == 4

    def test_repeated_source_record_collapses(self) -> None:
        p = posting(JobSource.INDEED, "dup")
        assert len(deduplicate([p, p.model_copy()])) == 1

    def test_representative_prefers_verified_then_exact_then_earliest(self) -> None:
        unverified = posting(JobSource.LINKEDIN, "u", published_at=None, apply_url=None)
        approx = posting(JobSource.INDEED, "a", published_at="2 days ago", apply_url=None)
        exact_late = posting(JobSource.ZIPRECRUITER, "e1", apply_url=None)
        exact_early = posting(
            JobSource.ZIPRECRUITER,
            "e2",
            apply_url=None,
            published_at=(RUN_AT - timedelta(hours=5)).isoformat(),
        )
        job = deduplicate([unverified, approx, exact_late, exact_early])[0]
        assert job.posting.external_id == "e2"

    def test_group_key_is_order_independent(self) -> None:
        ats = "https://boards.greenhouse.io/acme/jobs/1"
        a = posting(JobSource.LINKEDIN, "a", title="Platform Engineer", apply_url=ats)
        b = posting(JobSource.INDEED, "b", title="Engineer, Platform", apply_url=ats)
        assert deduplicate([a, b])[0].canonical_key == deduplicate([b, a])[0].canonical_key

    def test_empty(self) -> None:
        assert deduplicate([]) == []


def test_application_key_is_deterministic_and_resume_specific() -> None:
    user, r1, r2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    key = canonical_job_key(posting(JobSource.INDEED, "1"))
    assert application_key(user, r1, key) == application_key(user, r1, key)
    assert application_key(user, r1, key) != application_key(user, r2, key)
    assert len(application_key(user, r1, key)) == 64
