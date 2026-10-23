from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

from sqlalchemy.orm import Session

from job_agent.models.job import JobSource
from job_agent.models.matching import Recommendation
from job_agent.models.resume import ResumeProfile
from job_agent.persistence.orm import JobMatch, User
from job_agent.repositories.jobs import JobPostingRepository
from job_agent.repositories.matches import JobMatchRepository
from job_agent.services.deduplication import deduplicate
from job_agent.services.normalization import normalize_raw_posting
from job_agent.services.resume_service import ResumeService
from tests.conftest import RUN_AT, make_raw
from tests.unit.test_job_matcher import PREFS, FakeAssessor, assessment, matcher


def _job_id(session: Session, external_id: str) -> uuid.UUID:
    p = normalize_raw_posting(make_raw(external_id), JobSource.INDEED, now=RUN_AT)
    return JobPostingRepository(session).persist_group(deduplicate([p])[0], now=RUN_AT).id


def _decide(score: float):  # type: ignore[no-untyped-def]
    p = normalize_raw_posting(make_raw("x"), JobSource.INDEED, now=RUN_AT)
    return asyncio.run(
        matcher(FakeAssessor(assessment(score))).match(ResumeProfile(skills=["Python"]), PREFS, p)
    )


def test_save_upserts_and_lists(
    session: Session, user: User, resume_service: ResumeService, sample_docx: bytes
) -> None:
    rid = resume_service.create_resume(user.id, "a.docx", sample_docx).resume.id
    repo = JobMatchRepository(session)
    j1, j2 = _job_id(session, "j1"), _job_id(session, "j2")

    first = repo.save(resume_id=rid, resume_version=1, job_posting_id=j1, decision=_decide(55))
    repo.save(resume_id=rid, resume_version=1, job_posting_id=j2, decision=_decide(80))
    assert first.recommendation is Recommendation.SKIP and first.score == 55

    rescored = _decide(72)
    rescored.evaluated_at = RUN_AT + timedelta(hours=1)
    again = repo.save(resume_id=rid, resume_version=1, job_posting_id=j1, decision=rescored)
    assert again.id == first.id and again.score == 72
    assert session.query(JobMatch).count() == 2
    assert again.decision["prompt_version"] == "match-v1"
    assert again.criterion_scores["required_skills"] == 72

    assert [m.score for m in repo.list_for_resume(rid)] == [80, 72]
    assert [m.score for m in repo.list_for_resume(rid, min_score=75)] == [80]
    assert len(repo.list_for_resume(rid, recommendation=Recommendation.SKIP)) == 0
    assert repo.list_for_resume(uuid.uuid4()) == []


def test_new_resume_version_gets_its_own_match(
    session: Session, user: User, resume_service: ResumeService, sample_docx: bytes
) -> None:
    rid = resume_service.create_resume(user.id, "a.docx", sample_docx).resume.id
    repo = JobMatchRepository(session)
    jid = _job_id(session, "j1")
    repo.save(resume_id=rid, resume_version=1, job_posting_id=jid, decision=_decide(60))
    repo.save(resume_id=rid, resume_version=2, job_posting_id=jid, decision=_decide(70))
    assert sorted(m.resume_version for m in repo.list_for_resume(rid)) == [1, 2]
    # Matched jobs count as tracked, so later runs skip them for this resume.
    assert JobPostingRepository(session).tracked_canonical_keys(rid)
