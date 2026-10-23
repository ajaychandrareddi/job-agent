"""End-to-end workflow harness: real graph + real database, scripted sources/LLM/ATS."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from job_agent.config import Settings
from job_agent.job_sources.base import JobSourceAdapter
from job_agent.job_sources.indeed import IndeedAdapter
from job_agent.job_sources.linkedin import LinkedInAdapter
from job_agent.job_sources.mock import StaticFeedClient
from job_agent.job_sources.ziprecruiter import ZipRecruiterAdapter
from job_agent.models.application import ApplicationQuestion
from job_agent.models.job import ApplicationMethod, JobPosting, JobSource, RawPosting
from job_agent.models.matching import LLMMatchAssessment
from job_agent.models.resume import JobSearchPreferences, WorkflowStatus
from job_agent.persistence.db import create_db_engine, create_session_factory, session_scope
from job_agent.persistence.file_store import InMemoryResumeFileStore
from job_agent.persistence.orm import (
    Application,
    ApprovedAnswer,
    Base,
    JobMatch,
    ResumeWorkflow,
    SubmissionAttempt,
    WorkflowExecution,
)
from job_agent.repositories.resumes import UserRepository
from job_agent.services.application_service import (
    ApplicationPackage,
    LookupOutcome,
    LookupResult,
    SubmissionOutcome,
    SubmissionRegistry,
    SubmissionResult,
)
from job_agent.services.job_matcher import Assessment, JobMatcher
from job_agent.services.lifecycle import LifecycleAction
from job_agent.services.resume_parser import ResumeParser
from job_agent.services.resume_service import ResumeService
from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.runner import RunResult, WorkflowRunner
from tests.conftest import RUN_AT, SAMPLE_RESUME_LINES, make_docx
from tests.db_guard import assert_test_database
from tests.unit.test_job_matcher import assessment

PREFS = JobSearchPreferences(
    preferred_titles=["Senior Software Engineer"],
    locations=["New York, NY"],
    requires_sponsorship=False,
    authorized_to_work_countries=["United States"],
)


class ScriptedAssessor:
    """LLM stand-in: score chosen by a substring of the posting title (default 80)."""

    model_name = "scripted"

    def __init__(self) -> None:
        self.scores: dict[str, float] = {}
        self.calls = 0
        self.on_assess: Callable[[], None] | None = None

    async def assess(self, profile_json: str, posting_json: str) -> Assessment:
        self.calls += 1
        if self.on_assess:
            self.on_assess()
        score = next((v for k, v in self.scores.items() if k in posting_json), 80.0)
        result: LLMMatchAssessment = assessment(score)
        return Assessment(result, 10, 5)


class FakeIntegration:
    """A permitted ATS integration stand-in."""

    name = "fake-ats"

    def __init__(self) -> None:
        self.calls: list[tuple[ApplicationPackage, str]] = []
        self.form: list[ApplicationQuestion] = []
        self.form_error: BaseException | None = None
        self.results: list[SubmissionResult | BaseException] = []
        self.lookups: list[LookupResult | BaseException] = []
        self.lookup_calls = 0
        self.delay = 0.0

    def supports(self, posting: JobPosting) -> bool:
        return posting.application_method is ApplicationMethod.EXTERNAL_ATS

    async def questions(self, posting: JobPosting) -> list[ApplicationQuestion]:
        if self.form_error:
            raise self.form_error
        return list(self.form)

    async def lookup(
        self, posting: JobPosting, *, idempotency_key: str, submitted_at: datetime
    ) -> LookupResult:
        self.lookup_calls += 1
        if self.lookups:
            r = self.lookups.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r
        return LookupResult(LookupOutcome.UNKNOWN)

    async def submit(
        self, package: ApplicationPackage, *, idempotency_key: str
    ) -> SubmissionResult:
        self.calls.append((package, idempotency_key))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.results:
            r = self.results.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r
        return SubmissionResult(SubmissionOutcome.CONFIRMED, external_ref=f"CONF-{len(self.calls)}")


@dataclass
class World:
    engine: Engine
    factory: sessionmaker[Session]
    now: datetime = RUN_AT
    clients: dict[JobSource, StaticFeedClient] = field(default_factory=dict)
    assessor: ScriptedAssessor = field(default_factory=ScriptedAssessor)
    integration: FakeIntegration = field(default_factory=FakeIntegration)
    settings: Settings = field(default_factory=lambda: Settings(_env_file=None))  # type: ignore[call-arg]
    unavailable: set[JobSource] = field(default_factory=set)
    user_id: uuid.UUID | None = None
    file_store: InMemoryResumeFileStore = field(default_factory=InMemoryResumeFileStore)
    listeners: list[Any] = field(default_factory=list)

    def clock(self) -> datetime:
        return self.now

    def feed(self, source: JobSource, *raws: RawPosting, **kw: Any) -> StaticFeedClient:
        client = StaticFeedClient(list(raws), **kw)
        self.clients[source] = client
        return client

    def adapters(self) -> list[JobSourceAdapter]:
        out: list[JobSourceAdapter] = []
        for source, cls in (
            (JobSource.LINKEDIN, LinkedInAdapter),
            (JobSource.INDEED, IndeedAdapter),
            (JobSource.ZIPRECRUITER, ZipRecruiterAdapter),
        ):
            client = (
                None if source in self.unavailable else self.clients.get(source, StaticFeedClient())
            )
            out.append(cls(client, max_retries=0, clock=self.clock))
        return out

    def deps(self) -> WorkflowDeps:
        async def no_sleep(_: float) -> None:
            return None

        matcher = JobMatcher(self.assessor, self.settings, clock=self.clock, sleep=no_sleep)
        return WorkflowDeps(
            session_factory=self.factory,
            settings=self.settings,
            adapters=self.adapters(),
            matcher=matcher,
            submissions=SubmissionRegistry([self.integration]),
            clock=self.clock,
            submit_timeout_seconds=0.2,
        )

    def runner(self, **kw: Any) -> WorkflowRunner:
        return WorkflowRunner(self.deps(), **kw)

    def run(self, resume_id: uuid.UUID, **kw: Any) -> RunResult:
        return asyncio.run(self.runner().run(resume_id, **kw))

    # ----------------------------------------------------------- data helpers

    def create_resume(
        self, *, extra_line: str = "", prefs: JobSearchPreferences = PREFS, activate: bool = True
    ) -> uuid.UUID:
        with session_scope(self.factory) as s:
            user = UserRepository(s).get_or_create("auth0|e2e")
            self.user_id = user.id
            svc = self.resume_service(s)
            lines = [*SAMPLE_RESUME_LINES, extra_line] if extra_line else SAMPLE_RESUME_LINES
            rid = svc.create_resume(
                user.id, f"{uuid.uuid4()}.docx", make_docx(lines), preferences=prefs
            ).resume.id
            if activate:
                svc.change_status(user.id, rid, LifecycleAction.ACTIVATE)
        return rid

    def set_status(self, resume_id: uuid.UUID, action: LifecycleAction) -> None:
        with session_scope(self.factory) as s:
            self.resume_service(s).change_status(self.user_id, resume_id, action)  # type: ignore[arg-type]

    def resume_service(self, s: Session) -> ResumeService:
        return ResumeService(
            s, ResumeParser(), self.file_store, status_listeners=list(self.listeners)
        )

    def approve_answer(self, key: str, answer: str) -> None:
        with session_scope(self.factory) as s:
            s.add(ApprovedAnswer(user_id=self.user_id, question_key=key, answer=answer))

    def applications(self, resume_id: uuid.UUID) -> dict[str, Application]:
        """Applications keyed by the job title's external id (via the match/posting)."""
        from job_agent.persistence.orm import JobPostingRecord

        with session_scope(self.factory) as s:
            rows = s.execute(
                sa.select(Application, JobPostingRecord.external_id)
                .join(JobPostingRecord, JobPostingRecord.id == Application.job_posting_id)
                .where(Application.resume_id == resume_id)
            ).all()
            for app, _ in rows:
                s.expunge(app)
            return {ext: app for app, ext in rows}

    def attempts(self) -> list[SubmissionAttempt]:
        with session_scope(self.factory) as s:
            rows = list(s.scalars(sa.select(SubmissionAttempt)))
            for r in rows:
                s.expunge(r)
            return rows

    def executions(self, resume_id: uuid.UUID) -> list[WorkflowExecution]:
        with session_scope(self.factory) as s:
            rows = list(
                s.scalars(
                    sa.select(WorkflowExecution)
                    .where(WorkflowExecution.resume_id == resume_id)
                    .order_by(WorkflowExecution.started_at)
                )
            )
            for r in rows:
                s.expunge(r)
            return rows

    def workflow(self, resume_id: uuid.UUID) -> ResumeWorkflow:
        with session_scope(self.factory) as s:
            wf = s.scalar(sa.select(ResumeWorkflow).where(ResumeWorkflow.resume_id == resume_id))
            assert wf is not None
            s.expunge(wf)
            return wf

    def match_count(self) -> int:
        with session_scope(self.factory) as s:
            return s.scalar(sa.select(sa.func.count(JobMatch.id))) or 0


@pytest.fixture
def world(tmp_path: Any) -> Iterator[World]:
    url = os.environ.get("TEST_DATABASE_URL_UNIT") or f"sqlite:///{tmp_path / 'e2e.db'}"
    assert_test_database(url)
    engine = create_db_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield World(engine=engine, factory=create_session_factory(engine))
    Base.metadata.drop_all(engine)
    engine.dispose()


__all__ = ["PREFS", "World", "WorkflowStatus"]
