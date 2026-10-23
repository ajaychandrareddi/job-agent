"""Application container: everything a request handler needs, built once per process."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from job_agent.config import Settings
from job_agent.persistence.db import create_db_engine, create_session_factory, session_scope
from job_agent.persistence.file_store import LocalResumeFileStore, ResumeFileStore
from job_agent.runtime import build_parser, build_workflow_deps
from job_agent.scheduler.scheduler_service import SchedulerService
from job_agent.security.auth import Authenticator, RateLimiter, build_authenticator
from job_agent.services.application_actions import ApplicationActions
from job_agent.services.resume_parser import ResumeParser
from job_agent.services.resume_service import ResumeService, StatusListener
from job_agent.workflows.deps import WorkflowDeps


@dataclass
class AppContainer:
    settings: Settings
    engine: Engine
    factory: sessionmaker[Session]
    file_store: ResumeFileStore
    parser: ResumeParser
    deps: WorkflowDeps
    actions: ApplicationActions
    authenticator: Authenticator
    limiter: RateLimiter
    scheduler: SchedulerService | None = None
    owns_engine: bool = field(default=True)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with session_scope(self.factory) as s:
            yield s

    def resume_service(self, s: Session) -> ResumeService:
        listeners: list[StatusListener] = (
            [self.scheduler.on_status_change] if self.scheduler else []
        )
        return ResumeService(
            s,
            self.parser,
            self.file_store,
            status_listeners=listeners,
            resume_retention_days=self.settings.resume_retention_days,
        )


def build_container(settings: Settings) -> AppContainer:
    engine = create_db_engine(settings.database_url, echo=settings.db_echo)
    factory = create_session_factory(engine)
    file_store = LocalResumeFileStore(settings.resume_storage_dir)
    deps = build_workflow_deps(settings, factory, file_store)
    return AppContainer(
        settings=settings,
        engine=engine,
        factory=factory,
        file_store=file_store,
        parser=build_parser(settings),
        deps=deps,
        actions=ApplicationActions(deps),
        authenticator=build_authenticator(settings),
        limiter=RateLimiter(settings.api_rate_limit_per_minute),
    )
