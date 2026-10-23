"""Dependencies injected into graph nodes (no global state)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.orm import Session, sessionmaker

from job_agent.config import Settings
from job_agent.job_sources.base import JobSourceAdapter
from job_agent.persistence.db import session_scope
from job_agent.repositories.resumes import WorkflowRepository
from job_agent.services.answers import AnswerGenerator
from job_agent.services.application_service import SubmissionRegistry
from job_agent.services.job_matcher import JobMatcher
from job_agent.services.lifecycle import may_run


@dataclass
class RunLimits:
    """Per-execution concurrency primitives, created in the running event loop."""

    search: asyncio.Semaphore
    match: asyncio.Semaphore
    submit_lock: asyncio.Lock


@dataclass
class WorkflowDeps:
    session_factory: sessionmaker[Session]
    settings: Settings
    adapters: Sequence[JobSourceAdapter]
    matcher: JobMatcher
    submissions: SubmissionRegistry = field(default_factory=SubmissionRegistry)
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    submit_timeout_seconds: float = 120.0
    answer_generator: AnswerGenerator | None = None
    _limits: dict[str, RunLimits] = field(default_factory=dict)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with session_scope(self.session_factory) as s:
            yield s

    def limits(self, execution_id: str) -> RunLimits:
        if execution_id not in self._limits:
            self._limits[execution_id] = RunLimits(
                search=asyncio.Semaphore(self.settings.max_concurrent_job_searches),
                match=asyncio.Semaphore(self.settings.max_concurrent_matches),
                submit_lock=asyncio.Lock(),
            )
        return self._limits[execution_id]

    def release(self, execution_id: str) -> None:
        self._limits.pop(execution_id, None)

    def workflow_active(self, resume_id: str) -> bool:
        """Stop check. Reads the committed status in a fresh transaction."""
        with self.session() as s:
            wf = WorkflowRepository(s).get_by_resume(uuid.UUID(resume_id))
            return wf is not None and may_run(wf.status)
