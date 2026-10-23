"""Wire up a scheduler process: ``python -m job_agent.scheduler``."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from job_agent.config import Settings, get_settings
from job_agent.observability.logging import configure_logging, get_logger
from job_agent.persistence.db import create_db_engine, create_session_factory
from job_agent.persistence.file_store import LocalResumeFileStore
from job_agent.runtime import (
    NoLLMAssessor,
    build_answer_generator,
    build_matcher,
    build_submissions,
    build_workflow_deps,
)
from job_agent.scheduler.scheduler_service import SchedulerService
from job_agent.services.application_actions import ApplicationActions
from job_agent.workflows.runner import WorkflowRunner, postgres_checkpointer

__all__ = [
    "NoLLMAssessor",
    "build_answer_generator",
    "build_matcher",
    "build_submissions",
    "main",
    "scheduler_runtime",
]

log = get_logger(__name__)


@asynccontextmanager
async def scheduler_runtime(settings: Settings) -> AsyncIterator[SchedulerService]:
    engine = create_db_engine(settings.database_url, echo=settings.db_echo)
    factory = create_session_factory(engine)
    file_store = LocalResumeFileStore(settings.resume_storage_dir)
    deps = build_workflow_deps(settings, factory, file_store)
    actions = ApplicationActions(deps)
    try:
        async with postgres_checkpointer(settings.database_url) as saver:
            service = SchedulerService(
                session_factory=factory,
                settings=settings,
                runner=WorkflowRunner(deps, checkpointer=saver),
                file_store=file_store,
                reconcile_submissions=actions.reconcile_due,
            )
            await service.start()
            try:
                yield service
            finally:
                await service.shutdown()
    finally:
        engine.dispose()


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with scheduler_runtime(settings):
        await stop.wait()
