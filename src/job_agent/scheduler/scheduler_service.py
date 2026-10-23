"""Per-resume scheduling with APScheduler.

Source of truth is the database, never the scheduler's memory:

* on start and every ``SCHEDULER_RECONCILE_SECONDS`` the job list is rebuilt from
  ``resume_workflows``: one daily job (17:00 ``JOB_SEARCH_TIMEZONE``) per ACTIVE resume,
  none for PAUSED / STOPPED / DRAFT; stable job ids + ``replace_existing`` mean no
  duplicate registrations;
* lifecycle changes made in this process are applied right after their transaction
  commits (a rolled-back activation schedules nothing); changes made by other
  processes (e.g. the API) are picked up by the next reconcile;
* each run takes a database lease on the resume (renewed by a heartbeat), so several
  scheduler instances never run the same resume at once; the partial unique index on
  RUNNING executions is a second guard;
* misfires: ``misfire_grace_time`` and ``coalesce=True`` on every job; on start-up a
  run missed by more than the grace period is handled by ``SCHEDULER_MISSED_RUN_ACTION``
  (one catch-up run, or skip) and never by replaying each missed day;
* executions left RUNNING by a dead process (lease expired) are resumed from their
  LangGraph checkpoint, or closed as FAILED/CANCELLED if that is not possible.
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from time import monotonic
from typing import Any, Protocol

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import event
from sqlalchemy.orm import Session, object_session, sessionmaker

from job_agent.config import MissedRunAction, Settings
from job_agent.models.resume import WorkflowStatus
from job_agent.observability import metrics
from job_agent.observability.logging import get_logger
from job_agent.persistence.db import session_scope
from job_agent.persistence.file_store import ResumeFileStore
from job_agent.persistence.orm import ExecutionStatus, ResumeWorkflow
from job_agent.repositories.executions import ExecutionRepository
from job_agent.repositories.resumes import WorkflowRepository
from job_agent.services.retention import RetentionService
from job_agent.workflows.runner import RunResult

log = get_logger(__name__)


class RunKind(StrEnum):
    INITIAL = "initial"
    SCHEDULED = "daily"
    CATCH_UP = "catchup"
    RECOVERY = "recover"


_EXECUTION_LABEL = {RunKind.SCHEDULED: "SCHEDULED", RunKind.CATCH_UP: "CATCH_UP"}
_RESUME_JOB_KINDS = (RunKind.SCHEDULED, RunKind.INITIAL, RunKind.CATCH_UP)


class Runner(Protocol):
    async def run(
        self, resume_id: uuid.UUID, *, requested_trigger: str | None = None
    ) -> RunResult: ...

    async def resume(self, execution_id: uuid.UUID) -> RunResult: ...


def job_id(kind: RunKind, key: str) -> str:
    return f"{kind.value}:{key}"


class SchedulerService:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        settings: Settings,
        runner: Runner,
        file_store: ResumeFileStore | None = None,
        instance_id: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        scheduler: AsyncIOScheduler | None = None,
        reconcile_submissions: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self.reconcile_submissions = reconcile_submissions
        self.factory = session_factory
        self.settings = settings
        self.runner = runner
        self.file_store = file_store
        self.clock = clock
        self.instance_id = instance_id or (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )
        self.scheduler = scheduler or AsyncIOScheduler(
            timezone=settings.tz,
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": settings.scheduler_misfire_grace_seconds,
            },
        )
        self._slots: asyncio.Semaphore | None = None
        self.started = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._slots = asyncio.Semaphore(self.settings.scheduler_max_workers)
        self.scheduler.start()
        self.started = True
        await self.reconcile(startup=True)
        self.scheduler.add_job(
            self.reconcile,
            IntervalTrigger(seconds=self.settings.scheduler_reconcile_seconds),
            id="system:reconcile",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_maintenance,
            CronTrigger(
                hour=self.settings.scheduler_maintenance_hour, minute=15, timezone=self.settings.tz
            ),
            id="system:maintenance",
            replace_existing=True,
        )
        if self.reconcile_submissions is not None:
            self.scheduler.add_job(
                self.reconcile_submissions,
                IntervalTrigger(seconds=self.settings.reconcile_interval_seconds),
                id="system:reconcile-submissions",
                replace_existing=True,
            )
        log.info("scheduler_started", instance=self.instance_id)

    async def shutdown(self) -> None:
        if self.started:
            self.scheduler.shutdown(wait=False)
            self.started = False
        log.info("scheduler_stopped", instance=self.instance_id)

    # -------------------------------------------------------------------- triggers

    @property
    def daily_trigger(self) -> CronTrigger:
        return CronTrigger(
            hour=self.settings.job_search_hour,
            minute=self.settings.job_search_minute,
            timezone=self.settings.tz,
        )

    def next_daily_run(self, now: datetime | None = None) -> datetime:
        """Next daily fire time. APScheduler fires by wall-clock time, so that is the
        default reference (the injectable clock is for run bookkeeping only)."""
        now = now or datetime.now(UTC)
        nxt: datetime | None = self.daily_trigger.get_next_fire_time(None, now)
        assert nxt is not None
        return nxt.astimezone(UTC)

    def previous_daily_run(self, now: datetime) -> datetime:
        """Most recent scheduled time at or before ``now`` (DST-safe via zoneinfo)."""
        tz = self.settings.tz
        local = now.astimezone(tz)
        at = time(self.settings.job_search_hour, self.settings.job_search_minute)
        candidate = datetime.combine(local.date(), at, tzinfo=tz)
        if candidate > local:
            candidate = datetime.combine(local.date() - timedelta(days=1), at, tzinfo=tz)
        return candidate.astimezone(UTC)

    def next_run_for(self, resume_id: uuid.UUID) -> datetime | None:
        job = self.scheduler.get_job(job_id(RunKind.SCHEDULED, str(resume_id)))
        if job is None:
            return None
        nxt = getattr(job, "next_run_time", None)
        return nxt.astimezone(UTC) if nxt else self.next_daily_run()

    # ---------------------------------------------------------------- registration

    def _register_daily(self, resume_id: str) -> None:
        jid = job_id(RunKind.SCHEDULED, resume_id)
        if self.scheduler.get_job(jid) is None:
            self.scheduler.add_job(
                self._run_job,
                self.daily_trigger,
                id=jid,
                replace_existing=True,
                kwargs={"resume_id": resume_id, "kind": RunKind.SCHEDULED},
            )

    def _schedule_now(self, resume_id: str, kind: RunKind, *, slot: RunKind | None = None) -> None:
        """Run once as soon as possible. ``slot`` names the job id (one pending one-off
        run per slot per resume); ``kind`` is how the run is labelled."""
        self.scheduler.add_job(
            self._run_job,
            DateTrigger(),  # now, by the scheduler's own clock
            id=job_id(slot or kind, resume_id),
            replace_existing=True,
            misfire_grace_time=None,
            kwargs={"resume_id": resume_id, "kind": kind},
        )

    def _unregister(self, resume_id: str) -> None:
        for kind in _RESUME_JOB_KINDS:
            jid = job_id(kind, resume_id)
            if self.scheduler.get_job(jid) is not None:
                self.scheduler.remove_job(jid)

    def registered_resumes(self) -> set[str]:
        prefix = f"{RunKind.SCHEDULED.value}:"
        return {j.id[len(prefix) :] for j in self.scheduler.get_jobs() if j.id.startswith(prefix)}

    # ----------------------------------------------------------- lifecycle listener

    def on_status_change(
        self, workflow: ResumeWorkflow, previous: WorkflowStatus, new: WorkflowStatus
    ) -> None:
        """``ResumeService`` status listener. Applied only after the change commits."""
        resume_id = str(workflow.resume_id)
        initial_done = workflow.initial_search_completed
        if new is WorkflowStatus.ACTIVE:
            workflow.next_run_at = self.next_daily_run()

        def apply(_session: Any = None) -> None:
            if new is WorkflowStatus.ACTIVE:
                self._register_daily(resume_id)
                if not initial_done:
                    self._schedule_now(resume_id, RunKind.INITIAL)
            else:
                self._unregister(resume_id)
            log.info("schedule_updated", resume_id=resume_id, status=new.value)

        session = object_session(workflow)
        if session is None:
            apply()
            return

        def discard(_session: Any) -> None:
            if event.contains(session, "after_commit", apply):
                event.remove(session, "after_commit", apply)

        event.listen(session, "after_commit", apply, once=True)
        event.listen(session, "after_rollback", discard, once=True)

    # -------------------------------------------------------------------- reconcile

    async def reconcile(self, *, startup: bool = False) -> None:
        now = self.clock()
        grace = timedelta(seconds=self.settings.scheduler_misfire_grace_seconds)
        to_run: list[tuple[str, RunKind]] = []
        with session_scope(self.factory) as s:
            repo = WorkflowRepository(s)
            executions = ExecutionRepository(s)
            active = list(repo.list_by_status(WorkflowStatus.ACTIVE))
            desired = {str(w.resume_id) for w in active}
            for wf in s.query(ResumeWorkflow).filter(
                ResumeWorkflow.status != WorkflowStatus.ACTIVE,
                ResumeWorkflow.next_run_at.is_not(None),
            ):
                wf.next_run_at = None
            for wf in active:
                rid = str(wf.resume_id)
                self._register_daily(rid)
                wf.next_run_at = self.next_run_for(wf.resume_id)
                if executions.has_running(wf.resume_id):
                    continue
                if not wf.initial_search_completed:
                    # Never-run initial search (e.g. activated by another process, or the
                    # process died first). A failed initial search waits for the daily run
                    # instead of being retried on every restart.
                    if wf.last_execution_at is None and not self._pending(rid):
                        to_run.append((rid, RunKind.INITIAL))
                elif startup:
                    due = self.previous_daily_run(now)
                    if wf.last_execution_at is None or wf.last_execution_at < due:
                        if now - due <= grace:
                            to_run.append((rid, RunKind.SCHEDULED))
                        elif self.settings.scheduler_missed_run_action is MissedRunAction.RUN_ONCE:
                            to_run.append((rid, RunKind.CATCH_UP))
                        else:
                            log.info("missed_run_skipped", resume_id=rid, due=due.isoformat())
        for registered in self.registered_resumes() - desired:
            self._unregister(registered)
        for rid, kind in to_run:
            # A late-but-within-grace daily run is labelled SCHEDULED, but uses the
            # one-off catch-up slot so it never replaces the recurring daily job.
            slot = RunKind.CATCH_UP if kind is RunKind.SCHEDULED else kind
            self._schedule_now(rid, kind, slot=slot)
        await self.recover_stale_executions()

    def _pending(self, resume_id: str) -> bool:
        return self.scheduler.get_job(job_id(RunKind.INITIAL, resume_id)) is not None

    # --------------------------------------------------------------------- running

    async def _run_job(self, resume_id: str, kind: RunKind) -> RunResult | None:
        assert self._slots is not None, "scheduler not started"
        async with self._slots:
            rid = uuid.UUID(resume_id)
            if not self._acquire(rid):
                log.info("run_skipped_lease_held", resume_id=resume_id, kind=kind.value)
                return None
            heartbeat = asyncio.create_task(self._heartbeat(rid))
            result: RunResult | None = None
            started = monotonic()
            try:
                result = await self.runner.run(rid, requested_trigger=_EXECUTION_LABEL.get(kind))
                log.info(
                    "scheduled_run_finished",
                    resume_id=resume_id,
                    kind=kind.value,
                    status=result.status,
                    halt=result.summary.get("halt_reason"),
                )
            except Exception as exc:  # a failed run must not kill the scheduler
                log.error(
                    "scheduled_run_failed",
                    resume_id=resume_id,
                    kind=kind.value,
                    error=type(exc).__name__,
                )
            finally:
                heartbeat.cancel()
                self._release(rid)
                metrics.scheduler_run_duration.record(
                    monotonic() - started,
                    {"kind": kind.value, "status": str(result.status) if result else "error"},
                )
            return result

    async def _recover(self, execution_id: str, resume_id: str) -> RunResult | None:
        assert self._slots is not None, "scheduler not started"
        async with self._slots:
            rid = uuid.UUID(resume_id)
            if not self._acquire(rid):
                return None
            heartbeat = asyncio.create_task(self._heartbeat(rid))
            try:
                result = await self.runner.resume(uuid.UUID(execution_id))
                log.info("execution_recovered", execution_id=execution_id, status=result.status)
                return result
            except Exception as exc:
                with session_scope(self.factory) as s:
                    ExecutionRepository(s).close_abandoned(
                        uuid.UUID(execution_id),
                        ExecutionStatus.FAILED,
                        f"abandoned after a crash; could not resume ({type(exc).__name__})",
                        self.clock(),
                    )
                log.warning(
                    "execution_abandoned", execution_id=execution_id, error=type(exc).__name__
                )
                return None
            finally:
                heartbeat.cancel()
                self._release(rid)

    async def recover_stale_executions(self) -> None:
        """RUNNING executions whose lease has expired belong to a dead process."""
        now = self.clock()
        to_resume: list[tuple[str, str]] = []
        with session_scope(self.factory) as s:
            repo = WorkflowRepository(s)
            executions = ExecutionRepository(s)
            for execution, wf in executions.running():
                if not repo.lease_expired(wf, now):
                    continue  # a live instance is running it
                if wf.status is not WorkflowStatus.ACTIVE:
                    executions.close_abandoned(
                        execution.id,
                        ExecutionStatus.CANCELLED,
                        "abandoned after a crash; workflow is no longer active",
                        now,
                    )
                else:
                    to_resume.append((str(execution.id), str(wf.resume_id)))
        for execution_id, resume_id in to_resume:
            self.scheduler.add_job(
                self._recover,
                DateTrigger(),
                id=job_id(RunKind.RECOVERY, execution_id),
                replace_existing=True,
                misfire_grace_time=None,
                kwargs={"execution_id": execution_id, "resume_id": resume_id},
            )

    # ---------------------------------------------------------------------- leases

    def _acquire(self, resume_id: uuid.UUID) -> bool:
        with session_scope(self.factory) as s:
            return WorkflowRepository(s).try_acquire_lease(
                resume_id, self.instance_id, self.clock(), self.settings.scheduler_run_lease_seconds
            )

    def _release(self, resume_id: uuid.UUID) -> None:
        with session_scope(self.factory) as s:
            WorkflowRepository(s).release_lease(resume_id, self.instance_id)
            wf = WorkflowRepository(s).get_by_resume(resume_id)
            if wf is not None:
                wf.next_run_at = (
                    self.next_run_for(resume_id) if wf.status is WorkflowStatus.ACTIVE else None
                )

    async def _heartbeat(self, resume_id: uuid.UUID) -> None:
        interval = self.settings.scheduler_run_lease_seconds / 3
        while True:
            await asyncio.sleep(interval)
            self.renew(resume_id)

    def renew(self, resume_id: uuid.UUID) -> bool:
        with session_scope(self.factory) as s:
            return WorkflowRepository(s).renew_lease(
                resume_id, self.instance_id, self.clock(), self.settings.scheduler_run_lease_seconds
            )

    # ------------------------------------------------------------------ maintenance

    async def run_maintenance(self) -> dict[str, int]:
        if self.file_store is None:
            return {}
        with session_scope(self.factory) as s:
            return RetentionService(s, self.file_store).purge(self.clock())
