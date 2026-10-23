"""Scheduler service tests (spec §7, §2C/§2D, §12 cases 10, 12, 16, 17)."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from job_agent.config import MissedRunAction, Settings
from job_agent.models.application import ApplicationStatus
from job_agent.models.job import JobSource
from job_agent.models.resume import WorkflowStatus
from job_agent.persistence.db import session_scope
from job_agent.persistence.orm import (
    Application,
    ExecutionStatus,
    Resume,
    ResumeVersion,
    ResumeWorkflow,
    WorkflowExecution,
)
from job_agent.repositories.resumes import WorkflowRepository
from job_agent.scheduler.bootstrap import NoLLMAssessor, build_matcher
from job_agent.scheduler.scheduler_service import RunKind, SchedulerService, job_id
from job_agent.services.application_service import ApplicationService
from job_agent.services.lifecycle import LifecycleAction
from job_agent.workflows.runner import RunResult, WorkflowRunner
from tests.conftest import RUN_AT, make_raw
from tests.e2e.conftest import World

# RUN_AT is 2026-10-01 21:00 UTC == 17:00 America/New_York (EDT).
FIRE = RUN_AT


class FakeRunner:
    def __init__(self) -> None:
        self.runs: list[tuple[uuid.UUID, str | None]] = []
        self.resumed: list[uuid.UUID] = []
        self.delay = 0.0
        self.error: BaseException | None = None
        self.resume_error: BaseException | None = None
        self.active = 0
        self.peak = 0

    async def run(self, resume_id: uuid.UUID, *, requested_trigger: str | None = None) -> RunResult:
        self.runs.append((resume_id, requested_trigger))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            return RunResult(uuid.uuid4(), ExecutionStatus.SUCCEEDED)
        finally:
            self.active -= 1

    async def resume(self, execution_id: uuid.UUID) -> RunResult:
        self.resumed.append(execution_id)
        if self.resume_error:
            raise self.resume_error
        return RunResult(execution_id, ExecutionStatus.SUCCEEDED)


def service(
    world: World,
    runner: Any = None,
    *,
    clock: Callable[[], datetime] | None = None,
    instance: str = "inst-a",
    **settings_kw: Any,
) -> SchedulerService:
    settings = Settings(_env_file=None, **settings_kw)  # type: ignore[call-arg]
    return SchedulerService(
        session_factory=world.factory,
        settings=settings,
        runner=runner or FakeRunner(),
        file_store=world.file_store,
        instance_id=instance,
        clock=clock or world.clock,
    )


async def wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    end = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def mark_ran(world: World, resume_id: uuid.UUID, *, last: datetime, initial: bool = True) -> None:
    with session_scope(world.factory) as s:
        wf = WorkflowRepository(s).get_by_resume(resume_id)
        assert wf is not None
        wf.initial_search_completed = initial
        wf.initial_search_completed_at = last if initial else None
        wf.last_execution_at = last


async def started(svc: SchedulerService) -> SchedulerService:
    await svc.start()
    return svc


# --------------------------------------------------------------------------- registration


def test_registers_daily_job_per_active_resume_only(world: World) -> None:
    a = world.create_resume()
    b = world.create_resume(extra_line="Intern at Bar, Jun 2015 - Aug 2015")
    paused = world.create_resume(extra_line="Intern at Baz, Jun 2014 - Aug 2014")
    world.set_status(paused, LifecycleAction.PAUSE)
    draft = world.create_resume(extra_line="Intern at Qux, Jun 2013 - Aug 2013", activate=False)
    runner = FakeRunner()

    async def scenario() -> None:
        svc = await started(service(world, runner))
        try:
            assert svc.registered_resumes() == {str(a), str(b)}
            job = svc.scheduler.get_job(job_id(RunKind.SCHEDULED, str(a)))
            fields = {f.name: str(f) for f in job.trigger.fields}
            assert (fields["hour"], fields["minute"]) == ("17", "0")
            assert str(job.trigger.timezone) == "America/New_York"
            assert (job.misfire_grace_time, job.coalesce, job.max_instances) == (3600, True, 1)
            next_run = job.next_run_time.astimezone(UTC)
            assert world.workflow(a).next_run_at == next_run
            assert next_run.astimezone(Settings(_env_file=None).tz).hour == 17  # type: ignore[call-arg]
            assert world.workflow(paused).next_run_at is None
            assert world.workflow(draft).next_run_at is None
            # Never-run initial searches start right away (activated before this process).
            await wait_for(lambda: len(runner.runs) == 2)
            assert {r for r, _ in runner.runs} == {a, b}
            assert all(label is None for _, label in runner.runs)  # INITIAL label from graph
            # Reconcile is idempotent: no duplicate registrations, no repeat initial runs.
            await svc.reconcile()
            await svc.reconcile()
            assert len([j for j in svc.scheduler.get_jobs() if j.id.startswith("daily:")]) == 2
        finally:
            await svc.shutdown()

    asyncio.run(scenario())


def test_previous_daily_run_handles_dst(world: World) -> None:
    svc = service(world)
    # Before 17:00 local -> yesterday's 17:00. EDT (UTC-4) vs EST (UTC-5).
    assert svc.previous_daily_run(datetime(2026, 7, 1, 20, tzinfo=UTC)) == datetime(
        2026, 6, 30, 21, tzinfo=UTC
    )
    assert svc.previous_daily_run(datetime(2026, 12, 1, 22, 30, tzinfo=UTC)) == datetime(
        2026, 12, 1, 22, tzinfo=UTC
    )
    # DST ends 2026-11-01: 17:00 that day is EST.
    assert svc.previous_daily_run(datetime(2026, 11, 1, 23, tzinfo=UTC)) == datetime(
        2026, 11, 1, 22, tzinfo=UTC
    )
    assert svc.next_daily_run(datetime(2026, 11, 1, 23, tzinfo=UTC)) == datetime(
        2026, 11, 2, 22, tzinfo=UTC
    )


def test_configured_time_and_timezone(world: World) -> None:
    svc = service(
        world, job_search_timezone="America/Los_Angeles", job_search_hour=9, job_search_minute=30
    )
    nxt = svc.next_daily_run(datetime(2026, 10, 1, 12, tzinfo=UTC))
    assert nxt == datetime(2026, 10, 1, 16, 30, tzinfo=UTC)  # 09:30 PDT


# --------------------------------------------------------------------------- lifecycle at runtime


def test_activation_registers_after_commit_and_runs_initial(world: World) -> None:
    rid = world.create_resume(activate=False)
    runner = FakeRunner()

    async def scenario() -> None:
        svc = await started(service(world, runner))
        world.listeners.append(svc.on_status_change)
        try:
            assert svc.registered_resumes() == set()
            world.set_status(rid, LifecycleAction.ACTIVATE)
            assert svc.registered_resumes() == {str(rid)}
            assert world.workflow(rid).next_run_at == svc.next_run_for(rid)
            await wait_for(lambda: runner.runs == [(rid, None)])  # initial 30-day run
        finally:
            await svc.shutdown()

    asyncio.run(scenario())


def test_rolled_back_activation_schedules_nothing(world: World) -> None:
    rid = world.create_resume(activate=False)

    async def scenario() -> None:
        svc = await started(service(world))
        world.listeners.append(svc.on_status_change)
        try:
            s = world.factory()
            try:
                world.resume_service(s).change_status(
                    world.user_id,  # type: ignore[arg-type]
                    rid,
                    LifecycleAction.ACTIVATE,
                )
                s.rollback()
            finally:
                s.close()
            assert svc.registered_resumes() == set()
            assert svc.scheduler.get_job(job_id(RunKind.INITIAL, str(rid))) is None
        finally:
            await svc.shutdown()

    asyncio.run(scenario())


def test_pause_resume_stop_are_independent_per_resume(world: World) -> None:  # case 10
    a = world.create_resume()
    b = world.create_resume(extra_line="Intern at Bar, Jun 2015 - Aug 2015")
    for rid in (a, b):
        mark_ran(world, rid, last=FIRE + timedelta(minutes=5))
    runner = FakeRunner()

    async def scenario() -> None:
        svc = await started(service(world, runner, clock=lambda: FIRE + timedelta(hours=1)))
        world.listeners.append(svc.on_status_change)
        try:
            world.set_status(b, LifecycleAction.PAUSE)
            assert svc.registered_resumes() == {str(a)}
            assert world.workflow(b).next_run_at is None
            world.set_status(b, LifecycleAction.RESUME)
            assert svc.registered_resumes() == {str(a), str(b)}
            world.set_status(b, LifecycleAction.STOP)
            assert svc.registered_resumes() == {str(a)}
            await asyncio.sleep(0.05)
            assert runner.runs == []  # initial searches were done; nothing extra ran
        finally:
            await svc.shutdown()
        # Restart: stopped stays stopped, active is re-registered.
        svc2 = await started(service(world, runner, clock=lambda: FIRE + timedelta(hours=1)))
        try:
            assert svc2.registered_resumes() == {str(a)}
            assert world.workflow(b).status is WorkflowStatus.STOPPED
        finally:
            await svc2.shutdown()

    asyncio.run(scenario())


def test_changes_from_another_process_are_picked_up_by_reconcile(world: World) -> None:
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE + timedelta(minutes=5))

    async def scenario() -> None:
        svc = await started(service(world, clock=lambda: FIRE + timedelta(hours=1)))
        try:
            world.set_status(rid, LifecycleAction.PAUSE)  # no listener: "API process"
            assert svc.registered_resumes() == {str(rid)}
            await svc.reconcile()
            assert svc.registered_resumes() == set()
            assert world.workflow(rid).next_run_at is None
        finally:
            await svc.shutdown()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- restarts & misfires


def run_startup(world: World, runner: FakeRunner, now: datetime, **kw: Any) -> None:
    async def scenario() -> None:
        svc = await started(service(world, runner, clock=lambda: now, **kw))
        try:
            await asyncio.sleep(0.1)  # let any immediate one-off jobs fire
        finally:
            await svc.shutdown()

    asyncio.run(scenario())


def test_restart_does_not_repeat_initial_search(world: World) -> None:  # cases 12, 16
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE + timedelta(minutes=5))
    runner = FakeRunner()
    run_startup(world, runner, FIRE + timedelta(hours=2))
    run_startup(world, runner, FIRE + timedelta(hours=3))
    assert runner.runs == []


def test_failed_initial_search_waits_for_daily_run(world: World) -> None:
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE - timedelta(hours=2), initial=False)
    runner = FakeRunner()
    run_startup(world, runner, FIRE - timedelta(hours=1))
    assert runner.runs == []  # not retried on every restart


def test_late_start_within_grace_runs_scheduled_once(world: World) -> None:  # case 16
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE - timedelta(days=1) + timedelta(minutes=5))
    runner = FakeRunner()
    run_startup(world, runner, FIRE + timedelta(minutes=30))
    assert runner.runs == [(rid, "SCHEDULED")]


def test_outage_beyond_grace_runs_one_catch_up(world: World) -> None:  # case 16
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE - timedelta(days=4))  # three daily runs missed
    runner = FakeRunner()
    run_startup(world, runner, FIRE + timedelta(hours=5))
    assert runner.runs == [(rid, "CATCH_UP")]  # once, not once per missed day


def test_outage_beyond_grace_skip_policy(world: World) -> None:
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE - timedelta(days=4))
    runner = FakeRunner()
    run_startup(
        world, runner, FIRE + timedelta(hours=5), scheduler_missed_run_action=MissedRunAction.SKIP
    )
    assert runner.runs == []


def test_catch_up_window_follows_publication_policy(world: World) -> None:
    """A catch-up run still searches only the last 24h by default (no backlog)."""
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE - timedelta(days=4))
    with session_scope(world.factory) as s:
        wf = WorkflowRepository(s).get_by_resume(rid)
        assert wf is not None
        wf.last_window_end = FIRE - timedelta(days=4)
    world.now = FIRE + timedelta(hours=5)
    client = world.feed(JobSource.LINKEDIN, make_raw("x"))
    result = asyncio.run(WorkflowRunner(world.deps()).run(rid, requested_trigger="CATCH_UP"))
    assert client.calls[0][1] == world.now - timedelta(hours=24)
    assert world.executions(rid)[0].trigger == "CATCH_UP"
    assert result.summary["window_trigger"] == "SCHEDULED"


# --------------------------------------------------------------------------- coordination


def test_lease_blocks_second_instance_and_reentry(world: World) -> None:  # case 17
    rid = world.create_resume()
    a_runner, b_runner = FakeRunner(), FakeRunner()

    async def scenario() -> None:
        a = service(world, a_runner, instance="inst-a")
        b = service(world, b_runner, instance="inst-b")
        a._slots = asyncio.Semaphore(4)
        b._slots = asyncio.Semaphore(4)
        assert a._acquire(rid)
        assert not a._acquire(rid)  # not re-entrant within one instance either
        assert await b._run_job(str(rid), RunKind.SCHEDULED) is None
        assert b_runner.runs == []
        a._release(rid)
        assert await b._run_job(str(rid), RunKind.SCHEDULED) is not None
        assert b_runner.runs == [(rid, "SCHEDULED")]

    asyncio.run(scenario())


def test_expired_lease_can_be_taken_over_and_heartbeat_renews(world: World) -> None:
    rid = world.create_resume()
    now = {"t": FIRE}
    a = service(world, instance="inst-a", clock=lambda: now["t"], scheduler_run_lease_seconds=60)
    b = service(world, instance="inst-b", clock=lambda: now["t"], scheduler_run_lease_seconds=60)
    assert a._acquire(rid)
    now["t"] = FIRE + timedelta(seconds=50)
    assert a.renew(rid)  # heartbeat keeps it alive
    now["t"] = FIRE + timedelta(seconds=100)
    assert not b._acquire(rid)
    now["t"] = FIRE + timedelta(seconds=200)  # owner died: lease expired
    assert b._acquire(rid)
    assert not a.renew(rid)


def test_max_workers_limits_concurrent_runs(world: World) -> None:
    rids = [
        world.create_resume(extra_line=f"Intern at C{i}, Jun 2015 - Aug 2015") for i in range(3)
    ]
    runner = FakeRunner()
    runner.delay = 0.05

    async def scenario() -> None:
        svc = service(world, runner, scheduler_max_workers=1)
        svc._slots = asyncio.Semaphore(1)
        await asyncio.gather(*(svc._run_job(str(r), RunKind.SCHEDULED) for r in rids))

    asyncio.run(scenario())
    assert runner.peak == 1 and len(runner.runs) == 3


def test_run_failure_releases_lease(world: World) -> None:
    rid = world.create_resume()
    runner = FakeRunner()
    runner.error = RuntimeError("boom")

    async def scenario() -> None:
        svc = service(world, runner)
        svc._slots = asyncio.Semaphore(1)
        assert await svc._run_job(str(rid), RunKind.SCHEDULED) is None

    asyncio.run(scenario())
    assert world.workflow(rid).run_lock_owner is None


# --------------------------------------------------------------------------- stale executions


def add_running_execution(
    world: World, rid: uuid.UUID, *, lease_owner: str | None, lease_expires: datetime | None
) -> uuid.UUID:
    with session_scope(world.factory) as s:
        wf = s.scalar(sa.select(ResumeWorkflow).where(ResumeWorkflow.resume_id == rid))
        assert wf is not None
        ex = WorkflowExecution(
            resume_id=rid,
            workflow_id=wf.id,
            resume_version=1,
            trigger="SCHEDULED",
            status="RUNNING",
            search_window_start=FIRE - timedelta(hours=24),
            search_window_end=FIRE,
            thread_id="t",
        )
        s.add(ex)
        s.flush()
        WorkflowRepository(s).release_lease(rid, "x")
        s.execute(
            sa.update(ResumeWorkflow)
            .where(ResumeWorkflow.resume_id == rid)
            .values(run_lock_owner=lease_owner, run_lock_expires_at=lease_expires)
        )
        return ex.id


@pytest.mark.parametrize("resume_error", [None, LookupError("no checkpoint")])
def test_stale_execution_is_resumed_or_failed(
    world: World, resume_error: BaseException | None
) -> None:
    rid = world.create_resume()
    mark_ran(world, rid, last=FIRE - timedelta(hours=1))
    ex_id = add_running_execution(
        world, rid, lease_owner="dead", lease_expires=FIRE - timedelta(minutes=1)
    )
    runner = FakeRunner()
    runner.resume_error = resume_error
    run_startup(world, runner, FIRE + timedelta(minutes=10))
    assert runner.resumed == [ex_id]
    assert runner.runs == []  # running execution blocks a new run for the resume
    status = world.executions(rid)[0].status
    assert status == (
        ExecutionStatus.RUNNING if resume_error is None else ExecutionStatus.FAILED
    )  # (FakeRunner doesn't touch the row)


def test_stale_execution_of_stopped_resume_is_cancelled(world: World) -> None:
    rid = world.create_resume()
    world.set_status(rid, LifecycleAction.STOP)
    add_running_execution(world, rid, lease_owner=None, lease_expires=None)
    runner = FakeRunner()
    run_startup(world, runner, FIRE)
    assert runner.resumed == []
    assert world.executions(rid)[0].status == ExecutionStatus.CANCELLED


def test_live_execution_is_left_alone(world: World) -> None:
    rid = world.create_resume()
    add_running_execution(
        world, rid, lease_owner="alive", lease_expires=FIRE + timedelta(minutes=10)
    )
    runner = FakeRunner()
    run_startup(world, runner, FIRE)
    assert runner.resumed == [] and runner.runs == []
    assert world.executions(rid)[0].status == ExecutionStatus.RUNNING


# --------------------------------------------------------------------------- end to end


def test_scheduler_drives_real_workflow(world: World) -> None:
    """Activation -> immediate 30-day initial run -> daily job registered."""
    rid = world.create_resume(activate=False)
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    runner = WorkflowRunner(world.deps())

    async def scenario() -> None:
        svc = await started(service(world, runner))
        world.listeners.append(svc.on_status_change)
        try:
            world.set_status(rid, LifecycleAction.ACTIVATE)
            await wait_for(lambda: world.workflow(rid).initial_search_completed)
            await wait_for(lambda: world.workflow(rid).run_lock_owner is None)
        finally:
            await svc.shutdown()

    asyncio.run(scenario())
    (execution,) = world.executions(rid)
    assert execution.trigger == "INITIAL" and execution.status == ExecutionStatus.SUCCEEDED
    assert world.applications(rid)["good"].status is ApplicationStatus.APPLIED
    assert world.workflow(rid).next_run_at is not None


# --------------------------------------------------------------------------- retention


def test_stop_and_reactivate_set_retention(world: World) -> None:
    rid = world.create_resume()
    world.set_status(rid, LifecycleAction.STOP)
    with session_scope(world.factory) as s:
        resume = s.get(Resume, rid)
        assert resume is not None and resume.purge_after is not None
        assert resume.purge_after - datetime.now(UTC) > timedelta(days=364)
    world.set_status(rid, LifecycleAction.ACTIVATE)
    with session_scope(world.factory) as s:
        resume = s.get(Resume, rid)
        assert resume is not None and resume.purge_after is None


def test_maintenance_purges_expired_data(world: World) -> None:
    rid = world.create_resume()
    world.feed(
        JobSource.LINKEDIN,
        make_raw("old"),
        make_raw("open", title="Platform Senior Software Engineer"),
    )
    world.integration.results = [TimeoutError()]  # first submission ends unresolved
    world.run(rid)
    apps = world.applications(rid)
    unresolved = [a for a in apps.values() if a.status is ApplicationStatus.NEEDS_RECONCILIATION]
    assert len(unresolved) == 1
    assert world.file_store.files
    with session_scope(world.factory) as s:
        s.execute(sa.update(Application).values(purge_after=FIRE - timedelta(days=1)))
        s.execute(
            sa.update(Resume).where(Resume.id == rid).values(purge_after=FIRE - timedelta(days=1))
        )

    result = asyncio.run(service(world).run_maintenance())
    assert result == {"resumes": 1, "applications": 1}
    assert world.file_store.files == {}
    remaining = world.applications(rid)
    assert [a.status for a in remaining.values()] == [ApplicationStatus.NEEDS_RECONCILIATION]
    with session_scope(world.factory) as s:
        version = s.scalar(sa.select(ResumeVersion).where(ResumeVersion.resume_id == rid))
        assert version is not None and version.extracted_text is None and version.profile == {}
        resume = s.get(Resume, rid)
        assert resume is not None and resume.deleted_at is not None
        assert ApplicationService(s).get(remaining[next(iter(remaining))].id) is not None


# --------------------------------------------------------------------------- bootstrap


def test_no_llm_configured_means_review_not_apply() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    matcher = build_matcher(settings)
    assert isinstance(matcher._assessor, NoLLMAssessor)
    from tests.unit.test_job_matcher import PREFS, PROFILE, posting

    decision = asyncio.run(matcher.match(PROFILE, PREFS, posting()))
    assert decision.recommendation.value == "REVIEW" and not decision.eligible_for_auto_apply


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_scheduler_runtime_on_postgres() -> None:
    """The production wiring (Postgres engine + checkpointer + scheduler) starts, rebuilds
    its schedule from the database and shuts down cleanly."""
    from job_agent.persistence.db import create_db_engine, create_session_factory
    from job_agent.persistence.file_store import InMemoryResumeFileStore
    from job_agent.persistence.orm import Base
    from job_agent.repositories.resumes import UserRepository
    from job_agent.scheduler.bootstrap import scheduler_runtime
    from job_agent.services.resume_parser import ResumeParser
    from job_agent.services.resume_service import ResumeService
    from tests.conftest import SAMPLE_RESUME_LINES, make_docx
    from tests.e2e.conftest import PREFS

    url = os.environ["TEST_DATABASE_URL"]
    from tests.db_guard import assert_test_database

    assert_test_database(url)
    engine = create_db_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as s:
            user = UserRepository(s).get_or_create("auth0|pg-runtime")
            svc = ResumeService(s, ResumeParser(), InMemoryResumeFileStore())
            rid = svc.create_resume(
                user.id, "a.docx", make_docx(SAMPLE_RESUME_LINES), preferences=PREFS
            ).resume.id
            svc.change_status(user.id, rid, LifecycleAction.ACTIVATE)
            wf = WorkflowRepository(s).get_by_resume(rid)
            assert wf is not None
            wf.initial_search_completed = True
            wf.last_execution_at = datetime.now(UTC)
        settings = Settings(_env_file=None, database_url=url)  # type: ignore[call-arg]

        async def scenario() -> tuple[set[str], datetime | None]:
            async with scheduler_runtime(settings) as runtime:
                return runtime.registered_resumes(), runtime.next_run_for(rid)

        registered, next_run = asyncio.run(scenario())
        assert registered == {str(rid)} and next_run is not None
        with session_scope(factory) as s:
            stored = WorkflowRepository(s).get_by_resume(rid)
            assert stored is not None and stored.next_run_at == next_run
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_process_main_runs_until_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m job_agent.scheduler`` starts the runtime and stops cleanly on SIGTERM."""
    import signal
    from contextlib import asynccontextmanager

    from job_agent.scheduler import bootstrap

    events: list[str] = []

    @asynccontextmanager
    async def fake_runtime(settings: Settings):  # type: ignore[no-untyped-def]
        events.append("started")
        try:
            yield None
        finally:
            events.append("stopped")

    monkeypatch.setattr(bootstrap, "scheduler_runtime", fake_runtime)
    monkeypatch.setattr(bootstrap, "get_settings", lambda: Settings(_env_file=None, log_json=False))  # type: ignore[call-arg]

    async def scenario() -> None:
        asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(bootstrap.main(), timeout=5)

    asyncio.run(scenario())
    assert events == ["started", "stopped"]
