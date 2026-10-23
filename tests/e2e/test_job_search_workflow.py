"""End-to-end LangGraph workflow tests (spec §12 cases noted inline)."""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

import pytest

from job_agent.job_sources.base import TransientSourceError
from job_agent.models.application import ApplicationStatus, SubmissionAttemptStatus
from job_agent.models.job import JobSource
from job_agent.models.resume import JobSearchPreferences
from job_agent.persistence.orm import ExecutionStatus
from job_agent.services.application_service import SubmissionOutcome, SubmissionResult
from job_agent.services.lifecycle import LifecycleAction
from job_agent.workflows.nodes.application_nodes import STOPPED_BEFORE_SUBMISSION
from job_agent.workflows.nodes.search_nodes import SearchNodes
from job_agent.workflows.state import HaltReason
from tests.conftest import RUN_AT, make_raw
from tests.e2e.conftest import PREFS, World

A = ApplicationStatus


def q(key: str, label: str, *, required: bool = True):  # type: ignore[no-untyped-def]
    from job_agent.models.application import ApplicationQuestion

    return ApplicationQuestion(key=key, label=label, field_name=f"field_{key}", required=required)


def standard_feed(world: World) -> None:
    world.feed(
        JobSource.LINKEDIN,
        make_raw("good"),  # 80 -> APPLY
        make_raw("low", title="Data Engineer Senior Software Engineer"),
        make_raw(
            "nodate",
            title="Platform Senior Software Engineer",
            published_at=None,
            apply_url="https://boards.greenhouse.io/acme/jobs/nodate",
        ),
    )
    # Same job as "good" on another board (shared ATS URL) -> deduplicated.
    world.feed(
        JobSource.INDEED,
        make_raw("good-indeed", apply_url="https://boards.greenhouse.io/acme/jobs/good"),
    )
    world.assessor.scores = {"Data Engineer": 59}


# --------------------------------------------------------------------------- happy path


def test_initial_run_end_to_end(world: World) -> None:  # cases 1, 3, 6, 7
    rid = world.create_resume()
    standard_feed(world)
    result = world.run(rid)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert world.clients[JobSource.LINKEDIN].calls[0][1] == RUN_AT - timedelta(days=30)
    apps = world.applications(rid)
    assert apps["good"].status is A.APPLIED and apps["good"].external_confirmation_ref == "CONF-1"
    assert apps["low"].status is A.SKIPPED
    assert apps["nodate"].status is A.PENDING_REVIEW  # missing date: never auto-applied
    assert "good-indeed" not in apps  # duplicate posting folded into "good"
    assert len(world.integration.calls) == 1

    s = result.summary
    assert s["trigger"] == "INITIAL" and s["discovered_by_source"]["linkedin"] == 3
    assert (s["total_discovered"], s["unique_after_dedup"], s["evaluated"]) == (4, 3, 3)
    assert (s["applied"], s["pending_review"], s["skipped"]["count"]) == (1, 1, 1)
    assert s["at_or_above_threshold"] == 2 and len(s["application_ids"]) == 3
    assert s["sources"]["ziprecruiter"]["status"] == "ok"
    assert {j["title"] for j in s["eligible_jobs"]} == {
        "Senior Software Engineer",
        "Platform Senior Software Engineer",
    }

    wf = world.workflow(rid)
    assert wf.initial_search_completed and wf.last_execution_status == ExecutionStatus.SUCCEEDED
    assert wf.last_window_end == RUN_AT
    (execution,) = world.executions(rid)
    assert (
        execution.applied_count,
        execution.skipped_count,
        execution.pending_review_count,
        execution.discovered_count,
        execution.unique_count,
    ) == (1, 1, 1, 4, 3)
    assert {str(a.execution_id) for a in apps.values()} == {str(result.execution_id)}


def test_recurring_run_after_restart(world: World) -> None:  # cases 2, 8, 12
    rid = world.create_resume()
    standard_feed(world)
    world.run(rid)

    world.now = RUN_AT + timedelta(days=1)  # next day, new process (fresh runner/deps)
    fresh = make_raw(
        "fresh",
        title="Staff Senior Software Engineer",
        published_at=(world.now - timedelta(hours=3)).isoformat(),
    )
    # "good" resurfaces on ZipRecruiter: already applied by this resume -> not re-evaluated.
    world.feed(
        JobSource.ZIPRECRUITER,
        fresh,
        make_raw(
            "good-zr",
            published_at=(world.now - timedelta(hours=1)).isoformat(),
            apply_url="https://boards.greenhouse.io/acme/jobs/good",
        ),
    )
    world.clients.pop(JobSource.LINKEDIN)
    world.clients.pop(JobSource.INDEED)
    result = world.run(rid)

    assert result.summary["trigger"] == "SCHEDULED"
    assert world.clients[JobSource.ZIPRECRUITER].calls[-1][1] == world.now - timedelta(hours=24)
    assert result.summary["already_tracked"] == 1 and result.summary["evaluated"] == 1
    assert world.applications(rid)["fresh"].status is A.APPLIED
    assert len(world.integration.calls) == 2  # "good" was not submitted again
    assert [e.trigger for e in world.executions(rid)] == ["INITIAL", "SCHEDULED"]


# --------------------------------------------------------------------------- independence


def test_paused_resume_does_not_affect_another(world: World) -> None:  # case 10
    a = world.create_resume()
    b = world.create_resume(extra_line="Intern at Bar, Jun 2015 - Aug 2015")
    world.set_status(b, LifecycleAction.PAUSE)
    world.feed(JobSource.LINKEDIN, make_raw("good"))

    async def both():  # type: ignore[no-untyped-def]
        runner = world.runner()
        return await asyncio.gather(runner.run(a), runner.run(b))

    ra, rb = asyncio.run(both())
    assert ra.status == ExecutionStatus.SUCCEEDED
    assert rb.status == ExecutionStatus.SKIPPED and rb.summary["halt_reason"] == (
        HaltReason.NOT_ACTIVE
    )
    assert world.executions(b) == [] and world.applications(b) == {}
    assert world.applications(a)["good"].status is A.APPLIED


def test_different_resumes_apply_to_same_job_independently(world: World) -> None:  # case 9
    a = world.create_resume()
    b = world.create_resume(extra_line="Intern at Bar, Jun 2015 - Aug 2015")
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    world.run(a)
    world.run(b)
    app_a, app_b = world.applications(a)["good"], world.applications(b)["good"]
    assert app_a.status is A.APPLIED and app_b.status is A.APPLIED
    assert app_a.application_key != app_b.application_key
    assert {p.resume_id for p, _ in world.integration.calls} == {a, b}


def test_missing_and_stopped_resumes_halt(world: World) -> None:
    import uuid

    assert world.run(uuid.uuid4()).summary["halt_reason"] == HaltReason.RESUME_NOT_FOUND
    draft = world.create_resume(activate=False)
    assert world.run(draft).summary["halt_reason"] == HaltReason.NOT_ACTIVE


# --------------------------------------------------------------------------- stop signal


def test_stop_before_submission_prevents_submission(world: World) -> None:  # case 11
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    # The user stops the workflow while the job is being evaluated.
    world.assessor.on_assess = lambda: world.set_status(rid, LifecycleAction.STOP)
    result = world.run(rid)

    assert world.integration.calls == [] and world.attempts() == []
    app = world.applications(rid)["good"]
    assert app.status is A.PENDING_REVIEW
    assert app.status_reason in (
        STOPPED_BEFORE_SUBMISSION,
        "workflow stopped or paused before application; not submitted",
    )
    assert result.status == ExecutionStatus.CANCELLED
    assert not world.workflow(rid).initial_search_completed  # initial search will rerun


def test_stop_check_is_immediately_before_submit(world: World) -> None:
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    nodes_module = __import__(
        "job_agent.workflows.nodes.application_nodes", fromlist=["ApplicationNodes"]
    )
    original = nodes_module.ApplicationNodes.validate_application_answers

    async def stop_after_validation(self, state):  # type: ignore[no-untyped-def]
        out = await original(self, state)
        world.set_status(rid, LifecycleAction.PAUSE)  # arrives after all earlier checks
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            nodes_module.ApplicationNodes, "validate_application_answers", stop_after_validation
        )
        world.run(rid)
    assert world.integration.calls == []
    assert world.applications(rid)["good"].status_reason == STOPPED_BEFORE_SUBMISSION


# --------------------------------------------------------------------------- sources


def test_failed_source_does_not_fail_search(world: World) -> None:  # case 14
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    world.feed(JobSource.INDEED, errors=[TransientSourceError()])
    result = world.run(rid)
    assert result.status == ExecutionStatus.PARTIAL
    assert result.summary["sources"]["indeed"]["status"] == "failed"
    assert world.applications(rid)["good"].status is A.APPLIED
    assert world.workflow(rid).initial_search_completed  # something was searched


def test_nothing_searchable_fails_and_keeps_initial_pending(world: World) -> None:
    rid = world.create_resume()
    world.unavailable = {JobSource.LINKEDIN, JobSource.INDEED, JobSource.ZIPRECRUITER}
    result = world.run(rid)
    assert result.status == ExecutionStatus.FAILED
    assert result.summary["sources"]["linkedin"]["manual_search_url"]
    wf = world.workflow(rid)
    assert not wf.initial_search_completed and wf.last_window_end is None


# --------------------------------------------------------------------------- application rules


def test_missing_answers_route_to_review(world: World) -> None:  # case 15
    rid = world.create_resume()
    world.integration.form = [
        q("work_authorization_us", "Are you authorized to work in the US?"),
        q("requires_sponsorship", "Will you require visa sponsorship?"),
    ]
    world.approve_answer("work_authorization_us", "Yes")
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    world.run(rid)
    app = world.applications(rid)["good"]
    assert app.status is A.PENDING_REVIEW and world.integration.calls == []
    assert app.error_details is not None
    assert app.error_details["missing_questions"] == ["requires_sponsorship"]
    assert app.error_details["review_kind"] == "missing_answers"


def test_approved_answers_are_submitted(world: World) -> None:
    rid = world.create_resume()
    world.integration.form = [q("work_authorization_us", "Are you authorized to work in the US?")]
    world.approve_answer("work_authorization_us", "Yes")
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    world.run(rid)
    (package, key) = world.integration.calls[0]
    assert package.answers == {"field_work_authorization_us": "Yes"} and key.endswith(":1")
    assert package.resume_file_ref.endswith(".docx")


def test_unsupported_method_is_manual_action(world: World) -> None:
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("easy", apply_method="easy_apply", apply_url=None))
    world.run(rid)
    app = world.applications(rid)["easy"]
    assert app.status is A.APPLICATION_UNAVAILABLE and world.integration.calls == []
    assert app.error_details is not None and app.error_details["manual_url"].startswith("https://")


@pytest.mark.parametrize(
    ("result", "status", "attempt_status"),
    [
        (TimeoutError(), A.NEEDS_RECONCILIATION, SubmissionAttemptStatus.UNCERTAIN),
        (
            SubmissionResult(SubmissionOutcome.CONFIRMED),
            A.NEEDS_RECONCILIATION,
            SubmissionAttemptStatus.UNCERTAIN,
        ),  # "success" without a reference is not proof
        (
            SubmissionResult(SubmissionOutcome.REJECTED, detail="position filled"),
            A.FAILED,
            SubmissionAttemptStatus.REJECTED,
        ),
        (ConnectionResetError(), A.NEEDS_RECONCILIATION, SubmissionAttemptStatus.UNCERTAIN),
    ],
)
def test_submission_outcomes(
    world: World, result: object, status: ApplicationStatus, attempt_status: SubmissionAttemptStatus
) -> None:
    rid = world.create_resume()
    world.integration.results = [result]  # type: ignore[list-item]
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    world.run(rid)
    assert world.applications(rid)["good"].status is status
    (attempt,) = world.attempts()
    assert attempt.status is attempt_status


def test_slow_submission_times_out_without_resubmitting(world: World) -> None:  # case 13
    rid = world.create_resume()
    world.integration.delay = 1.0  # longer than submit_timeout_seconds
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    world.run(rid)
    assert world.applications(rid)["good"].status is A.NEEDS_RECONCILIATION
    world.now = RUN_AT + timedelta(days=1)
    world.integration.delay = 0
    world.feed(
        JobSource.LINKEDIN,
        make_raw("good", published_at=(world.now - timedelta(hours=1)).isoformat()),
    )
    world.run(rid)
    assert len(world.integration.calls) == 1  # never automatically resubmitted


def test_per_run_application_cap(world: World) -> None:
    prefs = JobSearchPreferences.model_validate(
        {**PREFS.model_dump(), "max_applications_per_run": 1}
    )
    rid = world.create_resume(prefs=prefs)
    world.feed(
        JobSource.LINKEDIN,
        *[make_raw(f"j{i}", title=f"Senior Software Engineer {i}") for i in range(3)],
    )
    result = world.run(rid)
    assert len(world.integration.calls) == 1
    assert (result.summary["applied"], result.summary["pending_review"]) == (1, 2)


# --------------------------------------------------------------------------- concurrency & recovery


def test_concurrent_executions_do_not_duplicate(world: World) -> None:  # case 17
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"), delay_seconds=0.05)

    async def twice():  # type: ignore[no-untyped-def]
        runner = world.runner()
        return await asyncio.gather(runner.run(rid), runner.run(rid))

    results = asyncio.run(twice())
    assert sorted(str(r.status) for r in results) == sorted(
        [ExecutionStatus.SUCCEEDED, ExecutionStatus.SKIPPED]
    )
    assert any(r.summary.get("halt_reason") == HaltReason.ALREADY_RUNNING for r in results)
    assert len(world.integration.calls) == 1 and len(world.attempts()) == 1


class SimulatedCrash(BaseException):
    """Stands in for the process dying (not an ordinary, containable error)."""


def test_resume_from_checkpoint_after_crash(world: World) -> None:
    rid = world.create_resume()
    world.feed(
        JobSource.LINKEDIN,
        make_raw("good"),
        make_raw("low", title="Data Engineer Senior Software Engineer"),
    )
    world.assessor.scores = {"Data Engineer": 40}
    original = SearchNodes.generate_run_summary
    crashed = {"n": 0}

    def crash_once(self, state):  # type: ignore[no-untyped-def]
        if not crashed["n"]:
            crashed["n"] += 1
            raise SimulatedCrash
        return original(self, state)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SearchNodes, "generate_run_summary", crash_once)
        runner = world.runner()  # nodes are bound when the graph is built
        execution_id = __import__("uuid").uuid4()
        with pytest.raises(SimulatedCrash):
            asyncio.run(runner.run(rid, execution_id=execution_id))
        assert world.executions(rid)[0].status == ExecutionStatus.RUNNING
        calls_before = world.assessor.calls
        result = asyncio.run(runner.resume(execution_id))

    assert result.status == ExecutionStatus.SUCCEEDED and result.summary["applied"] == 1
    assert world.assessor.calls == calls_before  # completed jobs were not re-evaluated
    assert len(world.integration.calls) == 1 and world.match_count() == 2
    assert world.workflow(rid).initial_search_completed
    assert asyncio.run(runner.resume(execution_id)).status == ExecutionStatus.SUCCEEDED


def test_crash_after_confirmed_submission_uses_checkpointed_result(world: World) -> None:
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    nodes_module = __import__(
        "job_agent.workflows.nodes.application_nodes", fromlist=["ApplicationNodes"]
    )
    original = nodes_module.ApplicationNodes.persist_application_result
    crashed = {"n": 0}

    def die_once(self, state):  # type: ignore[no-untyped-def]
        if not crashed["n"]:
            crashed["n"] += 1
            raise SimulatedCrash  # dies after the confirmed reply was checkpointed
        return original(self, state)

    execution_id = __import__("uuid").uuid4()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nodes_module.ApplicationNodes, "persist_application_result", die_once)
        runner = world.runner()
        with pytest.raises(SimulatedCrash):
            asyncio.run(runner.run(rid, execution_id=execution_id))
        assert world.applications(rid)["good"].status is A.SUBMITTING
        result = asyncio.run(runner.resume(execution_id))

    # The platform's confirmation survived in the checkpoint: recorded, not resubmitted.
    assert len(world.integration.calls) == 1
    app = world.applications(rid)["good"]
    assert app.status is A.APPLIED and app.external_confirmation_ref == "CONF-1"
    assert result.summary["applied"] == 1


def test_crash_during_platform_call_is_never_resubmitted(world: World) -> None:  # case 13
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    # The process dies while the request is in flight: we cannot know if it arrived.
    world.integration.results = [SimulatedCrash()]
    execution_id = __import__("uuid").uuid4()
    runner = world.runner()
    with pytest.raises(SimulatedCrash):
        asyncio.run(runner.run(rid, execution_id=execution_id))
    assert world.applications(rid)["good"].status is A.SUBMITTING
    result = asyncio.run(runner.resume(execution_id))

    assert len(world.integration.calls) == 1  # not submitted a second time
    assert world.applications(rid)["good"].status is A.NEEDS_RECONCILIATION
    assert [a.status for a in world.attempts()] == [SubmissionAttemptStatus.UNCERTAIN]
    assert result.summary["needs_reconciliation"] == 1


def test_unexpected_job_error_is_contained(world: World) -> None:
    rid = world.create_resume()
    world.feed(
        JobSource.LINKEDIN,
        make_raw("good"),
        make_raw("boom", title="Boom Senior Software Engineer"),
    )

    def explode() -> None:
        if world.assessor.calls == 2:
            raise ValueError("bug")

    world.assessor.on_assess = explode
    result = world.run(rid)
    assert result.status == ExecutionStatus.PARTIAL and result.summary["errors"] == 1
    assert result.summary["applied"] == 1


def test_resume_unknown_execution(world: World) -> None:
    import uuid

    with pytest.raises(LookupError):
        asyncio.run(world.runner().resume(uuid.uuid4()))


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_postgres_checkpointer_resume(world: World) -> None:
    from job_agent.workflows.runner import WorkflowRunner, postgres_checkpointer

    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    execution_id = __import__("uuid").uuid4()
    original = SearchNodes.generate_run_summary
    crashed = {"n": 0}

    def crash_once(self, state):  # type: ignore[no-untyped-def]
        if not crashed["n"]:
            crashed["n"] += 1
            raise SimulatedCrash
        return original(self, state)

    async def scenario() -> str:
        url = os.environ["TEST_DATABASE_URL"]
        async with postgres_checkpointer(url) as saver:
            with pytest.raises(SimulatedCrash):
                await WorkflowRunner(world.deps(), checkpointer=saver).run(
                    rid, execution_id=execution_id
                )
        # New process: new runner, new connection, same durable checkpoint.
        async with postgres_checkpointer(url) as saver:
            result = await WorkflowRunner(world.deps(), checkpointer=saver).resume(execution_id)
            return str(result.status)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SearchNodes, "generate_run_summary", crash_once)
        assert asyncio.run(scenario()) == ExecutionStatus.SUCCEEDED
    assert len(world.integration.calls) == 1


def test_missing_application_url_is_manual_action(world: World) -> None:
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("nourl", apply_url=None, apply_method="external"))
    world.run(rid)
    app = world.applications(rid)["nourl"]
    assert app.status is A.APPLICATION_UNAVAILABLE and world.integration.calls == []
    assert "no application URL" in (app.status_reason or "")


class PeakTracker:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def hold(self, seconds: float) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(seconds)
        self.active -= 1


def test_graph_concurrency_limits(world: World) -> None:
    from job_agent.config import Settings

    world.settings = Settings(
        _env_file=None,
        max_concurrent_job_searches=1,  # type: ignore[call-arg]
        max_concurrent_matches=2,
    )
    rid = world.create_resume()
    searches, matches = PeakTracker(), PeakTracker()
    for source in JobSource:
        client = world.feed(
            source, make_raw(f"{source.value}-1", title=f"Senior Software Engineer {source.value}")
        )
        original = client.search

        async def slow_search(*args, _orig=original, **kw):  # type: ignore[no-untyped-def]
            await searches.hold(0.02)
            return await _orig(*args, **kw)

        client.search = slow_search  # type: ignore[method-assign]
    world.feed(
        JobSource.LINKEDIN,
        *[make_raw(f"m{i}", title=f"Senior Software Engineer {i}") for i in range(6)],
    )
    original_assess = world.assessor.assess

    async def slow_assess(*args, **kw):  # type: ignore[no-untyped-def]
        await matches.hold(0.02)
        return await original_assess(*args, **kw)

    world.assessor.assess = slow_assess  # type: ignore[method-assign]
    result = world.run(rid)
    assert result.summary["evaluated"] == 8
    assert searches.peak == 1 and matches.peak == 2
