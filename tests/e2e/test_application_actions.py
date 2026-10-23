"""Phase 6 end-to-end: reconciliation, safe retry, manual packages, verified answers,
and the Greenhouse integration driven through the real workflow graph."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta
from typing import Any

import httpx
import pytest
import sqlalchemy as sa

from job_agent.config import Settings
from job_agent.integrations.greenhouse import GreenhouseIntegration
from job_agent.models.application import (
    ApplicationQuestion,
    ApplicationStatus,
    QuestionKind,
    SubmissionAttemptStatus,
)
from job_agent.models.job import JobSource
from job_agent.persistence.db import session_scope
from job_agent.persistence.orm import Application, SubmissionAttempt
from job_agent.services.answers import GeneratedAnswer, GroundedStatement
from job_agent.services.application_actions import ApplicationActions
from job_agent.services.application_service import (
    LookupOutcome,
    LookupResult,
    SubmissionOutcome,
    SubmissionRegistry,
    SubmissionResult,
)
from job_agent.services.errors import Conflict, NotFound
from job_agent.services.lifecycle import LifecycleAction
from tests.conftest import RUN_AT, make_raw
from tests.e2e.conftest import World

A = ApplicationStatus


def actions(world: World) -> ApplicationActions:
    return ApplicationActions(world.deps())


def one_app(world: World, rid: uuid.UUID, ext: str = "good") -> Application:
    return world.applications(rid)[ext]


def run_job(world: World, *, result: Any = None, **raw: Any) -> tuple[uuid.UUID, Application]:
    rid = world.create_resume()
    if result is not None:
        world.integration.results = [result]
    world.feed(JobSource.LINKEDIN, make_raw("good", **raw))
    world.run(rid)
    return rid, one_app(world, rid)


# --------------------------------------------------------------------------- retry


def test_retry_after_retryable_rejection_applies(world: World) -> None:
    rid, app = run_job(
        world,
        result=SubmissionResult(SubmissionOutcome.REJECTED, detail="rate limited", retryable=True),
    )
    assert app.status is A.FAILED
    out = asyncio.run(actions(world).retry(world.user_id, app.id))  # type: ignore[arg-type]
    assert out.status is A.APPLIED and out.application_id == app.id
    after = one_app(world, rid)
    assert after.retry_count == 1 and after.external_confirmation_ref == "CONF-2"
    assert [a.status for a in sorted(world.attempts(), key=lambda a: a.attempt_number)] == [
        SubmissionAttemptStatus.REJECTED,
        SubmissionAttemptStatus.CONFIRMED,
    ]
    assert len(world.applications(rid)) == 1  # same row: no duplicate application


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (SubmissionResult(SubmissionOutcome.REJECTED, detail="position filled"), "not retryable"),
        (TimeoutError(), "reconcile it first"),
        (None, "cannot be retried"),  # APPLIED
    ],
)
def test_unsafe_retries_are_refused(world: World, result: Any, message: str) -> None:
    _, app = run_job(world, result=result)
    with pytest.raises(Conflict, match=message):
        asyncio.run(actions(world).retry(world.user_id, app.id))  # type: ignore[arg-type]
    assert len(world.integration.calls) == 1


def test_retry_after_missing_answer_is_approved(world: World) -> None:
    world.integration.form = [
        ApplicationQuestion(
            key="requires_sponsorship",
            label="Will you require sponsorship?",
            field_name="q_sponsor",
            kind=QuestionKind.BOOLEAN,
            required=True,
            options=[],
            sensitive=True,
        )
    ]
    rid, app = run_job(world)
    assert app.status is A.PENDING_REVIEW and world.integration.calls == []
    world.approve_answer("requires_sponsorship", "No")
    out = asyncio.run(actions(world).retry(world.user_id, app.id))  # type: ignore[arg-type]
    assert out.status is A.APPLIED
    assert world.integration.calls[0][0].answers == {"q_sponsor": "No"}


def test_match_review_is_not_retryable(world: World) -> None:
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("nodate", published_at=None))
    world.run(rid)
    app = one_app(world, rid, "nodate")
    assert app.status is A.PENDING_REVIEW
    with pytest.raises(Conflict, match="needs your review"):
        asyncio.run(actions(world).retry(world.user_id, app.id))  # type: ignore[arg-type]


def test_retry_limit_and_ownership_and_paused_workflow(world: World) -> None:
    world.settings = Settings(_env_file=None, max_application_retries=1)  # type: ignore[call-arg]
    retryable = SubmissionResult(SubmissionOutcome.REJECTED, detail="busy", retryable=True)
    rid, app = run_job(world, result=retryable)
    with pytest.raises(NotFound):
        asyncio.run(actions(world).retry(uuid.uuid4(), app.id))
    world.set_status(rid, LifecycleAction.PAUSE)
    with pytest.raises(Conflict, match="resume the workflow"):
        asyncio.run(actions(world).retry(world.user_id, app.id))  # type: ignore[arg-type]
    world.set_status(rid, LifecycleAction.RESUME)
    world.integration.results = [retryable]
    assert asyncio.run(actions(world).retry(world.user_id, app.id)).status is A.FAILED  # type: ignore[arg-type]
    with pytest.raises(Conflict, match="retry limit"):
        asyncio.run(actions(world).retry(world.user_id, app.id))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- reconcile


def uncertain(world: World) -> tuple[uuid.UUID, Application]:
    rid, app = run_job(world, result=TimeoutError())
    assert app.status is A.NEEDS_RECONCILIATION
    return rid, app


def test_reconcile_found_marks_applied(world: World) -> None:
    rid, app = uncertain(world)
    world.integration.lookups = [LookupResult(LookupOutcome.FOUND, external_ref="GH-1")]
    out = asyncio.run(actions(world).reconcile(app.id, user_id=world.user_id))
    assert out.status is A.APPLIED
    after = one_app(world, rid)
    assert after.external_confirmation_ref == "GH-1" and after.reconcile_attempts == 1
    assert after.submission_evidence is not None
    assert after.submission_evidence["verification"] == "reconciliation_lookup"
    assert [a.status for a in world.attempts()] == [SubmissionAttemptStatus.CONFIRMED]
    assert len(world.integration.calls) == 1  # never resubmitted


def test_reconcile_not_found_waits_for_indexing_then_allows_retry(world: World) -> None:
    rid, app = uncertain(world)
    world.integration.lookups = [LookupResult(LookupOutcome.NOT_FOUND)] * 2
    young = asyncio.run(actions(world).reconcile(app.id))
    assert young.status is A.NEEDS_RECONCILIATION  # too soon to trust "not found"
    world.now = RUN_AT + timedelta(hours=1)
    old = asyncio.run(actions(world).reconcile(app.id))
    assert old.status is A.FAILED
    assert asyncio.run(actions(world).retry(world.user_id, app.id)).status is A.APPLIED  # type: ignore[arg-type]


def test_reconcile_unknown_escalates_to_review(world: World) -> None:
    world.settings = Settings(_env_file=None, reconcile_max_attempts=2)  # type: ignore[call-arg]
    rid, app = uncertain(world)
    world.integration.lookups = [RuntimeError("api down"), LookupResult(LookupOutcome.UNKNOWN)]
    assert asyncio.run(actions(world).reconcile(app.id)).status is A.NEEDS_RECONCILIATION
    assert asyncio.run(actions(world).reconcile(app.id)).status is A.PENDING_REVIEW
    after = one_app(world, rid)
    assert after.error_details is not None and after.error_details["review_kind"] == (
        "reconciliation"
    )
    with pytest.raises(Conflict):  # the user must check the employer's site first
        asyncio.run(actions(world).retry(world.user_id, app.id))  # type: ignore[arg-type]
    assert asyncio.run(actions(world).reconcile(app.id)).detail == "nothing to reconcile"


def test_reconcile_without_integration_cannot_confirm(world: World) -> None:
    rid, app = uncertain(world)
    deps = world.deps()
    deps.submissions = SubmissionRegistry([])
    out = asyncio.run(ApplicationActions(deps).reconcile(app.id))
    assert out.status is A.NEEDS_RECONCILIATION and "no integration" in out.detail


def test_interrupted_submission_is_reconciled(world: World) -> None:
    rid, app = run_job(world)
    with session_scope(world.factory) as s:  # simulate a crash mid-submission
        row = s.get(Application, app.id)
        assert row is not None
        row.status = A.SUBMITTING
        attempt = s.scalar(sa.select(SubmissionAttempt))
        assert attempt is not None
        attempt.status = SubmissionAttemptStatus.STARTED
    in_flight = asyncio.run(actions(world).reconcile(app.id))
    assert in_flight.status is A.SUBMITTING  # may still be running: left alone
    world.now = RUN_AT + timedelta(hours=1)
    world.integration.lookups = [LookupResult(LookupOutcome.FOUND, external_ref="R-9")]
    assert asyncio.run(actions(world).reconcile(app.id)).status is A.APPLIED


def test_reconcile_due_sweeps_only_due_records(world: World) -> None:
    rid, app = uncertain(world)
    world.integration.lookups = [LookupResult(LookupOutcome.UNKNOWN)] * 3
    assert asyncio.run(actions(world).reconcile_due()) == {"NEEDS_RECONCILIATION": 1}
    assert asyncio.run(actions(world).reconcile_due()) == {}  # not due again yet
    world.now = RUN_AT + timedelta(hours=1)
    assert asyncio.run(actions(world).reconcile_due()) == {"NEEDS_RECONCILIATION": 1}
    assert one_app(world, rid).reconcile_attempts == 2


# --------------------------------------------------------------------------- packages & answers


def test_manual_package_for_unsupported_application(world: World) -> None:
    rid = world.create_resume()
    world.approve_answer("work_authorization_us", "Yes")
    world.feed(JobSource.LINKEDIN, make_raw("easy", apply_method="easy_apply"))
    world.run(rid)
    app = one_app(world, rid, "easy")
    assert app.status is A.APPLICATION_UNAVAILABLE
    pkg = (app.submission_evidence or {})["manual_package"]
    assert pkg["apply_url"].startswith("https://boards.greenhouse.io/")
    assert pkg["match_score"] == 80 and pkg["resume_file"].endswith(".docx")
    assert pkg["approved_answers_available"] == ["work_authorization_us"]
    assert "Yes" not in json.dumps(pkg)  # keys only, never sensitive values


def test_generated_answer_is_submitted_and_recorded(world: World) -> None:
    class Gen:
        async def generate(self, question: str, profile: Any, posting: Any) -> GeneratedAnswer:
            return GeneratedAnswer(
                statements=[
                    GroundedStatement(
                        text="I build backend data platforms in healthcare.",
                        # Must be in the verified profile, not merely in the resume file.
                        evidence="Backend engineer building data platforms in healthcare.",
                    )
                ]
            )

    world.integration.form = [
        ApplicationQuestion(
            key="why_interested",
            label="Why do you want to join?",
            field_name="q_why",
            kind=QuestionKind.LONG_TEXT,
            required=True,
        )
    ]
    rid = world.create_resume()
    world.feed(JobSource.LINKEDIN, make_raw("good"))
    deps = world.deps()
    deps.answer_generator = Gen()
    from job_agent.workflows.runner import WorkflowRunner

    asyncio.run(WorkflowRunner(deps).run(rid))
    app = one_app(world, rid)
    assert app.status is A.APPLIED
    assert world.integration.calls[0][0].answers == {
        "q_why": "I build backend data platforms in healthcare."
    }
    answers = (app.submission_evidence or {})["answers"]
    assert answers == [
        {
            "key": "why_interested",
            "field_name": "q_why",
            "source": "generated",
            "value": "I build backend data platforms in healthcare.",
        }
    ]


def test_question_loading_failure_is_retryable_review(world: World) -> None:
    world.integration.form_error = RuntimeError("ats down")
    rid, app = run_job(world)
    assert app.status is A.PENDING_REVIEW
    assert (app.error_details or {})["review_kind"] == "integration_error"
    world.integration.form_error = None
    assert asyncio.run(actions(world).retry(world.user_id, app.id)).status is A.APPLIED  # type: ignore[arg-type]


# --------------------------------------------------------------------------- greenhouse


def test_greenhouse_through_the_workflow(world: World) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "questions": [
                        {
                            "required": True,
                            "label": "First Name",
                            "fields": [{"name": "first_name", "type": "input_text"}],
                        },
                        {
                            "required": True,
                            "label": "Email",
                            "fields": [{"name": "email", "type": "input_text"}],
                        },
                        {
                            "required": True,
                            "label": "Resume",
                            "fields": [{"name": "resume", "type": "input_file"}],
                        },
                        {
                            "required": True,
                            "label": "Are you authorized to work in the US?",
                            "fields": [
                                {
                                    "name": "question_9",
                                    "type": "multi_value_single_select",
                                    "values": [
                                        {"label": "Yes", "value": 1},
                                        {"label": "No", "value": 0},
                                    ],
                                }
                            ],
                        },
                    ]
                },
            )
        return httpx.Response(200, json={"success": "Candidate saved", "application_id": 42})

    gh = GreenhouseIntegration(
        {"acme": "key"},
        file_store=world.file_store,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    rid = world.create_resume()
    for key, value in (
        ("first_name", "Jane"),
        ("email", "jane@example.com"),
        ("work_authorization_us", "Yes"),
    ):
        world.approve_answer(key, value)
    world.feed(
        JobSource.LINKEDIN, make_raw("gh", apply_url="https://boards.greenhouse.io/acme/jobs/555")
    )
    deps = world.deps()
    deps.submissions = SubmissionRegistry([gh])
    from job_agent.workflows.runner import WorkflowRunner

    asyncio.run(WorkflowRunner(deps).run(rid))
    app = one_app(world, rid, "gh")
    assert app.status is A.APPLIED and app.external_confirmation_ref == "greenhouse:acme:42"
    body = json.loads(requests[-1].content)
    assert (body["first_name"], body["email"], body["question_9"]) == (
        "Jane",
        "jane@example.com",
        "1",
    )
    assert body["resume_content"] and body["resume_content_filename"].endswith(".docx")
    evidence = app.submission_evidence or {}
    assert evidence["integration"] == "greenhouse"
    assert evidence["verification"] == "greenhouse_application_id"
    assert "jane@example.com" not in json.dumps(evidence)  # approved values not duplicated


def test_scheduler_registers_submission_reconciliation(world: World) -> None:
    from job_agent.scheduler.scheduler_service import SchedulerService

    calls: list[int] = []

    async def sweep() -> None:
        calls.append(1)

    async def scenario() -> list[str]:
        svc = SchedulerService(
            session_factory=world.factory,
            settings=world.settings,
            runner=None,  # type: ignore[arg-type]
            reconcile_submissions=sweep,
        )
        await svc.start()
        try:
            return [j.id for j in svc.scheduler.get_jobs()]
        finally:
            await svc.shutdown()

    assert "system:reconcile-submissions" in asyncio.run(scenario())
