"""Direct tests of per-job node branches that the end-to-end flows rarely reach."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from job_agent.models.application import ApplicationStatus
from job_agent.models.job import JobSource
from job_agent.models.matching import HardConstraintViolation, MatchDecision, Recommendation
from job_agent.persistence.db import session_scope
from job_agent.persistence.orm import Application, ResumeWorkflow, WorkflowExecution
from job_agent.repositories.jobs import JobPostingRepository
from job_agent.repositories.matches import JobMatchRepository
from job_agent.services.application_service import ApplicationService
from job_agent.services.deduplication import deduplicate
from job_agent.services.normalization import normalize_raw_posting
from job_agent.workflows.nodes.job_nodes import JobNodes
from job_agent.workflows.state import JobRoute, OutcomeKind
from tests.conftest import RUN_AT, make_raw
from tests.e2e.conftest import World
from tests.unit.test_job_matcher import scores

A = ApplicationStatus


def decision(
    score: float = 80,
    rec: Recommendation = Recommendation.APPLY,
    *,
    eligible: bool | None = None,
    review: list[str] | None = None,
    violations: bool = False,
) -> MatchDecision:
    return MatchDecision(
        overall_score=score,
        criterion_scores=scores(score),
        matching_skills=[],
        missing_required_skills=[],
        resume_evidence=[],
        experience_alignment="",
        potential_disqualifiers=[],
        hard_constraint_violations=[HardConstraintViolation(code="x", detail="excluded")]
        if violations
        else [],
        review_reasons=review or [],
        explanation="because",
        recommendation=rec,
        eligible_for_auto_apply=rec is Recommendation.APPLY if eligible is None else eligible,
        threshold=60,
        evaluated_at=RUN_AT,
    )


@pytest.mark.parametrize(
    ("d", "date_reason", "route", "reason_part"),
    [
        (decision(90, violations=True), None, JobRoute.SKIP, "excluded"),
        (decision(59, Recommendation.SKIP), None, JobRoute.SKIP, "below threshold"),
        (
            decision(70, Recommendation.REVIEW, review=["thin evidence"]),
            None,
            JobRoute.REVIEW,
            "thin evidence",
        ),
        (decision(70, Recommendation.REVIEW), None, JobRoute.REVIEW, "because"),
        (decision(80), "publication_date_straddles_window", JobRoute.REVIEW, "publication date"),
        (decision(80), None, JobRoute.APPLY, "meets threshold"),
        (decision(80, eligible=False), None, JobRoute.REVIEW, "inconsistent"),
    ],
)
def test_apply_match_threshold(
    world: World, d: MatchDecision, date_reason: str | None, route: JobRoute, reason_part: str
) -> None:
    out = JobNodes(world.deps()).apply_match_threshold(
        {"decision": d.model_dump(mode="json"), "job": {"date_review_reason": date_reason}}
    )
    assert out["route"] == route.value and reason_part in out["reason"]


# --------------------------------------------------------------------------- existing apps


def _setup(world: World) -> dict[str, Any]:
    rid = world.create_resume()
    posting = normalize_raw_posting(make_raw("j1"), JobSource.LINKEDIN, now=RUN_AT)
    group = deduplicate([posting])[0]
    with session_scope(world.factory) as s:
        record = JobPostingRepository(s).persist_group(group, now=RUN_AT)
        match = JobMatchRepository(s).save(
            resume_id=rid, resume_version=1, job_posting_id=record.id, decision=decision()
        )
        wf = s.scalar(sa.select(ResumeWorkflow).where(ResumeWorkflow.resume_id == rid))
        assert wf is not None
        executions = []
        for i in range(2):
            ex = WorkflowExecution(
                resume_id=rid,
                workflow_id=wf.id,
                resume_version=1,
                trigger="SCHEDULED",
                status="SUCCEEDED",
                search_window_start=RUN_AT - timedelta(hours=24 + i),
                search_window_end=RUN_AT,
                thread_id=str(uuid.uuid4()),
            )
            s.add(ex)
            s.flush()
            executions.append(str(ex.id))
        return {
            "user_id": str(world.user_id),
            "resume_id": str(rid),
            "resume_version": 1,
            "match_id": str(match.id),
            "execution_ids": executions,
            "job": {
                "job_posting_id": str(record.id),
                "canonical_key": group.canonical_key,
                "posting": posting.model_dump(mode="json"),
            },
        }


def _state(base: dict[str, Any], execution: int, route: JobRoute) -> dict[str, Any]:
    return {
        **{k: v for k, v in base.items() if k != "execution_ids"},
        "execution_id": base["execution_ids"][execution],
        "route": route.value,
        "reason": "r",
    }


def _set_status(world: World, app_id: str, status: ApplicationStatus) -> None:
    with session_scope(world.factory) as s:
        app = s.get(Application, uuid.UUID(app_id))
        assert app is not None
        app.status = status  # direct write: simulating state left by a crash


def test_application_from_another_execution_is_duplicate(world: World) -> None:
    base = _setup(world)
    nodes = JobNodes(world.deps())
    first = nodes.check_existing_applications(_state(base, 0, JobRoute.APPLY))
    assert first["final_status"] == A.MATCHED.value
    other = nodes.check_existing_applications(_state(base, 1, JobRoute.APPLY))
    assert other["route"] == JobRoute.DUPLICATE.value
    assert other["final_status"] == OutcomeKind.DUPLICATE.value
    assert other["application_id"] == first["application_id"]


@pytest.mark.parametrize(
    ("left_as", "route", "final"),
    [
        (A.MATCHED, JobRoute.APPLY, A.MATCHED),
        (A.READY_TO_APPLY, JobRoute.APPLY, A.READY_TO_APPLY),
        (A.SKIPPED, JobRoute.HALT, A.SKIPPED),
        (A.SUBMITTING, JobRoute.HALT, A.NEEDS_RECONCILIATION),
    ],
)
def test_resumed_execution_picks_up_its_own_application(
    world: World, left_as: ApplicationStatus, route: JobRoute, final: ApplicationStatus
) -> None:
    base = _setup(world)
    nodes = JobNodes(world.deps())
    created = nodes.check_existing_applications(_state(base, 0, JobRoute.APPLY))
    _set_status(world, created["application_id"], left_as)
    again = nodes.check_existing_applications(_state(base, 0, JobRoute.APPLY))
    assert again["route"] == route.value and again["final_status"] == final.value
    assert again["application_id"] == created["application_id"]


def test_resumed_execution_with_open_attempt_is_reconciled(world: World) -> None:
    base = _setup(world)
    nodes = JobNodes(world.deps())
    created = nodes.check_existing_applications(_state(base, 0, JobRoute.APPLY))
    with session_scope(world.factory) as s:
        svc = ApplicationService(s)
        app = svc.get(uuid.UUID(created["application_id"]))
        assert app is not None
        svc.transition(app, A.READY_TO_APPLY)
        svc.start_attempt(app, RUN_AT)
    again = nodes.check_existing_applications(_state(base, 0, JobRoute.APPLY))
    assert again["final_status"] == A.NEEDS_RECONCILIATION.value
    assert [a.status.value for a in world.attempts()] == ["UNCERTAIN"]


@pytest.mark.parametrize(
    ("route", "status"), [(JobRoute.SKIP, A.SKIPPED), (JobRoute.REVIEW, A.PENDING_REVIEW)]
)
def test_new_application_status_follows_route(
    world: World, route: JobRoute, status: ApplicationStatus
) -> None:
    base = _setup(world)
    out = JobNodes(world.deps()).check_existing_applications(_state(base, 0, route))
    assert out["final_status"] == status.value
