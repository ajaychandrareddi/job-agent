"""Per-job nodes: matching, threshold routing and application bookkeeping.

All nodes are idempotent so a job can safely be re-run after a crash: the match is an
upsert, the application row is get-or-create on its unique key, and submission is
guarded by the durable attempt record.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from job_agent.models.application import ApplicationStatus
from job_agent.models.job import JobPosting
from job_agent.models.matching import MatchDecision, Recommendation
from job_agent.models.resume import JobSearchPreferences, ResumeProfile
from job_agent.observability import metrics
from job_agent.observability.logging import get_logger
from job_agent.persistence.orm import Application
from job_agent.repositories.matches import JobMatchRepository
from job_agent.services.application_service import ApplicationService
from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.state import JobRoute, JobState, OutcomeKind

log = get_logger(__name__)


class JobNodes:
    def __init__(self, deps: WorkflowDeps) -> None:
        self.d = deps

    async def calculate_job_match(self, state: JobState) -> dict[str, Any]:
        profile = ResumeProfile.model_validate(state["profile"])
        prefs = JobSearchPreferences.model_validate(state["preferences"])
        posting = JobPosting.model_validate(state["job"]["posting"])
        async with self.d.limits(state["execution_id"]).match:
            decision = await self.d.matcher.match(profile, prefs, posting)
        with self.d.session() as s:
            match = JobMatchRepository(s).save(
                resume_id=uuid.UUID(state["resume_id"]),
                resume_version=state["resume_version"],
                job_posting_id=uuid.UUID(state["job"]["job_posting_id"]),
                decision=decision,
                execution_id=uuid.UUID(state["execution_id"]),
            )
            match_id = str(match.id)
        return {"decision": decision.model_dump(mode="json"), "match_id": match_id}

    def apply_match_threshold(self, state: JobState) -> dict[str, Any]:
        """Deterministic routing. The LLM never decides this."""
        d = MatchDecision.model_validate(state["decision"])
        threshold = self.d.settings.job_match_threshold
        date_reason = state["job"].get("date_review_reason")
        if d.hard_constraint_violations:
            return _route(JobRoute.SKIP, "; ".join(v.detail for v in d.hard_constraint_violations))
        if d.recommendation is Recommendation.SKIP or d.overall_score < threshold:
            return _route(JobRoute.SKIP, f"score {d.overall_score:g} below threshold {threshold:g}")
        if d.recommendation is Recommendation.REVIEW:
            return _route(JobRoute.REVIEW, "; ".join(d.review_reasons) or d.explanation)
        if date_reason:
            return _route(JobRoute.REVIEW, f"publication date needs review: {date_reason}")
        if d.recommendation is Recommendation.APPLY and d.eligible_for_auto_apply:
            return _route(JobRoute.APPLY, f"score {d.overall_score:g} meets threshold")
        return _route(JobRoute.REVIEW, "inconsistent match decision")

    def check_existing_applications(self, state: JobState) -> dict[str, Any]:
        route = JobRoute(state["route"])
        execution_id = uuid.UUID(state["execution_id"])
        with self.d.session() as s:
            svc = ApplicationService(s)
            app, created = svc.get_or_create(
                user_id=uuid.UUID(state["user_id"]),
                resume_id=uuid.UUID(state["resume_id"]),
                resume_version=state["resume_version"],
                job_posting_id=uuid.UUID(state["job"]["job_posting_id"]),
                canonical_key=state["job"]["canonical_key"],
                match_id=uuid.UUID(state["match_id"]),
                execution_id=execution_id,
                purge_after=self.d.clock()
                + timedelta(days=self.d.settings.application_retention_days),
            )
            own_retry = not created and app.execution_id == execution_id
            if not created and not own_retry:
                metrics.application_outcomes.add(1, {"status": OutcomeKind.DUPLICATE.value})
                return {
                    "route": JobRoute.DUPLICATE.value,
                    "application_id": str(app.id),
                    "final_status": OutcomeKind.DUPLICATE.value,
                    "reason": f"already tracked for this resume as {app.status.value}",
                }
            if own_retry and app.status is not ApplicationStatus.DISCOVERED:
                return self._resume_own(svc, app, state)
            reason = state.get("reason")
            if route is JobRoute.SKIP:
                svc.transition(app, ApplicationStatus.SKIPPED, reason=reason)
            elif route is JobRoute.REVIEW:
                svc.transition(
                    app,
                    ApplicationStatus.PENDING_REVIEW,
                    reason=reason,
                    details={"review_kind": "match_review"},
                )
            else:
                svc.transition(app, ApplicationStatus.MATCHED, reason=reason)
            if route is JobRoute.REVIEW:
                metrics.review_routed.add(1, {"kind": "match_review"})
            if route in (JobRoute.SKIP, JobRoute.REVIEW):
                metrics.application_outcomes.add(1, {"status": app.status.value})
            return {"application_id": str(app.id), "final_status": app.status.value}

    def _resume_own(
        self, svc: ApplicationService, app: Application, state: JobState
    ) -> dict[str, Any]:
        """This execution was interrupted after creating the application; pick up safely."""
        base = {"application_id": str(app.id)}
        attempt = svc.open_attempt(app.id)
        if attempt is not None or app.status is ApplicationStatus.SUBMITTING:
            detail = "execution interrupted during submission; outcome unknown, not resubmitted"
            if attempt is not None:
                svc.mark_uncertain(app, attempt, self.d.clock(), detail)
            else:
                svc.transition(app, ApplicationStatus.NEEDS_RECONCILIATION, reason=detail)
            return {
                **base,
                "route": JobRoute.HALT.value,
                "reason": detail,
                "final_status": ApplicationStatus.NEEDS_RECONCILIATION.value,
            }
        if app.status in (ApplicationStatus.MATCHED, ApplicationStatus.READY_TO_APPLY):
            return {**base, "route": JobRoute.APPLY.value, "final_status": app.status.value}
        return {
            **base,
            "route": JobRoute.HALT.value,
            "final_status": app.status.value,
            "reason": app.status_reason or app.status.value,
        }

    def record_outcome(self, state: JobState) -> dict[str, Any]:
        return {"job_outcomes": [build_outcome(state)]}


def _route(route: JobRoute, reason: str) -> dict[str, Any]:
    return {"route": route.value, "reason": reason[:500]}


def build_outcome(state: JobState, *, error: str | None = None) -> dict[str, Any]:
    job = state.get("job", {})
    posting = job.get("posting", {})
    decision = state.get("decision") or {}
    sources = [posting.get("source")] + [d.get("source") for d in job.get("duplicates", [])]
    submission = state.get("submission") or {}
    return {
        "canonical_key": job.get("canonical_key"),
        "job_posting_id": job.get("job_posting_id"),
        "title": posting.get("title"),
        "company": posting.get("company_name"),
        "sources": [s for s in dict.fromkeys(sources) if s],
        "canonical_url": posting.get("canonical_url"),
        "application_url": posting.get("application_url"),
        "score": decision.get("overall_score"),
        "recommendation": decision.get("recommendation"),
        "application_id": state.get("application_id"),
        "status": OutcomeKind.ERROR.value if error else state.get("final_status"),
        "reason": error or state.get("reason"),
        "submitted": bool(submission.get("attempt_id")),
        "stopped": bool(
            state.get("route") == JobRoute.HALT.value and "stopped" in (state.get("reason") or "")
        ),
    }
