"""Application subgraph nodes (spec §4 steps 1-10).

verify_resume_active -> confirm_eligibility -> validate_application_method ->
prepare_application -> validate_application_answers -> submit_application ->
verify_submission -> persist_application_result

Every early exit records *why* (``error_details.review_kind``) so the retry service can
tell a fixable review (e.g. a missing approved answer) from one that needs a human
decision (e.g. an uncertain publication date). Answer *values* never enter workflow
checkpoints: only question keys and answer provenance do.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from job_agent.models.application import (
    AnswerSource,
    ApplicationQuestion,
    ApplicationStatus,
    ResolvedAnswer,
)
from job_agent.models.job import ApplicationAvailability, ApplicationMethod, JobPosting
from job_agent.models.matching import MatchDecision, Recommendation
from job_agent.models.resume import JobSearchPreferences, ResumeProfile
from job_agent.observability import metrics
from job_agent.observability.logging import get_logger
from job_agent.persistence.orm import Application, ResumeVersion, SubmissionAttempt
from job_agent.services.answers import AnswerResolver, build_answer_values
from job_agent.services.application_service import (
    ApplicationPackage,
    ApplicationService,
    SubmissionInProgress,
    SubmissionIntegration,
    SubmissionOutcome,
    SubmissionResult,
)
from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.state import JobRoute, JobState

log = get_logger(__name__)

STOPPED_BEFORE_APPLICATION = "workflow stopped or paused before application; not submitted"
STOPPED_BEFORE_SUBMISSION = "workflow stopped or paused before submission; not submitted"


class ReviewKind:
    """Why an application is waiting for the user. RETRYABLE kinds may be retried once
    the cause is fixed; the others need a human decision on the job itself."""

    STOPPED = "stopped"
    RUN_CAP = "run_cap"
    MISSING_ANSWERS = "missing_answers"
    ANSWERS_WITHDRAWN = "answers_withdrawn"
    INTEGRATION_ERROR = "integration_error"
    ELIGIBILITY = "eligibility"
    MATCH_REVIEW = "match_review"
    RECONCILIATION = "reconciliation"

    RETRYABLE = frozenset({STOPPED, RUN_CAP, MISSING_ANSWERS, ANSWERS_WITHDRAWN, INTEGRATION_ERROR})


def manual_package(
    app: Application,
    posting: JobPosting,
    version: ResumeVersion | None,
    decision: dict[str, Any] | None,
    approved_keys: list[str],
    reason: str,
) -> dict[str, Any]:
    """What the user needs to finish the application by hand (no sensitive values)."""
    decision = decision or {}
    return {
        "apply_url": str(posting.application_url or posting.canonical_url),
        "posting_url": str(posting.canonical_url),
        "source": posting.source.value,
        "application_method": posting.application_method.value,
        "resume_version": app.resume_version,
        "resume_file": version.original_filename if version else None,
        "match_score": decision.get("overall_score"),
        "matching_skills": decision.get("matching_skills", []),
        "resume_evidence": decision.get("resume_evidence", []),
        "approved_answers_available": sorted(approved_keys),
        "reason": reason,
    }


class ApplicationNodes:
    def __init__(self, deps: WorkflowDeps) -> None:
        self.d = deps
        self.resolver = AnswerResolver(
            deps.answer_generator if deps.settings.generate_application_answers else None
        )

    # ---------------------------------------------------------------- helpers

    def _end(
        self,
        state: JobState,
        status: ApplicationStatus,
        reason: str,
        details: dict[str, Any] | None = None,
        *,
        manual: bool = False,
    ) -> dict[str, Any]:
        with self.d.session() as s:
            svc = ApplicationService(s)
            app = svc.get(uuid.UUID(state["application_id"]))
            assert app is not None
            if app.status is not status:
                svc.transition(app, status, reason=reason, details=details)
                metrics.application_outcomes.add(1, {"status": status.value})
            if manual:
                posting = JobPosting.model_validate(state["job"]["posting"])
                approved = svc.all_approved_answers(uuid.UUID(state["user_id"]))
                svc.merge_evidence(
                    app,
                    {
                        "manual_package": manual_package(
                            app,
                            posting,
                            self._version(s, state),
                            state.get("decision"),
                            list(approved),
                            reason,
                        )
                    },
                )
        return {"route": JobRoute.HALT.value, "reason": reason, "final_status": status.value}

    def _review(
        self, state: JobState, reason: str, kind: str, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        metrics.review_routed.add(1, {"kind": kind})
        return self._end(
            state,
            ApplicationStatus.PENDING_REVIEW,
            reason,
            {"review_kind": kind, **(extra or {})},
            manual=True,
        )

    def _integration(self, state: JobState) -> SubmissionIntegration | None:
        return self.d.submissions.for_posting(JobPosting.model_validate(state["job"]["posting"]))

    @staticmethod
    def _version(s: Session, state: JobState) -> ResumeVersion | None:
        return s.scalar(
            sa.select(ResumeVersion).where(
                ResumeVersion.resume_id == uuid.UUID(state["resume_id"]),
                ResumeVersion.version == state["resume_version"],
            )
        )

    # ------------------------------------------------------------------ nodes

    def verify_resume_active(self, state: JobState) -> dict[str, Any]:
        if not self.d.workflow_active(state["resume_id"]):
            return self._review(state, STOPPED_BEFORE_APPLICATION, ReviewKind.STOPPED)
        return {"route": JobRoute.CONTINUE.value}

    def confirm_eligibility(self, state: JobState) -> dict[str, Any]:
        d = MatchDecision.model_validate(state["decision"])
        ok = (
            d.recommendation is Recommendation.APPLY
            and d.eligible_for_auto_apply
            and not d.hard_constraint_violations
            and d.overall_score >= self.d.settings.job_match_threshold
        )
        if not ok:
            return self._review(
                state, "eligibility re-check failed before application", ReviewKind.ELIGIBILITY
            )
        return {"route": JobRoute.CONTINUE.value}

    def validate_application_method(self, state: JobState) -> dict[str, Any]:
        posting = JobPosting.model_validate(state["job"]["posting"])
        manual_url = str(posting.application_url or posting.canonical_url)
        details = {
            "manual_url": manual_url,
            "application_method": posting.application_method.value,
            "source": posting.source.value,
        }
        if posting.application_availability is ApplicationAvailability.CLOSED:
            return self._end(
                state,
                ApplicationStatus.APPLICATION_UNAVAILABLE,
                "posting is closed to applications",
                details,
            )
        if posting.application_url is None and posting.application_method is not (
            ApplicationMethod.API
        ):
            return self._end(
                state,
                ApplicationStatus.APPLICATION_UNAVAILABLE,
                f"posting has no application URL; complete it manually at {manual_url}",
                details,
                manual=True,
            )
        integration = self._integration(state)
        if integration is None:
            return self._end(
                state,
                ApplicationStatus.APPLICATION_UNAVAILABLE,
                f"no permitted automated submission for {posting.application_method.value} "
                f"applications on {posting.source.value}; complete it manually at {manual_url}",
                details,
                manual=True,
            )
        return {"route": JobRoute.CONTINUE.value, "integration": integration.name}

    async def prepare_application(self, state: JobState) -> dict[str, Any]:
        integration = self._integration(state)
        assert integration is not None
        posting = JobPosting.model_validate(state["job"]["posting"])
        try:
            questions = await integration.questions(posting)
        except Exception as exc:
            return self._review(
                state,
                f"could not load application questions ({type(exc).__name__})",
                ReviewKind.INTEGRATION_ERROR,
            )
        return {
            "route": JobRoute.CONTINUE.value,
            "questions": [q.model_dump(mode="json") for q in questions],
        }

    async def validate_application_answers(self, state: JobState) -> dict[str, Any]:
        questions = [ApplicationQuestion.model_validate(q) for q in state.get("questions", [])]
        with self.d.session() as s:
            approved = ApplicationService(s).all_approved_answers(uuid.UUID(state["user_id"]))
        resolution = await self.resolver.resolve(
            questions,
            approved=approved,
            profile=ResumeProfile.model_validate(state["profile"]),
            posting=JobPosting.model_validate(state["job"]["posting"]),
        )
        evidence = {
            "integration": state.get("integration"),
            "questions": [q.key for q in questions],
            "answers": [a.model_dump(mode="json") for a in resolution.answers],
        }
        with self.d.session() as s:
            svc = ApplicationService(s)
            app = svc.get(uuid.UUID(state["application_id"]))
            assert app is not None
            svc.merge_evidence(app, evidence)
        if not resolution.ok:
            parts = []
            if resolution.missing:
                parts.append("needs approved answers for: " + ", ".join(resolution.missing))
            if resolution.problems:
                parts.append("; ".join(resolution.problems))
            return self._review(
                state,
                " | ".join(parts),
                ReviewKind.MISSING_ANSWERS,
                {"missing_questions": resolution.missing, "problems": resolution.problems},
            )
        with self.d.session() as s:
            svc = ApplicationService(s)
            app = svc.get(uuid.UUID(state["application_id"]))
            assert app is not None
            svc.transition(app, ApplicationStatus.READY_TO_APPLY, reason="all answers verified")
        return {"route": JobRoute.CONTINUE.value}

    async def submit_application(self, state: JobState) -> dict[str, Any]:
        app_id = uuid.UUID(state["application_id"])
        now = self.d.clock()
        with self.d.session() as s:
            svc = ApplicationService(s)
            app = svc.get(app_id)
            assert app is not None
            open_attempt = svc.open_attempt(app_id)
            if open_attempt is not None:
                # A previous run got as far as submitting and then died. Never resubmit.
                detail = "previous submission attempt has an unknown outcome; not resubmitted"
                svc.mark_uncertain(app, open_attempt, now, detail)
                return {
                    "route": JobRoute.HALT.value,
                    "reason": detail,
                    "final_status": ApplicationStatus.NEEDS_RECONCILIATION.value,
                    "submission": {"attempt_id": str(open_attempt.id)},
                }
            if app.status is not ApplicationStatus.READY_TO_APPLY:
                return {
                    "route": JobRoute.HALT.value,
                    "final_status": app.status.value,
                    "reason": f"application is {app.status.value}; not submitted",
                }

        integration = self._integration(state)
        assert integration is not None
        prefs = JobSearchPreferences.model_validate(state["preferences"])
        limits = self.d.limits(state["execution_id"])
        async with limits.submit_lock:
            # Explicit stop check immediately before the external side effect.
            if not self.d.workflow_active(state["resume_id"]):
                return self._review(state, STOPPED_BEFORE_SUBMISSION, ReviewKind.STOPPED)
            with self.d.session() as s:
                svc = ApplicationService(s)
                started = datetime.fromisoformat(state["execution_started_at"])
                if svc.attempts_started_since(uuid.UUID(state["resume_id"]), started) >= (
                    prefs.max_applications_per_run
                ):
                    reason = f"per-run application limit ({prefs.max_applications_per_run}) reached"
                    return self._review(state, reason, ReviewKind.RUN_CAP)
                app = svc.get(app_id)
                assert app is not None
                package = self._package(s, state, app)
                if package is None:
                    return self._review(
                        state,
                        "an approved answer was withdrawn before submission",
                        ReviewKind.ANSWERS_WITHDRAWN,
                    )
                try:
                    attempt = svc.start_attempt(app, now)
                except SubmissionInProgress:
                    return {
                        "route": JobRoute.HALT.value,
                        "final_status": ApplicationStatus.NEEDS_RECONCILIATION.value,
                        "reason": "another submission attempt is open",
                    }
                attempt_id, idem_key = attempt.id, attempt.idempotency_key
                metrics.submissions_started.add(1, {"integration": integration.name})
            # Session committed here: the attempt is durable before the platform is called.

        log.info("application_submitting", application_id=str(app_id), integration=integration.name)
        try:
            result = await asyncio.wait_for(
                integration.submit(package, idempotency_key=idem_key),
                timeout=self.d.submit_timeout_seconds,
            )
        except TimeoutError:
            result = SubmissionResult(SubmissionOutcome.UNCERTAIN, detail="submission timed out")
        except Exception as exc:  # outcome unknown: the request may have reached the platform
            result = SubmissionResult(
                SubmissionOutcome.UNCERTAIN, detail=f"submission error: {type(exc).__name__}"
            )
        return {
            "route": JobRoute.CONTINUE.value,
            "submission": {
                "attempt_id": str(attempt_id),
                "outcome": result.outcome.value,
                "external_ref": result.external_ref,
                "detail": result.detail,
                "retryable": result.retryable,
                "verification": result.verification,
            },
        }

    def verify_submission(self, state: JobState) -> dict[str, Any]:
        sub = dict(state.get("submission") or {})
        if sub.get("outcome") == SubmissionOutcome.CONFIRMED.value and not sub.get("external_ref"):
            sub["outcome"] = SubmissionOutcome.UNCERTAIN.value
            sub["detail"] = "platform reported success without a confirmation reference"
        return {"submission": sub}

    def persist_application_result(self, state: JobState) -> dict[str, Any]:
        sub = state["submission"] or {}
        with self.d.session() as s:
            svc = ApplicationService(s)
            app = svc.get(uuid.UUID(state["application_id"]))
            attempt = s.get(SubmissionAttempt, uuid.UUID(sub["attempt_id"]))
            assert app is not None and attempt is not None
            status = svc.record_result(
                app,
                attempt,
                SubmissionResult(
                    SubmissionOutcome(sub["outcome"]),
                    sub.get("external_ref"),
                    sub.get("detail"),
                    retryable=bool(sub.get("retryable")),
                    verification=sub.get("verification"),
                ),
                self.d.clock(),
            )
        metrics.application_outcomes.add(1, {"status": status.value})
        log.info("application_result", application_id=state["application_id"], status=status.value)
        return {"final_status": status.value, "reason": sub.get("detail") or status.value}

    def _package(self, s: Session, state: JobState, app: Application) -> ApplicationPackage | None:
        version = self._version(s, state)
        assert version is not None
        resolved = [
            ResolvedAnswer.model_validate(a)
            for a in (app.submission_evidence or {}).get("answers", [])
        ]
        approved = ApplicationService(s).all_approved_answers(uuid.UUID(state["user_id"]))
        try:
            questions = [ApplicationQuestion.model_validate(q) for q in state.get("questions", [])]
            values = build_answer_values(resolved, approved, questions)
        except KeyError:
            return None
        if not any(a.source is AnswerSource.RESUME_FILE for a in resolved):
            log.info("application_without_resume_field", application_id=str(app.id))
        return ApplicationPackage(
            application_id=app.id,
            resume_id=uuid.UUID(state["resume_id"]),
            resume_version=state["resume_version"],
            resume_file_ref=version.storage_ref,
            posting=JobPosting.model_validate(state["job"]["posting"]),
            answers=values,
            resume_filename=version.original_filename,
        )
