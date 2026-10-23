"""Post-run application actions: reconciliation of uncertain submissions and safe retry.

Reconciliation (automatic, and on demand):
* SUBMITTING with a STARTED attempt older than the submit timeout -> treated as
  uncertain (the process died mid-call);
* NEEDS_RECONCILIATION -> ask the integration whether the application exists:
  FOUND (with reference) -> APPLIED; NOT_FOUND (and the attempt is old enough for the
  platform to have indexed it) -> FAILED, retryable; UNKNOWN -> try again later, and
  after ``RECONCILE_MAX_ATTEMPTS`` -> PENDING_REVIEW for the user to check by hand.
  Never resubmits.

Retry (user-initiated, ownership-checked):
* only FAILED with a retryable rejection, or PENDING_REVIEW whose cause is fixable
  (stopped, per-run cap, missing/withdrawn answers, question-loading error);
* never for uncertain outcomes (reconcile first), APPLIED, SKIPPED, DUPLICATE or
  match-driven reviews; at most ``MAX_APPLICATION_RETRIES`` times;
* re-runs the *same* application subgraph on the *same* row, so every stop check,
  eligibility re-check, answer check, duplicate guard and durable-attempt rule applies.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session

from job_agent.models.application import ApplicationStatus, SubmissionAttemptStatus
from job_agent.models.job import JobPosting
from job_agent.observability import metrics
from job_agent.observability.logging import get_logger
from job_agent.persistence.orm import (
    Application,
    JobMatch,
    JobPostingRecord,
    Resume,
    ResumeVersion,
    SubmissionAttempt,
)
from job_agent.services.application_service import (
    ApplicationService,
    LookupOutcome,
    LookupResult,
)
from job_agent.services.errors import Conflict, NotFound
from job_agent.workflows.application_graph import build_application_graph
from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.nodes.application_nodes import ReviewKind
from job_agent.workflows.state import JobState

log = get_logger(__name__)

UNCERTAIN = (ApplicationStatus.NEEDS_RECONCILIATION, ApplicationStatus.SUBMITTING)


@dataclass
class ActionResult:
    application_id: uuid.UUID
    status: ApplicationStatus
    detail: str


class ApplicationActions:
    def __init__(self, deps: WorkflowDeps) -> None:
        self.d = deps
        self._graph = build_application_graph(deps)

    # ------------------------------------------------------------------ helpers

    def _load(
        self, s: Session, application_id: uuid.UUID, user_id: uuid.UUID | None
    ) -> Application:
        stmt = sa.select(Application).where(Application.id == application_id)
        if user_id is not None:
            stmt = stmt.where(Application.user_id == user_id)
        app = s.scalar(stmt.with_for_update())
        if app is None:
            raise NotFound("application not found")
        return app

    @staticmethod
    def _latest_attempt(s: Session, app: Application) -> SubmissionAttempt | None:
        return s.scalar(
            sa.select(SubmissionAttempt)
            .where(SubmissionAttempt.application_id == app.id)
            .order_by(SubmissionAttempt.attempt_number.desc())
            .limit(1)
        )

    @staticmethod
    def _posting(s: Session, app: Application) -> JobPosting:
        record = s.get(JobPostingRecord, app.job_posting_id)
        assert record is not None
        return JobPosting.model_validate(record.normalized)

    # ------------------------------------------------------------------ retry

    async def retry(self, user_id: uuid.UUID, application_id: uuid.UUID) -> ActionResult:
        with self.d.session() as s:
            app = self._load(s, application_id, user_id)
            self._check_retryable(s, app)
            state = self._retry_state(s, app)
            svc = ApplicationService(s)
            app.retry_count += 1
            metrics.application_retries.add(1)
            svc.transition(
                app, ApplicationStatus.READY_TO_APPLY, reason=f"manual retry {app.retry_count}"
            )
        log.info("application_retry", application_id=str(application_id))
        try:
            await self._graph.ainvoke(state)
        finally:
            self.d.release(state["execution_id"])
        with self.d.session() as s:
            app = self._load(s, application_id, user_id)
            return ActionResult(app.id, app.status, app.status_reason or app.status.value)

    def _check_retryable(self, s: Session, app: Application) -> None:
        if app.status in UNCERTAIN:
            raise Conflict("the last submission's outcome is unknown; reconcile it first")
        if app.status is ApplicationStatus.READY_TO_APPLY:
            raise Conflict("a retry is already in progress")
        if app.retry_count >= self.d.settings.max_application_retries:
            raise Conflict(f"retry limit ({self.d.settings.max_application_retries}) reached")
        if ApplicationService(s).open_attempt(app.id) is not None:
            raise Conflict("a submission attempt is still open")
        if app.status is ApplicationStatus.FAILED:
            attempt = self._latest_attempt(s, app)
            summary = (attempt.response_summary or {}) if attempt else {}
            if attempt is None or not summary.get("retryable"):
                raise Conflict("the platform rejected this application; it is not retryable")
        elif app.status is ApplicationStatus.PENDING_REVIEW:
            kind = (app.error_details or {}).get("review_kind")
            if kind not in ReviewKind.RETRYABLE:
                raise Conflict("this application needs your review of the job, not a retry")
        else:
            raise Conflict(f"an application that is {app.status.value} cannot be retried")
        if not self.d.workflow_active(str(app.resume_id)):
            raise Conflict("resume the workflow before retrying")

    def _retry_state(self, s: Session, app: Application) -> JobState:
        posting = self._posting(s, app)
        match = s.get(JobMatch, app.match_id) if app.match_id else None
        if match is None:
            raise Conflict("no match evaluation is recorded for this application")
        resume = s.get(Resume, app.resume_id)
        version = s.scalar(
            sa.select(ResumeVersion).where(
                ResumeVersion.resume_id == app.resume_id,
                ResumeVersion.version == app.resume_version,
            )
        )
        assert resume is not None and version is not None
        record = s.get(JobPostingRecord, app.job_posting_id)
        assert record is not None
        now = self.d.clock()
        return {
            "execution_id": f"retry-{app.id}-{app.retry_count + 1}",
            "execution_started_at": now.isoformat(),
            "user_id": str(app.user_id),
            "resume_id": str(app.resume_id),
            "resume_version": app.resume_version,
            "profile": version.profile,
            "preferences": resume.preferences,
            "decision": match.decision,
            "match_id": str(match.id),
            "application_id": str(app.id),
            "job": {
                "posting": posting.model_dump(mode="json"),
                "job_posting_id": str(app.job_posting_id),
                "canonical_key": record.canonical_key,
                "duplicates": [],
                "member_keys": [record.canonical_key],
            },
        }

    # ------------------------------------------------------------ reconciliation

    async def reconcile(
        self, application_id: uuid.UUID, *, user_id: uuid.UUID | None = None
    ) -> ActionResult:
        now = self.d.clock()
        with self.d.session() as s:
            app = self._load(s, application_id, user_id)
            if app.status not in UNCERTAIN:
                return ActionResult(app.id, app.status, "nothing to reconcile")
            svc = ApplicationService(s)
            attempt = self._latest_attempt(s, app)
            if attempt is None:
                svc.transition(
                    app,
                    ApplicationStatus.NEEDS_RECONCILIATION
                    if app.status is ApplicationStatus.SUBMITTING
                    else app.status,
                )
                return ActionResult(app.id, app.status, "no submission attempt recorded")
            if app.status is ApplicationStatus.SUBMITTING:
                in_flight_until = attempt.started_at + timedelta(
                    seconds=self.d.submit_timeout_seconds * 2
                )
                if now < in_flight_until:
                    return ActionResult(app.id, app.status, "submission still in progress")
                svc.mark_uncertain(app, attempt, now, "submission interrupted; outcome unknown")
            posting = self._posting(s, app)
            attempt_id, idem_key, started = attempt.id, attempt.idempotency_key, attempt.started_at

        integration = self.d.submissions.for_posting(posting)
        if integration is None:
            result = LookupResult(
                LookupOutcome.UNKNOWN, detail="no integration can look this application up"
            )
        else:
            try:
                result = await asyncio.wait_for(
                    integration.lookup(posting, idempotency_key=idem_key, submitted_at=started),
                    timeout=self.d.submit_timeout_seconds,
                )
            except Exception as exc:
                result = LookupResult(
                    LookupOutcome.UNKNOWN, detail=f"lookup failed: {type(exc).__name__}"
                )
        return self._apply_lookup(application_id, attempt_id, started, result, now)

    def _apply_lookup(
        self,
        application_id: uuid.UUID,
        attempt_id: uuid.UUID,
        started: datetime,
        result: LookupResult,
        now: datetime,
    ) -> ActionResult:
        s_ = self.d.settings
        with self.d.session() as s:
            svc = ApplicationService(s)
            app = self._load(s, application_id, None)
            attempt = s.get(SubmissionAttempt, attempt_id)
            assert attempt is not None
            app.reconcile_attempts += 1
            app.last_reconciled_at = now
            if result.outcome is LookupOutcome.FOUND and result.external_ref:
                attempt.status = SubmissionAttemptStatus.CONFIRMED
                attempt.external_ref = result.external_ref
                attempt.finished_at = attempt.finished_at or now
                svc.transition(app, ApplicationStatus.APPLIED, reason="confirmed by reconciliation")
                app.submitted_at = app.submitted_at or started
                app.external_confirmation_ref = result.external_ref
                svc.merge_evidence(
                    app,
                    {
                        "confirmation_ref": result.external_ref,
                        "verification": "reconciliation_lookup",
                    },
                )
            elif result.outcome is LookupOutcome.NOT_FOUND and now - started >= timedelta(
                seconds=s_.reconcile_not_found_min_age_seconds
            ):
                attempt.status = SubmissionAttemptStatus.REJECTED
                attempt.response_summary = {
                    **(attempt.response_summary or {}),
                    "retryable": True,
                    "reconciled": "not_found",
                }
                svc.transition(
                    app,
                    ApplicationStatus.FAILED,
                    reason="platform has no record of the submission; safe to retry",
                )
            elif app.reconcile_attempts >= s_.reconcile_max_attempts:
                svc.transition(
                    app,
                    ApplicationStatus.PENDING_REVIEW,
                    reason="could not verify the submission automatically; check the "
                    "employer's site before applying again",
                    details={
                        "review_kind": ReviewKind.RECONCILIATION,
                        "attempt_id": str(attempt_id),
                        "detail": result.detail,
                    },
                )
            metrics.reconciliations.add(1, {"outcome": result.outcome.value})
            log.info(
                "application_reconciled",
                application_id=str(application_id),
                outcome=result.outcome.value,
                status=app.status.value,
                attempts=app.reconcile_attempts,
            )
            return ActionResult(app.id, app.status, result.detail or result.outcome.value)

    async def reconcile_due(self) -> dict[str, int]:
        """Scheduler entry point: reconcile everything that is due."""
        now = self.d.clock()
        interval = timedelta(seconds=self.d.settings.reconcile_interval_seconds)
        stale_submit = now - timedelta(seconds=self.d.submit_timeout_seconds * 2)
        with self.d.session() as s:
            ids = list(
                s.scalars(
                    sa.select(Application.id).where(
                        sa.or_(
                            sa.and_(
                                Application.status == ApplicationStatus.NEEDS_RECONCILIATION,
                                Application.reconcile_attempts
                                < self.d.settings.reconcile_max_attempts,
                                sa.or_(
                                    Application.last_reconciled_at.is_(None),
                                    Application.last_reconciled_at <= now - interval,
                                ),
                            ),
                            sa.and_(
                                Application.status == ApplicationStatus.SUBMITTING,
                                Application.updated_at <= stale_submit,
                            ),
                        )
                    )
                )
            )
        counts: dict[str, int] = {}
        for app_id in ids:
            try:
                result = await self.reconcile(app_id)
            except Exception as exc:  # one bad record must not stop the sweep
                log.error("reconcile_failed", application_id=str(app_id), error=type(exc).__name__)
                counts["error"] = counts.get("error", 0) + 1
                continue
            counts[result.status.value] = counts.get(result.status.value, 0) + 1
        return counts
