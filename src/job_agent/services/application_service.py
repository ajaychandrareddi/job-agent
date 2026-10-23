"""Application records, the submission state machine and submission integrations.

A job-site submission and a database commit can never be atomic. The protocol:

1. commit a ``SubmissionAttempt`` row (STARTED) and set the application to SUBMITTING
   *before* calling the platform; a partial unique index allows only one open
   (STARTED/UNCERTAIN) attempt per application;
2. call the platform with a stable idempotency key;
3. record the outcome: CONFIRMED with a platform reference -> APPLIED;
   definitive rejection -> FAILED; anything else (timeout, crash, ambiguous reply,
   "success" without a reference) -> UNCERTAIN / NEEDS_RECONCILIATION.

If the process dies between 1 and 3, the open attempt is found on restart and the
application goes to NEEDS_RECONCILIATION instead of being submitted again.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from job_agent.models.application import (
    ApplicationQuestion,
    ApplicationStatus,
    SubmissionAttemptStatus,
    assert_transition,
)
from job_agent.models.job import JobPosting
from job_agent.persistence.orm import Application, ApprovedAnswer, SubmissionAttempt
from job_agent.services.deduplication import application_key

# --------------------------------------------------------------------------- integrations


class SubmissionOutcome(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class SubmissionResult:
    outcome: SubmissionOutcome
    external_ref: str | None = None
    detail: str | None = None
    # For REJECTED: the platform definitively did not accept the application and a
    # later attempt may succeed (e.g. rate limited). Final rejections are not retryable.
    retryable: bool = False
    # How success was verified (stored as evidence), e.g. "http_200_success_body".
    verification: str | None = None


class LookupOutcome(StrEnum):
    FOUND = "found"  # platform confirms the application exists
    NOT_FOUND = "not_found"  # platform definitively has no such application
    UNKNOWN = "unknown"  # platform cannot tell us (or lookup unsupported)


@dataclass(frozen=True)
class LookupResult:
    outcome: LookupOutcome
    external_ref: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ApplicationPackage:
    application_id: uuid.UUID
    resume_id: uuid.UUID
    resume_version: int
    resume_file_ref: str
    posting: JobPosting
    # field_name -> value, built from approved answers, resume facts and validated
    # generated text. Never contains invented values.
    answers: dict[str, str] = field(default_factory=dict)
    resume_filename: str = "resume.pdf"


class SubmissionIntegration(Protocol):
    """A *permitted* programmatic application channel (e.g. an ATS candidate API).

    Implementations must never automate a platform against its terms, solve CAPTCHAs
    or reuse browser sessions. Anything needing those stays a manual task.
    """

    name: str

    def supports(self, posting: JobPosting) -> bool: ...

    async def questions(self, posting: JobPosting) -> list[ApplicationQuestion]: ...

    async def submit(
        self, package: ApplicationPackage, *, idempotency_key: str
    ) -> SubmissionResult: ...

    async def lookup(
        self, posting: JobPosting, *, idempotency_key: str, submitted_at: datetime
    ) -> LookupResult:
        """Reconciliation: did an earlier, uncertain submission reach the platform?"""
        ...


class SubmissionRegistry:
    def __init__(self, integrations: Sequence[SubmissionIntegration] = ()) -> None:
        self._integrations = list(integrations)

    def for_posting(self, posting: JobPosting) -> SubmissionIntegration | None:
        return next((i for i in self._integrations if i.supports(posting)), None)


# --------------------------------------------------------------------------- service


class SubmissionInProgress(RuntimeError):
    """An open (STARTED/UNCERTAIN) attempt already exists for this application."""


class ApplicationService:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get(self, application_id: uuid.UUID) -> Application | None:
        return self.s.get(Application, application_id)

    def get_or_create(
        self,
        *,
        user_id: uuid.UUID,
        resume_id: uuid.UUID,
        resume_version: int,
        job_posting_id: uuid.UUID,
        canonical_key: str,
        match_id: uuid.UUID | None,
        execution_id: uuid.UUID | None = None,
        purge_after: datetime | None = None,
    ) -> tuple[Application, bool]:
        """Return (application, created). Uniqueness is enforced by the database."""
        key = application_key(user_id, resume_id, canonical_key)
        existing = self._find(key, resume_id, job_posting_id)
        if existing is not None:
            return existing, False
        app = Application(
            application_key=key,
            user_id=user_id,
            resume_id=resume_id,
            resume_version=resume_version,
            job_posting_id=job_posting_id,
            match_id=match_id,
            execution_id=execution_id,
            purge_after=purge_after,
            status=ApplicationStatus.DISCOVERED,
        )
        try:
            with self.s.begin_nested():
                self.s.add(app)
            return app, True
        except IntegrityError:  # created concurrently
            found = self._find(key, resume_id, job_posting_id)
            assert found is not None
            return found, False

    def _find(
        self, key: str, resume_id: uuid.UUID, job_posting_id: uuid.UUID
    ) -> Application | None:
        return self.s.scalar(
            sa.select(Application)
            .where(
                sa.or_(
                    Application.application_key == key,
                    sa.and_(
                        Application.resume_id == resume_id,
                        Application.job_posting_id == job_posting_id,
                    ),
                )
            )
            .limit(1)
        )

    def transition(
        self,
        app: Application,
        status: ApplicationStatus,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        assert_transition(app.status, status)
        app.status = status
        if reason is not None:
            app.status_reason = reason[:500]
        if details is not None:
            app.error_details = details
        self.s.flush()

    # ------------------------------------------------------------- attempts

    def open_attempt(self, application_id: uuid.UUID) -> SubmissionAttempt | None:
        return self.s.scalar(
            sa.select(SubmissionAttempt).where(
                SubmissionAttempt.application_id == application_id,
                SubmissionAttempt.status.in_(
                    [SubmissionAttemptStatus.STARTED, SubmissionAttemptStatus.UNCERTAIN]
                ),
            )
        )

    def start_attempt(self, app: Application, now: datetime) -> SubmissionAttempt:
        """Durably record the intent to submit. Caller must commit before submitting."""
        if self.open_attempt(app.id) is not None:
            raise SubmissionInProgress(str(app.id))
        number = (
            self.s.scalar(
                sa.select(sa.func.max(SubmissionAttempt.attempt_number)).where(
                    SubmissionAttempt.application_id == app.id
                )
            )
            or 0
        ) + 1
        attempt = SubmissionAttempt(
            application_id=app.id,
            attempt_number=number,
            idempotency_key=f"{app.id}:{number}",
            status=SubmissionAttemptStatus.STARTED,
            started_at=now,
        )
        try:
            with self.s.begin_nested():
                self.s.add(attempt)
        except IntegrityError as exc:
            raise SubmissionInProgress(str(app.id)) from exc
        self.transition(app, ApplicationStatus.SUBMITTING)
        app.current_attempt_id = attempt.id
        self.s.flush()
        return attempt

    def mark_uncertain(
        self, app: Application, attempt: SubmissionAttempt, now: datetime, detail: str
    ) -> None:
        attempt.status = SubmissionAttemptStatus.UNCERTAIN
        attempt.finished_at = attempt.finished_at or now
        attempt.error = detail[:2000]
        if app.status is not ApplicationStatus.NEEDS_RECONCILIATION:
            self.transition(
                app,
                ApplicationStatus.NEEDS_RECONCILIATION,
                reason=detail,
                details={"attempt_id": str(attempt.id), "detail": detail},
            )

    def record_result(
        self, app: Application, attempt: SubmissionAttempt, result: SubmissionResult, now: datetime
    ) -> ApplicationStatus:
        attempt.finished_at = now
        attempt.response_summary = {
            "outcome": result.outcome.value,
            "detail": result.detail,
            "retryable": result.retryable,
            "verification": result.verification,
        }
        if result.outcome is SubmissionOutcome.CONFIRMED and result.external_ref:
            attempt.status = SubmissionAttemptStatus.CONFIRMED
            attempt.external_ref = result.external_ref
            self.transition(app, ApplicationStatus.APPLIED, reason="submission confirmed")
            app.submitted_at = now
            app.external_confirmation_ref = result.external_ref
            self.merge_evidence(
                app,
                {
                    "confirmation_ref": result.external_ref,
                    "verification": result.verification or "platform_reference",
                },
            )
        elif result.outcome is SubmissionOutcome.REJECTED:
            attempt.status = SubmissionAttemptStatus.REJECTED
            attempt.error = result.detail
            self.transition(
                app,
                ApplicationStatus.FAILED,
                reason=result.detail or "rejected",
                details={"attempt_id": str(attempt.id), "detail": result.detail},
            )
        else:
            detail = result.detail or (
                "platform reported success without a confirmation reference"
                if result.outcome is SubmissionOutcome.CONFIRMED
                else "outcome unknown"
            )
            self.mark_uncertain(app, attempt, now, detail)
        self.s.flush()
        return app.status

    def merge_evidence(self, app: Application, data: dict[str, Any]) -> None:
        app.submission_evidence = {**(app.submission_evidence or {}), **data}
        self.s.flush()

    def all_approved_answers(self, user_id: uuid.UUID) -> dict[str, str]:
        rows = self.s.scalars(sa.select(ApprovedAnswer).where(ApprovedAnswer.user_id == user_id))
        return {r.question_key: r.answer for r in rows}

    def attempts_started_since(self, resume_id: uuid.UUID, since: datetime) -> int:
        return (
            self.s.scalar(
                sa.select(sa.func.count(SubmissionAttempt.id))
                .join(Application, Application.id == SubmissionAttempt.application_id)
                .where(Application.resume_id == resume_id, SubmissionAttempt.started_at >= since)
            )
            or 0
        )

    # -------------------------------------------------------------- answers

    def approved_answers(self, user_id: uuid.UUID, keys: Sequence[str]) -> dict[str, str]:
        if not keys:
            return {}
        rows = self.s.scalars(
            sa.select(ApprovedAnswer).where(
                ApprovedAnswer.user_id == user_id, ApprovedAnswer.question_key.in_(list(keys))
            )
        )
        return {r.question_key: r.answer for r in rows}
