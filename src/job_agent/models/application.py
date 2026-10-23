"""Application status machine and related models."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from job_agent.models.common import AwareDatetime


class ApplicationStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    MATCHED = "MATCHED"
    SKIPPED = "SKIPPED"
    PENDING_REVIEW = "PENDING_REVIEW"
    READY_TO_APPLY = "READY_TO_APPLY"
    SUBMITTING = "SUBMITTING"
    APPLIED = "APPLIED"
    FAILED = "FAILED"
    DUPLICATE = "DUPLICATE"
    APPLICATION_UNAVAILABLE = "APPLICATION_UNAVAILABLE"
    # Submission was sent but the outcome could not be verified. Never auto-resubmitted.
    NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"


class SubmissionAttemptStatus(StrEnum):
    STARTED = "STARTED"  # durable intent recorded *before* the external call
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"  # platform definitively rejected; safe to retry if retryable
    UNCERTAIN = "UNCERTAIN"  # timeout / crash: must be reconciled before any retry


# Allowed transitions. Anything else is a programming error.
APPLICATION_TRANSITIONS: dict[ApplicationStatus, frozenset[ApplicationStatus]] = {
    ApplicationStatus.DISCOVERED: frozenset(
        {
            ApplicationStatus.MATCHED,
            ApplicationStatus.SKIPPED,
            ApplicationStatus.PENDING_REVIEW,
            ApplicationStatus.DUPLICATE,
            ApplicationStatus.APPLICATION_UNAVAILABLE,
        }
    ),
    ApplicationStatus.MATCHED: frozenset(
        {
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.PENDING_REVIEW,
            ApplicationStatus.SKIPPED,
            ApplicationStatus.DUPLICATE,
            ApplicationStatus.APPLICATION_UNAVAILABLE,
        }
    ),
    ApplicationStatus.PENDING_REVIEW: frozenset(
        {
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.SKIPPED,
            ApplicationStatus.APPLICATION_UNAVAILABLE,
        }
    ),
    ApplicationStatus.READY_TO_APPLY: frozenset(
        {
            ApplicationStatus.SUBMITTING,
            ApplicationStatus.PENDING_REVIEW,
            ApplicationStatus.SKIPPED,
            ApplicationStatus.APPLICATION_UNAVAILABLE,
        }
    ),
    ApplicationStatus.SUBMITTING: frozenset(
        {
            ApplicationStatus.APPLIED,
            ApplicationStatus.FAILED,
            ApplicationStatus.NEEDS_RECONCILIATION,
        }
    ),
    ApplicationStatus.NEEDS_RECONCILIATION: frozenset(
        {
            ApplicationStatus.APPLIED,
            ApplicationStatus.FAILED,
            ApplicationStatus.PENDING_REVIEW,
        }
    ),
    ApplicationStatus.FAILED: frozenset(
        {
            ApplicationStatus.READY_TO_APPLY,
            ApplicationStatus.PENDING_REVIEW,
        }
    ),
    ApplicationStatus.SKIPPED: frozenset({ApplicationStatus.PENDING_REVIEW}),
    ApplicationStatus.APPLICATION_UNAVAILABLE: frozenset({ApplicationStatus.PENDING_REVIEW}),
    ApplicationStatus.APPLIED: frozenset(),
    ApplicationStatus.DUPLICATE: frozenset(),
}


class InvalidStatusTransition(ValueError):
    pass


def assert_transition(current: ApplicationStatus, new: ApplicationStatus) -> None:
    if new == current:
        return
    if new not in APPLICATION_TRANSITIONS[current]:
        raise InvalidStatusTransition(f"illegal application transition {current} -> {new}")


class ApplicationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    resume_id: UUID
    resume_version: int
    job_posting_id: UUID
    status: ApplicationStatus
    external_confirmation_ref: str | None
    submitted_at: AwareDatetime | None
    last_reconciled_at: AwareDatetime | None
    error_details: dict[str, object] | None
    created_at: AwareDatetime
    updated_at: AwareDatetime


# --------------------------------------------------------------------------- questions


class QuestionKind(StrEnum):
    TEXT = "text"
    LONG_TEXT = "long_text"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    BOOLEAN = "boolean"
    FILE = "file"


class QuestionOption(BaseModel):
    label: str
    value: str


class ApplicationQuestion(BaseModel):
    """A question on an application form, as reported by a submission integration."""

    key: str  # canonical key (e.g. "work_authorization_us") or "custom:<slug>"
    label: str
    field_name: str  # the platform's own field name
    kind: QuestionKind = QuestionKind.TEXT
    required: bool = False
    options: list[QuestionOption] = []
    sensitive: bool = False  # never generated; only a user-approved answer may be used


class AnswerSource(StrEnum):
    APPROVED = "approved"  # stored, user-approved answer
    RESUME_FACT = "resume_fact"  # read from the verified resume profile
    GENERATED = "generated"  # LLM text grounded in resume facts, validated
    RESUME_FILE = "resume_file"


class ResolvedAnswer(BaseModel):
    key: str
    field_name: str
    source: AnswerSource
    # Only generated and resume-fact values are stored; approved values are read from
    # approved_answers at submission time so they are never duplicated.
    value: str | None = None
