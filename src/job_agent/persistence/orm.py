"""SQLAlchemy ORM schema.

The application database is the source of truth for application history and
uniqueness. LangGraph checkpoints (separate tables managed by
``langgraph-checkpoint-postgres``) hold only recoverable in-flight workflow state.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from job_agent.models.application import ApplicationStatus, SubmissionAttemptStatus
from job_agent.models.common import utcnow
from job_agent.models.job import JobSource
from job_agent.models.matching import Recommendation
from job_agent.models.resume import ResumeFileType, WorkflowStatus
from job_agent.persistence.types import JSONType, UTCDateTime, enum_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {datetime: UTCDateTime(), dict[str, Any]: JSONType}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )


class ExecutionStatus:
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"  # finished, but some sources/jobs failed
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"  # stop/pause observed mid-run
    SKIPPED = "SKIPPED"  # e.g. workflow not active when the trigger fired

    ALL = (RUNNING, SUCCEEDED, PARTIAL, FAILED, CANCELLED, SKIPPED)


class ExecutionTrigger:
    INITIAL = "INITIAL"
    SCHEDULED = "SCHEDULED"
    CATCH_UP = "CATCH_UP"
    MANUAL = "MANUAL"

    ALL = (INITIAL, SCHEDULED, CATCH_UP, MANUAL)


# --------------------------------------------------------------------------- #


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    external_subject: Mapped[str] = mapped_column(sa.String(255), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(sa.String(320))
    timezone: Mapped[str] = mapped_column(sa.String(64), default="America/New_York")
    preferences: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    resumes: Mapped[list[Resume]] = relationship(back_populates="user")


class Resume(TimestampMixin, Base):
    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    current_version: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=1)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    preferences_updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    purge_after: Mapped[datetime | None] = mapped_column(UTCDateTime(), index=True)

    user: Mapped[User] = relationship(back_populates="resumes")
    versions: Mapped[list[ResumeVersion]] = relationship(
        back_populates="resume", order_by="ResumeVersion.version", cascade="all, delete-orphan"
    )
    workflow: Mapped[ResumeWorkflow | None] = relationship(
        back_populates="resume", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (sa.UniqueConstraint("user_id", "label", name="uq_resumes_user_label"),)


class ResumeVersion(Base):
    """Immutable snapshot of one uploaded resume file and its parsed profile."""

    __tablename__ = "resume_versions"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    original_filename: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    file_type: Mapped[ResumeFileType] = mapped_column(
        enum_column(ResumeFileType, "resume_file_type"), nullable=False
    )
    content_sha256: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    storage_ref: Mapped[str] = mapped_column(sa.String(1024), nullable=False)
    # Protected text; null once purged under the retention policy.
    extracted_text: Mapped[str | None] = mapped_column(sa.Text)
    profile: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    parse_warnings: Mapped[list[str]] = mapped_column(JSONType, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, nullable=False)

    resume: Mapped[Resume] = relationship(back_populates="versions")

    __table_args__ = (
        sa.UniqueConstraint("resume_id", "version", name="uq_resume_versions_resume_version"),
        sa.Index("ix_resume_versions_sha", "content_sha256"),
    )


class ResumeWorkflow(Base):
    __tablename__ = "resume_workflows"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    status: Mapped[WorkflowStatus] = mapped_column(
        enum_column(WorkflowStatus, "workflow_status"),
        nullable=False,
        default=WorkflowStatus.DRAFT,
        index=True,
    )
    status_changed_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    initial_search_completed: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    initial_search_completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    next_run_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    last_execution_status: Mapped[str | None] = mapped_column(sa.String(32))
    last_execution_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    # Upper bound of the last *successfully covered* search window (catch-up policy).
    last_window_end: Mapped[datetime | None] = mapped_column(UTCDateTime())
    # DB-backed coordination across scheduler instances.
    run_lock_owner: Mapped[str | None] = mapped_column(sa.String(128))
    run_lock_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    row_version: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=1)

    resume: Mapped[Resume] = relationship(back_populates="workflow")

    __mapper_args__ = {"version_id_col": row_version}


class JobPostingRecord(Base):
    __tablename__ = "job_postings"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    source: Mapped[JobSource] = mapped_column(enum_column(JobSource, "job_source"), nullable=False)
    external_id: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    # Cross-source identity used for deduplication and the application key.
    canonical_key: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    company_name: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    location: Mapped[str | None] = mapped_column(sa.String(500))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), index=True)
    source_updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    publication_date_unverified: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    canonical_url: Mapped[str] = mapped_column(sa.String(2048), nullable=False)
    application_url: Mapped[str | None] = mapped_column(sa.String(2048))
    normalized: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    first_discovered_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    __table_args__ = (
        sa.UniqueConstraint("source", "external_id", name="uq_job_postings_source_external"),
    )


class JobMatch(Base):
    __tablename__ = "job_matches"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    resume_version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    job_posting_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )
    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("workflow_executions.id", ondelete="SET NULL")
    )
    score: Mapped[float] = mapped_column(sa.Numeric(5, 2, asdecimal=False), nullable=False)
    criterion_scores: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    matching_skills: Mapped[list[str]] = mapped_column(JSONType, default=list)
    missing_skills: Mapped[list[str]] = mapped_column(JSONType, default=list)
    evidence: Mapped[list[str]] = mapped_column(JSONType, default=list)
    recommendation: Mapped[Recommendation] = mapped_column(
        enum_column(Recommendation, "match_recommendation"), nullable=False
    )
    decision: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    __table_args__ = (
        sa.UniqueConstraint(
            "resume_id", "resume_version", "job_posting_id", name="uq_job_matches_resume_job"
        ),
        sa.CheckConstraint("score >= 0 AND score <= 100", name="score_range"),
        sa.Index("ix_job_matches_resume_score", "resume_id", "score"),
    )


class Application(TimestampMixin, Base):
    __tablename__ = "applications"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    # sha256(user_id | resume_id | canonical job key) — the idempotency anchor.
    application_key: Mapped[str] = mapped_column(sa.String(64), nullable=False, unique=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    resume_version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    job_posting_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("job_postings.id", ondelete="RESTRICT"), nullable=False
    )
    match_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("job_matches.id", ondelete="SET NULL")
    )
    # Execution that created this record (links run summaries to application records and
    # lets a resumed execution recognise its own in-flight applications).
    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("workflow_executions.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[ApplicationStatus] = mapped_column(
        enum_column(ApplicationStatus, "application_status"),
        nullable=False,
        default=ApplicationStatus.DISCOVERED,
    )
    current_attempt_id: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid)
    external_confirmation_ref: Mapped[str | None] = mapped_column(sa.String(255))
    submitted_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    last_reconciled_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    retry_count: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    reconcile_attempts: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    # Supporting evidence: integration, answer provenance (no sensitive values),
    # verification method, manual-application package.
    submission_evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    status_reason: Mapped[str | None] = mapped_column(sa.String(500))
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    purge_after: Mapped[datetime | None] = mapped_column(UTCDateTime(), index=True)

    attempts: Mapped[list[SubmissionAttempt]] = relationship(
        back_populates="application", order_by="SubmissionAttempt.attempt_number"
    )

    __table_args__ = (
        sa.UniqueConstraint("resume_id", "job_posting_id", name="uq_applications_resume_job"),
        sa.Index("ix_applications_resume_status", "resume_id", "status"),
    )


class SubmissionAttempt(Base):
    """Durable record written *before* any external submission call."""

    __tablename__ = "submission_attempts"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    application_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("applications.id", ondelete="CASCADE"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(sa.String(128), nullable=False, unique=True)
    status: Mapped[SubmissionAttemptStatus] = mapped_column(
        enum_column(SubmissionAttemptStatus, "submission_attempt_status"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    external_ref: Mapped[str | None] = mapped_column(sa.String(255))
    response_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(sa.Text)

    application: Mapped[Application] = relationship(back_populates="attempts")

    __table_args__ = (
        sa.UniqueConstraint("application_id", "attempt_number", name="uq_attempts_app_number"),
        # At most one in-flight or unresolved attempt per application.
        sa.Index(
            "uq_attempts_one_open_per_app",
            "application_id",
            unique=True,
            postgresql_where=sa.text("status IN ('STARTED', 'UNCERTAIN')"),
            sqlite_where=sa.text("status IN ('STARTED', 'UNCERTAIN')"),
        ),
    )


class ApprovedAnswer(TimestampMixin, Base):
    """User-approved answers to recurring application questions (sensitive)."""

    __tablename__ = "approved_answers"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    question_key: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    answer: Mapped[str] = mapped_column(sa.Text, nullable=False)
    is_sensitive: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    approved_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    __table_args__ = (
        sa.UniqueConstraint("user_id", "question_key", name="uq_approved_answers_user_key"),
    )


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("resume_workflows.id", ondelete="CASCADE"), nullable=False
    )
    resume_version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    trigger: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    search_window_start: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    search_window_end: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False, default="RUNNING")
    discovered_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    unique_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    matched_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    submitted_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    applied_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    pending_review_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    failed_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(sa.Text)
    # LangGraph thread id used for checkpointing this execution.
    thread_id: Mapped[str] = mapped_column(sa.String(128), nullable=False)

    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('RUNNING','SUCCEEDED','PARTIAL','FAILED','CANCELLED','SKIPPED')",
            name="status_valid",
        ),
        sa.CheckConstraint(
            "trigger IN ('INITIAL','SCHEDULED','CATCH_UP','MANUAL')", name="trigger_valid"
        ),
        sa.CheckConstraint("search_window_start < search_window_end", name="window_order"),
        sa.Index("ix_workflow_executions_resume_started", "resume_id", "started_at"),
        # Prevents overlapping executions of the same resume, even across instances.
        sa.Index(
            "uq_workflow_executions_one_running",
            "resume_id",
            unique=True,
            postgresql_where=sa.text("status = 'RUNNING'"),
            sqlite_where=sa.text("status = 'RUNNING'"),
        ),
    )
