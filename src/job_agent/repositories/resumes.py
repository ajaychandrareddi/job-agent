"""Repositories for users, resumes, resume versions and workflows.

All reads that take a ``user_id`` are ownership-scoped: a resume owned by
another user is indistinguishable from a missing one.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from job_agent.models.common import utcnow
from job_agent.models.resume import JobSearchPreferences, ParsedResume, WorkflowStatus
from job_agent.persistence.orm import Resume, ResumeVersion, ResumeWorkflow, User

# Namespace for deterministic resume ids: uuid5(NS, "<user_id>:<first upload sha256>").
RESUME_ID_NAMESPACE = uuid.UUID("6f1d3c52-8a2e-4c1b-9f57-0b6a2b7d9e11")


def stable_resume_id(user_id: uuid.UUID, content_sha256: str) -> uuid.UUID:
    return uuid.uuid5(RESUME_ID_NAMESPACE, f"{user_id}:{content_sha256}")


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get(self, user_id: uuid.UUID) -> User | None:
        return self.s.get(User, user_id)

    def get_or_create(
        self, external_subject: str, *, email: str | None = None, timezone: str = "America/New_York"
    ) -> User:
        user = self.s.scalar(sa.select(User).where(User.external_subject == external_subject))
        if user:
            return user
        user = User(
            external_subject=external_subject, email=email, timezone=timezone, preferences={}
        )
        try:
            with self.s.begin_nested():
                self.s.add(user)
        except IntegrityError:  # concurrent creation
            user = self.s.scalar(sa.select(User).where(User.external_subject == external_subject))
            assert user is not None
        return user


class ResumeRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get_for_user(self, user_id: uuid.UUID, resume_id: uuid.UUID) -> Resume | None:
        return self.s.scalar(
            sa.select(Resume)
            .where(Resume.id == resume_id, Resume.user_id == user_id, Resume.deleted_at.is_(None))
            .options(selectinload(Resume.workflow))
        )

    def list_for_user(self, user_id: uuid.UUID) -> Sequence[Resume]:
        return self.s.scalars(
            sa.select(Resume)
            .where(Resume.user_id == user_id, Resume.deleted_at.is_(None))
            .options(selectinload(Resume.workflow))
            .order_by(Resume.created_at)
        ).all()

    def find_version_by_hash(self, user_id: uuid.UUID, sha256: str) -> ResumeVersion | None:
        return self.s.scalar(
            sa.select(ResumeVersion)
            .join(Resume)
            .where(
                Resume.user_id == user_id,
                ResumeVersion.content_sha256 == sha256,
                Resume.deleted_at.is_(None),
            )
            .order_by(ResumeVersion.created_at.desc())
            .limit(1)
        )

    def current_version(self, resume: Resume) -> ResumeVersion:
        version = self.s.scalar(
            sa.select(ResumeVersion).where(
                ResumeVersion.resume_id == resume.id,
                ResumeVersion.version == resume.current_version,
            )
        )
        if version is None:
            raise LookupError(f"resume {resume.id} has no version {resume.current_version}")
        return version

    def create(
        self,
        *,
        user_id: uuid.UUID,
        label: str,
        filename: str,
        parsed: ParsedResume,
        storage_ref: str,
        preferences: JobSearchPreferences,
        resume_id: uuid.UUID | None = None,
    ) -> Resume:
        now = utcnow()
        resume = Resume(
            id=resume_id or stable_resume_id(user_id, parsed.content_sha256),
            user_id=user_id,
            label=label,
            current_version=1,
            preferences=preferences.model_dump(mode="json"),
            preferences_updated_at=now,
            created_at=now,
            updated_at=now,
        )
        self.s.add(resume)
        self.s.add(self._version(resume.id, 1, filename, parsed, storage_ref))
        self.s.add(
            ResumeWorkflow(
                resume_id=resume.id,
                status=WorkflowStatus.DRAFT,
                status_changed_at=now,
                initial_search_completed=False,
            )
        )
        self.s.flush()
        return resume

    def add_version(
        self, resume: Resume, *, filename: str, parsed: ParsedResume, storage_ref: str
    ) -> ResumeVersion:
        new_version = resume.current_version + 1
        version = self._version(resume.id, new_version, filename, parsed, storage_ref)
        self.s.add(version)
        resume.current_version = new_version
        resume.updated_at = utcnow()
        self.s.flush()
        return version

    def update_preferences(self, resume: Resume, preferences: JobSearchPreferences) -> Resume:
        resume.preferences = preferences.model_dump(mode="json")
        resume.preferences_updated_at = utcnow()
        resume.updated_at = resume.preferences_updated_at
        self.s.flush()
        return resume

    @staticmethod
    def _version(
        resume_id: uuid.UUID, version: int, filename: str, parsed: ParsedResume, storage_ref: str
    ) -> ResumeVersion:
        return ResumeVersion(
            resume_id=resume_id,
            version=version,
            original_filename=filename,
            file_type=parsed.file_type,
            content_sha256=parsed.content_sha256,
            storage_ref=storage_ref,
            extracted_text=parsed.text,
            profile=parsed.profile.model_dump(mode="json"),
            parse_warnings=list(parsed.warnings),
        )


class WorkflowRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get_by_resume(
        self, resume_id: uuid.UUID, *, for_update: bool = False
    ) -> ResumeWorkflow | None:
        stmt = sa.select(ResumeWorkflow).where(ResumeWorkflow.resume_id == resume_id)
        if for_update:
            stmt = stmt.with_for_update()
        return self.s.scalar(stmt)

    def list_by_status(self, status: WorkflowStatus) -> Sequence[ResumeWorkflow]:
        return self.s.scalars(
            sa.select(ResumeWorkflow)
            .join(Resume)
            .where(ResumeWorkflow.status == status, Resume.deleted_at.is_(None))
        ).all()

    def set_status(self, workflow: ResumeWorkflow, status: WorkflowStatus) -> ResumeWorkflow:
        workflow.status = status
        workflow.status_changed_at = utcnow()
        if status is not WorkflowStatus.ACTIVE:
            workflow.next_run_at = None
        self.s.flush()
        return workflow

    # ------------------------------------------------------------- run leases
    # Core UPDATEs (not ORM flushes) so acquisition is atomic across processes and
    # never conflicts with ORM writes to other columns of the same row.

    def try_acquire_lease(
        self, resume_id: uuid.UUID, owner: str, now: datetime, ttl_seconds: int
    ) -> bool:
        result = self.s.execute(
            sa.update(ResumeWorkflow)
            .where(
                ResumeWorkflow.resume_id == resume_id,
                sa.or_(
                    ResumeWorkflow.run_lock_owner.is_(None),
                    ResumeWorkflow.run_lock_expires_at < now,
                ),
            )
            .values(run_lock_owner=owner, run_lock_expires_at=now + timedelta(seconds=ttl_seconds))
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))

    def renew_lease(
        self, resume_id: uuid.UUID, owner: str, now: datetime, ttl_seconds: int
    ) -> bool:
        result = self.s.execute(
            sa.update(ResumeWorkflow)
            .where(ResumeWorkflow.resume_id == resume_id, ResumeWorkflow.run_lock_owner == owner)
            .values(run_lock_expires_at=now + timedelta(seconds=ttl_seconds))
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))

    def release_lease(self, resume_id: uuid.UUID, owner: str) -> None:
        self.s.execute(
            sa.update(ResumeWorkflow)
            .where(ResumeWorkflow.resume_id == resume_id, ResumeWorkflow.run_lock_owner == owner)
            .values(run_lock_owner=None, run_lock_expires_at=None)
            .execution_options(synchronize_session=False)
        )

    def lease_expired(self, workflow: ResumeWorkflow, now: datetime) -> bool:
        return workflow.run_lock_owner is None or (
            workflow.run_lock_expires_at is not None and workflow.run_lock_expires_at < now
        )

    def mark_initial_search_completed(
        self, workflow: ResumeWorkflow, completed_at: datetime
    ) -> ResumeWorkflow:
        workflow.initial_search_completed = True
        workflow.initial_search_completed_at = completed_at
        self.s.flush()
        return workflow
