"""Resume ingestion, versioning, preferences and lifecycle orchestration."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session

from job_agent.models.common import utcnow
from job_agent.models.resume import (
    JobSearchPreferences,
    ParsedResume,
    ResumeProfile,
    WorkflowStatus,
)
from job_agent.observability.logging import get_logger
from job_agent.persistence.file_store import ResumeFileStore
from job_agent.persistence.orm import Resume, ResumeVersion, ResumeWorkflow
from job_agent.repositories.resumes import (
    ResumeRepository,
    WorkflowRepository,
    stable_resume_id,
)
from job_agent.security.uploads import UploadRejected, sanitize_filename
from job_agent.services.errors import Conflict, InvalidInput, NotFound
from job_agent.services.lifecycle import LifecycleAction, LifecycleError, next_status
from job_agent.services.resume_parser import ResumeParser, sha256_hex
from job_agent.services.text_extraction import TextExtractionError

log = get_logger(__name__)

StatusListener = Callable[[ResumeWorkflow, WorkflowStatus, WorkflowStatus], None]


@dataclass
class ResumeUploadResult:
    resume: Resume
    version: ResumeVersion
    created: bool
    warnings: list[str] = field(default_factory=list)


def suggest_preferences(profile: ResumeProfile) -> JobSearchPreferences:
    """Suggestions only — never persisted as approved without the user's confirmation."""
    return JobSearchPreferences(
        preferred_titles=profile.job_titles[:3],
        keywords=(profile.skills + profile.technologies)[:10],
    )


class ResumeService:
    def __init__(
        self,
        session: Session,
        parser: ResumeParser,
        file_store: ResumeFileStore,
        *,
        status_listeners: list[StatusListener] | None = None,
        resume_retention_days: int = 365,
    ) -> None:
        self.resume_retention_days = resume_retention_days
        self.s = session
        self.parser = parser
        self.files = file_store
        self.resumes = ResumeRepository(session)
        self.workflows = WorkflowRepository(session)
        self._listeners = status_listeners or []

    # ------------------------------------------------------------------ upload

    def create_resume(
        self,
        user_id: uuid.UUID,
        filename: str,
        content: bytes,
        *,
        label: str | None = None,
        preferences: JobSearchPreferences | None = None,
    ) -> ResumeUploadResult:
        safe_name = sanitize_filename(filename)
        sha = sha256_hex(content)
        existing = self.resumes.find_version_by_hash(user_id, sha)
        if existing is not None:  # idempotent re-upload
            resume = self.resumes.get_for_user(user_id, existing.resume_id)
            assert resume is not None
            return ResumeUploadResult(
                resume=resume,
                version=existing,
                created=False,
                warnings=["identical resume already uploaded"],
            )

        parsed = self._parse(safe_name, content)
        resume_id = stable_resume_id(user_id, parsed.content_sha256)
        if self.s.get(Resume, resume_id) is not None:
            # Same content was uploaded before and later deleted; issue a fresh id.
            resume_id = uuid.uuid4()
        ref = self.files.save(
            user_id, resume_id, 1, parsed.content_sha256, content, parsed.file_type.value
        )
        resume = self.resumes.create(
            user_id=user_id,
            label=self._unique_label(user_id, label or safe_name.rsplit(".", 1)[0]),
            filename=safe_name,
            parsed=parsed,
            storage_ref=ref,
            preferences=preferences or JobSearchPreferences(),
            resume_id=resume_id,
        )
        log.info("resume_created", resume_id=str(resume.id), warnings=len(parsed.warnings))
        return ResumeUploadResult(
            resume=resume,
            version=self.resumes.current_version(resume),
            created=True,
            warnings=parsed.warnings,
        )

    def replace_resume_file(
        self, user_id: uuid.UUID, resume_id: uuid.UUID, filename: str, content: bytes
    ) -> ResumeUploadResult:
        """Upload a new version. The resume id (and so application history and
        duplicate protection, which are keyed on it) is unchanged."""
        resume = self.get_resume(user_id, resume_id)
        sha = sha256_hex(content)
        current = self.resumes.current_version(resume)
        if current.content_sha256 == sha:
            return ResumeUploadResult(
                resume=resume,
                version=current,
                created=False,
                warnings=["file identical to current version"],
            )
        other = self.resumes.find_version_by_hash(user_id, sha)
        if other is not None and other.resume_id != resume.id:
            raise Conflict("this file is already uploaded as a different resume")

        safe_name = sanitize_filename(filename)
        parsed = self._parse(safe_name, content)
        ref = self.files.save(
            user_id,
            resume.id,
            resume.current_version + 1,
            parsed.content_sha256,
            content,
            parsed.file_type.value,
        )
        version = self.resumes.add_version(
            resume, filename=safe_name, parsed=parsed, storage_ref=ref
        )
        log.info("resume_version_added", resume_id=str(resume.id), version=version.version)
        return ResumeUploadResult(
            resume=resume, version=version, created=True, warnings=parsed.warnings
        )

    # ------------------------------------------------------------- read/update

    def get_resume(self, user_id: uuid.UUID, resume_id: uuid.UUID) -> Resume:
        resume = self.resumes.get_for_user(user_id, resume_id)
        if resume is None:
            raise NotFound("resume not found")
        return resume

    def update_preferences(
        self, user_id: uuid.UUID, resume_id: uuid.UUID, preferences: JobSearchPreferences
    ) -> Resume:
        resume = self.get_resume(user_id, resume_id)
        return self.resumes.update_preferences(resume, preferences)

    # --------------------------------------------------------------- lifecycle

    def change_status(
        self, user_id: uuid.UUID, resume_id: uuid.UUID, action: LifecycleAction
    ) -> tuple[ResumeWorkflow, bool]:
        resume = self.get_resume(user_id, resume_id)
        workflow = self.workflows.get_by_resume(resume.id, for_update=True)
        if workflow is None:
            raise NotFound("workflow not found")
        try:
            target, changed = next_status(workflow.status, action)
        except LifecycleError as exc:
            raise Conflict(str(exc)) from exc
        if target is WorkflowStatus.ACTIVE and changed:
            prefs = JobSearchPreferences.model_validate(resume.preferences)
            if not prefs.preferred_titles:
                raise InvalidInput(
                    "set at least one preferred job title before activating this resume"
                )
        if not changed:
            return workflow, False
        previous = workflow.status
        self.workflows.set_status(workflow, target)
        # Retention: a stopped resume's files/text are purged after the retention period
        # unless it is reactivated first (see services/retention.py).
        if target is WorkflowStatus.STOPPED:
            resume.purge_after = utcnow() + timedelta(days=self.resume_retention_days)
        elif target is WorkflowStatus.ACTIVE:
            resume.purge_after = None
        log.info(
            "workflow_status_changed",
            resume_id=str(resume.id),
            previous=previous.value,
            current=target.value,
        )
        for listener in self._listeners:
            listener(workflow, previous, target)
        return workflow, True

    # ---------------------------------------------------------------- helpers

    def _parse(self, filename: str, content: bytes) -> ParsedResume:
        try:
            return self.parser.parse(filename, content)
        except (UploadRejected, TextExtractionError) as exc:
            raise InvalidInput(str(exc)) from exc

    def _unique_label(self, user_id: uuid.UUID, base: str) -> str:
        base = base.strip()[:180] or "Resume"
        taken = set(self.s.scalars(sa.select(Resume.label).where(Resume.user_id == user_id)))
        if base not in taken:
            return base
        n = 2
        while f"{base} ({n})" in taken:
            n += 1
        return f"{base} ({n})"
