"""Data-retention purge (policy in docs/architecture.md).

* Resumes: ``purge_after`` is set when a workflow is STOPPED (now + RESUME_RETENTION_DAYS)
  and cleared on reactivation. When due: stored files are deleted, extracted text and
  parsed profiles are erased, and the resume is soft-deleted.
* Applications: ``purge_after`` = creation + APPLICATION_RETENTION_DAYS. When due the
  record and its attempts are deleted, except unresolved submissions
  (SUBMITTING / NEEDS_RECONCILIATION), which are kept until resolved.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from job_agent.models.application import ApplicationStatus
from job_agent.observability.logging import get_logger
from job_agent.persistence.file_store import ResumeFileStore
from job_agent.persistence.orm import Application, Resume, ResumeVersion, SubmissionAttempt

log = get_logger(__name__)

UNRESOLVED = (ApplicationStatus.SUBMITTING, ApplicationStatus.NEEDS_RECONCILIATION)


class RetentionService:
    def __init__(self, session: Session, file_store: ResumeFileStore) -> None:
        self.s = session
        self.files = file_store

    def purge(self, now: datetime) -> dict[str, int]:
        result = {
            "resumes": self._purge_resumes(now),
            "applications": self._purge_applications(now),
        }
        log.info("retention_purge", **result)
        return result

    def _purge_resumes(self, now: datetime) -> int:
        due = self.s.scalars(
            sa.select(Resume).where(
                Resume.purge_after.is_not(None),
                Resume.purge_after <= now,
                Resume.deleted_at.is_(None),
            )
        ).all()
        for resume in due:
            versions = self.s.scalars(
                sa.select(ResumeVersion).where(ResumeVersion.resume_id == resume.id)
            ).all()
            for version in versions:
                self.files.delete(version.storage_ref)
                version.extracted_text = None
                version.profile = {}
                version.parse_warnings = []
            resume.deleted_at = now
        self.s.flush()
        return len(due)

    def _purge_applications(self, now: datetime) -> int:
        ids = list(
            self.s.scalars(
                sa.select(Application.id).where(
                    Application.purge_after.is_not(None),
                    Application.purge_after <= now,
                    Application.status.not_in(UNRESOLVED),
                )
            )
        )
        if ids:
            self.s.execute(
                sa.delete(SubmissionAttempt).where(SubmissionAttempt.application_id.in_(ids))
            )
            self.s.execute(sa.delete(Application).where(Application.id.in_(ids)))
        return len(ids)
