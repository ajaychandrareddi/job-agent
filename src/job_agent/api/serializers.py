"""ORM -> API schema conversion (only whitelisted fields leave the server)."""

from __future__ import annotations

from job_agent.api.schemas import ApplicationOut, JobBrief
from job_agent.persistence.orm import Application, JobPostingRecord


def job_brief(job: JobPostingRecord) -> JobBrief:
    return JobBrief(
        id=job.id,
        source=job.source,
        title=job.title,
        company_name=job.company_name,
        location=job.location,
        canonical_url=job.canonical_url,
        application_url=job.application_url,
        published_at=job.published_at,
    )


def application_out(app: Application, job: JobPostingRecord) -> ApplicationOut:
    details = app.error_details or {}
    evidence = app.submission_evidence or {}
    return ApplicationOut(
        id=app.id,
        resume_id=app.resume_id,
        resume_version=app.resume_version,
        status=app.status,
        status_reason=app.status_reason,
        review_kind=details.get("review_kind"),
        missing_questions=list(details.get("missing_questions") or []),
        manual_package=evidence.get("manual_package"),
        external_confirmation_ref=app.external_confirmation_ref,
        submitted_at=app.submitted_at,
        last_reconciled_at=app.last_reconciled_at,
        retry_count=app.retry_count,
        reconcile_attempts=app.reconcile_attempts,
        created_at=app.created_at,
        updated_at=app.updated_at,
        job=job_brief(job),
    )
