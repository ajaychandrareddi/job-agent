"""Resume, preference, lifecycle, schedule and history endpoints."""

from __future__ import annotations

import json
import uuid
from typing import Annotated

import sqlalchemy as sa
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from sqlalchemy.orm import Session

from job_agent.api.dependencies import Container, CurrentUser, Page, read_upload
from job_agent.api.schemas import (
    ApplicationOut,
    ExecutionOut,
    LifecycleOut,
    MatchOut,
    ResumeDetailOut,
    ResumeOut,
    ScheduleOut,
    UploadOut,
    WorkflowOut,
)
from job_agent.api.schemas import (
    Page as PageOut,
)
from job_agent.api.serializers import application_out, job_brief
from job_agent.models.application import ApplicationStatus
from job_agent.models.matching import Recommendation
from job_agent.models.resume import JobSearchPreferences, ResumeProfile, WorkflowStatus
from job_agent.persistence.orm import (
    Application,
    JobMatch,
    JobPostingRecord,
    Resume,
    WorkflowExecution,
)
from job_agent.repositories.resumes import ResumeRepository
from job_agent.services.errors import InvalidInput
from job_agent.services.lifecycle import LifecycleAction
from job_agent.services.resume_service import suggest_preferences

router = APIRouter(prefix="/api/v1/resumes", tags=["resumes"])


def _workflow(resume: Resume) -> WorkflowOut | None:
    return WorkflowOut.model_validate(resume.workflow) if resume.workflow else None


def _summary(resume: Resume) -> ResumeOut:
    return ResumeOut(
        id=resume.id,
        label=resume.label,
        current_version=resume.current_version,
        created_at=resume.created_at,
        updated_at=resume.updated_at,
        workflow=_workflow(resume),
    )


def _detail(s: Session, resume: Resume) -> ResumeDetailOut:
    version = ResumeRepository(s).current_version(resume)
    return ResumeDetailOut(
        **_summary(resume).model_dump(),
        original_filename=version.original_filename,
        file_type=version.file_type,
        profile=ResumeProfile.model_validate(version.profile),
        preferences=JobSearchPreferences.model_validate(resume.preferences),
        parse_warnings=version.parse_warnings or [],
    )


def _parse_preferences(raw: str | None) -> JobSearchPreferences | None:
    if raw is None or not raw.strip():
        return None
    try:
        return JobSearchPreferences.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise InvalidInput(f"invalid preferences JSON: {str(exc)[:300]}") from exc


# --------------------------------------------------------------------------- resumes


@router.post(
    "",
    response_model=UploadOut,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Identical resume already uploaded"}},
)
async def upload_resume(
    request: Request,
    response: Response,
    user: CurrentUser,
    c: Container,
    file: Annotated[UploadFile, File(description="PDF or DOCX resume")],
    label: Annotated[str | None, Form(max_length=200)] = None,
    preferences: Annotated[str | None, Form(description="JobSearchPreferences JSON")] = None,
) -> UploadOut:
    content = await read_upload(request, file, c.settings.max_upload_bytes)
    prefs = _parse_preferences(preferences)

    def create() -> UploadOut:  # parsing (and any LLM call) runs off the event loop
        with c.session() as s:
            result = c.resume_service(s).create_resume(
                user.id, file.filename or "resume", content, label=label, preferences=prefs
            )
            detail = _detail(s, result.resume)
            return UploadOut(
                resume=detail,
                created=result.created,
                warnings=result.warnings,
                suggested_preferences=suggest_preferences(detail.profile),
            )

    out = await run_in_threadpool(create)
    if not out.created:
        response.status_code = status.HTTP_200_OK
    return out


@router.get("", response_model=list[ResumeOut])
def list_resumes(user: CurrentUser, c: Container) -> list[ResumeOut]:
    with c.session() as s:
        return [_summary(r) for r in ResumeRepository(s).list_for_user(user.id)]


@router.get("/{resume_id}", response_model=ResumeDetailOut)
def get_resume(resume_id: uuid.UUID, user: CurrentUser, c: Container) -> ResumeDetailOut:
    with c.session() as s:
        return _detail(s, c.resume_service(s).get_resume(user.id, resume_id))


@router.put("/{resume_id}/preferences", response_model=ResumeDetailOut)
def update_preferences(
    resume_id: uuid.UUID, body: JobSearchPreferences, user: CurrentUser, c: Container
) -> ResumeDetailOut:
    with c.session() as s:
        resume = c.resume_service(s).update_preferences(user.id, resume_id, body)
        return _detail(s, resume)


@router.put("/{resume_id}/file", response_model=UploadOut)
async def replace_resume_file(
    request: Request,
    resume_id: uuid.UUID,
    user: CurrentUser,
    c: Container,
    file: Annotated[UploadFile, File(description="New PDF or DOCX version")],
) -> UploadOut:
    content = await read_upload(request, file, c.settings.max_upload_bytes)

    def replace() -> UploadOut:
        with c.session() as s:
            result = c.resume_service(s).replace_resume_file(
                user.id, resume_id, file.filename or "resume", content
            )
            detail = _detail(s, result.resume)
            return UploadOut(
                resume=detail,
                created=result.created,
                warnings=result.warnings,
                suggested_preferences=suggest_preferences(detail.profile),
            )

    return await run_in_threadpool(replace)


# --------------------------------------------------------------------------- lifecycle


def _lifecycle(
    resume_id: uuid.UUID, user: CurrentUser, c: Container, action: LifecycleAction
) -> LifecycleOut:
    with c.session() as s:
        workflow, changed = c.resume_service(s).change_status(user.id, resume_id, action)
        return LifecycleOut(workflow=WorkflowOut.model_validate(workflow), changed=changed)


@router.post("/{resume_id}/activate", response_model=LifecycleOut)
def activate(resume_id: uuid.UUID, user: CurrentUser, c: Container) -> LifecycleOut:
    """Start (or restart after STOP). The first activation runs the 30-day initial search."""
    return _lifecycle(resume_id, user, c, LifecycleAction.ACTIVATE)


@router.post("/{resume_id}/pause", response_model=LifecycleOut)
def pause(resume_id: uuid.UUID, user: CurrentUser, c: Container) -> LifecycleOut:
    return _lifecycle(resume_id, user, c, LifecycleAction.PAUSE)


@router.post("/{resume_id}/resume", response_model=LifecycleOut)
def resume_workflow(resume_id: uuid.UUID, user: CurrentUser, c: Container) -> LifecycleOut:
    return _lifecycle(resume_id, user, c, LifecycleAction.RESUME)


@router.post("/{resume_id}/stop", response_model=LifecycleOut)
def stop(resume_id: uuid.UUID, user: CurrentUser, c: Container) -> LifecycleOut:
    """Stop permanently until re-activated. Submissions already sent are not withdrawn."""
    return _lifecycle(resume_id, user, c, LifecycleAction.STOP)


@router.get("/{resume_id}/schedule", response_model=ScheduleOut)
def schedule(resume_id: uuid.UUID, user: CurrentUser, c: Container) -> ScheduleOut:
    with c.session() as s:
        wf = c.resume_service(s).get_resume(user.id, resume_id).workflow
        if wf is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "workflow not found")
        st = c.settings
        return ScheduleOut(
            status=wf.status,
            scheduled=wf.status is WorkflowStatus.ACTIVE,
            next_run_at=wf.next_run_at if wf.status is WorkflowStatus.ACTIVE else None,
            timezone=st.job_search_timezone,
            daily_time=f"{st.job_search_hour:02d}:{st.job_search_minute:02d}",
            initial_search_completed=wf.initial_search_completed,
            last_execution_status=wf.last_execution_status,
            last_execution_at=wf.last_execution_at,
        )


# --------------------------------------------------------------------------- history


@router.get("/{resume_id}/executions", response_model=PageOut[ExecutionOut])
def executions(
    resume_id: uuid.UUID, user: CurrentUser, c: Container, page: Page
) -> PageOut[ExecutionOut]:
    with c.session() as s:
        c.resume_service(s).get_resume(user.id, resume_id)  # ownership
        where = WorkflowExecution.resume_id == resume_id
        total = s.scalar(sa.select(sa.func.count(WorkflowExecution.id)).where(where)) or 0
        rows = s.scalars(
            sa.select(WorkflowExecution)
            .where(where)
            .order_by(WorkflowExecution.started_at.desc())
            .limit(page.limit)
            .offset(page.offset)
        ).all()
        return PageOut[ExecutionOut](
            items=[ExecutionOut.model_validate(r) for r in rows],
            total=total,
            limit=page.limit,
            offset=page.offset,
        )


@router.get("/{resume_id}/matches", response_model=PageOut[MatchOut])
def matches(
    resume_id: uuid.UUID,
    user: CurrentUser,
    c: Container,
    page: Page,
    recommendation: Recommendation | None = None,
    min_score: float | None = None,
) -> PageOut[MatchOut]:
    with c.session() as s:
        c.resume_service(s).get_resume(user.id, resume_id)
        conds = [JobMatch.resume_id == resume_id]
        if recommendation is not None:
            conds.append(JobMatch.recommendation == recommendation)
        if min_score is not None:
            conds.append(JobMatch.score >= min_score)
        total = s.scalar(sa.select(sa.func.count(JobMatch.id)).where(*conds)) or 0
        rows = s.execute(
            sa.select(JobMatch, JobPostingRecord)
            .join(JobPostingRecord, JobPostingRecord.id == JobMatch.job_posting_id)
            .where(*conds)
            .order_by(JobMatch.score.desc(), JobMatch.evaluated_at.desc())
            .limit(page.limit)
            .offset(page.offset)
        ).all()
        items = [
            MatchOut(
                id=m.id,
                resume_version=m.resume_version,
                score=m.score,
                recommendation=m.recommendation,
                criterion_scores=m.criterion_scores,
                matching_skills=m.matching_skills,
                missing_skills=m.missing_skills,
                evidence=m.evidence,
                explanation=m.decision.get("explanation"),
                review_reasons=m.decision.get("review_reasons", []),
                hard_constraint_violations=m.decision.get("hard_constraint_violations", []),
                evaluated_at=m.evaluated_at,
                job=job_brief(j),
            )
            for m, j in rows
        ]
        return PageOut[MatchOut](items=items, total=total, limit=page.limit, offset=page.offset)


@router.get("/{resume_id}/applications", response_model=PageOut[ApplicationOut])
def applications(
    resume_id: uuid.UUID,
    user: CurrentUser,
    c: Container,
    page: Page,
    status_filter: Annotated[ApplicationStatus | None, Query(alias="status")] = None,
) -> PageOut[ApplicationOut]:
    with c.session() as s:
        c.resume_service(s).get_resume(user.id, resume_id)
        conds = [Application.resume_id == resume_id, Application.user_id == user.id]
        if status_filter is not None:
            conds.append(Application.status == status_filter)
        total = s.scalar(sa.select(sa.func.count(Application.id)).where(*conds)) or 0
        rows = s.execute(
            sa.select(Application, JobPostingRecord)
            .join(JobPostingRecord, JobPostingRecord.id == Application.job_posting_id)
            .where(*conds)
            .order_by(Application.updated_at.desc())
            .limit(page.limit)
            .offset(page.offset)
        ).all()
        return PageOut[ApplicationOut](
            items=[application_out(a, j) for a, j in rows],
            total=total,
            limit=page.limit,
            offset=page.offset,
        )
