"""Application detail, retry and reconciliation endpoints."""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from fastapi import APIRouter

from job_agent.api.dependencies import Container, CurrentUser
from job_agent.api.schemas import ActionOut, ApplicationOut
from job_agent.api.serializers import application_out
from job_agent.persistence.orm import Application, JobPostingRecord
from job_agent.services.errors import NotFound

router = APIRouter(prefix="/api/v1/applications", tags=["applications"])


@router.get("/{application_id}", response_model=ApplicationOut)
def get_application(application_id: uuid.UUID, user: CurrentUser, c: Container) -> ApplicationOut:
    with c.session() as s:
        row = s.execute(
            sa.select(Application, JobPostingRecord)
            .join(JobPostingRecord, JobPostingRecord.id == Application.job_posting_id)
            .where(Application.id == application_id, Application.user_id == user.id)
        ).first()
        if row is None:
            raise NotFound("application not found")
        return application_out(row[0], row[1])


@router.post("/{application_id}/retry", response_model=ActionOut)
async def retry(application_id: uuid.UUID, user: CurrentUser, c: Container) -> ActionOut:
    """Retry only when it is safe: after a retryable rejection, or after fixing what put
    the application into review (e.g. approving a missing answer). Uncertain submissions
    must be reconciled first. Never creates a second application for the same job."""
    result = await c.actions.retry(user.id, application_id)
    return ActionOut(
        application_id=result.application_id, status=result.status, detail=result.detail
    )


@router.post("/{application_id}/reconcile", response_model=ActionOut)
async def reconcile(application_id: uuid.UUID, user: CurrentUser, c: Container) -> ActionOut:
    """Ask the platform whether an uncertain submission arrived. Never resubmits.
    Idempotent: a no-op for applications whose outcome is already known."""
    result = await c.actions.reconcile(application_id, user_id=user.id)
    return ActionOut(
        application_id=result.application_id, status=result.status, detail=result.detail
    )
