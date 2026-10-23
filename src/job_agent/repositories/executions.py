"""Queries over workflow executions used by the scheduler."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from job_agent.persistence.orm import ExecutionStatus, ResumeWorkflow, WorkflowExecution


class ExecutionRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def running(self) -> Sequence[tuple[WorkflowExecution, ResumeWorkflow]]:
        rows = self.s.execute(
            sa.select(WorkflowExecution, ResumeWorkflow)
            .join(ResumeWorkflow, ResumeWorkflow.id == WorkflowExecution.workflow_id)
            .where(WorkflowExecution.status == ExecutionStatus.RUNNING)
        ).all()
        return [(e, w) for e, w in rows]

    def has_running(self, resume_id: uuid.UUID) -> bool:
        return (
            self.s.scalar(
                sa.select(sa.func.count(WorkflowExecution.id)).where(
                    WorkflowExecution.resume_id == resume_id,
                    WorkflowExecution.status == ExecutionStatus.RUNNING,
                )
            )
            or 0
        ) > 0

    def close_abandoned(
        self, execution_id: uuid.UUID, status: str, error: str, now: datetime
    ) -> None:
        execution = self.s.get(WorkflowExecution, execution_id)
        if execution is None or execution.status != ExecutionStatus.RUNNING:
            return
        execution.status = status
        execution.error = error
        execution.completed_at = now
        self.s.flush()

    def history(self, resume_id: uuid.UUID, limit: int = 50) -> Sequence[WorkflowExecution]:
        return self.s.scalars(
            sa.select(WorkflowExecution)
            .where(WorkflowExecution.resume_id == resume_id)
            .order_by(WorkflowExecution.started_at.desc())
            .limit(limit)
        ).all()
