"""Request/response schemas for the REST API."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from job_agent.models.application import ApplicationStatus
from job_agent.models.common import AwareDatetime
from job_agent.models.job import JobSource
from job_agent.models.matching import Recommendation
from job_agent.models.resume import (
    JobSearchPreferences,
    ResumeFileType,
    ResumeProfile,
    WorkflowStatus,
)


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


class WorkflowOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: WorkflowStatus
    initial_search_completed: bool
    initial_search_completed_at: AwareDatetime | None
    next_run_at: AwareDatetime | None
    last_execution_status: str | None
    last_execution_at: AwareDatetime | None


class ResumeOut(BaseModel):
    id: uuid.UUID
    label: str
    current_version: int
    created_at: AwareDatetime
    updated_at: AwareDatetime
    workflow: WorkflowOut | None


class ResumeDetailOut(ResumeOut):
    original_filename: str
    file_type: ResumeFileType
    profile: ResumeProfile
    preferences: JobSearchPreferences
    parse_warnings: list[str]


class UploadOut(BaseModel):
    resume: ResumeDetailOut
    created: bool
    warnings: list[str]
    suggested_preferences: JobSearchPreferences = Field(
        description="Suggestions from the resume. Not applied until you approve them via "
        "PUT /resumes/{id}/preferences."
    )


class LifecycleOut(BaseModel):
    workflow: WorkflowOut
    changed: bool


class ScheduleOut(BaseModel):
    status: WorkflowStatus
    scheduled: bool
    next_run_at: AwareDatetime | None
    timezone: str
    daily_time: str
    initial_search_completed: bool
    last_execution_status: str | None
    last_execution_at: AwareDatetime | None


class ExecutionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    trigger: str
    status: str
    resume_version: int
    search_window_start: AwareDatetime
    search_window_end: AwareDatetime
    started_at: AwareDatetime
    completed_at: AwareDatetime | None
    discovered_count: int
    unique_count: int
    matched_count: int
    submitted_count: int
    applied_count: int
    skipped_count: int
    pending_review_count: int
    failed_count: int
    error: str | None
    summary: dict[str, Any] | None


class JobBrief(BaseModel):
    id: uuid.UUID
    source: JobSource
    title: str
    company_name: str
    location: str | None
    canonical_url: str
    application_url: str | None
    published_at: AwareDatetime | None


class MatchOut(BaseModel):
    id: uuid.UUID
    resume_version: int
    score: float
    recommendation: Recommendation
    criterion_scores: dict[str, Any]
    matching_skills: list[str]
    missing_skills: list[str]
    evidence: list[str]
    explanation: str | None
    review_reasons: list[str]
    hard_constraint_violations: list[dict[str, Any]]
    evaluated_at: AwareDatetime
    job: JobBrief


class ApplicationOut(BaseModel):
    id: uuid.UUID
    resume_id: uuid.UUID
    resume_version: int
    status: ApplicationStatus
    status_reason: str | None
    review_kind: str | None
    missing_questions: list[str]
    manual_package: dict[str, Any] | None
    external_confirmation_ref: str | None
    submitted_at: AwareDatetime | None
    last_reconciled_at: AwareDatetime | None
    retry_count: int
    reconcile_attempts: int
    created_at: AwareDatetime
    updated_at: AwareDatetime
    job: JobBrief


class ActionOut(BaseModel):
    application_id: uuid.UUID
    status: ApplicationStatus
    detail: str


class AnswerIn(BaseModel):
    answer: str = Field(min_length=1, max_length=5000)
    label: str | None = Field(
        None, max_length=500, description="Question wording; improves sensitivity detection."
    )


class AnswerOut(BaseModel):
    key: str
    sensitive: bool
    answer: str = Field(description="Masked for sensitive answers.")
    approved_at: AwareDatetime
