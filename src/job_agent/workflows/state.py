"""Typed LangGraph state.

State holds only JSON-safe values (ids as strings, timestamps as ISO strings, models
as ``model_dump(mode="json")`` dicts) so checkpoints serialise cleanly to PostgreSQL.
Each execution has its own thread (thread_id = execution id) and every value is
scoped to one resume: nothing is shared between resumes.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from job_agent.models.job import JobPosting
from job_agent.services.deduplication import DedupedJob


class HaltReason(StrEnum):
    RESUME_NOT_FOUND = "resume_not_found"
    NOT_ACTIVE = "workflow_not_active"
    NO_SEARCH_CRITERIA = "no_approved_search_criteria"
    ALREADY_RUNNING = "another_execution_running"


class JobRoute(StrEnum):
    APPLY = "apply"
    REVIEW = "review"
    SKIP = "skip"
    DUPLICATE = "duplicate"
    HALT = "halt"  # application path ended early (stop signal, unavailable, review)
    CONTINUE = "continue"


class OutcomeKind(StrEnum):
    """Per-job result reported in the run summary (application status or a run-level code)."""

    APPLIED = "APPLIED"
    SKIPPED = "SKIPPED"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPLICATION_UNAVAILABLE = "APPLICATION_UNAVAILABLE"
    FAILED = "FAILED"
    NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"
    DUPLICATE = "DUPLICATE"
    ERROR = "ERROR"  # unexpected processing error for this job (run continues)


class SearchInput(TypedDict, total=False):
    execution_id: str
    resume_id: str
    manual: bool
    # Label recorded on the execution (MANUAL / SCHEDULED / CATCH_UP); the initial
    # 30-day search is always recorded as INITIAL.
    requested_trigger: str | None


class SearchState(TypedDict, total=False):
    execution_id: str
    resume_id: str
    manual: bool
    requested_trigger: str | None
    user_id: str
    resume_version: int
    profile: dict[str, Any]
    preferences: dict[str, Any]
    halt_reason: str | None
    execution_created: bool
    window_start: str
    window_end: str
    trigger: str
    started_at: str
    source_results: Annotated[list[dict[str, Any]], operator.add]
    postings: list[dict[str, Any]]
    jobs: list[dict[str, Any]]
    date_filter_counts: dict[str, int]
    to_process: list[dict[str, Any]]
    already_tracked: int
    job_outcomes: Annotated[list[dict[str, Any]], operator.add]
    status: str
    summary: dict[str, Any]


class JobState(TypedDict, total=False):
    """State of one job's matching + application path (isolated per job)."""

    execution_id: str
    execution_started_at: str
    user_id: str
    resume_id: str
    resume_version: int
    profile: dict[str, Any]
    preferences: dict[str, Any]
    job: dict[str, Any]
    decision: dict[str, Any]
    match_id: str
    route: str
    reason: str
    application_id: str
    integration: str | None
    questions: list[dict[str, Any]]  # ApplicationQuestion dumps (no answer values)
    submission: dict[str, Any] | None
    final_status: str
    job_outcomes: Annotated[list[dict[str, Any]], operator.add]


class JobOutput(TypedDict, total=False):
    job_outcomes: Annotated[list[dict[str, Any]], operator.add]


def dump_job(job: DedupedJob, *, date_review_reason: str | None = None) -> dict[str, Any]:
    return {
        "canonical_key": job.canonical_key,
        "member_keys": sorted(job.member_keys),
        "posting": job.posting.model_dump(mode="json"),
        "duplicates": [d.model_dump(mode="json") for d in job.duplicates],
        "date_review_reason": date_review_reason,
    }


def load_job(data: dict[str, Any]) -> DedupedJob:
    return DedupedJob(
        canonical_key=data["canonical_key"],
        posting=JobPosting.model_validate(data["posting"]),
        duplicates=[JobPosting.model_validate(d) for d in data["duplicates"]],
        member_keys=frozenset(data["member_keys"]),
    )
