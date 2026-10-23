"""Resume, search-preference and workflow-lifecycle domain models."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from job_agent.models.common import AwareDatetime


class WorkArrangement(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    UNKNOWN = "unknown"


class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    UNKNOWN = "unknown"


class WorkflowStatus(StrEnum):
    """Independent lifecycle state of one resume's search/apply workflow."""

    DRAFT = "DRAFT"  # created, never activated
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class ResumeFileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"


# --------------------------------------------------------------------------- #
# Structured resume profile (LLM structured-output target)
# --------------------------------------------------------------------------- #


class Position(BaseModel):
    """One employment entry, exactly as stated on the resume."""

    title: str = Field(description="Job title exactly as written on the resume.")
    company: str | None = Field(None, description="Employer name as written.")
    start_date: date | None = Field(
        None, description="Start date if stated. Use the 1st of the month if only month/year."
    )
    end_date: date | None = Field(
        None, description="End date if stated; null if current or not stated."
    )
    is_current: bool = Field(False, description="True only if the resume says present/current.")
    highlights: list[str] = Field(
        default_factory=list, description="Accomplishment bullets copied from the resume."
    )


class Education(BaseModel):
    institution: str | None = None
    degree: str | None = Field(None, description="Degree as written, e.g. 'B.S. Computer Science'")
    field_of_study: str | None = None
    graduation_year: int | None = Field(None, ge=1900, le=2100)


class ResumeProfile(BaseModel):
    """Structured representation of a resume.

    Every value must be directly supported by the resume text. The parser
    removes items it cannot ground in the source text.
    """

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(None, description="Candidate name if present.")
    headline: str | None = Field(None, description="Professional headline/summary line, if any.")
    summary: str | None = Field(None, description="Professional summary, if present.")
    skills: list[str] = Field(default_factory=list, description="Skills explicitly listed.")
    technologies: list[str] = Field(
        default_factory=list, description="Tools, languages, frameworks, platforms named."
    )
    certifications: list[str] = Field(default_factory=list)
    job_titles: list[str] = Field(default_factory=list, description="Titles held.")
    positions: list[Position] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    accomplishments: list[str] = Field(default_factory=list)
    domains: list[str] = Field(
        default_factory=list,
        description="Industries/domains explicitly mentioned (e.g. 'healthcare', 'fintech').",
    )
    years_of_experience: float | None = Field(
        None,
        ge=0,
        le=70,
        description="Total professional experience in years, only if derivable from the resume.",
    )

    @field_validator(
        "skills", "technologies", "certifications", "job_titles", "accomplishments", "domains"
    )
    @classmethod
    def _dedupe_strip(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in values:
            item = " ".join(raw.split())
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                out.append(item)
        return out


class ParsedResume(BaseModel):
    """Output of the resume parser: profile + provenance and warnings."""

    profile: ResumeProfile
    text: str
    content_sha256: str
    file_type: ResumeFileType
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# User-approved search preferences and hard constraints
# --------------------------------------------------------------------------- #


class SalaryRequirement(BaseModel):
    minimum_annual: int | None = Field(None, ge=0)
    currency: str = Field("USD", min_length=3, max_length=3)
    # When True, a posting with no salary info is allowed (not a hard disqualifier).
    allow_unspecified: bool = True


class JobSearchPreferences(BaseModel):
    """Search criteria and hard constraints the user has explicitly approved."""

    model_config = ConfigDict(extra="forbid")

    preferred_titles: list[str] = Field(default_factory=list, max_length=25)
    keywords: list[str] = Field(default_factory=list, max_length=50)
    locations: list[str] = Field(default_factory=list, max_length=25)
    excluded_locations: list[str] = Field(default_factory=list, max_length=50)
    work_arrangements: list[WorkArrangement] = Field(
        default_factory=lambda: [
            WorkArrangement.REMOTE,
            WorkArrangement.HYBRID,
            WorkArrangement.ONSITE,
        ]
    )
    employment_types: list[EmploymentType] = Field(
        default_factory=lambda: [EmploymentType.FULL_TIME]
    )
    salary: SalaryRequirement = Field(default_factory=SalaryRequirement)
    excluded_companies: list[str] = Field(default_factory=list, max_length=200)
    excluded_keywords: list[str] = Field(default_factory=list, max_length=100)
    # Work authorization facts are only ever taken from explicit user input.
    authorized_to_work_countries: list[str] = Field(default_factory=list)
    requires_sponsorship: bool | None = None
    max_applications_per_run: int = Field(25, ge=0, le=200)

    @model_validator(mode="after")
    def _no_overlap(self) -> JobSearchPreferences:
        inc = {loc.lower() for loc in self.locations}
        exc = {loc.lower() for loc in self.excluded_locations}
        if inc & exc:
            raise ValueError(f"locations both included and excluded: {sorted(inc & exc)}")
        return self


# --------------------------------------------------------------------------- #
# Read models
# --------------------------------------------------------------------------- #


class ResumeWorkflowView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    resume_id: UUID
    status: WorkflowStatus
    initial_search_completed: bool
    initial_search_completed_at: AwareDatetime | None
    next_run_at: AwareDatetime | None
    last_execution_status: str | None
    last_execution_at: AwareDatetime | None


class ResumeView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    label: str
    current_version: int
    original_filename: str
    file_type: ResumeFileType
    profile: ResumeProfile
    preferences: JobSearchPreferences
    created_at: AwareDatetime
    updated_at: AwareDatetime
