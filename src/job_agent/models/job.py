"""Normalised job-posting model shared by every job-source adapter."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from job_agent.models.common import AwareDatetime
from job_agent.models.resume import EmploymentType, WorkArrangement


class JobSource(StrEnum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    ZIPRECRUITER = "ziprecruiter"


class ApplicationMethod(StrEnum):
    API = "api"  # a permitted, programmatic application integration
    EASY_APPLY = "easy_apply"  # platform-native apply; only automated if the platform permits it
    EXTERNAL_ATS = "external_ats"  # company career site / ATS
    EMAIL = "email"
    MANUAL_ONLY = "manual_only"
    UNKNOWN = "unknown"


class ApplicationAvailability(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class PublicationPrecision(StrEnum):
    EXACT = "exact"  # full timestamp from the source
    DAY = "day"  # date only; posted some time during that day
    APPROXIMATE = "approximate"  # relative text such as "3 days ago"


class SalaryRange(BaseModel):
    minimum: float | None = Field(None, ge=0)
    maximum: float | None = Field(None, ge=0)
    currency: str = "USD"
    period: str = Field("year", pattern="^(year|month|week|day|hour)$")

    @model_validator(mode="after")
    def _ordered(self) -> SalaryRange:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("salary minimum exceeds maximum")
        return self


class SearchCriteria(BaseModel):
    """What adapters receive. Derived from a resume's approved preferences."""

    titles: list[str]
    keywords: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    work_arrangements: list[WorkArrangement] = Field(default_factory=list)
    employment_types: list[EmploymentType] = Field(default_factory=list)


class JobPosting(BaseModel):
    """A posting as reported by a source. Fields that the source did not
    provide stay ``None`` — adapters must never fabricate them."""

    model_config = ConfigDict(extra="forbid")

    source: JobSource
    external_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    company_name: str = Field(min_length=1, max_length=500)
    location: str | None = None
    work_arrangement: WorkArrangement = WorkArrangement.UNKNOWN
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    description: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_requirements: str | None = None
    min_years_experience: float | None = Field(None, ge=0)
    education_requirements: list[str] = Field(default_factory=list)
    certification_requirements: list[str] = Field(default_factory=list)
    salary: SalaryRange | None = None
    # Earliest possible publication time stated by the source (never the update time).
    published_at: AwareDatetime | None = None
    # Latest possible publication time; equals published_at for EXACT precision.
    published_at_latest: AwareDatetime | None = None
    published_at_precision: PublicationPrecision = PublicationPrecision.EXACT
    updated_at: AwareDatetime | None = None
    canonical_url: HttpUrl
    application_url: HttpUrl | None = None
    application_method: ApplicationMethod = ApplicationMethod.UNKNOWN
    application_availability: ApplicationAvailability = ApplicationAvailability.UNKNOWN
    # True when the adapter could not verify the publication date.
    publication_date_unverified: bool = False
    raw_metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _flag_missing_date(self) -> JobPosting:
        if self.published_at is None:
            self.publication_date_unverified = True
            self.published_at_latest = None
        elif self.published_at_latest is None:
            self.published_at_latest = self.published_at
        elif self.published_at_latest < self.published_at:
            raise ValueError("published_at_latest precedes published_at")
        return self

    def is_published_within(self, start: datetime, end: datetime) -> bool:
        """True only if the whole possible publication interval lies in [start, end]."""
        if self.published_at is None or self.published_at_latest is None:
            return False
        return start <= self.published_at and self.published_at_latest <= end


class SourceStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"  # error while searching (network, auth, rate limit exhausted)
    UNAVAILABLE = "unavailable"  # no permitted automated access configured


class SourceSearchResult(BaseModel):
    """Result of one adapter call. Failures are reported, never hidden or papered over."""

    source: JobSource
    status: SourceStatus = SourceStatus.OK
    postings: list[JobPosting] = Field(default_factory=list)
    raw_count: int = 0
    rejected_count: int = 0
    rejection_reasons: list[str] = Field(default_factory=list)
    error: str | None = None
    # Set when the source cannot be searched automatically under its terms.
    limitation: str | None = None
    manual_search_url: HttpUrl | None = None
    attempts: int = 0
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status is SourceStatus.OK


class RawPosting(BaseModel):
    """Source-neutral record handed over by a platform client before normalisation.

    A platform client (built once API access is approved) maps the platform's real
    response into this shape. Every field except the identifiers is optional so that
    clients never have to invent values they did not receive.
    """

    model_config = ConfigDict(extra="ignore")

    external_id: str
    title: str | None = None
    company_name: str | None = None
    location: str | None = None
    remote_type: str | None = None  # platform's own label, e.g. "remote", "hybrid"
    employment_type: str | None = None
    description: str | None = None  # may contain HTML
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_requirements: str | None = None
    education_requirements: list[str] = Field(default_factory=list)
    certification_requirements: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    salary_text: str | None = None
    # ISO-8601 string, epoch seconds/milliseconds, aware datetime, or relative text.
    published_at: object | None = None
    updated_at: object | None = None
    url: str | None = None
    apply_url: str | None = None
    apply_method: str | None = None
    is_open: bool | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
