"""Application settings, loaded from environment variables / .env files."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class AppEnv(StrEnum):
    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PROD = "prod"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    VERTEX = "vertex"
    ANTHROPIC = "anthropic"
    FAKE = "fake"


class MissingPublicationDatePolicy(StrEnum):
    """What to do with postings whose publication date cannot be verified."""

    EXCLUDE_FROM_AUTO_APPLY = "exclude_from_auto_apply"  # default: flag + route to review
    SKIP = "skip"


class ApiAuthMode(StrEnum):
    STATIC_TOKEN = "static_token"  # noqa: S105  dev/test only; refused when APP_ENV=prod
    JWT = "jwt"


class MissedRunPolicy(StrEnum):
    """How recurring runs treat time not covered because runs were missed (outage)."""

    LAST_WINDOW_ONLY = "last_window_only"  # default: search only the normal 24h window
    COVER_GAP = "cover_gap"  # search back to the last covered window, capped


class MissedRunAction(StrEnum):
    """What the scheduler does on start-up when a daily run was missed beyond the
    misfire grace period (e.g. after an outage)."""

    RUN_ONCE = "run_once"  # one catch-up run now (never one run per missed day)
    SKIP = "skip"  # wait for the next scheduled time


class MatchingWeights(BaseModel):
    """Rubric weights (percent). Must sum to 100.

    Set via env/.env as MATCHING_WEIGHTS__<NAME>, e.g. MATCHING_WEIGHTS__REQUIRED_SKILLS=40.
    """

    model_config = ConfigDict(extra="forbid")

    required_skills: float = 30
    experience_seniority: float = 25
    responsibilities_alignment: float = 20
    preferred_skills: float = 10
    education_certifications: float = 5
    domain_experience: float = 10

    @model_validator(mode="after")
    def _sum_to_100(self) -> MatchingWeights:
        total = sum(self.model_dump().values())
        if abs(total - 100) > 1e-6:
            raise ValueError(f"matching weights must sum to 100, got {total}")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    app_env: AppEnv = AppEnv.DEV
    database_url: str = "postgresql+psycopg://job_agent:job_agent@localhost:5432/job_agent"
    db_echo: bool = False

    llm_provider: LLMProvider = LLMProvider.FAKE
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"

    job_search_timezone: str = "America/New_York"
    job_search_hour: int = Field(17, ge=0, le=23)
    job_search_minute: int = Field(0, ge=0, le=59)
    initial_search_days: int = Field(30, ge=1, le=90)
    recurring_search_hours: int = Field(24, ge=1, le=168)
    job_match_threshold: float = Field(60, ge=0, le=100)
    max_concurrent_job_searches: int = Field(3, ge=1, le=16)
    max_application_retries: int = Field(2, ge=0, le=10)
    missing_publication_date_policy: MissingPublicationDatePolicy = (
        MissingPublicationDatePolicy.EXCLUDE_FROM_AUTO_APPLY
    )
    missed_run_policy: MissedRunPolicy = MissedRunPolicy.LAST_WINDOW_ONLY
    max_catch_up_hours: int = Field(72, ge=1, le=24 * 30)
    future_date_tolerance_seconds: int = Field(300, ge=0)

    # Job sources. Real access needs approved credentials; see docs/integrations.md.
    linkedin_enabled: bool = True
    indeed_enabled: bool = True
    ziprecruiter_enabled: bool = True
    source_timeout_seconds: float = Field(30.0, gt=0)
    source_max_retries: int = Field(2, ge=0, le=5)
    source_max_retry_after_seconds: float = Field(120.0, ge=0)
    source_max_results: int = Field(500, ge=1, le=5000)
    max_description_chars: int = Field(20_000, ge=1000)

    # Matching
    match_llm_retries: int = Field(1, ge=0, le=3)
    max_concurrent_matches: int = Field(4, ge=1, le=32)
    match_max_description_chars: int = Field(12_000, ge=1000)
    # Max gap between the LLM's required-skills score and deterministic skill coverage
    # (when the source lists required skills) before the match is sent to review.
    match_consistency_tolerance: float = Field(40, ge=0, le=100)

    scheduler_misfire_grace_seconds: int = Field(3600, ge=0)
    scheduler_max_workers: int = Field(4, ge=1)
    scheduler_missed_run_action: MissedRunAction = MissedRunAction.RUN_ONCE
    scheduler_reconcile_seconds: int = Field(300, ge=10)
    # Lease that stops two scheduler instances from running the same resume; renewed
    # every third of its length while a run is in progress.
    scheduler_run_lease_seconds: int = Field(900, ge=30)
    scheduler_maintenance_hour: int = Field(3, ge=0, le=23)

    # Applications
    generate_application_answers: bool = True
    reconcile_interval_seconds: int = Field(900, ge=60)
    reconcile_max_attempts: int = Field(5, ge=1, le=50)
    # A "not found" from the platform is trusted only after this long (indexing delay).
    reconcile_not_found_min_age_seconds: int = Field(900, ge=0)
    submit_timeout_seconds: float = Field(120.0, gt=0)
    # Greenhouse Job Board API: only for boards whose EMPLOYER issued you a key.
    # Format: "board_token:api_key,board2:key2". Empty = integration disabled.
    greenhouse_board_keys: Annotated[dict[str, SecretStr], NoDecode] = Field(default_factory=dict)
    greenhouse_api_base: str = "https://boards-api.greenhouse.io/v1"

    max_upload_bytes: int = Field(5 * 1024 * 1024, ge=1024)
    resume_storage_dir: Path = Path("./var/resumes")
    resume_retention_days: int = Field(365, ge=1)
    application_retention_days: int = Field(730, ge=1)

    api_auth_mode: ApiAuthMode = ApiAuthMode.STATIC_TOKEN
    # static_token (dev/test only): env format "token1:subject1,token2:subject2". Stored
    # as subject -> token so tokens stay masked (SecretStr) in any settings dump. Each
    # subject is a stable user identity (e.g. the IdP "sub"); users are created on use.
    api_static_tokens: Annotated[dict[str, SecretStr], NoDecode] = Field(default_factory=dict)
    # jwt: verify bearer JWTs from your identity provider. Use a shared secret (HS256)
    # or the provider's public key (RS256/ES256). The "sub" claim is the user identity.
    api_jwt_secret: SecretStr | None = None
    api_jwt_public_key: str | None = None
    api_jwt_algorithms: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["RS256"])
    api_jwt_issuer: str | None = None
    api_jwt_audience: str | None = None
    api_rate_limit_per_minute: int = Field(120, ge=1)
    api_max_page_size: int = Field(100, ge=1, le=1000)
    # Run the scheduler inside the API process (single-process deployments). Lifecycle
    # changes then reach the scheduler immediately instead of on the next reconcile.
    api_run_scheduler: bool = False

    log_level: str = "INFO"
    log_json: bool = True

    matching_weights: MatchingWeights = Field(default_factory=MatchingWeights)

    @field_validator("job_search_timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {v}") from exc
        return v

    @field_validator("api_static_tokens", mode="before")
    @classmethod
    def _parse_tokens(cls, v: object) -> object:
        if isinstance(v, str):
            pairs = [p.strip() for p in v.split(",") if p.strip()]
            out: dict[str, str] = {}
            for pair in pairs:
                token, _, subject = pair.partition(":")
                if not token or not subject:
                    raise ValueError("API_STATIC_TOKENS entries must be 'token:subject'")
                if len(token) < 16:
                    raise ValueError("API_STATIC_TOKENS tokens must be at least 16 characters")
                out[subject] = token
            return out
        return v

    @field_validator("api_jwt_algorithms", mode="before")
    @classmethod
    def _parse_algorithms(cls, v: object) -> object:
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v

    @model_validator(mode="after")
    def _auth_is_safe(self) -> Settings:
        if self.api_auth_mode is ApiAuthMode.STATIC_TOKEN and self.app_env is AppEnv.PROD:
            raise ValueError("API_AUTH_MODE=static_token is not allowed when APP_ENV=prod")
        if self.api_auth_mode is ApiAuthMode.JWT:
            if not (self.api_jwt_secret or self.api_jwt_public_key):
                raise ValueError("API_AUTH_MODE=jwt needs API_JWT_SECRET or API_JWT_PUBLIC_KEY")
            if "none" in {a.lower() for a in self.api_jwt_algorithms}:
                raise ValueError("the 'none' JWT algorithm is not allowed")
        return self

    @field_validator("greenhouse_board_keys", mode="before")
    @classmethod
    def _parse_board_keys(cls, v: object) -> object:
        if isinstance(v, str):
            out: dict[str, str] = {}
            for pair in (p.strip() for p in v.split(",") if p.strip()):
                board, _, key = pair.partition(":")
                if not board or not key:
                    raise ValueError("GREENHOUSE_BOARD_KEYS entries must be 'board_token:api_key'")
                out[board] = key
            return out
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.job_search_timezone)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
