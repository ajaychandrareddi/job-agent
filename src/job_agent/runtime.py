"""Shared wiring for the API and scheduler processes."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from job_agent.config import LLMProvider, Settings
from job_agent.integrations.greenhouse import GreenhouseIntegration
from job_agent.job_sources.registry import build_adapters
from job_agent.llm.factory import build_chat_model
from job_agent.persistence.file_store import ResumeFileStore
from job_agent.services.answers import AnswerGenerator, LLMAnswerGenerator
from job_agent.services.application_service import SubmissionIntegration, SubmissionRegistry
from job_agent.services.job_matcher import (
    Assessment,
    AssessmentFailed,
    JobMatcher,
    LLMMatchAssessor,
    MatchAssessor,
)
from job_agent.services.resume_extractors import LLMResumeExtractor
from job_agent.services.resume_parser import ResumeParser
from job_agent.workflows.deps import WorkflowDeps


class NoLLMAssessor:
    """Used when no LLM provider is configured: every job goes to REVIEW (never APPLY)."""

    model_name: str | None = "none"

    async def assess(self, profile_json: str, posting_json: str) -> Assessment:
        raise AssessmentFailed("no LLM provider is configured (LLM_PROVIDER=fake)")


def llm_enabled(settings: Settings) -> bool:
    return settings.llm_provider is not LLMProvider.FAKE


def build_matcher(settings: Settings) -> JobMatcher:
    assessor: MatchAssessor = (
        LLMMatchAssessor(build_chat_model(settings), model_name=settings.llm_model or None)
        if llm_enabled(settings)
        else NoLLMAssessor()
    )
    return JobMatcher(assessor, settings)


def build_parser(settings: Settings) -> ResumeParser:
    extractor = LLMResumeExtractor(build_chat_model(settings)) if llm_enabled(settings) else None
    return ResumeParser(extractor, max_upload_bytes=settings.max_upload_bytes)


def build_answer_generator(settings: Settings) -> AnswerGenerator | None:
    if not llm_enabled(settings) or not settings.generate_application_answers:
        return None  # free-text questions then go to review
    return LLMAnswerGenerator(build_chat_model(settings))


def build_submissions(settings: Settings, file_store: ResumeFileStore) -> SubmissionRegistry:
    """Only integrations the platform permits. Greenhouse is enabled per board, and only
    for boards whose employer issued a key (GREENHOUSE_BOARD_KEYS)."""
    integrations: list[SubmissionIntegration] = []
    if settings.greenhouse_board_keys:
        integrations.append(
            GreenhouseIntegration(
                {b: k.get_secret_value() for b, k in settings.greenhouse_board_keys.items()},
                file_store=file_store,
                api_base=settings.greenhouse_api_base,
            )
        )
    return SubmissionRegistry(integrations)


def build_workflow_deps(
    settings: Settings, factory: sessionmaker[Session], file_store: ResumeFileStore
) -> WorkflowDeps:
    return WorkflowDeps(
        session_factory=factory,
        settings=settings,
        adapters=build_adapters(settings),  # approved platform clients plug in here
        matcher=build_matcher(settings),
        submissions=build_submissions(settings, file_store),
        submit_timeout_seconds=settings.submit_timeout_seconds,
        answer_generator=build_answer_generator(settings),
    )
