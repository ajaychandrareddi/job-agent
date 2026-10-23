"""Job-matching schemas.

``LLMMatchAssessment`` is the structured-output target for the LLM. The final
``MatchDecision`` is produced by deterministic Python code (threshold + hard
constraints), never by the LLM.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from job_agent.models.common import AwareDatetime


class Recommendation(StrEnum):
    APPLY = "APPLY"
    SKIP = "SKIP"
    REVIEW = "REVIEW"


class CriterionScores(BaseModel):
    """Each criterion scored 0-100; the weighted total is computed in code."""

    model_config = ConfigDict(extra="forbid")

    required_skills: float = Field(ge=0, le=100)
    experience_seniority: float = Field(ge=0, le=100)
    responsibilities_alignment: float = Field(ge=0, le=100)
    preferred_skills: float = Field(ge=0, le=100)
    education_certifications: float = Field(ge=0, le=100)
    domain_experience: float = Field(ge=0, le=100)


class LLMMatchAssessment(BaseModel):
    """Semantic evaluation produced by the LLM."""

    model_config = ConfigDict(extra="forbid")

    criterion_scores: CriterionScores
    matching_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    resume_evidence: list[str] = Field(
        default_factory=list, description="Short quotes from the resume supporting the scores."
    )
    experience_alignment: str = ""
    potential_disqualifiers: list[str] = Field(default_factory=list)
    explanation: str = Field("", max_length=2000)
    insufficient_evidence: bool = Field(
        False, description="True if the posting or resume lacks information to judge fit."
    )


class HardConstraintViolation(BaseModel):
    code: str
    detail: str


class MatchDecision(BaseModel):
    """Deterministic final result persisted to ``job_matches``."""

    overall_score: float = Field(ge=0, le=100)
    criterion_scores: CriterionScores
    matching_skills: list[str]
    missing_required_skills: list[str]
    resume_evidence: list[str]
    experience_alignment: str
    potential_disqualifiers: list[str]
    hard_constraint_violations: list[HardConstraintViolation] = Field(default_factory=list)
    # Reasons a qualifying score still cannot be auto-applied (routes to REVIEW).
    review_reasons: list[str] = Field(default_factory=list)
    explanation: str
    recommendation: Recommendation
    eligible_for_auto_apply: bool
    threshold: float
    evaluated_at: AwareDatetime
    # Set when the LLM evaluation failed; the decision is then REVIEW with score 0.
    evaluation_error: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
