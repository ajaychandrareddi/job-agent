"""Persistence of match decisions (one per resume version x job posting)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from job_agent.models.matching import MatchDecision, Recommendation
from job_agent.persistence.orm import JobMatch


class JobMatchRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get(
        self, resume_id: uuid.UUID, resume_version: int, job_posting_id: uuid.UUID
    ) -> JobMatch | None:
        return self.s.scalar(
            sa.select(JobMatch).where(
                JobMatch.resume_id == resume_id,
                JobMatch.resume_version == resume_version,
                JobMatch.job_posting_id == job_posting_id,
            )
        )

    def save(
        self,
        *,
        resume_id: uuid.UUID,
        resume_version: int,
        job_posting_id: uuid.UUID,
        decision: MatchDecision,
        execution_id: uuid.UUID | None = None,
    ) -> JobMatch:
        """Insert or replace the evaluation for this resume version and job."""
        match = self.get(resume_id, resume_version, job_posting_id)
        if match is None:
            match = JobMatch(
                resume_id=resume_id, resume_version=resume_version, job_posting_id=job_posting_id
            )
            self._apply(match, decision, execution_id)
            try:
                with self.s.begin_nested():
                    self.s.add(match)
                return match
            except IntegrityError:  # concurrent evaluation of the same job
                match = self.get(resume_id, resume_version, job_posting_id)
                assert match is not None
        self._apply(match, decision, execution_id)
        self.s.flush()
        return match

    @staticmethod
    def _apply(m: JobMatch, d: MatchDecision, execution_id: uuid.UUID | None) -> None:
        m.score = d.overall_score
        m.criterion_scores = d.criterion_scores.model_dump()
        m.matching_skills = d.matching_skills
        m.missing_skills = d.missing_required_skills
        m.evidence = d.resume_evidence
        m.recommendation = d.recommendation
        m.decision = d.model_dump(mode="json")
        m.evaluated_at = d.evaluated_at
        if execution_id is not None:
            m.execution_id = execution_id

    def list_for_resume(
        self,
        resume_id: uuid.UUID,
        *,
        recommendation: Recommendation | None = None,
        min_score: float | None = None,
    ) -> Sequence[JobMatch]:
        stmt = sa.select(JobMatch).where(JobMatch.resume_id == resume_id)
        if recommendation is not None:
            stmt = stmt.where(JobMatch.recommendation == recommendation)
        if min_score is not None:
            stmt = stmt.where(JobMatch.score >= min_score)
        return self.s.scalars(stmt.order_by(JobMatch.score.desc(), JobMatch.evaluated_at)).all()
