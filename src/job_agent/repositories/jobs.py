"""Persistence of job postings and the 'already tracked for this resume' check."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from job_agent.models.job import JobPosting, JobSource
from job_agent.persistence.orm import Application, JobMatch, JobPostingRecord
from job_agent.services.deduplication import DedupedJob


class JobPostingRepository:
    def __init__(self, session: Session) -> None:
        self.s = session

    def get_by_source_id(self, source: JobSource, external_id: str) -> JobPostingRecord | None:
        return self.s.scalar(
            sa.select(JobPostingRecord).where(
                JobPostingRecord.source == source, JobPostingRecord.external_id == external_id
            )
        )

    def _existing_members(self, job: DedupedJob) -> list[JobPostingRecord]:
        conditions = [
            sa.and_(
                JobPostingRecord.source == p.source, JobPostingRecord.external_id == p.external_id
            )
            for p in job.all_postings
        ]
        return list(
            self.s.scalars(
                sa.select(JobPostingRecord)
                .where(sa.or_(*conditions))
                .order_by(JobPostingRecord.first_discovered_at)
            )
        )

    def resolve_canonical_key(self, job: DedupedJob) -> str:
        """Reuse the key already stored for this job so identity is stable across runs."""
        existing = self._existing_members(job)
        if existing:
            return existing[0].canonical_key
        stored = self.s.scalar(
            sa.select(JobPostingRecord.canonical_key)
            .where(JobPostingRecord.canonical_key.in_(job.member_keys or {job.canonical_key}))
            .order_by(JobPostingRecord.first_discovered_at)
            .limit(1)
        )
        return stored or job.canonical_key

    def persist_group(self, job: DedupedJob, *, now: datetime) -> JobPostingRecord:
        """Upsert every posting in the group under one canonical key.

        Returns the record of the representative posting.
        """
        key = self.resolve_canonical_key(job)
        job.canonical_key = key
        records = {(p.source, p.external_id): self._upsert(p, key, now) for p in job.all_postings}
        return records[(job.posting.source, job.posting.external_id)]

    def _upsert(self, posting: JobPosting, key: str, now: datetime) -> JobPostingRecord:
        record = self.get_by_source_id(posting.source, posting.external_id)
        if record is None:
            record = JobPostingRecord(
                source=posting.source,
                external_id=posting.external_id,
                canonical_key=key,
                first_discovered_at=now,
            )
            self._apply(record, posting, now, is_new=True)
            try:
                with self.s.begin_nested():
                    self.s.add(record)
                return record
            except IntegrityError:  # inserted concurrently by another run
                record = self.get_by_source_id(posting.source, posting.external_id)
                assert record is not None
        self._apply(record, posting, now, is_new=False)
        self.s.flush()
        return record

    @staticmethod
    def _apply(record: JobPostingRecord, p: JobPosting, now: datetime, *, is_new: bool) -> None:
        record.title = p.title
        record.company_name = p.company_name
        record.location = p.location
        record.description = p.description
        record.canonical_url = str(p.canonical_url)
        record.application_url = str(p.application_url) if p.application_url else None
        record.source_updated_at = p.updated_at
        record.normalized = p.model_dump(mode="json")
        record.last_seen_at = now
        # Publication time is immutable once known: a later edit or repost must not make
        # an old job look new. It is only filled in if it was previously unknown.
        if is_new or record.published_at is None:
            record.published_at = p.published_at
            record.publication_date_unverified = p.publication_date_unverified

    # ------------------------------------------------------------------ tracking

    def tracked_canonical_keys(self, resume_id: uuid.UUID) -> set[str]:
        """Canonical keys of jobs this resume has already matched or applied to."""
        applied = (
            sa.select(JobPostingRecord.canonical_key)
            .join(Application, Application.job_posting_id == JobPostingRecord.id)
            .where(Application.resume_id == resume_id)
        )
        matched = (
            sa.select(JobPostingRecord.canonical_key)
            .join(JobMatch, JobMatch.job_posting_id == JobPostingRecord.id)
            .where(JobMatch.resume_id == resume_id)
        )
        return set(self.s.scalars(sa.union(applied, matched)))

    def partition_untracked(
        self, resume_id: uuid.UUID, jobs: Iterable[DedupedJob]
    ) -> tuple[list[DedupedJob], list[DedupedJob]]:
        """Split jobs into (new for this resume, already tracked by this resume)."""
        tracked = self.tracked_canonical_keys(resume_id)
        new: list[DedupedJob] = []
        seen: list[DedupedJob] = []
        for job in jobs:
            keys = set(job.member_keys) | {job.canonical_key}
            keys |= {r.canonical_key for r in self._existing_members(job)}
            (seen if keys & tracked else new).append(job)
        return new, seen
