"""User-approved answers to recurring application questions (spec §4).

These are the only source for sensitive questions (work authorisation, sponsorship,
clearance, demographics, ...) and for identity fields. Keys are canonical question
keys (see ``services/answers.py``) or ``custom:<slug>``. Values are never logged.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import Session

from job_agent.models.common import utcnow
from job_agent.observability.logging import get_logger
from job_agent.persistence.orm import ApprovedAnswer
from job_agent.services.answers import _CANONICAL, IDENTITY_KEYS, is_sensitive
from job_agent.services.errors import InvalidInput, NotFound

log = get_logger(__name__)

KNOWN_KEYS = frozenset(k for k, _ in _CANONICAL) | IDENTITY_KEYS
_CUSTOM = re.compile(r"^custom:[a-z0-9_]{1,80}$")
MAX_ANSWER_CHARS = 5000


def validate_key(key: str) -> str:
    key = key.strip()
    if key in KNOWN_KEYS or _CUSTOM.match(key):
        return key
    raise InvalidInput(f"unknown question key '{key[:80]}'; use a known key or custom:<slug>")


class ApprovedAnswerService:
    def __init__(self, session: Session) -> None:
        self.s = session

    def upsert(
        self, user_id: uuid.UUID, key: str, answer: str, *, label: str | None = None
    ) -> ApprovedAnswer:
        key = validate_key(key)
        value = answer.strip()
        if not value:
            raise InvalidInput("answer must not be empty")
        if len(value) > MAX_ANSWER_CHARS:
            raise InvalidInput(f"answer exceeds {MAX_ANSWER_CHARS} characters")
        row = self.s.scalar(
            sa.select(ApprovedAnswer).where(
                ApprovedAnswer.user_id == user_id, ApprovedAnswer.question_key == key
            )
        )
        sensitive = is_sensitive(key, label or key.replace("_", " "))
        if row is None:
            row = ApprovedAnswer(
                user_id=user_id, question_key=key, answer=value, is_sensitive=sensitive
            )
            self.s.add(row)
        else:
            row.answer = value
            row.is_sensitive = sensitive
            row.approved_at = utcnow()
        self.s.flush()
        log.info("approved_answer_saved", key=key, sensitive=sensitive)  # never the value
        return row

    def list(self, user_id: uuid.UUID) -> Sequence[ApprovedAnswer]:
        return self.s.scalars(
            sa.select(ApprovedAnswer)
            .where(ApprovedAnswer.user_id == user_id)
            .order_by(ApprovedAnswer.question_key)
        ).all()

    def delete(self, user_id: uuid.UUID, key: str) -> None:
        row = self.s.scalar(
            sa.select(ApprovedAnswer).where(
                ApprovedAnswer.user_id == user_id, ApprovedAnswer.question_key == key
            )
        )
        if row is None:
            raise NotFound("approved answer not found")
        self.s.delete(row)
        self.s.flush()
        log.info("approved_answer_deleted", key=key)
