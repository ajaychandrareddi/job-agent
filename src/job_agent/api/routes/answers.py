"""User-approved answers to recurring application questions."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from job_agent.api.dependencies import Container, CurrentUser
from job_agent.api.schemas import AnswerIn, AnswerOut
from job_agent.persistence.orm import ApprovedAnswer
from job_agent.services.approved_answers import ApprovedAnswerService

router = APIRouter(prefix="/api/v1/answers", tags=["approved answers"])

MASK = "********"


def _out(row: ApprovedAnswer) -> AnswerOut:
    return AnswerOut(
        key=row.question_key,
        sensitive=row.is_sensitive,
        answer=MASK if row.is_sensitive else row.answer,
        approved_at=row.approved_at,
    )


@router.get("", response_model=list[AnswerOut])
def list_answers(user: CurrentUser, c: Container) -> list[AnswerOut]:
    with c.session() as s:
        return [_out(r) for r in ApprovedAnswerService(s).list(user.id)]


@router.put("/{key}", response_model=AnswerOut)
def put_answer(key: str, body: AnswerIn, user: CurrentUser, c: Container) -> AnswerOut:
    """Approve an answer. Keys: e.g. ``first_name``, ``email``, ``work_authorization_us``,
    ``requires_sponsorship``, ``salary_expectation`` or ``custom:<slug>``."""
    with c.session() as s:
        return _out(ApprovedAnswerService(s).upsert(user.id, key, body.answer, label=body.label))


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_answer(key: str, user: CurrentUser, c: Container) -> Response:
    with c.session() as s:
        ApprovedAnswerService(s).delete(user.id, key)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
