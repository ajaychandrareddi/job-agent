"""Deterministic per-resume workflow lifecycle rules.

    DRAFT  --activate-->  ACTIVE
    ACTIVE --pause----->  PAUSED  --resume--> ACTIVE
    ACTIVE|PAUSED|DRAFT --stop--> STOPPED --activate--> ACTIVE (explicit reactivation)

Repeating an operation that is already in effect is a no-op (idempotent).
"""

from __future__ import annotations

from enum import StrEnum

from job_agent.models.resume import WorkflowStatus as S


class LifecycleAction(StrEnum):
    ACTIVATE = "activate"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"


class LifecycleError(ValueError):
    """The requested lifecycle action is not allowed from the current state."""


_RULES: dict[LifecycleAction, tuple[frozenset[S], S]] = {
    LifecycleAction.ACTIVATE: (frozenset({S.DRAFT, S.STOPPED}), S.ACTIVE),
    LifecycleAction.PAUSE: (frozenset({S.ACTIVE}), S.PAUSED),
    LifecycleAction.RESUME: (frozenset({S.PAUSED}), S.ACTIVE),
    LifecycleAction.STOP: (frozenset({S.DRAFT, S.ACTIVE, S.PAUSED}), S.STOPPED),
}


def next_status(current: S, action: LifecycleAction) -> tuple[S, bool]:
    """Return (new_status, changed). Raises LifecycleError if not allowed."""
    allowed_from, target = _RULES[action]
    if current == target:
        return current, False
    if current not in allowed_from:
        hint = " (use 'resume' for a paused workflow)" if current is S.PAUSED else ""
        raise LifecycleError(f"cannot {action.value} a workflow that is {current.value}{hint}")
    return target, True


def may_run(status: S) -> bool:
    """Only ACTIVE workflows may search or submit applications."""
    return status is S.ACTIVE
