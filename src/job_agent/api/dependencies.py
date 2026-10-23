"""Request dependencies: container access, authentication, rate limiting, paging."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from job_agent.api.container import AppContainer
from job_agent.repositories.resumes import UserRepository

bearer = HTTPBearer(auto_error=False, description="Bearer token (static dev token or JWT)")


def get_container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container


Container = Annotated[AppContainer, Depends(get_container)]


@dataclass(frozen=True)
class AuthUser:
    id: uuid.UUID
    subject: str


def _too_many(wait: float) -> HTTPException:
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        "rate limit exceeded",
        headers={"Retry-After": str(max(1, int(wait) + 1))},
    )


def current_user(
    request: Request,
    container: Container,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> AuthUser:
    subject = (
        container.authenticator.authenticate(credentials.credentials)
        if credentials and credentials.scheme.lower() == "bearer"
        else None
    )
    if subject is None:
        # Throttle repeated failures per client to slow token guessing.
        ip = request.client.host if request.client else "unknown"
        wait = container.limiter.acquire(f"authfail:{ip}")
        if wait is not None:
            raise _too_many(wait)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid or missing credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    wait = container.limiter.acquire(f"user:{subject}")
    if wait is not None:
        raise _too_many(wait)
    with container.session() as s:
        user = UserRepository(s).get_or_create(subject)
        return AuthUser(id=user.id, subject=subject)


CurrentUser = Annotated[AuthUser, Depends(current_user)]


@dataclass(frozen=True)
class Paging:
    limit: int
    offset: int


def paging(
    container: Container,
    limit: Annotated[int, Query(ge=1)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Paging:
    return Paging(limit=min(limit, container.settings.api_max_page_size), offset=offset)


Page = Annotated[Paging, Depends(paging)]

CHUNK = 64 * 1024


async def read_upload(request: Request, file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload with a hard size cap (never buffers more than the limit)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + CHUNK:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"upload exceeds {max_bytes} bytes")
    data = bytearray()
    while chunk := await file.read(CHUNK):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, f"upload exceeds {max_bytes} bytes"
            )
    return bytes(data)
