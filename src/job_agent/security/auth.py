"""API authentication and rate limiting.

* ``StaticTokenAuthenticator``: dev/test bearer tokens (refused when APP_ENV=prod by
  settings validation); constant-time comparison.
* ``JWTAuthenticator``: verifies bearer JWTs from an identity provider (signature,
  expiry, optional issuer/audience; ``sub`` required). The ``none`` algorithm is never
  accepted.

The authenticated *subject* is the stable user identity; ``users.external_subject``
stores it and every query is scoped to the resulting user id.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jwt

from job_agent.config import ApiAuthMode, Settings


class Authenticator(Protocol):
    def authenticate(self, token: str) -> str | None:
        """Return the subject for a valid token, else None."""
        ...


class StaticTokenAuthenticator:
    def __init__(self, tokens_by_subject: dict[str, str]) -> None:
        self._tokens = [(t.encode(), s) for s, t in tokens_by_subject.items()]

    def authenticate(self, token: str) -> str | None:
        candidate = token.encode()
        match: str | None = None
        for known, subject in self._tokens:  # compare all: no early exit timing signal
            if hmac.compare_digest(known, candidate):
                match = subject
        return match


class JWTAuthenticator:
    def __init__(
        self,
        *,
        key: str,
        algorithms: list[str],
        issuer: str | None = None,
        audience: str | None = None,
        leeway_seconds: int = 30,
    ) -> None:
        if any(a.lower() == "none" for a in algorithms):
            raise ValueError("the 'none' algorithm is not allowed")
        self._key = key
        self._algorithms = algorithms
        self._issuer = issuer
        self._audience = audience
        self._leeway = leeway_seconds

    def authenticate(self, token: str) -> str | None:
        try:
            claims = jwt.decode(
                token,
                self._key,
                algorithms=self._algorithms,
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={"require": ["exp", "sub"], "verify_aud": self._audience is not None},
            )
        except jwt.PyJWTError:
            return None
        subject = claims.get("sub")
        return subject if isinstance(subject, str) and subject else None


def build_authenticator(settings: Settings) -> Authenticator:
    if settings.api_auth_mode is ApiAuthMode.JWT:
        key = settings.api_jwt_public_key or (
            settings.api_jwt_secret.get_secret_value() if settings.api_jwt_secret else ""
        )
        return JWTAuthenticator(
            key=key,
            algorithms=settings.api_jwt_algorithms,
            issuer=settings.api_jwt_issuer,
            audience=settings.api_jwt_audience,
        )
    return StaticTokenAuthenticator(
        {s: t.get_secret_value() for s, t in settings.api_static_tokens.items()}
    )


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """In-process token bucket per key (per API instance; put a shared limiter at the
    gateway for multi-instance deployments)."""

    def __init__(
        self,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 100_000,
    ) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def acquire(self, key: str) -> float | None:
        """Consume one request. Returns None if allowed, else seconds to wait."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._buckets.clear()  # bounded memory under key floods
                bucket = self._buckets[key] = _Bucket(self.capacity, now)
            bucket.tokens = min(self.capacity, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return None
            return (1 - bucket.tokens) / self.rate
