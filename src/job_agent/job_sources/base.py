"""Common interface for job-source adapters.

Adapters must:
* use only official APIs / approved integrations permitted by the platform's terms;
* never bypass authentication, CAPTCHAs, rate limits or anti-automation controls
  (a rate-limit response is honoured: we wait or give up, never evade);
* never fabricate posting dates, URLs or descriptions (unknown fields stay ``None``);
* report failures and access limitations in ``SourceSearchResult`` instead of raising,
  so one failing source does not fail the whole search.

Real platform access is plugged in through ``PartnerFeedClient``: once a platform
grants API access, a client maps that platform's actual responses to ``RawPosting``.
Until then the adapter reports ``SourceStatus.UNAVAILABLE`` with a manual search link.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from job_agent.models.job import (
    JobSource,
    RawPosting,
    SearchCriteria,
    SourceSearchResult,
    SourceStatus,
)
from job_agent.observability import metrics
from job_agent.observability.logging import get_logger
from job_agent.services.normalization import NormalizationError, normalize_raw_posting

log = get_logger(__name__)

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], datetime]


class SourceError(Exception):
    """Base class for errors raised by platform clients."""


class TransientSourceError(SourceError):
    """Temporary failure (5xx, connection reset). Retried with backoff."""


class RateLimited(SourceError):
    """The platform asked us to slow down. Honoured, never circumvented."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__(f"rate limited (retry after {retry_after}s)")
        self.retry_after = retry_after


class SourceAuthError(SourceError):
    """Credentials missing, invalid or revoked. Not retried."""


class PartnerFeedClient(Protocol):
    """Permitted, credentialed access to one platform's job data."""

    async def search(
        self, criteria: SearchCriteria, posted_after: datetime, *, max_results: int
    ) -> list[RawPosting]: ...


class JobSourceAdapter(ABC):
    source: JobSource

    @abstractmethod
    async def search_jobs(
        self, search_criteria: SearchCriteria, posted_after: datetime
    ) -> SourceSearchResult:
        """Return postings published at or after ``posted_after`` (timezone-aware UTC).

        Sources filter coarsely; callers must still apply the exact date window.
        """

    @property
    def supports_automated_search(self) -> bool:
        return True

    @property
    def supports_automated_application(self) -> bool:
        return False


class PartnerFeedAdapter(JobSourceAdapter):
    """Shared retry, rate-limit, timeout and normalisation logic."""

    #: Human-readable reason shown when no permitted client is configured.
    unavailable_reason: str = "no approved API access is configured for this source"
    #: Timezone the platform documents for naive timestamps (None = unknown).
    source_tz: ZoneInfo | None = None

    def __init__(
        self,
        client: PartnerFeedClient | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        max_retry_after_seconds: float = 120.0,
        max_results: int = 500,
        max_description_chars: int = 20_000,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._max_retry_after = max_retry_after_seconds
        self._max_results = max_results
        self._max_description_chars = max_description_chars
        self._sleep = sleep
        self._clock = clock

    @property
    def supports_automated_search(self) -> bool:
        return self._client is not None

    @abstractmethod
    def manual_search_url(self, criteria: SearchCriteria, posted_after: datetime) -> str:
        """Public search page the user can open to continue manually."""

    async def search_jobs(
        self, search_criteria: SearchCriteria, posted_after: datetime
    ) -> SourceSearchResult:
        if posted_after.tzinfo is None:
            raise ValueError("posted_after must be timezone-aware")
        manual_url = self.manual_search_url(search_criteria, posted_after)
        if self._client is None:
            log.info("job_source_unavailable", source=self.source.value)
            return SourceSearchResult(
                source=self.source,
                status=SourceStatus.UNAVAILABLE,
                limitation=self.unavailable_reason,
                manual_search_url=manual_url,
            )

        started = time.monotonic()
        attempts = 0
        error: str | None = None
        raw: list[RawPosting] | None = None
        while attempts <= self._max_retries:
            attempts += 1
            try:
                raw = await asyncio.wait_for(
                    self._client.search(
                        search_criteria, posted_after, max_results=self._max_results
                    ),
                    timeout=self._timeout,
                )
                break
            except RateLimited as exc:
                wait = exc.retry_after if exc.retry_after is not None else 2.0**attempts
                log.warning(
                    "job_source_rate_limited",
                    source=self.source.value,
                    retry_after=exc.retry_after,
                    attempt=attempts,
                )
                error = "rate limited by source"
                metrics.source_rate_limited.add(1, {"source": self.source.value})
                if wait > self._max_retry_after or attempts > self._max_retries:
                    break
                await self._sleep(wait)
            except (TransientSourceError, TimeoutError) as exc:
                error = f"temporary failure: {type(exc).__name__}"
                log.warning(
                    "job_source_transient_error",
                    source=self.source.value,
                    attempt=attempts,
                    error=type(exc).__name__,
                )
                if attempts > self._max_retries:
                    break
                await self._sleep(2.0 ** (attempts - 1))
            except SourceAuthError:
                error = "authentication with source failed; check credentials"
                break
            except Exception as exc:  # unknown client failure: report, do not crash the run
                error = f"unexpected error: {type(exc).__name__}"
                log.error("job_source_error", source=self.source.value, error=type(exc).__name__)
                break

        duration = time.monotonic() - started
        if raw is None:
            return SourceSearchResult(
                source=self.source,
                status=SourceStatus.FAILED,
                error=error,
                manual_search_url=manual_url,
                attempts=attempts,
                duration_seconds=duration,
            )

        now = self._clock()
        postings, reasons = [], []
        for record in raw[: self._max_results]:
            try:
                postings.append(
                    normalize_raw_posting(
                        record,
                        self.source,
                        now=now,
                        max_description_chars=self._max_description_chars,
                        source_tz=self.source_tz,
                    )
                )
            except NormalizationError as exc:
                reasons.append(str(exc))
        log.info(
            "job_source_searched",
            source=self.source.value,
            raw=len(raw),
            normalized=len(postings),
            rejected=len(reasons),
            attempts=attempts,
        )
        return SourceSearchResult(
            source=self.source,
            status=SourceStatus.OK,
            postings=postings,
            raw_count=len(raw),
            rejected_count=len(reasons),
            rejection_reasons=sorted(set(reasons))[:20],
            manual_search_url=manual_url,
            attempts=attempts,
            duration_seconds=duration,
        )
