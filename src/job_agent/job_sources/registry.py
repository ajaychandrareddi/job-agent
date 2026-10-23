"""Build the configured set of adapters."""

from __future__ import annotations

from collections.abc import Mapping

from job_agent.config import Settings
from job_agent.job_sources.base import PartnerFeedAdapter, PartnerFeedClient
from job_agent.job_sources.indeed import IndeedAdapter
from job_agent.job_sources.linkedin import LinkedInAdapter
from job_agent.job_sources.ziprecruiter import ZipRecruiterAdapter
from job_agent.models.job import JobSource

_ADAPTERS: dict[JobSource, type[PartnerFeedAdapter]] = {
    JobSource.LINKEDIN: LinkedInAdapter,
    JobSource.INDEED: IndeedAdapter,
    JobSource.ZIPRECRUITER: ZipRecruiterAdapter,
}


def build_adapters(
    settings: Settings, clients: Mapping[JobSource, PartnerFeedClient] | None = None
) -> list[PartnerFeedAdapter]:
    """One adapter per enabled source. Sources without an approved client are still
    included so every run reports their limitation and manual link."""
    clients = clients or {}
    enabled = {
        JobSource.LINKEDIN: settings.linkedin_enabled,
        JobSource.INDEED: settings.indeed_enabled,
        JobSource.ZIPRECRUITER: settings.ziprecruiter_enabled,
    }
    return [
        cls(
            clients.get(source),
            timeout_seconds=settings.source_timeout_seconds,
            max_retries=settings.source_max_retries,
            max_retry_after_seconds=settings.source_max_retry_after_seconds,
            max_results=settings.source_max_results,
            max_description_chars=settings.max_description_chars,
        )
        for source, cls in _ADAPTERS.items()
        if enabled[source]
    ]
