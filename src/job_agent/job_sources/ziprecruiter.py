"""ZipRecruiter adapter.

ZipRecruiter job data is available through its publisher/partner programme, which
requires approval. Plug an approved client in as a ``PartnerFeedClient``; until then
the adapter reports UNAVAILABLE with a manual link.
"""

from __future__ import annotations

import math
from datetime import datetime
from urllib.parse import urlencode

from job_agent.job_sources.base import PartnerFeedAdapter
from job_agent.models.job import JobSource, SearchCriteria


class ZipRecruiterAdapter(PartnerFeedAdapter):
    source = JobSource.ZIPRECRUITER
    unavailable_reason = (
        "ZipRecruiter job data requires approved partner API access; "
        "search manually using the provided link"
    )

    def manual_search_url(self, criteria: SearchCriteria, posted_after: datetime) -> str:
        days = max(math.ceil((self._clock() - posted_after).total_seconds() / 86400), 1)
        params: dict[str, str] = {
            "search": criteria.titles[0] if criteria.titles else " ".join(criteria.keywords[:3]),
            "days": str(days),
        }
        if criteria.locations:
            params["location"] = criteria.locations[0]
        return "https://www.ziprecruiter.com/jobs-search?" + urlencode(params)
