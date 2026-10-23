"""Indeed adapter.

Indeed's legacy Publisher Job Search API is closed to new publishers; current access
is through approved partner programmes. Plug an approved client in as a
``PartnerFeedClient``; until then the adapter reports UNAVAILABLE with a manual link.
"""

from __future__ import annotations

import math
from datetime import datetime
from urllib.parse import urlencode

from job_agent.job_sources.base import PartnerFeedAdapter
from job_agent.models.job import JobSource, SearchCriteria
from job_agent.models.resume import WorkArrangement


class IndeedAdapter(PartnerFeedAdapter):
    source = JobSource.INDEED
    unavailable_reason = (
        "Indeed job data requires approved partner API access; "
        "search manually using the provided link"
    )

    def manual_search_url(self, criteria: SearchCriteria, posted_after: datetime) -> str:
        days = max(math.ceil((self._clock() - posted_after).total_seconds() / 86400), 1)
        query = " or ".join(f'"{t}"' for t in criteria.titles[:3]) or " ".join(
            criteria.keywords[:3]
        )
        params: dict[str, str] = {"q": query, "fromage": str(days)}
        if criteria.locations:
            params["l"] = criteria.locations[0]
        elif WorkArrangement.REMOTE in criteria.work_arrangements:
            params["l"] = "Remote"
        return "https://www.indeed.com/jobs?" + urlencode(params)
