"""LinkedIn adapter.

LinkedIn offers job search/apply APIs only to approved partners. Automating a
member's logged-in session or scraping violates LinkedIn's User Agreement, so this
adapter never does that. Without an approved ``PartnerFeedClient`` it reports
UNAVAILABLE plus a manual search link. Applications are always manual.
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlencode

from job_agent.job_sources.base import PartnerFeedAdapter
from job_agent.models.job import JobSource, SearchCriteria
from job_agent.models.resume import WorkArrangement

_REMOTE_FILTER = {
    WorkArrangement.ONSITE: "1",
    WorkArrangement.REMOTE: "2",
    WorkArrangement.HYBRID: "3",
}


class LinkedInAdapter(PartnerFeedAdapter):
    source = JobSource.LINKEDIN
    unavailable_reason = (
        "LinkedIn job data is available only to approved LinkedIn partners; "
        "search manually using the provided link"
    )

    def manual_search_url(self, criteria: SearchCriteria, posted_after: datetime) -> str:
        seconds = max(int((self._clock() - posted_after).total_seconds()), 3600)
        params: dict[str, str] = {
            "keywords": " OR ".join(criteria.titles[:3]) or " ".join(criteria.keywords[:3]),
            "f_TPR": f"r{seconds}",
        }
        if criteria.locations:
            params["location"] = criteria.locations[0]
        remote = [_REMOTE_FILTER[w] for w in criteria.work_arrangements if w in _REMOTE_FILTER]
        if remote:
            params["f_WT"] = ",".join(remote)
        return "https://www.linkedin.com/jobs/search/?" + urlencode(params)
