"""Cross-source deduplication and canonical job identity.

Two postings are the same job when either:
1. they share a canonical key: sha256(normalised company | title | location), or
2. they point at the same application URL (the employer's ATS link is often
   syndicated unchanged to every board).

Grouping is transitive (union-find), so A~B by key and B~C by URL puts A, B and C
together. Matching is deliberately exact after normalisation: fuzzy title matching
risks merging genuinely different roles, which would hide jobs from the user.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from job_agent.models.job import ApplicationMethod, JobPosting, JobSource
from job_agent.models.resume import WorkArrangement

_COMPANY_SUFFIXES = re.compile(
    r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|gmbh|"
    r"ag|sa|lp|llp|pllc|holdings|group)\b\.?",
    re.I,
)
_TITLE_ABBREVIATIONS = {
    "sr": "senior",
    "jr": "junior",
    "snr": "senior",
    "mgr": "manager",
    "eng": "engineer",
    "engr": "engineer",
    "dev": "developer",
    "swe": "software engineer",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "i": "1",
    "assoc": "associate",
    "dir": "director",
    "vp": "vice president",
    "admin": "administrator",
}
# Text in parentheses/after separators that boards append but that doesn't change the role.
_TITLE_NOISE = re.compile(r"\((remote|hybrid|on-?site|full[- ]time|contract|w2|urgent)\)", re.I)
_US_STATES = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
    "district of columbia": "dc",
}
_SOURCE_PRIORITY = {JobSource.LINKEDIN: 0, JobSource.INDEED: 1, JobSource.ZIPRECRUITER: 2}
_METHOD_PRIORITY = {
    ApplicationMethod.API: 0,
    ApplicationMethod.EXTERNAL_ATS: 1,
    ApplicationMethod.EASY_APPLY: 2,
    ApplicationMethod.EMAIL: 3,
    ApplicationMethod.MANUAL_ONLY: 4,
    ApplicationMethod.UNKNOWN: 5,
}


def _ascii_words(value: str) -> list[str]:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9+#]+", value.lower().replace("&", " and "))


def normalize_company(name: str) -> str:
    words = _ascii_words(_COMPANY_SUFFIXES.sub(" ", name))
    if words and words[0] == "the":
        words = words[1:]
    return " ".join(words)


def normalize_title(title: str) -> str:
    title = _TITLE_NOISE.sub(" ", title)
    words = [_TITLE_ABBREVIATIONS.get(w, w) for w in _ascii_words(title)]
    return " ".join(words)


def normalize_location(location: str | None, arrangement: WorkArrangement) -> str:
    if arrangement is WorkArrangement.REMOTE:
        return "remote"
    if not location:
        return ""
    text = location.lower()
    for name, code in _US_STATES.items():
        text = re.sub(rf"\b{name}\b", code, text)
    words = [
        w
        for w in _ascii_words(text)
        if w not in {"remote", "hybrid", "united", "states", "us", "usa", "in"}
    ]
    # City + region is enough; drop zip codes.
    words = [w for w in words if not w.isdigit()]
    return " ".join(words[:3])


def canonical_job_key(posting: JobPosting) -> str:
    parts = (
        normalize_company(posting.company_name),
        normalize_title(posting.title),
        normalize_location(posting.location, posting.work_arrangement),
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def application_key(user_id: uuid.UUID, resume_id: uuid.UUID, canonical_key: str) -> str:
    """Idempotency anchor for applications: one per user × resume × canonical job."""
    return hashlib.sha256(f"{user_id}|{resume_id}|{canonical_key}".encode()).hexdigest()


@dataclass
class DedupedJob:
    canonical_key: str
    posting: JobPosting  # representative
    duplicates: list[JobPosting] = field(default_factory=list)
    # Every canonical key in the group (members may normalise differently but share a URL).
    member_keys: frozenset[str] = frozenset()

    @property
    def sources(self) -> list[JobSource]:
        return sorted(
            {p.source for p in [self.posting, *self.duplicates]}, key=lambda s: _SOURCE_PRIORITY[s]
        )

    @property
    def all_postings(self) -> list[JobPosting]:
        return [self.posting, *self.duplicates]


def _representative_rank(p: JobPosting) -> tuple[int, int, float, int, int]:
    return (
        0 if not p.publication_date_unverified else 1,  # verified date first
        0 if p.published_at_latest == p.published_at else 1,  # exact precision first
        p.published_at.timestamp() if p.published_at else float("inf"),  # earliest post
        _METHOD_PRIORITY[p.application_method],
        _SOURCE_PRIORITY[p.source],
    )


def deduplicate(postings: list[JobPosting]) -> list[DedupedJob]:
    """Group duplicates and return one ``DedupedJob`` per real job, in input order."""
    # Identical (source, external_id) entries collapse first.
    unique: dict[tuple[JobSource, str], JobPosting] = {}
    for p in postings:
        unique.setdefault((p.source, p.external_id), p)
    items = list(unique.values())

    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    keys = [canonical_job_key(p) for p in items]
    first_by: dict[str, int] = {}
    for i, p in enumerate(items):
        anchors = [f"k:{keys[i]}"]
        if p.application_url is not None:
            anchors.append(f"u:{p.application_url}")
        for anchor in anchors:
            if anchor in first_by:
                union(i, first_by[anchor])
            else:
                first_by[anchor] = i

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)

    result: list[DedupedJob] = []
    for root in sorted(groups):
        members = groups[root]
        ordered = sorted(members, key=lambda i: _representative_rank(items[i]))
        best = ordered[0]
        group_keys = frozenset(keys[i] for i in members)
        result.append(
            DedupedJob(
                # Order-independent within a run. Across runs the repository reuses any key
                # already stored for a member posting (see JobPostingRepository.persist_group).
                canonical_key=min(group_keys),
                posting=items[best],
                duplicates=[items[i] for i in ordered[1:]],
                member_keys=group_keys,
            )
        )
    return result
