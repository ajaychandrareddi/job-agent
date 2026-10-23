"""Normalise ``RawPosting`` records from any source into ``JobPosting``.

Principles:
* Never invent data. A value the source did not provide stays ``None``/``UNKNOWN``.
* Publication time comes only from the source's *publication* field, never from
  its update/modification time.
* Job descriptions are untrusted: HTML and control characters are stripped and the
  text is length-capped. The text is kept verbatim otherwise and is only ever passed
  to the LLM as delimited data.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from job_agent.models.job import (
    ApplicationAvailability,
    ApplicationMethod,
    JobPosting,
    JobSource,
    PublicationPrecision,
    RawPosting,
    SalaryRange,
)
from job_agent.models.resume import EmploymentType, WorkArrangement
from job_agent.services.text_extraction import normalize_text


class NormalizationError(ValueError):
    """A raw record is unusable (e.g. no title or URL). It is skipped and counted."""


# --------------------------------------------------------------------------- HTML


class _TextExtractor(HTMLParser):
    _BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}
    _SKIP = {"script", "style", "noscript", "template", "iframe"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK and tag not in ("li", "br"):  # list items: single break
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def clean_description(text: str | None, *, max_chars: int) -> str:
    if not text:
        return ""
    if "<" in text and ">" in text:
        parser = _TextExtractor()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    else:
        text = unescape(text)
    cleaned = normalize_text(text)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + " [truncated]"
    return cleaned


def _clean_field(value: str | None, max_len: int = 500) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(normalize_text(unescape(value)).split())
    return cleaned[:max_len] or None


# --------------------------------------------------------------------------- URLs

_TRACKING_PARAMS = {
    "gclid",
    "fbclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "ref",
    "refid",
    "trk",
    "trkinfo",
    "trackingid",
    "src",
    "source",
    "from",
    "campaignid",
    "_ga",
}


def canonicalize_url(url: str | None) -> str | None:
    """Lower-case scheme/host, drop fragments and tracking parameters, sort the rest."""
    if not url:
        return None
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host if parts.port in (None, 80, 443) else f"{host}:{parts.port}"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        ("https" if parts.scheme.lower() == "https" else "http", netloc, path, urlencode(query), "")
    )


# --------------------------------------------------------------------------- dates

_RELATIVE = re.compile(
    r"(?P<n>\d+)\s*(?P<plus>\+)?\s*(?P<unit>minute|min|hour|hr|day|week|month)s?\s+ago", re.I
)
_UNIT = {
    "minute": timedelta(minutes=1),
    "min": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "hr": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
}
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MIN_YEAR = 2000


class ParsedTimestamp:
    __slots__ = ("earliest", "latest", "precision", "verified")

    def __init__(
        self,
        earliest: datetime | None,
        latest: datetime | None,
        precision: PublicationPrecision,
        verified: bool,
    ) -> None:
        self.earliest = earliest
        self.latest = latest
        self.precision = precision
        self.verified = verified


_MISSING = ParsedTimestamp(None, None, PublicationPrecision.EXACT, False)


def parse_timestamp(
    value: Any, *, now: datetime, source_tz: ZoneInfo | None = None
) -> ParsedTimestamp:
    """Parse a source timestamp into a UTC interval.

    ``source_tz`` is the documented timezone of naive timestamps for this source.
    Naive values without a known timezone cannot be verified.
    """
    if value is None or value == "":
        return _MISSING
    result: ParsedTimestamp
    if isinstance(value, datetime):
        result = _from_datetime(value, source_tz)
    elif isinstance(value, date):
        result = _from_day(value, source_tz)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        seconds = value / 1000 if value > 1e11 else value
        try:
            dt = datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return _MISSING
        result = ParsedTimestamp(dt, dt, PublicationPrecision.EXACT, True)
    elif isinstance(value, str):
        result = _from_string(value.strip(), now=now, source_tz=source_tz)
    else:
        return _MISSING
    if result.earliest is not None and result.earliest.year < _MIN_YEAR:
        return _MISSING
    return result


def _from_datetime(dt: datetime, source_tz: ZoneInfo | None) -> ParsedTimestamp:
    if dt.tzinfo is None or dt.utcoffset() is None:
        if source_tz is None:
            return ParsedTimestamp(
                dt.replace(tzinfo=UTC), dt.replace(tzinfo=UTC), PublicationPrecision.EXACT, False
            )
        dt = dt.replace(tzinfo=source_tz)
    dt = dt.astimezone(UTC)
    return ParsedTimestamp(dt, dt, PublicationPrecision.EXACT, True)


def _from_day(d: date, source_tz: ZoneInfo | None) -> ParsedTimestamp:
    start = datetime(d.year, d.month, d.day, tzinfo=source_tz or UTC)
    # Without the source timezone the day could start up to 14h earlier / end 12h later.
    pad = timedelta(0) if source_tz else timedelta(hours=14)
    earliest = (start - pad).astimezone(UTC)
    latest = (start + timedelta(days=1) - timedelta(microseconds=1) + pad).astimezone(UTC)
    return ParsedTimestamp(earliest, latest, PublicationPrecision.DAY, True)


def _from_string(text: str, *, now: datetime, source_tz: ZoneInfo | None) -> ParsedTimestamp:
    lowered = text.lower()
    if _DATE_ONLY.match(text):
        try:
            return _from_day(date.fromisoformat(text), source_tz)
        except ValueError:
            return _MISSING
    try:
        return _from_datetime(datetime.fromisoformat(text.replace("Z", "+00:00")), source_tz)
    except ValueError:
        pass
    if lowered in ("just posted", "just now", "today", "new"):
        return ParsedTimestamp(
            now - timedelta(hours=24), now, PublicationPrecision.APPROXIMATE, True
        )
    if lowered == "yesterday":
        return ParsedTimestamp(
            now - timedelta(hours=48),
            now - timedelta(hours=12),
            PublicationPrecision.APPROXIMATE,
            True,
        )
    m = _RELATIVE.search(lowered)
    if m:
        unit = _UNIT[m.group("unit").lower()]
        n = int(m.group("n"))
        latest = now - n * unit
        # "3 days ago" means anything from 3 to just under 4 days ago; "30+ days" is open-ended.
        earliest = latest - (unit if not m.group("plus") else timedelta(days=365))
        return ParsedTimestamp(earliest, latest, PublicationPrecision.APPROXIMATE, True)
    return _MISSING


# --------------------------------------------------------------------------- categories

_ARRANGEMENT_LABELS = {
    "remote": WorkArrangement.REMOTE,
    "fully remote": WorkArrangement.REMOTE,
    "work from home": WorkArrangement.REMOTE,
    "wfh": WorkArrangement.REMOTE,
    "hybrid": WorkArrangement.HYBRID,
    "hybrid remote": WorkArrangement.HYBRID,
    "onsite": WorkArrangement.ONSITE,
    "on-site": WorkArrangement.ONSITE,
    "on site": WorkArrangement.ONSITE,
    "in office": WorkArrangement.ONSITE,
    "in-person": WorkArrangement.ONSITE,
}
_EMPLOYMENT_LABELS = {
    "full-time": EmploymentType.FULL_TIME,
    "full time": EmploymentType.FULL_TIME,
    "fulltime": EmploymentType.FULL_TIME,
    "full_time": EmploymentType.FULL_TIME,
    "permanent": EmploymentType.FULL_TIME,
    "part-time": EmploymentType.PART_TIME,
    "part time": EmploymentType.PART_TIME,
    "part_time": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "contractor": EmploymentType.CONTRACT,
    "contract to hire": EmploymentType.CONTRACT,
    "c2h": EmploymentType.CONTRACT,
    "temporary": EmploymentType.TEMPORARY,
    "temp": EmploymentType.TEMPORARY,
    "seasonal": EmploymentType.TEMPORARY,
    "internship": EmploymentType.INTERNSHIP,
    "intern": EmploymentType.INTERNSHIP,
}
_METHOD_LABELS = {
    "api": ApplicationMethod.API,
    "easy_apply": ApplicationMethod.EASY_APPLY,
    "easy apply": ApplicationMethod.EASY_APPLY,
    "easyapply": ApplicationMethod.EASY_APPLY,
    "indeed_apply": ApplicationMethod.EASY_APPLY,
    "quick_apply": ApplicationMethod.EASY_APPLY,
    "1-click": ApplicationMethod.EASY_APPLY,
    "external": ApplicationMethod.EXTERNAL_ATS,
    "ats": ApplicationMethod.EXTERNAL_ATS,
    "company_site": ApplicationMethod.EXTERNAL_ATS,
    "email": ApplicationMethod.EMAIL,
    "manual": ApplicationMethod.MANUAL_ONLY,
}
_SOURCE_DOMAINS = {
    JobSource.LINKEDIN: "linkedin.com",
    JobSource.INDEED: "indeed.com",
    JobSource.ZIPRECRUITER: "ziprecruiter.com",
}


def _label(value: str | None) -> str:
    return " ".join((value or "").lower().replace("_", " ").split())


def detect_work_arrangement(
    remote_type: str | None, location: str | None, title: str | None
) -> WorkArrangement:
    label = _label(remote_type)
    if label in _ARRANGEMENT_LABELS:
        return _ARRANGEMENT_LABELS[label]
    for text in (_label(location), _label(title)):
        if re.search(r"\bhybrid\b", text):
            return WorkArrangement.HYBRID
        if re.search(r"\b(remote|work from home)\b", text) and not re.search(
            r"\b(not|no|non)[- ]remote\b", text
        ):
            return WorkArrangement.REMOTE
    return WorkArrangement.UNKNOWN  # an address alone does not prove on-site


def map_employment_type(label: str | None, title: str | None) -> EmploymentType:
    key = _label(label)
    if key in _EMPLOYMENT_LABELS:
        return _EMPLOYMENT_LABELS[key]
    if key.replace(" ", "-") in _EMPLOYMENT_LABELS:
        return _EMPLOYMENT_LABELS[key.replace(" ", "-")]
    t = _label(title)
    if re.search(r"\binter(n|nship)\b", t):
        return EmploymentType.INTERNSHIP
    if re.search(r"\bcontract(or)?\b", t):
        return EmploymentType.CONTRACT
    return EmploymentType.UNKNOWN


def map_application_method(
    label: str | None, apply_url: str | None, source: JobSource
) -> ApplicationMethod:
    key = (label or "").strip().lower()
    if key in _METHOD_LABELS:
        return _METHOD_LABELS[key]
    if not apply_url:
        return ApplicationMethod.UNKNOWN
    if apply_url.lower().startswith("mailto:"):
        return ApplicationMethod.EMAIL
    host = (urlsplit(apply_url).hostname or "").lower()
    if host and not host.endswith(_SOURCE_DOMAINS[source]):
        return ApplicationMethod.EXTERNAL_ATS
    return ApplicationMethod.UNKNOWN


# --------------------------------------------------------------------------- salary

_CUR = r"(?:\$|USD|£|GBP|€|EUR)"
_SALARY_TEXT = re.compile(
    rf"(?P<cur>{_CUR})\s*(?P<a>\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?)\s*(?P<ak>[kK])?"
    rf"\s*(?:-|–|—|to)\s*{_CUR}?\s*(?P<b>\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?)\s*(?P<bk>[kK])?"
    r"\s*(?:/|per|an|a)?\s*"
    r"(?P<per>year|yr|annum|annually|hour|hr|hourly|month|mo|week|wk|day)?\b",
    re.I,
)
_PERIOD = {
    "year": "year",
    "yr": "year",
    "annum": "year",
    "annually": "year",
    "hour": "hour",
    "hr": "hour",
    "hourly": "hour",
    "month": "month",
    "mo": "month",
    "week": "week",
    "wk": "week",
    "day": "day",
}
_CURRENCY = {"$": "USD", "usd": "USD", "£": "GBP", "gbp": "GBP", "€": "EUR", "eur": "EUR"}


def _num(value: str, k: str | None) -> float:
    n = float(value.replace(",", ""))
    return n * 1000 if k else n


def parse_salary(raw: RawPosting) -> SalaryRange | None:
    period = _PERIOD.get((raw.salary_period or "").strip().lower())
    currency = (raw.salary_currency or "USD").upper()[:3]
    if raw.salary_min is not None or raw.salary_max is not None:
        try:
            return SalaryRange(
                minimum=raw.salary_min,
                maximum=raw.salary_max,
                currency=currency,
                period=period or "year",
            )
        except ValidationError:
            return None
    if not raw.salary_text:
        return None
    m = _SALARY_TEXT.search(raw.salary_text)
    if not m:
        return None
    lo, hi = _num(m.group("a"), m.group("ak")), _num(m.group("b"), m.group("bk"))
    per = _PERIOD.get((m.group("per") or "").lower())
    if per is None:  # infer only when unambiguous
        per = "year" if lo >= 10_000 else "hour" if hi < 500 else None
    if per is None or lo > hi:
        return None
    return SalaryRange(
        minimum=lo, maximum=hi, currency=_CURRENCY[m.group("cur").lower()], period=per
    )


_YEARS = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-|–|to)?\s*(?:\d{1,2}\s*)?\+?\s*years?\s+(?:of\s+)?"
    r"(?:[\w/+#.-]+\s+){0,4}?experience",
    re.I,
)


def extract_min_years(description: str) -> float | None:
    """Smallest 'N years ... experience' requirement stated in the posting, if any."""
    values = [int(m.group(1)) for m in _YEARS.finditer(description)]
    values = [v for v in values if 0 < v <= 40]
    return float(min(values)) if values else None


_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def extract_experience_requirements(description: str, *, max_chars: int = 500) -> str | None:
    """Sentences from the posting that state an experience requirement, verbatim."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(description) if s.strip()]
    hits = [s for s in sentences if _YEARS.search(s)]
    if not hits:
        return None
    return " ".join(hits)[:max_chars]


def _clean_list(items: list[str], limit: int = 100) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items[:limit]:
        cleaned = _clean_field(item, 120)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


# --------------------------------------------------------------------------- entry point


def normalize_raw_posting(
    raw: RawPosting,
    source: JobSource,
    *,
    now: datetime,
    max_description_chars: int = 20_000,
    source_tz: ZoneInfo | None = None,
) -> JobPosting:
    title = _clean_field(raw.title)
    company = _clean_field(raw.company_name)
    canonical_url = canonicalize_url(raw.url)
    if not raw.external_id.strip():
        raise NormalizationError("missing external id")
    if not title or not company:
        raise NormalizationError("missing title or company")
    if canonical_url is None:
        raise NormalizationError("missing or invalid job URL")

    apply_url = canonicalize_url(raw.apply_url) if raw.apply_url else None
    if raw.apply_url and raw.apply_url.lower().startswith("mailto:"):
        apply_url = None  # kept only as the method; not a fetchable URL
    description = clean_description(raw.description, max_chars=max_description_chars)
    location = _clean_field(raw.location)

    published = parse_timestamp(raw.published_at, now=now, source_tz=source_tz)
    updated = parse_timestamp(raw.updated_at, now=now, source_tz=source_tz)
    metadata: dict[str, object] = {
        k: v
        for k, v in list(raw.metadata.items())[:50]
        if isinstance(v, str | int | float | bool) or v is None
    }
    if raw.published_at not in (None, "") and not published.verified:
        # Keep what the source said so reviewers can see why the date was not trusted.
        metadata["unverified_publication_value"] = str(raw.published_at)[:64]

    try:
        return JobPosting(
            source=source,
            external_id=raw.external_id.strip()[:255],
            title=title,
            company_name=company,
            location=location,
            work_arrangement=detect_work_arrangement(raw.remote_type, location, title),
            employment_type=map_employment_type(raw.employment_type, title),
            description=description,
            required_skills=_clean_list(raw.required_skills),
            preferred_skills=_clean_list(raw.preferred_skills),
            experience_requirements=(
                _clean_field(raw.experience_requirements, 500)
                or extract_experience_requirements(description)
            ),
            min_years_experience=extract_min_years(description),
            education_requirements=_clean_list(raw.education_requirements),
            certification_requirements=_clean_list(raw.certification_requirements),
            salary=parse_salary(raw),
            published_at=published.earliest if published.verified else None,
            published_at_latest=published.latest if published.verified else None,
            published_at_precision=published.precision,
            publication_date_unverified=not published.verified,
            updated_at=updated.latest if updated.verified else None,
            canonical_url=canonical_url,
            application_url=apply_url,
            application_method=map_application_method(raw.apply_method, raw.apply_url, source),
            application_availability=(
                ApplicationAvailability.UNKNOWN
                if raw.is_open is None
                else ApplicationAvailability.OPEN
                if raw.is_open
                else ApplicationAvailability.CLOSED
            ),
            raw_metadata=metadata,
        )
    except ValidationError as exc:
        raise NormalizationError(f"invalid posting: {exc.errors()[0]['msg']}") from exc
