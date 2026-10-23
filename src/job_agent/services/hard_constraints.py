"""Deterministic hard constraints. A violation blocks application regardless of score.

Two outcomes per check:
* **violation**: the posting clearly breaks an explicit user constraint -> SKIP;
* **uncertainty**: the posting may break one, but the data is missing or ambiguous ->
  a qualifying job goes to REVIEW instead of being auto-applied.

Work authorisation, sponsorship, citizenship and clearance are never assumed: they are
judged only from the user's explicit answers in ``JobSearchPreferences``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from job_agent.models.job import ApplicationAvailability, JobPosting, SalaryRange
from job_agent.models.matching import HardConstraintViolation
from job_agent.models.resume import EmploymentType, JobSearchPreferences, WorkArrangement
from job_agent.services.deduplication import normalize_company


class Constraint(StrEnum):
    EXCLUDED_COMPANY = "excluded_company"
    EXCLUDED_LOCATION = "excluded_location"
    LOCATION_MISMATCH = "location_mismatch"
    WORK_ARRANGEMENT = "work_arrangement"
    EMPLOYMENT_TYPE = "employment_type"
    SALARY_BELOW_MINIMUM = "salary_below_minimum"
    SALARY_UNSPECIFIED = "salary_unspecified"
    EXCLUDED_KEYWORD = "excluded_keyword"
    APPLICATION_CLOSED = "application_closed"
    SPONSORSHIP_UNAVAILABLE = "sponsorship_unavailable"
    WORK_AUTHORIZATION = "work_authorization"
    CITIZENSHIP_OR_CLEARANCE = "citizenship_or_clearance"


@dataclass
class ConstraintReport:
    violations: list[HardConstraintViolation] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def violate(self, code: Constraint, detail: str) -> None:
        self.violations.append(HardConstraintViolation(code=code.value, detail=detail))

    def unsure(self, code: Constraint, detail: str) -> None:
        self.uncertainties.append(f"{code.value}: {detail}")


_ANNUAL = {"year": 1, "month": 12, "week": 52, "day": 260, "hour": 2080}

_NO_SPONSORSHIP = re.compile(
    r"\b(no|not|unable to|cannot|can ?not|will not|won't|does not|do not)\s+"
    r"(provide\s+|offer\s+)?(visa\s+)?sponsor(ship)?\b"
    r"|\bsponsorship\s+(is\s+)?not\s+(available|provided|offered)\b"
    r"|\bwithout\s+(the\s+)?need\s+(for\s+)?(visa\s+)?sponsorship\b",
    re.I,
)
_US_AUTH = re.compile(
    r"\b(authori[sz]ed|eligible|legally able)\s+to\s+work\s+in\s+the\s+"
    r"(u\.?s\.?a?\.?|united states)\b",
    re.I,
)
_CITIZEN_OR_CLEARANCE = re.compile(
    r"\b(u\.?s\.?\s+citizen(ship)?\s+(is\s+)?required|must\s+be\s+a\s+u\.?s\.?\s+citizen|"
    r"(active|current|secret|top secret|ts/sci)\s+(security\s+)?clearance|"
    r"security\s+clearance\s+(is\s+)?required|ability\s+to\s+obtain\s+(a\s+)?"
    r"(security\s+)?clearance)\b",
    re.I,
)


def annual_salary_bounds(salary: SalaryRange) -> tuple[float | None, float | None]:
    factor = _ANNUAL[salary.period]
    lo = salary.minimum * factor if salary.minimum is not None else None
    hi = salary.maximum * factor if salary.maximum is not None else None
    return lo, hi


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#]+", text.lower()))


def _contains_phrase(haystack: str, phrase: str) -> bool:
    needle = _words(phrase)
    return (
        bool(needle)
        and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", _words(haystack))
        is not None
    )


def evaluate_hard_constraints(posting: JobPosting, prefs: JobSearchPreferences) -> ConstraintReport:
    r = ConstraintReport()
    _company(posting, prefs, r)
    _location(posting, prefs, r)
    _arrangement(posting, prefs, r)
    _employment(posting, prefs, r)
    _salary(posting, prefs, r)
    _keywords(posting, prefs, r)
    _availability(posting, r)
    _authorization(posting, prefs, r)
    return r


def _company(p: JobPosting, prefs: JobSearchPreferences, r: ConstraintReport) -> None:
    company = normalize_company(p.company_name)
    for excluded in prefs.excluded_companies:
        if company and company == normalize_company(excluded):
            r.violate(Constraint.EXCLUDED_COMPANY, f"{p.company_name} is excluded")
            return


def _location(p: JobPosting, prefs: JobSearchPreferences, r: ConstraintReport) -> None:
    location = p.location or ""
    for excluded in prefs.excluded_locations:
        if location and _contains_phrase(location, excluded):
            r.violate(Constraint.EXCLUDED_LOCATION, f"location '{location}' is excluded")
            return
    if not prefs.locations:
        return
    remote_ok = WorkArrangement.REMOTE in prefs.work_arrangements
    if p.work_arrangement is WorkArrangement.REMOTE and remote_ok:
        return
    if not location:
        r.unsure(Constraint.LOCATION_MISMATCH, "posting has no location")
        return
    if any(_contains_phrase(location, _city(pref)) for pref in prefs.locations):
        return
    if p.work_arrangement is WorkArrangement.UNKNOWN and remote_ok and "remote" in location.lower():
        return
    r.violate(Constraint.LOCATION_MISMATCH, f"location '{location}' is not in preferred locations")


def _city(pref: str) -> str:
    # "New York, NY" -> match on "New York"; a bare state/region is used as-is.
    return pref.split(",")[0].strip() or pref


def _arrangement(p: JobPosting, prefs: JobSearchPreferences, r: ConstraintReport) -> None:
    accepted = set(prefs.work_arrangements)
    restricts = (
        not {WorkArrangement.REMOTE, WorkArrangement.HYBRID, WorkArrangement.ONSITE} <= accepted
    )
    if p.work_arrangement is WorkArrangement.UNKNOWN:
        if restricts:
            r.unsure(Constraint.WORK_ARRANGEMENT, "posting does not state remote/hybrid/onsite")
    elif p.work_arrangement not in accepted:
        r.violate(
            Constraint.WORK_ARRANGEMENT,
            f"{p.work_arrangement.value} is not an accepted work arrangement",
        )


def _employment(p: JobPosting, prefs: JobSearchPreferences, r: ConstraintReport) -> None:
    accepted = set(prefs.employment_types)
    if p.employment_type is EmploymentType.UNKNOWN:
        if accepted and EmploymentType.UNKNOWN not in accepted:
            r.unsure(Constraint.EMPLOYMENT_TYPE, "posting does not state employment type")
    elif p.employment_type not in accepted:
        r.violate(
            Constraint.EMPLOYMENT_TYPE,
            f"{p.employment_type.value} is not an accepted employment type",
        )


def _salary(p: JobPosting, prefs: JobSearchPreferences, r: ConstraintReport) -> None:
    minimum = prefs.salary.minimum_annual
    if minimum is None:
        return
    if p.salary is None or (p.salary.minimum is None and p.salary.maximum is None):
        if not prefs.salary.allow_unspecified:
            r.violate(Constraint.SALARY_UNSPECIFIED, "posting has no salary and one is required")
        return
    if p.salary.currency.upper() != prefs.salary.currency.upper():
        r.unsure(
            Constraint.SALARY_BELOW_MINIMUM,
            f"salary in {p.salary.currency}, minimum set in {prefs.salary.currency}",
        )
        return
    lo, hi = annual_salary_bounds(p.salary)
    top = hi if hi is not None else lo
    if top is not None and top < minimum:
        r.violate(
            Constraint.SALARY_BELOW_MINIMUM,
            f"annual pay up to {top:,.0f} is below the minimum {minimum:,}",
        )


def _keywords(p: JobPosting, prefs: JobSearchPreferences, r: ConstraintReport) -> None:
    text = f"{p.title}\n{p.description}"
    for kw in prefs.excluded_keywords:
        if _contains_phrase(text, kw):
            r.violate(Constraint.EXCLUDED_KEYWORD, f"posting mentions excluded keyword '{kw}'")
            return


def _availability(p: JobPosting, r: ConstraintReport) -> None:
    if p.application_availability is ApplicationAvailability.CLOSED:
        r.violate(Constraint.APPLICATION_CLOSED, "the posting is closed to applications")


def _authorization(p: JobPosting, prefs: JobSearchPreferences, r: ConstraintReport) -> None:
    text = p.description
    if _NO_SPONSORSHIP.search(text):
        if prefs.requires_sponsorship is True:
            r.violate(
                Constraint.SPONSORSHIP_UNAVAILABLE,
                "posting does not sponsor visas and you require sponsorship",
            )
        elif prefs.requires_sponsorship is None:
            r.unsure(
                Constraint.SPONSORSHIP_UNAVAILABLE,
                "posting does not sponsor visas; your sponsorship need is not set",
            )
    if _US_AUTH.search(text):
        countries = {c.strip().lower() for c in prefs.authorized_to_work_countries}
        us = {"us", "usa", "u.s.", "united states", "united states of america"}
        if not countries:
            r.unsure(
                Constraint.WORK_AUTHORIZATION,
                "posting requires US work authorization; yours is not set",
            )
        elif not countries & us:
            r.violate(
                Constraint.WORK_AUTHORIZATION,
                "posting requires US work authorization, which you have not indicated",
            )
    if _CITIZEN_OR_CLEARANCE.search(text):
        # Never asserted on the user's behalf: always a human decision.
        r.unsure(
            Constraint.CITIZENSHIP_OR_CLEARANCE,
            "posting requires citizenship or a security clearance",
        )
