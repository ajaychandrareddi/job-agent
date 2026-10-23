from __future__ import annotations

import pytest

from job_agent.models.job import JobPosting, JobSource, SalaryRange
from job_agent.models.resume import (
    EmploymentType,
    JobSearchPreferences,
    SalaryRequirement,
    WorkArrangement,
)
from job_agent.services.hard_constraints import (
    Constraint,
    annual_salary_bounds,
    evaluate_hard_constraints,
)
from job_agent.services.normalization import normalize_raw_posting
from tests.conftest import RUN_AT, make_raw

W = WorkArrangement


def posting(**kw: object) -> JobPosting:
    return normalize_raw_posting(make_raw("1", **kw), JobSource.INDEED, now=RUN_AT)


def prefs(**kw: object) -> JobSearchPreferences:
    base: dict[str, object] = {
        "preferred_titles": ["Senior Software Engineer"],
        "locations": ["New York, NY"],
        "requires_sponsorship": False,
        "authorized_to_work_countries": ["United States"],
    }
    base.update(kw)
    return JobSearchPreferences.model_validate(base)


def codes(p: JobPosting, pr: JobSearchPreferences) -> tuple[set[str], set[str]]:
    r = evaluate_hard_constraints(p, pr)
    return {v.code for v in r.violations}, {u.split(":")[0] for u in r.uncertainties}


def test_clean_posting_passes() -> None:
    assert codes(posting(), prefs()) == (set(), set())


def test_excluded_company_matches_normalised_name() -> None:
    v, _ = codes(posting(), prefs(excluded_companies=["ACME HEALTH LLC"]))
    assert v == {Constraint.EXCLUDED_COMPANY}


def test_excluded_location() -> None:
    v, _ = codes(posting(location="Newark, NJ"), prefs(locations=[], excluded_locations=["Newark"]))
    assert v == {Constraint.EXCLUDED_LOCATION}
    assert codes(posting(location="New York, NY"), prefs(excluded_locations=["York"]))[0] == {
        Constraint.EXCLUDED_LOCATION
    }  # whole-word phrase match
    assert (
        codes(posting(location="Yorktown, NY"), prefs(locations=[], excluded_locations=["York"]))[0]
        == set()
    )


class TestLocationAndArrangement:
    def test_onsite_outside_preferred_locations(self) -> None:
        v, _ = codes(posting(location="Boston, MA"), prefs())
        assert v == {Constraint.LOCATION_MISMATCH}

    def test_remote_job_satisfies_location_when_remote_accepted(self) -> None:
        assert codes(posting(location="Remote, US"), prefs()) == (set(), set())

    def test_remote_job_rejected_when_remote_not_accepted(self) -> None:
        v, _ = codes(
            posting(location="Remote, US"),
            prefs(locations=[], work_arrangements=[W.ONSITE, W.HYBRID]),
        )
        assert v == {Constraint.WORK_ARRANGEMENT}

    def test_missing_location_is_uncertain(self) -> None:
        _, u = codes(posting(location=None), prefs())
        assert u == {Constraint.LOCATION_MISMATCH}

    def test_unknown_arrangement_uncertain_only_when_user_restricts(self) -> None:
        assert codes(posting(), prefs(work_arrangements=[W.REMOTE]))[1] == {
            Constraint.WORK_ARRANGEMENT
        }
        assert codes(posting(), prefs())[1] == set()


class TestEmploymentAndSalary:
    def test_employment_type(self) -> None:
        assert codes(posting(employment_type="Contract"), prefs())[0] == {
            Constraint.EMPLOYMENT_TYPE
        }
        assert codes(posting(employment_type=None), prefs())[1] == {Constraint.EMPLOYMENT_TYPE}
        both = prefs(employment_types=[EmploymentType.FULL_TIME, EmploymentType.CONTRACT])
        assert codes(posting(employment_type="Contract"), both) == (set(), set())

    @pytest.mark.parametrize(
        ("salary_text", "minimum", "expected"),
        [
            ("$40 - $45 an hour", 120_000, {Constraint.SALARY_BELOW_MINIMUM}),  # 93,600/yr
            ("$100,000 - $130,000 a year", 120_000, set()),  # range reaches the minimum
            ("$100,000 - $110,000 a year", 120_000, {Constraint.SALARY_BELOW_MINIMUM}),
            (None, 120_000, set()),  # unspecified allowed by default
        ],
    )
    def test_salary_minimum(
        self, salary_text: str | None, minimum: int, expected: set[str]
    ) -> None:
        pr = prefs(salary=SalaryRequirement(minimum_annual=minimum))
        assert codes(posting(salary_text=salary_text), pr)[0] == expected

    def test_salary_required(self) -> None:
        pr = prefs(salary=SalaryRequirement(minimum_annual=100_000, allow_unspecified=False))
        assert codes(posting(), pr)[0] == {Constraint.SALARY_UNSPECIFIED}

    def test_other_currency_is_uncertain(self) -> None:
        pr = prefs(salary=SalaryRequirement(minimum_annual=100_000))
        _, u = codes(posting(salary_min=50_000, salary_max=60_000, salary_currency="EUR"), pr)
        assert u == {Constraint.SALARY_BELOW_MINIMUM}

    def test_annualisation(self) -> None:
        assert annual_salary_bounds(SalaryRange(minimum=50, maximum=60, period="hour")) == (
            104_000,
            124_800,
        )
        assert annual_salary_bounds(SalaryRange(minimum=10_000, period="month")) == (120_000, None)


def test_excluded_keywords_and_closed_postings() -> None:
    assert codes(
        posting(description="Unpaid internship role"), prefs(excluded_keywords=["unpaid"])
    )[0] == {Constraint.EXCLUDED_KEYWORD}
    assert codes(posting(is_open=False), prefs())[0] == {Constraint.APPLICATION_CLOSED}


class TestAuthorization:
    no_sponsor = "We are unable to sponsor visas for this role."
    us_auth = "Candidates must be authorized to work in the United States."

    def test_no_sponsorship(self) -> None:
        p = posting(description=self.no_sponsor)
        assert codes(p, prefs(requires_sponsorship=True))[0] == {Constraint.SPONSORSHIP_UNAVAILABLE}
        assert codes(p, prefs(requires_sponsorship=None))[1] == {Constraint.SPONSORSHIP_UNAVAILABLE}
        assert codes(p, prefs(requires_sponsorship=False)) == (set(), set())

    def test_us_work_authorization(self) -> None:
        p = posting(description=self.us_auth)
        assert codes(p, prefs(authorized_to_work_countries=[]))[1] == {
            Constraint.WORK_AUTHORIZATION
        }
        assert codes(p, prefs(authorized_to_work_countries=["Canada"]))[0] == {
            Constraint.WORK_AUTHORIZATION
        }
        assert codes(p, prefs()) == (set(), set())

    @pytest.mark.parametrize(
        "text",
        [
            "Active Secret clearance required.",
            "Must be a U.S. citizen.",
            "Ability to obtain a security clearance.",
        ],
    )
    def test_citizenship_or_clearance_always_needs_a_human(self, text: str) -> None:
        assert codes(posting(description=text), prefs())[1] == {Constraint.CITIZENSHIP_OR_CLEARANCE}
