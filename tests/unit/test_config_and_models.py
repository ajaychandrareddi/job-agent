from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from job_agent.config import MatchingWeights, Settings
from job_agent.models.application import (
    ApplicationStatus,
    InvalidStatusTransition,
    assert_transition,
)
from job_agent.models.job import JobPosting, JobSource, SalaryRange
from job_agent.models.matching import CriterionScores
from job_agent.models.resume import JobSearchPreferences, ResumeProfile


class TestSettings:
    def test_defaults_match_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("JOB_SEARCH_TIMEZONE", "JOB_SEARCH_HOUR", "JOB_MATCH_THRESHOLD"):
            monkeypatch.delenv(var, raising=False)
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.job_search_timezone == "America/New_York"
        assert (s.job_search_hour, s.job_search_minute) == (17, 0)
        assert s.initial_search_days == 30
        assert s.recurring_search_hours == 24
        assert s.job_match_threshold == 60
        assert s.max_concurrent_job_searches == 3
        assert s.max_application_retries == 2

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JOB_SEARCH_TIMEZONE", "America/Chicago")
        monkeypatch.setenv("JOB_MATCH_THRESHOLD", "70")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.tz.key == "America/Chicago"
        assert s.job_match_threshold == 70

    def test_invalid_timezone_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown timezone"):
            Settings(_env_file=None, job_search_timezone="Mars/Olympus")  # type: ignore[call-arg]

    def test_static_tokens_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_STATIC_TOKENS", "a" * 16 + ":u1, " + "b" * 16 + ":u2")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        tokens = {k: v.get_secret_value() for k, v in s.api_static_tokens.items()}
        assert tokens == {"u1": "a" * 16, "u2": "b" * 16}
        assert "a" * 16 not in repr(s)  # never printed

    def test_short_static_tokens_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_STATIC_TOKENS", "short:u1")
        with pytest.raises(ValidationError, match="16 characters"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_auth_safety_rules(self) -> None:
        with pytest.raises(ValidationError, match="not allowed when APP_ENV=prod"):
            Settings(_env_file=None, app_env="prod")  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="needs API_JWT_SECRET"):
            Settings(_env_file=None, api_auth_mode="jwt")  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="'none'"):
            Settings(
                _env_file=None,
                api_auth_mode="jwt",
                api_jwt_secret="s" * 32,  # type: ignore[call-arg]
                api_jwt_algorithms="HS256,none",
            )
        ok = Settings(
            _env_file=None,
            app_env="prod",
            api_auth_mode="jwt",  # type: ignore[call-arg]
            api_jwt_secret="s" * 32,
            api_jwt_algorithms="HS256",
        )
        assert ok.api_jwt_algorithms == ["HS256"]

    def test_malformed_static_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_STATIC_TOKENS", "abc")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_weights_configurable_from_env_file(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            "MATCHING_WEIGHTS__REQUIRED_SKILLS=40\nMATCHING_WEIGHTS__DOMAIN_EXPERIENCE=0\n"
        )
        s = Settings(_env_file=env)  # type: ignore[call-arg]
        assert s.matching_weights.required_skills == 40
        assert s.matching_weights.domain_experience == 0

    def test_weights_from_env_must_still_sum_to_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MATCHING_WEIGHTS__REQUIRED_SKILLS", "50")
        with pytest.raises(ValidationError, match="sum to 100"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_env_example_documents_every_setting(self) -> None:
        root = Path(__file__).resolve().parents[2]
        lines = (root / ".env.example").read_text().splitlines()
        keys = {
            ln.split("=", 1)[0].strip()
            for ln in lines
            if "=" in ln and not ln.lstrip().startswith("#")
        }
        fields = {name.upper() for name in Settings.model_fields if name != "matching_weights"}
        weights = {f"MATCHING_WEIGHTS__{n.upper()}" for n in MatchingWeights.model_fields}
        assert fields - keys == set(), "settings missing from .env.example"
        assert keys - fields - weights == set(), ".env.example has unknown keys"
        assert weights <= keys
        Settings(_env_file=root / ".env.example")  # type: ignore[call-arg]  # parses cleanly

    def test_weights_must_sum_to_100(self) -> None:
        assert sum(MatchingWeights().model_dump().values()) == 100
        with pytest.raises(ValidationError, match="sum to 100"):
            MatchingWeights(required_skills=50)


class TestJobPosting:
    def _posting(self, **kw: object) -> JobPosting:
        base: dict[str, object] = dict(
            source=JobSource.INDEED,
            external_id="1",
            title="Engineer",
            company_name="Acme",
            canonical_url="https://example.com/jobs/1",
        )
        base.update(kw)
        return JobPosting(**base)  # type: ignore[arg-type]

    def test_missing_publication_date_is_flagged(self) -> None:
        assert self._posting().publication_date_unverified is True
        dated = self._posting(published_at=datetime(2026, 10, 1, tzinfo=UTC))
        assert dated.publication_date_unverified is False

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            self._posting(published_at=datetime(2026, 10, 1))  # noqa: DTZ001

    def test_timestamps_normalised_to_utc(self) -> None:
        est = timezone(timedelta(hours=-5))
        p = self._posting(published_at=datetime(2026, 10, 1, 17, tzinfo=est))
        assert p.published_at == datetime(2026, 10, 1, 22, tzinfo=UTC)
        assert p.published_at is not None and p.published_at.tzinfo == UTC

    def test_published_within(self) -> None:
        t = datetime(2026, 10, 1, 12, tzinfo=UTC)
        p = self._posting(published_at=t)
        assert p.is_published_within(t - timedelta(hours=24), t)
        assert not p.is_published_within(t + timedelta(seconds=1), t + timedelta(hours=1))
        assert not self._posting().is_published_within(t - timedelta(days=30), t)

    def test_salary_range_order(self) -> None:
        with pytest.raises(ValidationError):
            SalaryRange(minimum=200_000, maximum=100_000)


class TestResumeModels:
    def test_profile_dedupes_and_strips(self) -> None:
        p = ResumeProfile(skills=[" Python ", "python", "Go  lang", ""])
        assert p.skills == ["Python", "Go lang"]

    def test_profile_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ResumeProfile(clearance="TS/SCI")  # type: ignore[call-arg]

    def test_preferences_overlap_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both included and excluded"):
            JobSearchPreferences(locations=["Boston"], excluded_locations=["boston"])

    def test_criterion_scores_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CriterionScores(
                required_skills=101,
                experience_seniority=0,
                responsibilities_alignment=0,
                preferred_skills=0,
                education_certifications=0,
                domain_experience=0,
            )


class TestApplicationTransitions:
    def test_applied_is_terminal(self) -> None:
        with pytest.raises(InvalidStatusTransition):
            assert_transition(ApplicationStatus.APPLIED, ApplicationStatus.SUBMITTING)

    def test_submitting_cannot_silently_return_to_ready(self) -> None:
        # An in-flight submission must be resolved, never blindly retried.
        with pytest.raises(InvalidStatusTransition):
            assert_transition(ApplicationStatus.SUBMITTING, ApplicationStatus.READY_TO_APPLY)

    def test_uncertain_outcome_requires_reconciliation(self) -> None:
        assert_transition(ApplicationStatus.SUBMITTING, ApplicationStatus.NEEDS_RECONCILIATION)
        with pytest.raises(InvalidStatusTransition):
            assert_transition(ApplicationStatus.NEEDS_RECONCILIATION, ApplicationStatus.SUBMITTING)

    def test_same_status_is_noop(self) -> None:
        assert_transition(ApplicationStatus.MATCHED, ApplicationStatus.MATCHED)

    def test_every_status_has_rules(self) -> None:
        from job_agent.models.application import APPLICATION_TRANSITIONS

        assert set(APPLICATION_TRANSITIONS) == set(ApplicationStatus)
