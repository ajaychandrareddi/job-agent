from __future__ import annotations

import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest

from job_agent.config import LLMProvider, Settings
from job_agent.llm.factory import LLMConfigurationError, build_chat_model
from job_agent.models.resume import WorkflowStatus as S
from job_agent.persistence.file_store import LocalResumeFileStore
from job_agent.services.lifecycle import LifecycleAction as A
from job_agent.services.lifecycle import LifecycleError, may_run, next_status


@pytest.mark.parametrize(
    ("current", "action", "expected", "changed"),
    [
        (S.DRAFT, A.ACTIVATE, S.ACTIVE, True),
        (S.ACTIVE, A.ACTIVATE, S.ACTIVE, False),
        (S.ACTIVE, A.PAUSE, S.PAUSED, True),
        (S.PAUSED, A.PAUSE, S.PAUSED, False),
        (S.PAUSED, A.RESUME, S.ACTIVE, True),
        (S.ACTIVE, A.RESUME, S.ACTIVE, False),
        (S.ACTIVE, A.STOP, S.STOPPED, True),
        (S.PAUSED, A.STOP, S.STOPPED, True),
        (S.DRAFT, A.STOP, S.STOPPED, True),
        (S.STOPPED, A.STOP, S.STOPPED, False),
        (S.STOPPED, A.ACTIVATE, S.ACTIVE, True),
    ],
)
def test_allowed_transitions(current: S, action: A, expected: S, changed: bool) -> None:
    assert next_status(current, action) == (expected, changed)


@pytest.mark.parametrize(
    ("current", "action"),
    [
        (S.PAUSED, A.ACTIVATE),
        (S.STOPPED, A.PAUSE),
        (S.STOPPED, A.RESUME),
        (S.DRAFT, A.PAUSE),
        (S.DRAFT, A.RESUME),
    ],
)
def test_forbidden_transitions(current: S, action: A) -> None:
    with pytest.raises(LifecycleError):
        next_status(current, action)


def test_only_active_may_run() -> None:
    assert [s for s in S if may_run(s)] == [S.ACTIVE]


class TestLocalFileStore:
    def test_round_trip_and_permissions(self, tmp_path: Path) -> None:
        store = LocalResumeFileStore(tmp_path)
        ref = store.save(uuid4(), uuid4(), 1, "ab" * 32, b"data", "pdf")
        assert store.load(ref) == b"data"
        assert (tmp_path / ref).stat().st_mode & 0o777 == 0o600
        store.delete(ref)
        assert not (tmp_path / ref).exists()

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            LocalResumeFileStore(tmp_path).load("../../etc/passwd")


class TestLLMFactory:
    def test_fake_provider_has_no_model(self) -> None:
        with pytest.raises(LLMConfigurationError):
            build_chat_model(Settings(_env_file=None, llm_provider=LLMProvider.FAKE))  # type: ignore[call-arg]

    def test_missing_optional_extra_is_explained(self) -> None:
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            with pytest.raises(LLMConfigurationError, match="optional extra"):
                build_chat_model(Settings(_env_file=None, llm_provider=LLMProvider.OPENAI))  # type: ignore[call-arg]
        else:
            pytest.skip("langchain-openai is installed")

    def test_anthropic_model_built_without_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Current Claude models return 400 "temperature is deprecated" if it is sent.
        captured: dict[str, object] = {}

        class FakeChatAnthropic:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        module = types.ModuleType("langchain_anthropic")
        module.ChatAnthropic = FakeChatAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_anthropic", module)

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            llm_provider=LLMProvider.ANTHROPIC,
            llm_model="claude-haiku-5-5",
            llm_temperature=0.0,
        )
        build_chat_model(settings)
        assert captured["model_name"] == "claude-haiku-5-5"
        assert "temperature" not in captured


class TestDatabaseGuard:
    def test_refuses_non_test_postgres_database(self) -> None:
        from tests.db_guard import assert_test_database

        assert_test_database("sqlite://")
        assert_test_database("postgresql+psycopg://u@h/job_agent_test")
        with pytest.raises(pytest.fail.Exception, match="refusing to drop tables"):
            assert_test_database("postgresql+psycopg://u@h/job_agent")

    def test_ensure_test_db_refuses_non_test_names(self) -> None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "ensure_test_db.py"
        spec = importlib.util.spec_from_file_location("ensure_test_db", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.main("postgresql+psycopg://u@localhost:1/job_agent") == 1
