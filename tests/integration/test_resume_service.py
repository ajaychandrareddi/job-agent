from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from job_agent.models.application import ApplicationStatus, SubmissionAttemptStatus
from job_agent.models.job import JobSource
from job_agent.models.resume import JobSearchPreferences, WorkflowStatus
from job_agent.persistence.db import create_session_factory
from job_agent.persistence.file_store import InMemoryResumeFileStore
from job_agent.persistence.orm import (
    Application,
    JobPostingRecord,
    ResumeWorkflow,
    SubmissionAttempt,
    User,
    WorkflowExecution,
)
from job_agent.repositories.resumes import UserRepository, WorkflowRepository, stable_resume_id
from job_agent.services.errors import Conflict, InvalidInput, NotFound
from job_agent.services.lifecycle import LifecycleAction as A
from job_agent.services.resume_parser import ResumeParser, sha256_hex
from job_agent.services.resume_service import ResumeService, suggest_preferences
from tests.conftest import SAMPLE_RESUME_LINES, make_docx

NOW = datetime(2026, 10, 1, 21, tzinfo=UTC)


def _second_resume() -> bytes:
    return make_docx([*SAMPLE_RESUME_LINES, "Data Engineer at Foo, Jan 2014 - May 2016"])


class TestResumeCreation:
    def test_create_resume_persists_profile_version_and_draft_workflow(
        self,
        resume_service: ResumeService,
        user: User,
        sample_docx: bytes,
        approved_prefs: JobSearchPreferences,
        file_store: InMemoryResumeFileStore,
    ) -> None:
        result = resume_service.create_resume(
            user.id, "Jane CV.docx", sample_docx, preferences=approved_prefs
        )
        r = result.resume
        assert result.created
        assert r.id == stable_resume_id(user.id, sha256_hex(sample_docx))
        assert r.label == "Jane CV" and r.current_version == 1
        assert result.version.profile["full_name"] == "Jane Doe"
        assert result.version.storage_ref in file_store.files
        assert r.preferences["preferred_titles"] == ["Senior Software Engineer"]
        assert r.workflow is not None
        assert r.workflow.status is WorkflowStatus.DRAFT
        assert r.workflow.initial_search_completed is False

    def test_reupload_is_idempotent(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        first = resume_service.create_resume(user.id, "a.docx", sample_docx)
        again = resume_service.create_resume(user.id, "b.docx", sample_docx)
        assert not again.created and again.resume.id == first.resume.id
        assert len(resume_service.resumes.list_for_user(user.id)) == 1

    def test_same_file_for_different_users_gets_different_ids(
        self, resume_service: ResumeService, user: User, other_user: User, sample_docx: bytes
    ) -> None:
        a = resume_service.create_resume(user.id, "a.docx", sample_docx).resume
        b = resume_service.create_resume(other_user.id, "a.docx", sample_docx).resume
        assert a.id != b.id

    def test_labels_unique_per_user(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        resume_service.create_resume(user.id, "cv.docx", sample_docx)
        second = resume_service.create_resume(user.id, "cv.docx", _second_resume())
        assert second.resume.label == "cv (2)"

    def test_invalid_upload_maps_to_invalid_input(
        self, resume_service: ResumeService, user: User
    ) -> None:
        with pytest.raises(InvalidInput, match="unsupported file type"):
            resume_service.create_resume(user.id, "cv.txt", b"plain text resume")

    def test_suggested_preferences_are_from_profile(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        version = resume_service.create_resume(user.id, "a.docx", sample_docx).version
        from job_agent.models.resume import ResumeProfile

        prefs = suggest_preferences(ResumeProfile.model_validate(version.profile))
        assert prefs.preferred_titles == ["Senior Software Engineer", "Software Engineer"]


class TestVersioning:
    def test_new_version_keeps_resume_id(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        created = resume_service.create_resume(user.id, "a.docx", sample_docx).resume
        updated = resume_service.replace_resume_file(
            user.id, created.id, "a2.docx", _second_resume()
        )
        assert updated.created and updated.resume.id == created.id
        assert updated.resume.current_version == 2 and updated.version.version == 2
        assert resume_service.resumes.current_version(updated.resume).version == 2

    def test_identical_replacement_is_noop(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        created = resume_service.create_resume(user.id, "a.docx", sample_docx).resume
        result = resume_service.replace_resume_file(user.id, created.id, "a.docx", sample_docx)
        assert not result.created and result.resume.current_version == 1

    def test_replacement_with_another_resumes_file_conflicts(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        a = resume_service.create_resume(user.id, "a.docx", sample_docx).resume
        resume_service.create_resume(user.id, "b.docx", _second_resume())
        with pytest.raises(Conflict):
            resume_service.replace_resume_file(user.id, a.id, "b.docx", _second_resume())

    def test_preferences_update(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        r = resume_service.create_resume(user.id, "a.docx", sample_docx).resume
        prefs = JobSearchPreferences(
            preferred_titles=["Staff Engineer"], excluded_companies=["Initech"]
        )
        updated = resume_service.update_preferences(user.id, r.id, prefs)
        assert updated.preferences["excluded_companies"] == ["Initech"]
        assert updated.preferences_updated_at is not None


class TestLifecycle:
    def _make(
        self, svc: ResumeService, user: User, content: bytes, prefs: JobSearchPreferences
    ) -> uuid.UUID:
        return svc.create_resume(
            user.id, f"{uuid.uuid4()}.docx", content, preferences=prefs
        ).resume.id

    def test_activation_requires_approved_titles(
        self, resume_service: ResumeService, user: User, sample_docx: bytes
    ) -> None:
        r = resume_service.create_resume(user.id, "a.docx", sample_docx).resume
        with pytest.raises(InvalidInput, match="preferred job title"):
            resume_service.change_status(user.id, r.id, A.ACTIVATE)

    def test_resumes_are_independent(
        self,
        resume_service: ResumeService,
        user: User,
        sample_docx: bytes,
        approved_prefs: JobSearchPreferences,
    ) -> None:
        a = self._make(resume_service, user, sample_docx, approved_prefs)
        b = self._make(resume_service, user, _second_resume(), approved_prefs)
        c = self._make(
            resume_service,
            user,
            make_docx([*SAMPLE_RESUME_LINES, "Intern at Bar, Jun 2015 - Aug 2015"]),
            approved_prefs,
        )
        for rid in (a, b, c):
            resume_service.change_status(user.id, rid, A.ACTIVATE)
        resume_service.change_status(user.id, b, A.PAUSE)
        resume_service.change_status(user.id, c, A.STOP)

        repo = WorkflowRepository(resume_service.s)
        status = {rid: repo.get_by_resume(rid).status for rid in (a, b, c)}  # type: ignore[union-attr]
        assert status == {
            a: WorkflowStatus.ACTIVE,
            b: WorkflowStatus.PAUSED,
            c: WorkflowStatus.STOPPED,
        }
        assert [w.resume_id for w in repo.list_by_status(WorkflowStatus.ACTIVE)] == [a]

    def test_listener_notified_once(
        self,
        resume_service: ResumeService,
        user: User,
        sample_docx: bytes,
        approved_prefs: JobSearchPreferences,
    ) -> None:
        events: list[tuple[WorkflowStatus, WorkflowStatus]] = []
        resume_service._listeners.append(lambda wf, old, new: events.append((old, new)))
        rid = self._make(resume_service, user, sample_docx, approved_prefs)
        _, changed1 = resume_service.change_status(user.id, rid, A.ACTIVATE)
        _, changed2 = resume_service.change_status(user.id, rid, A.ACTIVATE)  # idempotent
        assert (changed1, changed2) == (True, False)
        assert events == [(WorkflowStatus.DRAFT, WorkflowStatus.ACTIVE)]

    def test_pause_clears_next_run_and_activate_on_paused_conflicts(
        self,
        resume_service: ResumeService,
        user: User,
        sample_docx: bytes,
        approved_prefs: JobSearchPreferences,
    ) -> None:
        rid = self._make(resume_service, user, sample_docx, approved_prefs)
        wf, _ = resume_service.change_status(user.id, rid, A.ACTIVATE)
        wf.next_run_at = NOW
        wf, _ = resume_service.change_status(user.id, rid, A.PAUSE)
        assert wf.next_run_at is None
        with pytest.raises(Conflict, match="use 'resume'"):
            resume_service.change_status(user.id, rid, A.ACTIVATE)
        wf, _ = resume_service.change_status(user.id, rid, A.RESUME)
        assert wf.status is WorkflowStatus.ACTIVE

    def test_stopped_and_initial_flag_survive_restart(
        self, engine: sa.Engine, sample_docx: bytes, approved_prefs: JobSearchPreferences
    ) -> None:
        factory = create_session_factory(engine)
        with factory() as s1:
            u = UserRepository(s1).get_or_create("auth0|restart")
            svc = ResumeService(s1, ResumeParser(), InMemoryResumeFileStore())
            rid = svc.create_resume(
                u.id, "a.docx", sample_docx, preferences=approved_prefs
            ).resume.id
            wf, _ = svc.change_status(u.id, rid, A.ACTIVATE)
            WorkflowRepository(s1).mark_initial_search_completed(wf, NOW)
            svc.change_status(u.id, rid, A.STOP)
            s1.commit()
            uid = u.id
        with factory() as s2:  # simulates a process restart
            wf2 = WorkflowRepository(s2).get_by_resume(rid)
            assert wf2 is not None
            assert wf2.status is WorkflowStatus.STOPPED
            assert wf2.initial_search_completed is True
            assert wf2.initial_search_completed_at == NOW
            svc2 = ResumeService(s2, ResumeParser(), InMemoryResumeFileStore())
            wf3, _ = svc2.change_status(uid, rid, A.ACTIVATE)  # explicit reactivation
            assert wf3.status is WorkflowStatus.ACTIVE and wf3.initial_search_completed


class TestAuthorizationScoping:
    def test_other_users_resume_is_not_found(
        self,
        resume_service: ResumeService,
        user: User,
        other_user: User,
        sample_docx: bytes,
        approved_prefs: JobSearchPreferences,
    ) -> None:
        rid = resume_service.create_resume(user.id, "a.docx", sample_docx).resume.id
        with pytest.raises(NotFound):
            resume_service.get_resume(other_user.id, rid)
        with pytest.raises(NotFound):
            resume_service.update_preferences(other_user.id, rid, approved_prefs)
        with pytest.raises(NotFound):
            resume_service.change_status(other_user.id, rid, A.STOP)
        with pytest.raises(NotFound):
            resume_service.replace_resume_file(other_user.id, rid, "x.docx", _second_resume())
        assert resume_service.resumes.list_for_user(other_user.id) == []


# --------------------------------------------------------------------------- constraints


@pytest.fixture
def resume_and_job(
    resume_service: ResumeService, user: User, sample_docx: bytes, session: Session
) -> tuple[uuid.UUID, uuid.UUID]:
    rid = resume_service.create_resume(user.id, "a.docx", sample_docx).resume.id
    job = JobPostingRecord(
        source=JobSource.LINKEDIN,
        external_id="li-1",
        canonical_key="k" * 64,
        title="SWE",
        company_name="Acme",
        canonical_url="https://x/1",
        normalized={},
    )
    session.add(job)
    session.flush()
    return rid, job.id


class TestDatabaseSafeguards:
    def _app(
        self, user: User, rid: uuid.UUID, job_id: uuid.UUID, key: str, version: int = 1
    ) -> Application:
        return Application(
            application_key=key,
            user_id=user.id,
            resume_id=rid,
            resume_version=version,
            job_posting_id=job_id,
            status=ApplicationStatus.DISCOVERED,
        )

    def test_duplicate_application_key_rejected(
        self, session: Session, user: User, resume_and_job: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        rid, jid = resume_and_job
        session.add(self._app(user, rid, jid, "a" * 64))
        session.flush()
        session.add(self._app(user, rid, jid, "a" * 64))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_new_resume_version_cannot_duplicate_application(
        self, session: Session, user: User, resume_and_job: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        rid, jid = resume_and_job
        session.add(self._app(user, rid, jid, "a" * 64, version=1))
        session.flush()
        session.add(self._app(user, rid, jid, "b" * 64, version=2))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_different_resumes_may_apply_to_same_job(
        self,
        session: Session,
        user: User,
        resume_service: ResumeService,
        resume_and_job: tuple[uuid.UUID, uuid.UUID],
    ) -> None:
        rid, jid = resume_and_job
        rid2 = resume_service.create_resume(user.id, "b.docx", _second_resume()).resume.id
        session.add_all([self._app(user, rid, jid, "a" * 64), self._app(user, rid2, jid, "b" * 64)])
        session.flush()

    def test_only_one_open_submission_attempt(
        self, session: Session, user: User, resume_and_job: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        rid, jid = resume_and_job
        app = self._app(user, rid, jid, "a" * 64)
        session.add(app)
        session.flush()

        def attempt(n: int, status: SubmissionAttemptStatus) -> SubmissionAttempt:
            return SubmissionAttempt(
                application_id=app.id,
                attempt_number=n,
                idempotency_key=f"{app.id}:{n}",
                status=status,
            )

        session.add_all(
            [
                attempt(1, SubmissionAttemptStatus.REJECTED),
                attempt(2, SubmissionAttemptStatus.UNCERTAIN),
            ]
        )
        session.flush()
        session.add(attempt(3, SubmissionAttemptStatus.STARTED))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_only_one_running_execution_per_resume(
        self, session: Session, user: User, resume_and_job: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        rid, _ = resume_and_job
        wf = session.scalar(sa.select(ResumeWorkflow).where(ResumeWorkflow.resume_id == rid))
        assert wf is not None

        def execution(status: str) -> WorkflowExecution:
            return WorkflowExecution(
                resume_id=rid,
                workflow_id=wf.id,
                resume_version=1,
                trigger="SCHEDULED",
                status=status,
                search_window_start=NOW - timedelta(hours=24),
                search_window_end=NOW,
                thread_id=str(uuid.uuid4()),
            )

        session.add_all([execution("SUCCEEDED"), execution("RUNNING")])
        session.flush()
        session.add(execution("RUNNING"))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_naive_datetime_rejected_by_column_type(self, session: Session, user: User) -> None:
        user.updated_at = datetime(2026, 1, 1)  # noqa: DTZ001
        with pytest.raises(StatementError, match="naive datetime"):
            session.flush()

    def test_datetimes_round_trip_as_aware_utc(self, engine: sa.Engine) -> None:
        factory = create_session_factory(engine)
        with factory() as s:
            u = UserRepository(s).get_or_create("auth0|tz")
            u.updated_at = datetime(2026, 1, 1, 12, tzinfo=UTC)
            s.commit()
            uid = u.id
        with factory() as s:
            loaded = s.get(User, uid)
            assert loaded is not None and loaded.updated_at.tzinfo is not None
            assert loaded.updated_at == datetime(2026, 1, 1, 12, tzinfo=UTC)

    def test_get_or_create_user_is_idempotent(self, session: Session) -> None:
        a = UserRepository(session).get_or_create("auth0|same")
        b = UserRepository(session).get_or_create("auth0|same")
        assert a.id == b.id


# --------------------------------------------------------------------------- approved answers


class TestApprovedAnswers:
    def test_upsert_list_delete_and_sensitivity(
        self, session: Session, user: User, other_user: User
    ) -> None:
        from job_agent.services.approved_answers import ApprovedAnswerService

        svc = ApprovedAnswerService(session)
        svc.upsert(user.id, "work_authorization_us", " Yes ")
        svc.upsert(user.id, "email", "jane@example.com")
        svc.upsert(user.id, "custom:gender", "Decline to self-identify", label="Gender")
        updated = svc.upsert(user.id, "work_authorization_us", "No")
        assert updated.answer == "No"
        rows = {r.question_key: r for r in svc.list(user.id)}
        assert set(rows) == {"work_authorization_us", "email", "custom:gender"}
        assert rows["work_authorization_us"].is_sensitive
        assert rows["custom:gender"].is_sensitive and not rows["email"].is_sensitive
        assert svc.list(other_user.id) == []  # scoped per user
        svc.delete(user.id, "email")
        with pytest.raises(NotFound):
            svc.delete(user.id, "email")
        with pytest.raises(NotFound):
            svc.delete(other_user.id, "work_authorization_us")

    @pytest.mark.parametrize("key", ["bogus", "custom:", "custom:Has Spaces", "x" * 300])
    def test_invalid_keys(self, session: Session, user: User, key: str) -> None:
        from job_agent.services.approved_answers import ApprovedAnswerService

        with pytest.raises(InvalidInput):
            ApprovedAnswerService(session).upsert(user.id, key, "v")

    @pytest.mark.parametrize("answer", ["", "   ", "x" * 5001])
    def test_invalid_answers(self, session: Session, user: User, answer: str) -> None:
        from job_agent.services.approved_answers import ApprovedAnswerService

        with pytest.raises(InvalidInput):
            ApprovedAnswerService(session).upsert(user.id, "email", answer)

    def test_values_are_never_logged(
        self, session: Session, user: User, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import structlog

        from job_agent.observability.logging import configure_logging
        from job_agent.services.approved_answers import ApprovedAnswerService

        structlog.reset_defaults()
        configure_logging("INFO", json=True)
        ApprovedAnswerService(session).upsert(user.id, "requires_sponsorship", "SECRET-VALUE")
        assert "SECRET-VALUE" not in capsys.readouterr().out
        structlog.reset_defaults()
