"""REST API tests against a real database (spec §9, case 18)."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from job_agent.api.container import AppContainer
from job_agent.config import Settings
from job_agent.main import create_app
from job_agent.models.job import JobSource
from job_agent.persistence.db import session_scope
from job_agent.persistence.orm import Application
from job_agent.scheduler.scheduler_service import SchedulerService
from job_agent.security.auth import RateLimiter, StaticTokenAuthenticator
from job_agent.services.application_actions import ApplicationActions
from job_agent.services.application_service import SubmissionOutcome, SubmissionResult
from job_agent.services.resume_parser import ResumeParser
from tests.conftest import SAMPLE_RESUME_LINES, make_docx, make_raw
from tests.e2e.conftest import World

ALICE, BOB = "alice-token-0123456789", "bob-token-0123456789x"
SAMPLE_DOCX = make_docx(SAMPLE_RESUME_LINES)
PREFS = {
    "preferred_titles": ["Senior Software Engineer"],
    "locations": ["New York, NY"],
    "requires_sponsorship": False,
    "authorized_to_work_countries": ["United States"],
}


def build(
    world: World,
    *,
    per_minute: int = 1000,
    max_upload: int = 5 * 1024 * 1024,
    scheduler: SchedulerService | None = None,
) -> AppContainer:
    world.settings = Settings(
        _env_file=None,
        max_upload_bytes=max_upload,  # type: ignore[call-arg]
        log_json=False,
    )
    deps = world.deps()
    return AppContainer(
        settings=world.settings,
        engine=world.engine,
        factory=world.factory,
        file_store=world.file_store,
        parser=ResumeParser(max_upload_bytes=max_upload),
        deps=deps,
        actions=ApplicationActions(deps),
        authenticator=StaticTokenAuthenticator({"alice": ALICE, "bob": BOB}),
        limiter=RateLimiter(per_minute),
        scheduler=scheduler,
        owns_engine=False,
    )


@pytest.fixture
def client(world: World) -> Iterator[TestClient]:
    with TestClient(create_app(container=build(world)), raise_server_exceptions=False) as c:
        yield c


def h(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def upload(
    client: TestClient, token: str = ALICE, lines: list[str] | None = None, **form: Any
) -> Any:
    # Generated once: python-docx embeds a timestamp, so regenerating would change bytes.
    content = make_docx(lines) if lines else SAMPLE_DOCX
    return client.post(
        "/api/v1/resumes",
        headers=h(token),
        data=form,
        files={"file": ("jane.docx", content, "application/octet-stream")},
    )


def active_resume(client: TestClient, token: str = ALICE) -> str:
    rid = upload(client, token).json()["resume"]["id"]
    assert (
        client.put(f"/api/v1/resumes/{rid}/preferences", headers=h(token), json=PREFS).status_code
        == 200
    )
    assert client.post(f"/api/v1/resumes/{rid}/activate", headers=h(token)).status_code == 200
    return str(rid)


# --------------------------------------------------------------------------- ops & auth


def test_health_endpoints_need_no_auth_and_set_security_headers(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.json() == {"status": "ok"} and r.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/readyz").json() == {"status": "ready"}


@pytest.mark.parametrize(
    "headers", [{}, h("wrong-token-0000000000"), {"Authorization": f"Basic {ALICE}"}]
)
def test_unauthenticated_requests_are_rejected(client: TestClient, headers: dict[str, str]) -> None:
    r = client.get("/api/v1/resumes", headers=headers)
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"


def test_rate_limits(world: World) -> None:
    with TestClient(create_app(container=build(world, per_minute=3))) as c:
        codes = [c.get("/api/v1/resumes", headers=h(ALICE)).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert int(c.get("/api/v1/resumes", headers=h(ALICE)).headers["Retry-After"]) >= 1
        assert c.get("/api/v1/resumes", headers=h(BOB)).status_code == 200  # per user
        fails = [c.get("/api/v1/resumes", headers=h("bad-" * 6)).status_code for _ in range(4)]
        assert fails == [401, 401, 401, 429]  # repeated bad tokens are throttled


# --------------------------------------------------------------------------- resumes


def test_upload_and_read_resume(client: TestClient) -> None:
    r = upload(client, label="Backend CV", preferences=json.dumps(PREFS))
    assert r.status_code == 201
    body = r.json()
    assert body["created"] and body["resume"]["label"] == "Backend CV"
    assert body["resume"]["workflow"]["status"] == "DRAFT"
    assert body["resume"]["preferences"]["preferred_titles"] == ["Senior Software Engineer"]
    assert "Python" in body["resume"]["profile"]["skills"]
    assert body["suggested_preferences"]["preferred_titles"][0] == "Senior Software Engineer"
    again = upload(client)
    assert again.status_code == 200 and not again.json()["created"]  # idempotent
    rid = body["resume"]["id"]
    assert [x["id"] for x in client.get("/api/v1/resumes", headers=h(ALICE)).json()] == [rid]
    detail = client.get(f"/api/v1/resumes/{rid}", headers=h(ALICE)).json()
    assert detail["file_type"] == "docx" and detail["current_version"] == 1


def test_new_resume_version(client: TestClient) -> None:
    rid = upload(client).json()["resume"]["id"]
    r = client.put(
        f"/api/v1/resumes/{rid}/file",
        headers=h(ALICE),
        files={
            "file": (
                "v2.docx",
                make_docx([*SAMPLE_RESUME_LINES, "Mentor at Foo, Jan 2021 - Present"]),
                "application/octet-stream",
            )
        },
    )
    assert r.status_code == 200 and r.json()["resume"]["current_version"] == 2
    assert r.json()["resume"]["id"] == rid


@pytest.mark.parametrize(
    ("files", "data", "code"),
    [
        ({"file": ("cv.txt", b"plain text", "text/plain")}, {}, 422),
        (
            {"file": ("cv.docx", make_docx(SAMPLE_RESUME_LINES), "x")},
            {"preferences": "{not json"},
            422,
        ),
        (
            {"file": ("cv.docx", make_docx(SAMPLE_RESUME_LINES), "x")},
            {"preferences": json.dumps({"locations": ["A"], "excluded_locations": ["a"]})},
            422,
        ),
        ({}, {}, 422),
    ],
)
def test_invalid_uploads(
    client: TestClient, files: dict[str, Any], data: dict[str, Any], code: int
) -> None:
    assert (
        client.post("/api/v1/resumes", headers=h(ALICE), files=files, data=data).status_code == code
    )


def test_upload_size_limit(world: World) -> None:
    with TestClient(create_app(container=build(world, max_upload=2048))) as c:
        r = upload(c)
        assert r.status_code == 413 and "2048" in r.json()["detail"]


def test_lifecycle_endpoints(client: TestClient) -> None:
    rid = upload(client).json()["resume"]["id"]
    base = f"/api/v1/resumes/{rid}"
    r = client.post(f"{base}/activate", headers=h(ALICE))
    assert r.status_code == 422 and "preferred job title" in r.json()["detail"]
    client.put(f"{base}/preferences", headers=h(ALICE), json=PREFS)
    first = client.post(f"{base}/activate", headers=h(ALICE)).json()
    assert first["changed"] and first["workflow"]["status"] == "ACTIVE"
    assert not client.post(f"{base}/activate", headers=h(ALICE)).json()["changed"]
    assert client.post(f"{base}/pause", headers=h(ALICE)).json()["workflow"]["status"] == ("PAUSED")
    conflict = client.post(f"{base}/activate", headers=h(ALICE))
    assert conflict.status_code == 409 and "resume" in conflict.json()["detail"]
    assert client.post(f"{base}/resume", headers=h(ALICE)).json()["workflow"]["status"] == (
        "ACTIVE"
    )
    schedule = client.get(f"{base}/schedule", headers=h(ALICE)).json()
    assert schedule["daily_time"] == "17:00" and schedule["timezone"] == "America/New_York"
    assert schedule["scheduled"] and not schedule["initial_search_completed"]
    assert client.post(f"{base}/stop", headers=h(ALICE)).json()["workflow"]["status"] == ("STOPPED")
    assert client.get(f"{base}/schedule", headers=h(ALICE)).json()["next_run_at"] is None


def test_activation_with_in_process_scheduler(world: World) -> None:
    svc = SchedulerService(session_factory=world.factory, settings=world.settings, runner=None)  # type: ignore[arg-type]
    with TestClient(create_app(container=build(world, scheduler=svc))) as c:
        rid = active_resume(c)
        assert svc.registered_resumes() == {rid}  # registered right after commit
        c.post(f"/api/v1/resumes/{rid}/pause", headers=h(ALICE))
        assert svc.registered_resumes() == set()


# --------------------------------------------------------------------------- history


def run_once(world: World, rid: str, **raw: Any) -> None:
    from job_agent.workflows.runner import WorkflowRunner

    world.feed(
        JobSource.LINKEDIN,
        make_raw("good"),
        make_raw("low", title="Data Engineer Senior Software Engineer"),
        make_raw("easy", title="Platform Senior Software Engineer", apply_method="easy_apply"),
    )
    world.assessor.scores = {"Data Engineer": 40}
    asyncio.run(WorkflowRunner(world.deps()).run(uuid.UUID(rid)))


def test_history_endpoints(client: TestClient, world: World) -> None:
    rid = active_resume(client)
    run_once(world, rid)
    base = f"/api/v1/resumes/{rid}"
    ex = client.get(f"{base}/executions", headers=h(ALICE)).json()
    assert ex["total"] == 1 and ex["items"][0]["trigger"] == "INITIAL"
    assert ex["items"][0]["summary"]["applied"] == 1
    matches = client.get(f"{base}/matches", headers=h(ALICE)).json()
    assert matches["total"] == 3 and matches["items"][0]["score"] >= matches["items"][-1]["score"]
    skips = client.get(
        f"{base}/matches", headers=h(ALICE), params={"recommendation": "SKIP"}
    ).json()
    assert [m["score"] for m in skips["items"]] == [40]
    assert (
        client.get(f"{base}/matches", headers=h(ALICE), params={"min_score": 60}).json()["total"]
        == 2
    )
    apps = client.get(f"{base}/applications", headers=h(ALICE)).json()
    assert sorted(a["status"] for a in apps["items"]) == [
        "APPLICATION_UNAVAILABLE",
        "APPLIED",
        "SKIPPED",
    ]
    page = client.get(
        f"{base}/applications", headers=h(ALICE), params={"limit": 1, "offset": 1}
    ).json()
    assert page["total"] == 3 and len(page["items"]) == 1 and page["limit"] == 1
    manual = client.get(
        f"{base}/applications", headers=h(ALICE), params={"status": "APPLICATION_UNAVAILABLE"}
    ).json()["items"][0]
    detail = client.get(f"/api/v1/applications/{manual['id']}", headers=h(ALICE)).json()
    assert detail["manual_package"]["apply_url"].startswith("https://")
    assert detail["job"]["title"] == "Platform Senior Software Engineer"
    assert (
        client.get(f"{base}/applications", headers=h(ALICE), params={"limit": 100_000}).json()[
            "limit"
        ]
        == 100
    )  # capped


def test_retry_and_reconcile_endpoints(client: TestClient, world: World) -> None:
    rid = active_resume(client)
    world.integration.results = [
        SubmissionResult(SubmissionOutcome.REJECTED, detail="busy", retryable=True)
    ]
    run_once(world, rid)
    failed = client.get(
        f"/api/v1/resumes/{rid}/applications", headers=h(ALICE), params={"status": "FAILED"}
    ).json()["items"][0]
    r = client.post(f"/api/v1/applications/{failed['id']}/retry", headers=h(ALICE))
    assert r.status_code == 200 and r.json()["status"] == "APPLIED"
    again = client.post(f"/api/v1/applications/{failed['id']}/retry", headers=h(ALICE))
    assert again.status_code == 409  # APPLIED: never resubmitted
    noop = client.post(f"/api/v1/applications/{failed['id']}/reconcile", headers=h(ALICE))
    assert noop.status_code == 200 and noop.json()["detail"] == "nothing to reconcile"


# --------------------------------------------------------------------------- isolation


def test_users_cannot_access_each_others_data(client: TestClient, world: World) -> None:
    """Spec case 18."""
    rid = active_resume(client, ALICE)
    run_once(world, rid)
    with session_scope(world.factory) as s:
        app_id = str(s.query(Application).first().id)  # type: ignore[union-attr]
    client.put("/api/v1/answers/email", headers=h(ALICE), json={"answer": "a@x.io"})
    base = f"/api/v1/resumes/{rid}"
    for method, path, kw in [
        ("get", base, {}),
        ("put", f"{base}/preferences", {"json": PREFS}),
        ("post", f"{base}/pause", {}),
        ("post", f"{base}/stop", {}),
        ("get", f"{base}/schedule", {}),
        ("get", f"{base}/executions", {}),
        ("get", f"{base}/matches", {}),
        ("get", f"{base}/applications", {}),
        ("get", f"/api/v1/applications/{app_id}", {}),
        ("post", f"/api/v1/applications/{app_id}/retry", {}),
        ("post", f"/api/v1/applications/{app_id}/reconcile", {}),
        ("delete", "/api/v1/answers/email", {}),
    ]:
        r = getattr(client, method)(path, headers=h(BOB), **kw)
        assert r.status_code == 404, (method, path, r.status_code)
    assert client.get("/api/v1/resumes", headers=h(BOB)).json() == []
    assert client.get("/api/v1/answers", headers=h(BOB)).json() == []
    # Alice's data is untouched.
    assert client.get(f"{base}/schedule", headers=h(ALICE)).json()["status"] == "ACTIVE"


# --------------------------------------------------------------------------- answers & errors


def test_approved_answers_endpoints(client: TestClient) -> None:
    put = client.put(
        "/api/v1/answers/requires_sponsorship",
        headers=h(ALICE),
        json={"answer": "No", "label": "Will you require sponsorship?"},
    )
    assert put.status_code == 200 and put.json()["sensitive"] and put.json()["answer"] == "********"
    client.put("/api/v1/answers/email", headers=h(ALICE), json={"answer": "a@x.io"})
    listed = {a["key"]: a for a in client.get("/api/v1/answers", headers=h(ALICE)).json()}
    assert listed["email"]["answer"] == "a@x.io"
    assert listed["requires_sponsorship"]["answer"] == "********"  # never echoed
    assert (
        client.put("/api/v1/answers/bogus", headers=h(ALICE), json={"answer": "x"}).status_code
        == 422
    )
    assert (
        client.put("/api/v1/answers/email", headers=h(ALICE), json={"answer": ""}).status_code
        == 422
    )
    assert client.delete("/api/v1/answers/email", headers=h(ALICE)).status_code == 204
    assert client.delete("/api/v1/answers/email", headers=h(ALICE)).status_code == 404


def test_unexpected_errors_do_not_leak_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from job_agent.repositories.resumes import ResumeRepository

    def boom(self: Any, user_id: Any) -> Any:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(ResumeRepository, "list_for_user", boom)
    r = client.get("/api/v1/resumes", headers=h(ALICE))
    assert r.status_code == 500 and r.json() == {"detail": "internal error"}


def test_unknown_resume_is_404(client: TestClient) -> None:
    assert client.get(f"/api/v1/resumes/{uuid.uuid4()}", headers=h(ALICE)).status_code == 404
    assert client.get("/api/v1/resumes/not-a-uuid", headers=h(ALICE)).status_code == 422


def test_openapi_lists_spec_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    expected = {
        "/api/v1/resumes": {"get", "post"},
        "/api/v1/resumes/{resume_id}": {"get"},
        "/api/v1/resumes/{resume_id}/preferences": {"put"},
        **{
            f"/api/v1/resumes/{{resume_id}}/{a}": {"post"}
            for a in ("activate", "pause", "resume", "stop")
        },
        **{
            f"/api/v1/resumes/{{resume_id}}/{g}": {"get"}
            for g in ("schedule", "executions", "matches", "applications")
        },
        "/api/v1/applications/{application_id}": {"get"},
        "/api/v1/applications/{application_id}/retry": {"post"},
        "/api/v1/applications/{application_id}/reconcile": {"post"},
    }
    for path, methods in expected.items():
        assert methods <= set(paths[path]), path
