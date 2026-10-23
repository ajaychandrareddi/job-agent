"""Greenhouse integration against a simulated Job Board API (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from job_agent.integrations.greenhouse import GreenhouseIntegration, parse_job_url
from job_agent.models.application import QuestionKind
from job_agent.models.job import JobSource
from job_agent.persistence.file_store import InMemoryResumeFileStore
from job_agent.services.application_service import (
    ApplicationPackage,
    LookupOutcome,
    SubmissionOutcome,
)
from job_agent.services.normalization import normalize_raw_posting
from tests.conftest import RUN_AT, make_raw

JOB_URL = "https://boards.greenhouse.io/acme/jobs/4012345"

QUESTIONS = {
    "id": 4012345,
    "title": "Senior Software Engineer",
    "questions": [
        {
            "required": True,
            "label": "First Name",
            "fields": [{"name": "first_name", "type": "input_text", "values": []}],
        },
        {
            "required": True,
            "label": "Email",
            "fields": [{"name": "email", "type": "input_text", "values": []}],
        },
        {
            "required": True,
            "label": "Resume/CV",
            "fields": [
                {"name": "resume", "type": "input_file", "values": []},
                {"name": "resume_text", "type": "textarea", "values": []},
            ],
        },
        {
            "required": False,
            "label": "Why do you want to join Acme?",
            "fields": [{"name": "question_111", "type": "textarea", "values": []}],
        },
        {
            "required": True,
            "label": "Will you now or in the future require sponsorship?",
            "fields": [
                {
                    "name": "question_222",
                    "type": "multi_value_single_select",
                    "values": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                }
            ],
        },
        {"required": False, "label": "empty", "fields": []},
    ],
    "compliance": [
        {
            "type": "eeoc",
            "questions": [
                {
                    "required": False,
                    "label": "Gender",
                    "fields": [
                        {
                            "name": "gender",
                            "type": "multi_value_single_select",
                            "values": [{"label": "Decline", "value": 3}],
                        }
                    ],
                }
            ],
        }
    ],
}


def posting(url: str = JOB_URL):  # type: ignore[no-untyped-def]
    return normalize_raw_posting(make_raw("g1", apply_url=url), JobSource.LINKEDIN, now=RUN_AT)


def integration(
    handler: Callable[[httpx.Request], httpx.Response], keys: dict[str, str] | None = None
) -> tuple[GreenhouseIntegration, InMemoryResumeFileStore]:
    store = InMemoryResumeFileStore()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GreenhouseIntegration(
        keys if keys is not None else {"acme": "secret-key"}, file_store=store, client=client
    ), store


def package(store: InMemoryResumeFileStore) -> ApplicationPackage:
    ref = store.save(uuid.uuid4(), uuid.uuid4(), 1, "ab" * 32, b"%PDF-resume", "pdf")
    return ApplicationPackage(
        application_id=uuid.uuid4(),
        resume_id=uuid.uuid4(),
        resume_version=1,
        resume_file_ref=ref,
        posting=posting(),
        answers={"first_name": "Jane", "email": "j@x.io", "question_222": "0"},
        resume_filename="jane.pdf",
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (JOB_URL, ("acme", "4012345")),
        ("https://job-boards.greenhouse.io/acme/jobs/77?gh_src=x", ("acme", "77")),
        ("https://boards.greenhouse.io/embed/job_app?for=acme&token=55", ("acme", "55")),
        ("https://boards.greenhouse.io/embed/job_app?for=acme&token=abc", None),
        ("https://jobs.lever.co/acme/123", None),
        (None, None),
    ],
)
def test_parse_job_url(url: str | None, expected: tuple[str, str] | None) -> None:
    assert parse_job_url(url) == expected


def test_supports_only_authorised_boards() -> None:
    gh, _ = integration(lambda r: httpx.Response(200), {"acme": "k"})
    assert gh.supports(posting())
    assert not gh.supports(posting("https://boards.greenhouse.io/other/jobs/1"))
    assert not gh.supports(posting("https://jobs.lever.co/acme/1"))
    with pytest.raises(ValueError):
        gh._url(posting("https://boards.greenhouse.io/other/jobs/1"))


def test_questions_are_mapped() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=QUESTIONS)

    gh, _ = integration(handler)
    questions = asyncio.run(gh.questions(posting()))
    assert str(seen[0].url) == (
        "https://boards-api.greenhouse.io/v1/boards/acme/jobs/4012345?questions=true"
    )
    by_key = {q.key: q for q in questions}
    assert list(by_key) == [
        "first_name",
        "email",
        "resume",
        "why_interested",
        "requires_sponsorship",
        "custom:gender",
    ]
    assert by_key["resume"].kind is QuestionKind.FILE and by_key["resume"].field_name == "resume"
    assert by_key["why_interested"].kind is QuestionKind.LONG_TEXT
    sponsor = by_key["requires_sponsorship"]
    assert sponsor.kind is QuestionKind.BOOLEAN and sponsor.sensitive and sponsor.required
    assert sponsor.field_name == "question_222"
    assert by_key["custom:gender"].sensitive and not by_key["custom:gender"].required


def test_questions_http_error_raises() -> None:
    gh, _ = integration(lambda r: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(gh.questions(posting()))


def test_submit_success_with_platform_id() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"success": "Candidate saved successfully", "application_id": 987}
        )

    gh, store = integration(handler)
    result = asyncio.run(gh.submit(package(store), idempotency_key="app:1"))
    assert result.outcome is SubmissionOutcome.CONFIRMED
    assert result.external_ref == "greenhouse:acme:987"
    assert result.verification == "greenhouse_application_id"
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/boards/acme/jobs/4012345"
    assert req.headers["authorization"] == "Basic " + base64.b64encode(b"secret-key:").decode()
    body = json.loads(req.content)
    assert body["first_name"] == "Jane" and body["question_222"] == "0"
    assert base64.b64decode(body["resume_content"]) == b"%PDF-resume"
    assert body["resume_content_filename"] == "jane.pdf"


def test_submit_success_without_id_uses_success_message() -> None:
    gh, store = integration(lambda r: httpx.Response(200, json={"success": "Candidate saved"}))
    result = asyncio.run(gh.submit(package(store), idempotency_key="app:1"))
    assert result.outcome is SubmissionOutcome.CONFIRMED
    assert result.external_ref == "greenhouse:acme:app:1"
    assert result.verification == "greenhouse_success_response"


@pytest.mark.parametrize(
    ("status", "body", "outcome", "retryable", "detail"),
    [
        (200, {"status": "ok"}, SubmissionOutcome.UNCERTAIN, False, "without a success"),
        (200, "not json", SubmissionOutcome.UNCERTAIN, False, "without a success"),
        (429, {}, SubmissionOutcome.REJECTED, True, "rate limited"),
        (401, {}, SubmissionOutcome.REJECTED, False, "board key"),
        (404, {}, SubmissionOutcome.REJECTED, False, "not found"),
        (422, {"error": "Email is invalid"}, SubmissionOutcome.REJECTED, False, "Email is invalid"),
        (500, {}, SubmissionOutcome.UNCERTAIN, False, "outcome unknown"),
    ],
)
def test_submit_response_interpretation(
    status: int, body: object, outcome: SubmissionOutcome, retryable: bool, detail: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    gh, store = integration(handler)
    result = asyncio.run(gh.submit(package(store), idempotency_key="k"))
    assert (result.outcome, result.retryable) == (outcome, retryable)
    assert detail in (result.detail or "")


def test_connection_failure_is_definitely_not_submitted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    gh, store = integration(handler)
    result = asyncio.run(gh.submit(package(store), idempotency_key="k"))
    assert result.outcome is SubmissionOutcome.REJECTED and result.retryable


def test_read_timeout_propagates_as_unknown_outcome() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    gh, store = integration(handler)
    with pytest.raises(httpx.ReadTimeout):  # the workflow records this as UNCERTAIN
        asyncio.run(gh.submit(package(store), idempotency_key="k"))


def test_lookup_is_unknown_and_client_closes() -> None:
    gh, _ = integration(lambda r: httpx.Response(200))
    result = asyncio.run(gh.lookup(posting(), idempotency_key="k", submitted_at=datetime.now(UTC)))
    assert result.outcome is LookupOutcome.UNKNOWN
    asyncio.run(gh.aclose())
