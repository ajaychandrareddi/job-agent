"""Greenhouse Job Board API submission integration.

PERMISSION MODEL: Greenhouse's Job Board API accepts application submissions only
when authenticated with the *employer's* Job Board API key (it exists so employers can
build their own career pages). This integration is therefore enabled per board, only
for boards listed in ``GREENHOUSE_BOARD_KEYS``, i.e. where the employer has issued you
a key. Without a key a Greenhouse job is a manual application.

VERIFICATION STATUS: request/response handling follows Greenhouse's public Job Board
API documentation and is tested against a simulated server only. It has NOT been
verified against the live API; do so in a sandbox board before relying on it.

Endpoints used:
* ``GET  {base}/boards/{board}/jobs/{job_id}?questions=true``: application questions
* ``POST {base}/boards/{board}/jobs/{job_id}``: submit (HTTP Basic, key as username)
The Job Board API offers no application lookup, so reconciliation returns UNKNOWN and
uncertain submissions end in PENDING_REVIEW for the user to check.
"""

from __future__ import annotations

import base64
import re
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from job_agent.models.application import ApplicationQuestion, QuestionKind, QuestionOption
from job_agent.models.job import JobPosting
from job_agent.persistence.file_store import ResumeFileStore
from job_agent.services.answers import canonical_key, is_sensitive
from job_agent.services.application_service import (
    ApplicationPackage,
    LookupOutcome,
    LookupResult,
    SubmissionOutcome,
    SubmissionResult,
)

HOSTS = ("boards.greenhouse.io", "job-boards.greenhouse.io")
_PATH = re.compile(r"^/(?P<board>[A-Za-z0-9_-]+)/jobs/(?P<job>\d+)")
_TYPES = {
    "input_text": QuestionKind.TEXT,
    "textarea": QuestionKind.LONG_TEXT,
    "input_file": QuestionKind.FILE,
    "multi_value_single_select": QuestionKind.SINGLE_SELECT,
    "multi_value_multi_select": QuestionKind.MULTI_SELECT,
    "input_hidden": QuestionKind.TEXT,
}
_STANDARD_FIELDS = {"first_name", "last_name", "email", "phone"}


def parse_job_url(url: str | None) -> tuple[str, str] | None:
    """(board_token, job_id) from a Greenhouse job or embed URL."""
    if not url:
        return None
    parts = urlsplit(url)
    if (parts.hostname or "").lower() not in HOSTS:
        return None
    m = _PATH.match(parts.path)
    if m:
        return m.group("board"), m.group("job")
    if parts.path.rstrip("/") == "/embed/job_app":
        q = parse_qs(parts.query)
        if q.get("for") and q.get("token") and q["token"][0].isdigit():
            return q["for"][0], q["token"][0]
    return None


class GreenhouseIntegration:
    name = "greenhouse"

    def __init__(
        self,
        board_keys: dict[str, str],
        *,
        file_store: ResumeFileStore,
        api_base: str = "https://boards-api.greenhouse.io/v1",
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._keys = dict(board_keys)
        self._files = file_store
        self._base = api_base.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    def _ids(self, posting: JobPosting) -> tuple[str, str] | None:
        return parse_job_url(str(posting.application_url) if posting.application_url else None)

    def supports(self, posting: JobPosting) -> bool:
        ids = self._ids(posting)
        return ids is not None and ids[0] in self._keys

    def _url(self, posting: JobPosting) -> tuple[str, str]:
        ids = self._ids(posting)
        if ids is None or ids[0] not in self._keys:
            raise ValueError("posting is not on an authorised Greenhouse board")
        board, job = ids
        return board, f"{self._base}/boards/{board}/jobs/{job}"

    # ------------------------------------------------------------------ questions

    async def questions(self, posting: JobPosting) -> list[ApplicationQuestion]:
        _, url = self._url(posting)
        resp = await self._client.get(url, params={"questions": "true"})
        resp.raise_for_status()
        data = resp.json()
        out = [q for raw in data.get("questions", []) if (q := self._question(raw, False))]
        # Compliance (EEOC etc.) questions are demographic: always sensitive.
        for group in data.get("compliance", []) or []:
            for raw in group.get("questions", []) or []:
                q = self._question(raw, True)
                if q:
                    out.append(q)
        return out

    @staticmethod
    def _question(raw: dict[str, Any], compliance: bool) -> ApplicationQuestion | None:
        fields = raw.get("fields") or []
        if not fields:
            return None
        label = str(raw.get("label") or fields[0].get("name") or "question")
        # Resume / cover letter offer file + text fields; use the file.
        field = next((f for f in fields if f.get("type") == "input_file"), fields[0])
        name = str(field.get("name"))
        kind = _TYPES.get(str(field.get("type")), QuestionKind.TEXT)
        if name in _STANDARD_FIELDS:
            key = name
        elif name == "resume":
            key = "resume"
        elif name == "cover_letter":
            key = "cover_letter"
        else:
            key = canonical_key(label)
        options = [
            QuestionOption(label=str(v.get("label")), value=str(v.get("value")))
            for v in field.get("values") or []
        ]
        if kind is QuestionKind.SINGLE_SELECT and {o.label.lower() for o in options} == {
            "yes",
            "no",
        }:
            kind = QuestionKind.BOOLEAN
        return ApplicationQuestion(
            key=key,
            label=label,
            field_name=name,
            kind=kind,
            required=bool(raw.get("required")),
            options=options,
            sensitive=compliance or is_sensitive(key, label),
        )

    # --------------------------------------------------------------------- submit

    async def submit(
        self, package: ApplicationPackage, *, idempotency_key: str
    ) -> SubmissionResult:
        board, url = self._url(package.posting)
        body: dict[str, Any] = dict(package.answers)
        body.pop("resume", None)
        body["resume_content"] = base64.b64encode(
            self._files.load(package.resume_file_ref)
        ).decode()
        body["resume_content_filename"] = package.resume_filename
        try:
            resp = await self._client.post(url, json=body, auth=(self._keys[board], ""))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # The request never reached Greenhouse: definitively not submitted.
            return SubmissionResult(
                SubmissionOutcome.REJECTED,
                retryable=True,
                detail=f"could not connect to Greenhouse ({type(exc).__name__})",
            )
        # Any other transport error (read timeout, reset) propagates: the caller records
        # the outcome as UNCERTAIN because the request may have been processed.
        return self._interpret(resp, board, idempotency_key)

    @staticmethod
    def _interpret(resp: httpx.Response, board: str, idempotency_key: str) -> SubmissionResult:
        code = resp.status_code
        try:
            data: Any = resp.json()
        except ValueError:
            data = None
        message = _message(data) or resp.reason_phrase
        if 200 <= code < 300:
            if isinstance(data, dict) and data.get("success") and not data.get("error"):
                platform_id = next(
                    (str(data[k]) for k in ("application_id", "candidate_id", "id") if data.get(k)),
                    None,
                )
                return SubmissionResult(
                    SubmissionOutcome.CONFIRMED,
                    external_ref=f"greenhouse:{board}:{platform_id or idempotency_key}",
                    detail=str(data.get("success"))[:200],
                    verification="greenhouse_application_id"
                    if platform_id
                    else "greenhouse_success_response",
                )
            # 2xx without an explicit success statement is not proof of submission.
            return SubmissionResult(
                SubmissionOutcome.UNCERTAIN, detail=f"HTTP {code} without a success confirmation"
            )
        if code == 429:
            return SubmissionResult(
                SubmissionOutcome.REJECTED, retryable=True, detail="rate limited by Greenhouse"
            )
        if code in (401, 403):
            return SubmissionResult(
                SubmissionOutcome.REJECTED,
                detail="Greenhouse refused the board key; check it with the employer",
            )
        if code == 404:
            return SubmissionResult(
                SubmissionOutcome.REJECTED, detail="job not found on the board (closed or removed)"
            )
        if 400 <= code < 500:
            return SubmissionResult(
                SubmissionOutcome.REJECTED,
                detail=f"Greenhouse rejected the application: {message}"[:500],
            )
        return SubmissionResult(
            SubmissionOutcome.UNCERTAIN,
            detail=f"Greenhouse server error HTTP {code}; outcome unknown",
        )

    async def lookup(
        self, posting: JobPosting, *, idempotency_key: str, submitted_at: datetime
    ) -> LookupResult:
        return LookupResult(
            LookupOutcome.UNKNOWN, detail="the Greenhouse Job Board API cannot look up applications"
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _message(data: Any) -> str | None:
    if isinstance(data, dict):
        for key in ("error", "errors", "message"):
            if data.get(key):
                return str(data[key])[:300]
    return None
