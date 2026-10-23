"""Spec §10 metrics are emitted during real workflow runs (OpenTelemetry SDK reader)."""

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from job_agent.job_sources.base import RateLimited
from job_agent.models.job import JobSource
from job_agent.services.application_actions import ApplicationActions
from job_agent.services.application_service import (
    LookupOutcome,
    LookupResult,
)
from tests.conftest import make_raw
from tests.e2e.conftest import World

_READER = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[_READER]))


def collected() -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    data = _READER.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out.setdefault(m.name, []).extend(m.data.data_points)
    return out


def total(points: list[Any], **attrs: str) -> float:
    return sum(
        getattr(p, "value", getattr(p, "sum", 0))
        for p in points
        if all(p.attributes.get(k) == v for k, v in attrs.items())
    )


def test_workflow_emits_spec_metrics(world: World) -> None:
    rid = world.create_resume()
    world.feed(
        JobSource.LINKEDIN,
        make_raw("good"),
        make_raw("nodate", title="Platform Senior Software Engineer", published_at=None),
        make_raw(
            "old", title="Staff Senior Software Engineer", published_at="2026-01-01T00:00:00Z"
        ),
    )
    world.feed(
        JobSource.INDEED,
        make_raw("dup", apply_url="https://boards.greenhouse.io/acme/jobs/good"),
        errors=[RateLimited(retry_after=99999)],
    )  # rate limited -> FAILED source
    world.feed(
        JobSource.ZIPRECRUITER,
        make_raw("dup-zr", apply_url="https://boards.greenhouse.io/acme/jobs/good"),
    )
    world.integration.results = [TimeoutError()]
    world.run(rid)
    app = world.applications(rid)["good"]
    world.integration.lookups = [LookupResult(LookupOutcome.FOUND, external_ref="R")]
    asyncio.run(ApplicationActions(world.deps()).reconcile(app.id))

    m = collected()
    assert total(m["job_agent.jobs.discovered"], source="linkedin") >= 3
    assert total(m["job_agent.source.searches"], source="indeed", status="failed") >= 1
    assert total(m["job_agent.source.rate_limited"], source="indeed") >= 1
    assert total(m["job_agent.jobs.duplicates"]) >= 1
    assert total(m["job_agent.jobs.date_filter"], outcome="excluded:published_before_window") >= 1
    assert total(m["job_agent.jobs.date_filter"], outcome="review:missing_publication_date") >= 1
    assert m["job_agent.match.score"] and m["job_agent.llm.latency"]
    assert total(m["job_agent.llm.tokens"], direction="input") >= 10
    assert total(m["job_agent.match.decisions"], recommendation="APPLY") >= 1
    assert total(m["job_agent.applications.submissions"], integration="fake-ats") >= 1
    assert total(m["job_agent.applications.outcomes"], status="NEEDS_RECONCILIATION") >= 1
    assert total(m["job_agent.applications.review"], kind="match_review") >= 1
    assert total(m["job_agent.applications.outcomes"], status="PENDING_REVIEW") >= 1
    assert total(m["job_agent.applications.reconciliations"], outcome="found") >= 1
    assert total(m["job_agent.executions.finished"], trigger="INITIAL") >= 1
    # No personal data in metric attributes.
    for points in m.values():
        for p in points:
            assert not any("@" in str(v) or "Jane" in str(v) for v in p.attributes.values())
