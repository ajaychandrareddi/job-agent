"""OpenTelemetry metrics and tracing (spec §10).

Instruments use the OpenTelemetry *API* only: they are no-ops until an SDK and exporter
are configured (``pip install 'job-agent[otel]'`` and the standard ``OTEL_*`` env vars,
or a ``MeterProvider`` set in code). Attributes never contain personal data, only
sources, statuses and reasons.
"""

from __future__ import annotations

from opentelemetry import metrics, trace

_meter = metrics.get_meter("job_agent", "1.0.0")
tracer = trace.get_tracer("job_agent", "1.0.0")

# Discovery
jobs_discovered = _meter.create_counter(
    "job_agent.jobs.discovered", unit="{posting}", description="Postings returned, by source"
)
postings_rejected = _meter.create_counter(
    "job_agent.jobs.rejected",
    unit="{posting}",
    description="Raw postings rejected during normalisation, by source",
)
source_searches = _meter.create_counter(
    "job_agent.source.searches", unit="{search}", description="Source searches, by status"
)
source_rate_limited = _meter.create_counter(
    "job_agent.source.rate_limited",
    unit="{event}",
    description="Rate-limit responses from job sources",
)
date_filter_results = _meter.create_counter(
    "job_agent.jobs.date_filter",
    unit="{job}",
    description="Publication-date filtering outcomes, by bucket and reason",
)
duplicates_folded = _meter.create_counter(
    "job_agent.jobs.duplicates",
    unit="{posting}",
    description="Duplicate postings folded into another posting",
)

# Matching / LLM
match_score = _meter.create_histogram(
    "job_agent.match.score", unit="1", description="Overall match score distribution (0-100)"
)
match_decisions = _meter.create_counter(
    "job_agent.match.decisions", unit="{job}", description="Match decisions, by recommendation"
)
llm_latency = _meter.create_histogram(
    "job_agent.llm.latency", unit="ms", description="LLM call latency, by purpose"
)
llm_tokens = _meter.create_counter(
    "job_agent.llm.tokens", unit="{token}", description="LLM tokens, by direction and purpose"
)
llm_failures = _meter.create_counter(
    "job_agent.llm.failures", unit="{call}", description="Failed or malformed LLM calls"
)

# Applications
submissions_started = _meter.create_counter(
    "job_agent.applications.submissions",
    unit="{attempt}",
    description="Submission attempts sent to a platform",
)
application_outcomes = _meter.create_counter(
    "job_agent.applications.outcomes",
    unit="{application}",
    description="Final application statuses (APPLIED, FAILED, NEEDS_RECONCILIATION, ...)",
)
review_routed = _meter.create_counter(
    "job_agent.applications.review",
    unit="{application}",
    description="Applications routed to review, by review kind",
)
reconciliations = _meter.create_counter(
    "job_agent.applications.reconciliations",
    unit="{lookup}",
    description="Reconciliation lookups, by outcome",
)
application_retries = _meter.create_counter(
    "job_agent.applications.retries", unit="{retry}", description="User-initiated retries"
)

# Workflows / scheduler
scheduler_run_duration = _meter.create_histogram(
    "job_agent.scheduler.run_duration",
    unit="s",
    description="Scheduled run duration, by kind and status",
)
executions_finished = _meter.create_counter(
    "job_agent.executions.finished",
    unit="{execution}",
    description="Finished executions, by status and trigger",
)
checkpoint_recoveries = _meter.create_counter(
    "job_agent.workflow.checkpoint_recoveries",
    unit="{execution}",
    description="Executions resumed from a checkpoint",
)
