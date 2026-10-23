# job-agent

An agentic job search and application workflow built with LangGraph, LangChain, FastAPI, PostgreSQL, and APScheduler. For each resume it searches job sources, scores every posting against that resume, and applies to postings that score 60 or higher, but only through permitted integrations and only with verified applicant information.

* Architecture, data model, state machines, and sequence diagrams: [docs/architecture.md](docs/architecture.md)
* Integrations that need credentials or approval: [docs/integrations.md](docs/integrations.md)
* Requirement-by-requirement acceptance review: [docs/acceptance.md](docs/acceptance.md)

> **Before you rely on automatic applications, read this.** No job board currently offers a *job-seeker-side* API that permits automated submission. LinkedIn, Indeed and ZipRecruiter require partner approval just to search, and Greenhouse accepts API applications only with an employer-issued key. Out of the box, the system searches whatever sources you have access to and scores every job. Jobs that qualify become **prepared manual applications**, each with the apply link, resume version, match evidence and your approved answers. It never fabricates results or bypasses platform controls.

## Build status

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation: config, models, schema and migrations, resume ingestion, repositories, lifecycle | ✅ done, tested |
| 2 | Job-source adapters, normalisation, deduplication, date filtering | ✅ done, tested. Real platform access still needs partner approval |
| 3 | Matching engine (LLM structured output + deterministic rules) | ✅ done, tested |
| 4 | LangGraph search graph, application subgraph, checkpointing | ✅ done, tested (PostgreSQL checkpointer included) |
| 5 | Scheduler (initial 30-day run, daily 17:00 ET, per-resume lifecycle) | ✅ done, tested (`python -m job_agent.scheduler`) |
| 6 | Application submission, verification, reconciliation | ✅ done, tested (Greenhouse integration: employer-key only, unverified live) |
| 7 | FastAPI routes, auth, metrics, deployment | ✅ done, tested (see the acceptance review) |

## Quick start (local)

```bash
cd job-agent
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env
```

Run the tests. They need no database server, because unit and integration tests use SQLite:

```bash
.venv/bin/pytest --cov
```

To run against PostgreSQL:

```bash
docker compose up -d db          # POSTGRES_HOST_PORT=55432 if 5432 is already taken
.venv/bin/alembic upgrade head
```

To run the full phase gate including the PostgreSQL checks. These use a **dedicated `job_agent_test` database**, created automatically. Tests drop tables, so they refuse any database whose name lacks "test".

```bash
TEST_DATABASE_URL=postgresql+psycopg://job_agent:job_agent@localhost:5432/job_agent_test scripts/validate.sh
```

To run the scheduler process locally (it needs PostgreSQL, which it also uses for the LangGraph checkpoints):

```bash
.venv/bin/python -m job_agent.scheduler
```

To start the full stack (database, migrations, API, scheduler):

```bash
docker compose up --build
```

`POSTGRES_HOST_PORT` and `API_HOST_PORT` override the published ports if 5432 or 8000 are taken. Before starting, set `API_STATIC_TOKENS` in `.env` (dev), or `API_AUTH_MODE=jwt` with your identity provider's settings (required when `APP_ENV=prod`).

## Using the API

Interactive docs: `http://localhost:8000/docs`. Every `/api/v1` call needs `Authorization: Bearer <token>`. Users are identified by the token's subject (JWT `sub`, or the subject paired with a static dev token) and can only see their own data.

```bash
# .env (dev only):  API_STATIC_TOKENS=dev-token-change-me-123:alice
TOKEN=dev-token-change-me-123
API=http://localhost:8000/api/v1

# 1. Upload a resume (PDF or DOCX). The response includes the parsed profile and
#    *suggested* preferences; nothing is approved until step 2.
curl -s -H "Authorization: Bearer $TOKEN" -F "file=@resume.pdf" -F "label=Backend" $API/resumes

# 2. Approve search preferences and hard constraints
curl -s -X PUT -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"preferred_titles":["Senior Software Engineer"],"locations":["New York, NY"],
       "work_arrangements":["remote","hybrid"],"employment_types":["full_time"],
       "salary":{"minimum_annual":150000},"requires_sponsorship":false,
       "authorized_to_work_countries":["United States"]}' \
  $API/resumes/$RESUME_ID/preferences

# 3. Approve answers to recurring application questions (sensitive ones are never generated)
curl -s -X PUT -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"answer":"Yes"}' $API/answers/work_authorization_us

# 4. Activate: runs the 30-day initial search, then daily at 17:00 America/New_York
curl -s -X POST -H "Authorization: Bearer $TOKEN" $API/resumes/$RESUME_ID/activate
curl -s -H "Authorization: Bearer $TOKEN" $API/resumes/$RESUME_ID/schedule

# 5. Results
curl -s -H "Authorization: Bearer $TOKEN" "$API/resumes/$RESUME_ID/executions"
curl -s -H "Authorization: Bearer $TOKEN" "$API/resumes/$RESUME_ID/matches?min_score=60"
curl -s -H "Authorization: Bearer $TOKEN" "$API/resumes/$RESUME_ID/applications?status=PENDING_REVIEW"
curl -s -H "Authorization: Bearer $TOKEN" $API/applications/$APPLICATION_ID   # includes manual_package

# 6. Lifecycle and recovery
curl -s -X POST -H "Authorization: Bearer $TOKEN" $API/resumes/$RESUME_ID/pause    # also /resume, /stop
curl -s -X POST -H "Authorization: Bearer $TOKEN" $API/applications/$APPLICATION_ID/reconcile
curl -s -X POST -H "Authorization: Bearer $TOKEN" $API/applications/$APPLICATION_ID/retry
```

| Endpoint | Notes |
|---|---|
| `POST /resumes` | 201 when created; 200 when the identical file was already uploaded (idempotent); 413 if over `MAX_UPLOAD_BYTES` |
| `PUT /resumes/{id}/file` | New version under the same resume id (no duplicate applications) |
| `POST /resumes/{id}/activate\|pause\|resume\|stop` | Idempotent; `changed:false` if already in that state; 409 for invalid transitions; 422 without approved titles |
| `GET /resumes/{id}/executions\|matches\|applications` | Paginated (`limit` ≤ `API_MAX_PAGE_SIZE`, `offset`); filters: `recommendation`, `min_score`, `status` |
| `POST /applications/{id}/retry` | Only when safe (retryable rejection or fixable review); 409 otherwise, including for uncertain outcomes |
| `POST /applications/{id}/reconcile` | Asks the platform; never resubmits; no-op if the outcome is known |
| `GET/PUT/DELETE /answers/{key}` | Approved answers; sensitive values are masked in responses |

Errors: 401 unauthenticated, 404 not found *or not yours*, 409 conflict, 413 too large, 422 invalid input, 429 rate limited (with `Retry-After`), 500 with no internal details.

## Observability

* **Logs:** structured JSON (structlog), with PII redaction. Resume text, answers, tokens, emails and phone numbers are never logged.
* **Metrics:** OpenTelemetry, as specified in §10. Discovery by source, rejected postings, rate limits, date-filter outcomes, duplicates, score distribution, decisions, LLM latency/tokens/failures, submissions, outcomes, reviews by kind, reconciliations, retries, scheduler run duration, finished executions, and checkpoint recoveries.
* **Traces:** a span per execution and per job match.
* **Export:** the `otel` extra (installed in the Docker image) plus standard `OTEL_*` variables, for example `OTEL_EXPORTER_OTLP_ENDPOINT`. Without an exporter, instruments are no-ops.

## LLM providers

Set `LLM_PROVIDER` to `openai`, `vertex`, `anthropic`, or `fake`, and install the matching extra, for example `uv pip install -e ".[vertex]"`. With `fake`, resume parsing uses the deterministic `HeuristicResumeExtractor`. Every provider's output goes through the same grounding filter.

## Project layout

```text
src/job_agent/
  config.py            settings (env / .env), matching weights, timezone
  models/              Pydantic domain models: resume, job, matching, application
  persistence/         SQLAlchemy ORM, portable types, sessions, file store
  repositories/        ownership-scoped data access
  services/            resume parsing + grounding, lifecycle, normalization, deduplication,
                       date filter / search window, job discovery
                       hard constraints, matcher, answers, approved answers, retention,
                       application service/actions (retry, reconcile)
  security/            upload validation, PII redaction, authentication, rate limiting
  observability/       structured logging with redaction; OpenTelemetry metrics/traces
  llm/                 provider-agnostic chat-model factory
  job_sources/         adapters (LinkedIn, Indeed, ZipRecruiter), retry/rate-limit base, mock client
  integrations/        permitted submission integrations (Greenhouse, employer-key only)
  workflows/           LangGraph search graph, per-job + application subgraphs, runner
  scheduler/           APScheduler service, process bootstrap (python -m job_agent.scheduler)
  api/                 FastAPI routes, schemas, auth dependencies, container
  runtime.py           shared wiring for the API and scheduler processes
  main.py              FastAPI app factory (uvicorn job_agent.main:create_app --factory)
migrations/            Alembic (0001 schema, 0002 execution link, 0003 reconciliation/evidence)
scripts/validate.sh    the phase gate (lint, format, mypy, tests + coverage, migrations, Postgres)
tests/unit, tests/integration, tests/e2e
```

## Key design decisions (Phase 1)

* **Stable resume identity.** `resume_id = uuid5(user_id, sha256(first upload))`. Re-uploading the same file returns the existing resume. A new file creates a new version under the same id. Applications are unique per `(resume_id, job)`, so a new version can never cause a duplicate application.
* **Anti-fabrication grounding.** Whatever the LLM returns, any skill, technology, title, employer, certification, degree, domain, or accomplishment that can't be found in the resume text is dropped and recorded as a parse warning. Years of experience are recomputed from stated dates, or kept only if the resume states them explicitly.
* **Preferences need approval.** Activation requires at least one user-approved preferred title. `suggest_preferences()` produces suggestions only.
* **Database-enforced safety.** Partial unique indexes allow at most one `RUNNING` execution per resume and one open (`STARTED`/`UNCERTAIN`) submission attempt per application. These hold across processes and scheduler instances.
* **UTC everywhere.** Naive datetimes are rejected by both the Pydantic models and the `UTCDateTime` column type.
* **Upload hardening.** The file type is detected from content rather than the extension. Size limits apply, and zip bombs, macros, encrypted PDFs, and path traversal are rejected. Files are stored with `0600` permissions.

## Key design decisions (Phase 2)

* **No fabricated results.** Without an approved `PartnerFeedClient`, a source returns `UNAVAILABLE`, a reason, and a link to its public search page. It never returns postings. Real API access plugs in as a client that maps that platform's responses to `RawPosting`. Nothing in this repo assumes a platform's API format.
* **Rate limits are honoured.** `Retry-After` is respected up to `SOURCE_MAX_RETRY_AFTER_SECONDS`, and beyond that the source gives up. Temporary errors back off exponentially. Auth errors are not retried. One failing source never fails the run.
* **Publication time is an interval.** Exact timestamps, date-only values, and relative text ("3 days ago") each become an earliest/latest range. A job is eligible only if the whole range is inside the window. Ranges that cross a window edge go to review. Naive timestamps with no documented source timezone are treated as unverified. Update and modification times are never used.
* **Search windows.** The first run covers 30 days and later runs cover 24 hours. After an outage the default `MISSED_RUN_POLICY=last_window_only` still searches only 24 hours. `cover_gap` backfills, capped at `MAX_CATCH_UP_HOURS`.
* **Deduplication.** Postings are the same job if their normalized company, title, and location match, or if they share an application URL. Grouping is transitive. The group key is order-independent, and the repository reuses any key already stored, so the identity used for application keys stays stable across runs.
* **Stored publication date is immutable.** A reposted or edited job cannot look new again.
* **"Already tracked" is per resume.** A job resurfacing on another board is recognized for the resume that already matched or applied, while other resumes can still evaluate it.

## Key design decisions (Phase 3)

* **The LLM scores, code decides.** The LLM returns six criterion scores with evidence, as `LLMMatchAssessment` (extra fields are rejected, so it can't sneak in its own recommendation). Python computes the weighted score in exact decimal arithmetic. This matters: naive float math can turn a true 60.00 into 59.99999999999999 and wrongly skip the job.
* **Outcome rules:**
  * A hard-constraint violation means SKIP at any score. Constraints are excluded company or location, location outside preferences, work arrangement, employment type, salary minimum (annualized), excluded keywords, closed postings, and sponsorship or work authorization.
  * A score below `JOB_MATCH_THRESHOLD` means SKIP.
  * A qualifying score still goes to REVIEW if there is uncertainty: a constraint the posting doesn't state, insufficient evidence, an unverified date, instruction-like posting text, LLM-cited skills not in the resume, or an LLM score that conflicts with listed-skill coverage.
  * Otherwise the result is APPLY.
* **Never assumed:** work authorization, sponsorship, citizenship and clearance are judged only from the user's explicit preferences. Citizenship and clearance requirements always need a human.
* **Prompt injection defense in depth.** The posting is delimited JSON in the human message; HTML is stripped and tag-like text neutralized; the system prompt says to ignore embedded instructions. Even a fully compliant model can't force an APPLY or bypass a constraint.
* **Failures are safe.** Malformed or failed LLM output is retried (`MATCH_LLM_RETRIES`), then becomes REVIEW with score 0.
* **Configurable rubric:** `MATCHING_WEIGHTS__<CRITERION>` in `.env` (must sum to 100).

## Key design decisions (Phase 4)

* **The graph orchestrates, services decide.** Nodes are thin wrappers around the Phase 1–3 services. State is JSON-safe and scoped to one resume, and each execution is its own checkpoint thread (`thread_id` = execution id).
* **Parallel and bounded.** One search node per source runs in parallel under `MAX_CONCURRENT_JOB_SEARCHES`. Each job is a `Send` into an isolated per-job subgraph, with LLM calls limited by `MAX_CONCURRENT_MATCHES`. The summary node is deferred until every job finishes. An unexpected error in one job is contained and reported as ERROR.
* **Stop checks.** Workflow status is re-read from the database at the start of the run, before the application path, and immediately before each submission. A run that observes a stop ends CANCELLED and does not mark the initial search complete.
* **Submission safety.** A durable attempt is committed *before* the platform call. APPLIED requires a confirmation reference. Timeouts, errors and "success" without a reference become NEEDS_RECONCILIATION and are never resubmitted automatically. After a crash, a resumed run uses the checkpointed result if the platform reply was saved; otherwise the open attempt routes the job to NEEDS_RECONCILIATION.
* **No overlapping runs** for the same resume: a partial unique index allows only one RUNNING execution, and the losing run halts with `another_execution_running`.
* **Alembic ignores LangGraph's checkpoint tables** (`persistence/schema_filters.py`), so autogenerate can never drop workflow state.

## Key design decisions (Phase 5)

* **The database is the schedule.** APScheduler's in-memory job list is rebuilt from `resume_workflows` at start and every `SCHEDULER_RECONCILE_SECONDS`. Each ACTIVE resume gets one daily job (stable id, `replace_existing`); PAUSED, STOPPED and DRAFT resumes get none. `next_run_at` is stored for the API.
* **Lifecycle changes apply after commit.** A rolled-back activation schedules nothing. Activating a resume starts the 30-day initial search immediately; pause and stop remove its jobs at once. A failed initial search isn't retried on every restart; it runs again at the next daily slot.
* **No double runs, even with several scheduler instances.** Each run takes a per-resume database lease, renewed by a heartbeat and not re-entrant. The one-RUNNING-execution index is a second guard. A dead instance's lease expires, and its RUNNING execution is resumed from its checkpoint or closed.
* **Misfires are explicit:**
  * `coalesce=True`, `max_instances=1`, `misfire_grace_time=SCHEDULER_MISFIRE_GRACE_SECONDS`.
  * After an outage, a run late within the grace period runs normally.
  * Beyond it, `SCHEDULER_MISSED_RUN_ACTION=run_once` does one catch-up run, never one per missed day.
  * That run still uses the publication-date window rules, so there's no stale backlog.
* **Retention:** a daily maintenance job purges stopped resumes' files, text and profiles after `RESUME_RETENTION_DAYS`, and applications after `APPLICATION_RETENTION_DAYS`. Unresolved submissions are kept.

## Key design decisions (Phase 6)

* **Answers come only from permitted sources:**
  * Sensitive questions (work authorization, sponsorship, clearance, criminal history, demographics, disability, veteran status, salary) and identity fields are answered **only** from answers you've approved (`ApprovedAnswerService`).
  * Factual questions come from the verified resume profile.
  * Job-specific free text is LLM-written as statements that each quote resume evidence. Unsupported evidence, new numbers, sensitive topics or unfamiliar claims reject it.
  * Anything else sends the application to PENDING_REVIEW, with a list of what's missing.
* **Qualified questions aren't guessed.** "Years of experience with Python?" is a different question from total experience, so it's never answered with the total.
* **Supporting evidence** is stored on every application: integration, answer provenance (keys and sources, never sensitive values), verification method, confirmation reference, and a manual-application package when automation isn't available.
* **Reconciliation never resubmits.** The integration's lookup decides: found → APPLIED; definitively not found (after an indexing delay) → FAILED-retryable; unknown → retried, then PENDING_REVIEW for you to check the employer's site.
* **Safe retry:** only after a retryable rejection or a fixable review, at most `MAX_APPLICATION_RETRIES` times, on the same application row, through the same subgraph with all stop, eligibility, answer and duplicate checks.
* **Greenhouse:** submission is supported only for boards whose **employer issued you a Job Board API key** (that is Greenhouse's permission model). It follows the public docs, is tested against a simulated server, and is **not verified against the live API**.
