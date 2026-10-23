# job-agent — project context

Agentic job search and application platform (Python 3.12, LangChain, LangGraph, FastAPI, PostgreSQL, APScheduler). The original specification is the master prompt from the first session; its structure is mirrored in `README.md` and `docs/architecture.md`. Build order: Phases 1–7.

## MANDATORY: phase completion gate

After completing **every** phase, and before starting the next one:

1. **Review requirements.** Re-read all original requirements that apply to the phase (spec sections, numbered test cases in §12, and acceptance criteria in §16). Check each one against the code, not against memory. List any gap.
2. **Run all checks:** `scripts/validate.sh`. For the PostgreSQL checks, point `TEST_DATABASE_URL` at a dedicated **`*_test`** database, never the application database; tests drop tables and refuse other names. This runs lint, format, strict mypy, tests with coverage of at least 85% overall **and per module** for business logic and workflow code (`scripts/check_module_coverage.py`; scopes: services, workflows, repositories, job_sources, models, scheduler, integrations, security), PostgreSQL migration rendering, and SQLite migration up/down. When `TEST_DATABASE_URL` is set, it also runs the PostgreSQL migration test and the full test suite on PostgreSQL. Those PostgreSQL checks are mandatory. Run any phase-specific checks too, such as a clean-environment install after dependency changes.
3. **Fix and rerun.** If any check fails or any gap is found, fix it and rerun **all** checks. Repeat until everything passes.
4. **Do not mark the phase complete** while any mandatory check is failing or unresolved. A check that cannot run in the current environment is recorded as **BLOCKED**, with the reason, never as passed.
5. **Update this file** with the phase status, validation results, fixes made, open items, and next steps. Only then proceed to the next phase.

Other standing rules:
- Never claim an external integration works until it has been tested against the real service.
- Don't commit unless the user asks. The enclosing git repo is rooted at the home directory.

## Commands

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"   # setup
scripts/validate.sh                                    # phase gate checks
.venv/bin/pytest --cov                                 # tests only
# Full gate with PostgreSQL. Port 5432 on this machine is taken by another
# container (autoarb-postgres), so this project's db runs on 55432:
POSTGRES_HOST_PORT=55432 docker compose up -d db
TEST_DATABASE_URL=postgresql+psycopg://job_agent:job_agent@localhost:55432/job_agent_test scripts/validate.sh
```

## Phase status

| Phase | Status | Last validated |
|---|---|---|
| 1 Foundation | ✅ Complete. All checks passing, including PostgreSQL | 2026-10-23 |
| 2 Job discovery | ✅ Complete. All checks passing, including PostgreSQL | 2026-10-23 |
| 3 Matching | ✅ Complete. All checks passing, including PostgreSQL | 2026-10-23 |
| 4 LangGraph workflows | ✅ Complete. All 9 checks passing, including PostgreSQL | 2026-10-23 |
| 5 Scheduling | ✅ Complete. All 9 checks passing, including PostgreSQL | 2026-10-23 |
| 6 Applications | ✅ Complete. All 9 checks passing, including PostgreSQL | 2026-10-23 |
| 7 API, deployment, observability | ✅ Complete. All 10 checks passing, including PostgreSQL; Docker deployment verified | 2026-10-23 |

### Validation log

**2026-10-23: gate run for Phase 7 (API, auth, metrics, deployment) and final acceptance review**

- Built:
  - `api/`: all §9 endpoints plus `PUT /resumes/{id}/file` and `/answers`. Pydantic schemas; pagination with a cap; domain errors → 404/409/422; generic 500s; security headers; uploads read with a hard cap (413); parsing off the event loop.
  - `security/auth.py`:
    - Static dev tokens: constant-time compare, stored subject → masked secret, at least 16 characters, refused when `APP_ENV=prod`.
    - JWT: signature, exp, iss, aud and `sub` required; the `none` algorithm and algorithm confusion are rejected.
    - Per-user token-bucket rate limiting, plus a per-IP limit on auth failures.
  - `observability/metrics.py`: every §10 metric (OpenTelemetry API, no-op without an SDK), with spans per execution and match.
  - `runtime.py`: shared wiring for the API and scheduler. An in-process scheduler option (`API_RUN_SCHEDULER`).
  - Docker: `.dockerignore` (keeps `.env` out of the build context), `HEALTHCHECK`, non-root user, `EXTRAS` build arg, configurable host ports.
  - Docs: `docs/acceptance.md`, a README API guide, and updated architecture sequences.
- Requirements reviewed: §9 (endpoints, auth, authorization, upload limits, rate limiting, idempotency, retry not bypassing safeguards), §10 (all metrics; no PII in logs or metric attributes), §11 (secrets never committed or sent to the Docker build context), §12 case 18, §15 deliverables 1–14, and §16 criteria 1–11. See `docs/acceptance.md`.
- Gaps found and fixed:
  - **The PostgreSQL test commands in CLAUDE.md and the README pointed at the application database, and the test fixtures drop all tables.** Following the docs against a real deployment would have erased its data. This was found when the Docker migration failed on a database the tests had emptied. Fixes:
    - `tests/db_guard.py` refuses non-`*test*` databases in every destructive fixture.
    - `scripts/ensure_test_db.py` creates `job_agent_test`, and `validate.sh` runs it first.
    - The docs are corrected.
  - **Static tokens were the unmasked dict keys** (they would print in a settings dump). They're now subject → `SecretStr`, with a minimum length.
  - **Matching-stage reviews, skips and duplicates weren't counted** in the review and outcome metrics (§10 "applications requiring review").
  - **No `.dockerignore`:** `.env` and `.venv` were sent to the Docker daemon on every build.
  - **No container healthcheck.**
  - **README lacked API examples** (§15.12), and **the architecture doc still said "[planned]"** (§15.13).
  - **Deprecated Starlette status constants.** Deprecations raised from our own code now fail the test suite (pyproject `filterwarnings`).
- Test/env defects found and fixed:
  - A `time` import clash in the scheduler.
  - Metrics test data with identical postings (correctly merged by dedup).
  - **A flaky test:** python-docx embeds a timestamp, so regenerating an "identical" resume could change its bytes. It's generated once now.
  - Gate ordering: the test database is now created before the test step.
  - A stale leftover from test runs in the dev database was cleared (version marker and checkpoint tables only, no application data).
- Results:
  - **All 10 checks PASS**: test-DB setup, lint, format, mypy --strict (80 files), **470 tests** at 97% coverage, every logic/workflow module ≥85%, PostgreSQL render, SQLite migration up/down, PostgreSQL tests, and the full suite on PostgreSQL (467 passed).
  - **Clean-environment install** from a fresh copy: all tests pass.
  - **`docker compose up --build`:**
    - Migrations 0001→0003 apply and the API reports healthy.
    - Over HTTP: 401 without a token; upload (201) → preferences → activate.
    - The separate scheduler container picked up the activation within 10s and ran INITIAL. It ended FAILED ("no job source could be searched"), because no partner access is configured, which is correct.
    - `next_run_at` is 17:00 America/New_York; no errors in the logs; the token never logged.

**2026-10-23: gate run for Phase 6 (applications)**

- Built:
  - `services/answers.py`: question classification, canonical keys, sensitivity detection; resume-fact answers; LLM-generated answers as evidence-quoted statements, validated against the profile (evidence, numbers, sensitive topics, vocabulary); `AnswerResolver`; option mapping.
  - `services/approved_answers.py`: upsert/list/delete with key validation and an automatic sensitive flag; values never logged.
  - `services/application_actions.py`:
    - `retry`: only FAILED-retryable or fixable reviews; re-runs the application subgraph on the same row.
    - `reconcile` / `reconcile_due`: FOUND → APPLIED; NOT_FOUND (after an indexing delay) → FAILED-retryable; UNKNOWN → retry later, then PENDING_REVIEW. Stale SUBMITTING is handled too.
  - `integrations/greenhouse.py`: Greenhouse Job Board API (questions, submit, response interpretation, connect-error vs uncertain); per-board, employer-issued keys only.
  - The application nodes now use integration questions, verified answers, `review_kind` tagging, a manual package (`submission_evidence.manual_package`, keys only) and supporting evidence.
  - Scheduler: a reconciliation interval job. Bootstrap wires Greenhouse, the answer generator and the reconciler.
  - Migration `0003`: `applications.reconcile_attempts` (server default 0) and `applications.submission_evidence`.
- Requirements reviewed: §4 (all 10 workflow steps, statuses, APPLIED only on confirmation, every safety rule including manual completion with prepared information), §6 (reconcile if possible, never resubmit blindly, mark for review, retry only when safe), §9 (retry/reconcile semantics: idempotent, no bypass of duplicate safeguards), §10 (no answer values in logs), §11, and §12 cases 13 and 15.
- Gaps found and fixed:
  - **A qualified question was answered with a fabricated value.** "Years of experience *with Python*?" mapped to *total* years. Qualified questions are now custom keys and need an approved answer.
  - **Approved answers to select questions were submitted as raw text** ("no") instead of the form's option value ("0"). They're now mapped at submission time, and a stale answer goes to review.
  - Approved answers had no service (§4 "store explicit user-approved answers"). `ApprovedAnswerService` is added.
  - The migration as autogenerated would fail on existing data (NOT NULL without a default). A server default was added.
  - The per-module coverage gate didn't include `integrations/` or `security/`; both are in scope now, at ≥92%.
- Test defects found and fixed:
  - A test enshrined the option-value bug.
  - A stop-check test wrapped a now-async node synchronously.
  - A stray no-op `pytest.raises` block and a no-op line.
  - The generated-answer test cited evidence present in the file but not in the verified profile; the correct rejection was kept.
- Results: **all 9 checks PASS**:
  - lint, format, mypy --strict (70 files);
  - **430 tests**, coverage 96%, every logic/workflow/scheduler/integration/security module ≥85%;
  - PostgreSQL render, SQLite migration up/down, PostgreSQL tests, and the full suite on PostgreSQL (427 passed).

**2026-10-23: gate run for Phase 5 (scheduling)**

- Built:
  - `scheduler/scheduler_service.py`:
    - APScheduler `AsyncIOScheduler` with one daily job per ACTIVE resume, at 17:00 in `JOB_SEARCH_TIMEZONE`.
    - Job list rebuilt from the DB at start and every `SCHEDULER_RECONCILE_SECONDS`, with stable ids and `replace_existing`.
    - Lifecycle listener applied only *after commit*.
    - Immediate initial run on activation.
    - DB run lease per resume with heartbeat, so multiple instances never double-run (non-re-entrant).
    - `coalesce`, `max_instances=1`, explicit `misfire_grace_time`, and a `SCHEDULER_MAX_WORKERS` semaphore.
    - Start-up missed-run handling: late within grace runs as SCHEDULED; beyond grace, `SCHEDULER_MISSED_RUN_ACTION` runs one CATCH_UP or skips. Never one run per missed day.
    - Stale RUNNING executions (expired lease) are resumed from checkpoint, otherwise closed as FAILED (or CANCELLED if the workflow is no longer active).
    - Daily retention maintenance.
  - `scheduler/bootstrap.py` and `__main__.py`: `python -m job_agent.scheduler`, a PostgreSQL checkpointer, SIGTERM handling. `NoLLMAssessor` when `LLM_PROVIDER=fake` (everything goes to REVIEW).
  - `repositories/executions.py`, plus lease methods in `WorkflowRepository`.
  - `services/retention.py`: the purge. Stopping a resume sets `purge_after`; reactivating clears it. Applications get `purge_after` at creation. Unresolved submissions are never purged.
  - `docker-compose.yml`: a `scheduler` service.
  - Runner/graph: `requested_trigger` labels executions SCHEDULED / CATCH_UP / MANUAL. INITIAL always wins.
- Requirements reviewed: all of §7, §2C (register after activation, persisted initial flag, no 30-day rerun on restart), §2D (independent lifecycle, stopped persists across restarts, next scheduled execution persisted in `resume_workflows.next_run_at`), §8 retention, §11 configuration, and §12 cases 10, 12, 16 and 17.
- Gaps found and fixed:
  - **`next_run_at` was computed from the injectable clock** while APScheduler fires by wall-clock time. The stored next execution could disagree with the real one, so it's now always wall-clock based.
  - **The per-module coverage gate didn't cover `scheduler/`**, although §12 names scheduling as a coverage priority. It's now in scope. That exposed an untested `bootstrap.main()` (79%), which now has a SIGTERM test; the `__main__.py` shim is omitted from coverage.
  - **10 settings were missing from `.env.example`**: six new scheduler ones, plus `DB_ECHO`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS` and `FUTURE_DATE_TOLERANCE_SECONDS` from earlier phases. They're documented, and a test now fails if any setting is undocumented or unknown.
- Test defects found and fixed: the PostgreSQL runtime test assumed tables existed in the shared database. It now builds its own schema and data and checks the real registration and `next_run_at`.
- Results: **all 9 checks PASS**:
  - lint, format, mypy --strict (66 files);
  - **350 tests**, coverage 97%, every logic/workflow/scheduler module ≥85%;
  - PostgreSQL render, SQLite migration up/down, PostgreSQL tests (migration, checkpointer resume, production scheduler runtime);
  - full suite on PostgreSQL (347 passed).

**2026-10-23: gate run for Phase 4 (LangGraph workflows)**

- Built:
  - `workflows/state.py`: typed, JSON-safe state.
  - `workflows/deps.py`: injected dependencies, per-execution semaphores, stop check.
  - `nodes/search_nodes.py`, `nodes/job_nodes.py`, `nodes/application_nodes.py`: every node named in §5.
  - `application_graph.py`: the per-job subgraph and the application subgraph.
  - `job_search_graph.py`: parallel source nodes, Send fan-out, deferred summary.
  - `runner.py`: run and resume, in-memory or PostgreSQL checkpointer.
  - `services/application_service.py`: durable submission attempts, outcome mapping, `SubmissionIntegration` protocol and registry.
  - Migration `0002`: `applications.execution_id`.
- Requirements reviewed: §5 (all named nodes, conditional routing, parallel search with limits, per-job isolation, resume-level isolation, stop checks before side effects), §6 (persistent checkpointer, stable ids, unique application key, durable attempts, no blind resubmission, DB authoritative), §2C/§2D (30-day first run, 24h recurring, tracked filter, initial flag set only after the outcome is recorded, pause and stop semantics), §4 (statuses, APPLIED only on confirmation with a reference), §10 (run summary contents), and §12 cases 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 and 17.
- Gaps found and fixed:
  - **Crash recovery mislabelled a resumed job's own in-flight application as DUPLICATE**, leaving it stuck in SUBMITTING. Fixed with `applications.execution_id` (migration 0002) and `JobNodes._resume_own`.
  - **Alembic autogenerate would DROP LangGraph's checkpoint tables** (they share the database). Fixed with the `persistence/schema_filters.py` filter in `migrations/env.py` and the drift test, plus a regression test.
  - §4 "validate the application URL": a missing URL now goes to APPLICATION_UNAVAILABLE.
  - §5 concurrency limits in the graph were untested; a test is added.
  - **§12 per-module coverage:** `job_nodes.py` was at 74.8%, hidden by the 96% total. It's now ≥85%, and `validate.sh` enforces the per-module floor from now on. All Phase 1–3 modules already met it.
- Test/env defects found and fixed:
  - The crash tests patched node methods after the graph was built. They're restructured.
  - One crash test expected NEEDS_RECONCILIATION where the checkpointed confirmation correctly gives APPLIED; it's split into two scenarios.
  - The PostgreSQL migration test was polluted by a previous run's leftovers. It now uses a throwaway database.
- Results: **all 9 checks PASS**:
  - lint, format, mypy --strict (61 files);
  - **322 tests**, coverage 96%, every logic/workflow module ≥85%;
  - PostgreSQL SQL render, SQLite migration up/down, PostgreSQL migration test (isolated database), and the full suite on PostgreSQL (320 passed), including **checkpoint resume with the real PostgreSQL checkpointer** across new connections.

**2026-10-23: gate run for Phase 3 (matching engine)**

- Built:
  - `services/hard_constraints.py`: violations block; uncertainties go to review.
  - `services/job_matcher.py`: LangChain structured output with `include_raw` for token usage; Decimal weighted score; threshold; review gates; prompt-injection defenses; retry, then REVIEW on failure.
  - `repositories/matches.py`: upsert per resume version and job.
  - `MatchDecision` gained `review_reasons`, `evaluation_error`, `model_name` and `prompt_version`.
- Requirements reviewed: all of §3 (rubric and configurability, per-job outputs, deterministic threshold, hard constraints for location, work authorization, employment type and salary, insufficient evidence goes to REVIEW, untrusted posting text, malformed LLM output), §10 (LLM latency and token logging, no PII in prompts or logs), §11 (configuration), and §12 cases 3, 4 and 5.
- Gaps found and fixed:
  - Rubric weights were not configurable through `.env` (`MatchingWeights` read only process env). They now nest in `Settings` as `MATCHING_WEIGHTS__<NAME>`, with tests.
  - The match settings were missing from `.env.example`; they're added.
- Test defects found and fixed:
  - The float-precision test used values that happen to sum exactly. It's replaced with a real case (per-term float math gives 59.99999999999999 for a true 60).
  - The delimiter-escape test was exercising HTML stripping. It now tests both defense layers separately.
- Gate failures found and fixed: ruff lint and format issues in the new tests. The full gate was then rerun.
- Results: **all 8 checks PASS**:
  - lint, format, mypy --strict (50 files);
  - **279 tests**, coverage **96%** (job_matcher 97%, hard_constraints 97%, matches repo 91%);
  - PostgreSQL SQL render, SQLite migration up/down, PostgreSQL migration test, and the full suite on PostgreSQL (278 passed).

**2026-10-23: PostgreSQL gate run (clears the BLOCKED item)**

- Environment:
  - Started Docker Desktop (server 20.10.22).
  - Brought up `postgres:16` with `POSTGRES_HOST_PORT=55432 docker compose up -d db`.
  - Created `.env` from `.env.example`, per the README setup step.
- Fixes:
  - `docker-compose.yml` hard-coded host port 5432, which is already used by another container (`autoarb-postgres`, left untouched). The host port is now `${POSTGRES_HOST_PORT:-5432}`.
  - `validate.sh` now also runs the full test suite on PostgreSQL when `TEST_DATABASE_URL` is set, so DB constraints and UTC handling are verified on the real engine.
  - README updated to match.
- Results:
  - **All 8 checks PASS:** lint, format, mypy --strict, 216 tests at 96% coverage, PostgreSQL SQL render, SQLite migration up/down, **PostgreSQL migration test** (upgrade, no schema drift, downgrade), and the **full suite on PostgreSQL** (215 passed).
  - Partial unique indexes (one RUNNING execution per resume, one open submission attempt per application) and UTC round-trips behave on PostgreSQL as tested on SQLite.

**2026-10-23: gate run for Phases 1 and 2** (first run under this rule)

- Requirements reviewed: §2A resume management, §2B job sources and `JobPosting` fields, §2C windows and publication dates, §8 data model and retention, §11 configuration, and §12 test cases 1, 2, 6, 7, 8 (tracking filter), 9, 10, 12 (window side), 14, 18 (service-level scoping).
- Gap found and fixed: `JobPosting.experience_requirements` (§2B) was never populated. Normalization now uses the source's own value, or else quotes the sentences in the description that state an experience requirement, word for word. `RawPosting.experience_requirements` was added, along with tests.
- Check-script defects found and fixed:
  - Invalid `alembic -q` flag.
  - A stdout redirect was swallowing the PASS/FAIL line for the PostgreSQL render check.
- Results after fixes:
  - Passing: lint, format, mypy --strict (47 files), **215 passed / 1 skipped**, coverage **96%**, PostgreSQL SQL render, SQLite migration up/down.
  - Clean-environment install from a fresh copy (README steps): 215 passed.
- **BLOCKED (resolved 2026-10-23, see the PostgreSQL gate run above):** PostgreSQL migration test (`tests/integration/test_migrations.py::test_migrations_postgres`). The Docker daemon isn't running. It must pass before Phase 4, because LangGraph checkpointing needs PostgreSQL.

### Open items

- [x] Stale RUNNING executions: handled by the scheduler (resume from checkpoint, or close). Done in Phase 5.
- [x] Retention purge job: `services/retention.py`, daily maintenance job. Done in Phase 5.
- [x] (Phase 7: set `API_RUN_SCHEDULER=true` for immediate effect in single-process deployments.) Lifecycle changes made by a *separate* API process reach the scheduler on its next reconcile (≤ `SCHEDULER_RECONCILE_SECONDS`, default 5 min), so the initial search can start up to 5 minutes after activation. Phase 7 can call the scheduler directly (same process) or lower the interval.
- [ ] Edge case: if an initial run is still running at 17:00, that day's daily run is skipped (the lease is held). The next 24h window then misses the few minutes between the initial run's end and 17:00. `MISSED_RUN_POLICY=cover_gap` closes this; the default does not.
- [x] Phase 6: reconciliation, safe retry, verified answers, approved-answer service, Greenhouse integration. Done.
- [ ] **Greenhouse integration is UNVERIFIED against the live API.** It follows the public Job Board API docs and is tested only against a simulated server. Its permission model requires an *employer-issued* board key (`GREENHOUSE_BOARD_KEYS`), so for a typical job seeker it stays disabled, and APPLY decisions become manual applications with a prepared package. Verify against a sandbox board before enabling.
- [ ] No permitted *job-seeker-side* automated submission channel exists for LinkedIn, Indeed or ZipRecruiter (`docs/integrations.md`).
- [ ] Generated answers are grounded against the parsed profile. With `LLM_PROVIDER=fake`, the heuristic parser doesn't extract bullet points, so there's little evidence to cite and no generator either: free-text questions go to review.
- [ ] Nodes call synchronous SQLAlchemy from async code. It's correct, and each node's DB work is short with no session spanning an `await`, but it blocks the event loop briefly. Consider an async engine or `asyncio.to_thread` if run throughput matters.

- [x] PostgreSQL migration test and full suite on PostgreSQL. Passed 2026-10-23.
- [ ] Real platform clients for LinkedIn, Indeed, and ZipRecruiter need partner approval (`docs/integrations.md`). Until then, adapters report UNAVAILABLE with a manual search link.
- [x] Metrics and traces for every §10 item: `observability/metrics.py`. Done in Phase 7.

### Next steps

All seven phases are complete; see `docs/acceptance.md`. Optional follow-ups, none required by the spec:
- **Platform access:** apply for LinkedIn, Indeed and ZipRecruiter partner access and implement `PartnerFeedClient`s. Each must pass a sandbox/live verification before being enabled.
- **Live checks:** verify the Greenhouse integration against an employer-provided sandbox board; verify each LLM provider end to end with real keys.
- **Production hardening:** a shared rate limiter at the gateway; a secret manager for `.env` values; JWT via your IdP (`API_AUTH_MODE=jwt`); an OTLP exporter for metrics and traces.

Previously planned for Phase 4 (done):
- A `StateGraph` with typed state, using the nodes from spec §5. Each one is a thin wrapper around the existing services: `JobDiscoveryService`, `JobPostingRepository.partition_untracked`, `JobMatcher.match`, and `JobMatchRepository.save`.
- A per-job subgraph via `Send`, with bounded concurrency.
- A PostgreSQL checkpointer (`langgraph-checkpoint-postgres`) with `thread_id` = execution id.
- Execution records in `workflow_executions`, and stop checks before side effects.
- An application subgraph skeleton routing to PENDING_REVIEW / APPLICATION_UNAVAILABLE. Real submission is Phase 6.
- Tests: resume isolation, stop signal before submission, source failure tolerance, restart/resume from checkpoint.

## Key design decisions (summary)

Full detail is in the "Key design decisions" sections of `README.md` and in `docs/architecture.md`.

- `resume_id = uuid5(user_id, sha256(first upload))`. Applications are unique per `(resume_id, job)`, so a new resume version can't duplicate an application.
- Resume profiles are passed through a grounding filter: anything not found in the resume text is dropped.
- Partial unique indexes: one RUNNING execution per resume, and one open submission attempt per application.
- Publication time is an earliest/latest interval. Only a range fully inside the window is eligible; missing or uncertain dates go to review.
- Deduplication uses the canonical key OR a shared application URL (union-find). Keys stay stable across runs through the repository.
- Matching:
  - The LLM produces criterion scores and evidence only. Code computes the score (exact Decimal) and applies the threshold, hard constraints and review gates.
  - Hard-constraint violations mean SKIP regardless of score. Uncertain constraints, insufficient evidence, an unverified date, instruction-like posting text, unsupported LLM-cited skills, or LLM/skill-coverage inconsistency mean REVIEW.
  - LLM failure after retries means REVIEW with score 0, and never APPLY.
