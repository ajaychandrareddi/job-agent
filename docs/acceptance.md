# Final acceptance review

**Reviewed:** 2026-10-23, against the original specification (§12 test cases, §15 deliverables, §16 acceptance criteria). **Gate result:** all 10 checks in `scripts/validate.sh` pass, including PostgreSQL 16:

- lint, format, mypy --strict;
- 470 tests with 97% coverage, and every logic/workflow module at 85% or more;
- migration rendering and SQLite up/down;
- PostgreSQL migration, checkpointer and runtime tests, and the full suite on PostgreSQL.

On top of the gate:
- **Clean-environment install:** a fresh copy installed by following the README passed every test.
- **Docker Compose deployment:** verified end to end over HTTP (details in §16.11 below).

Evidence below names the tests that prove each item (file paths under `tests/`).

## §16 Acceptance criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Multiple resumes managed independently | ✅ | `e2e/test_job_search_workflow.py::test_paused_resume_does_not_affect_another`, `e2e/test_scheduler.py::test_pause_resume_stop_are_independent_per_resume`, `integration/test_resume_service.py::TestLifecycle::test_resumes_are_independent` |
| 2 | Each new resume gets a 30-day initial search | ✅ | `e2e/test_job_search_workflow.py::test_initial_run_end_to_end` (asserts `posted_after = now − 30d`), `e2e/test_scheduler.py::test_activation_registers_after_commit_and_runs_initial`, `::test_scheduler_drives_real_workflow` |
| 3 | Daily runs at 17:00 in the configured timezone, covering 24h | ✅ | `e2e/test_scheduler.py::test_registers_daily_job_per_active_resume_only` (cron 17:00 America/New_York), `::test_previous_daily_run_handles_dst`, `::test_configured_time_and_timezone`, `e2e/test_job_search_workflow.py::test_recurring_run_after_restart` (asserts `posted_after = now − 24h`) |
| 4 | Only ≥60% and constraint-compliant jobs considered for application | ✅ | `unit/test_job_matcher.py::TestThreshold`, `::TestHardConstraintsOverrideScore`, `unit/test_hard_constraints.py` |
| 5 | No duplicate applications across retries, restarts and concurrency | ✅ | DB unique `application_key` and `(resume_id, job)`, plus partial unique indexes (one RUNNING execution, one open attempt): `integration/test_resume_service.py::TestDatabaseSafeguards`, `e2e/test_job_search_workflow.py::test_concurrent_executions_do_not_duplicate`, `::test_crash_during_platform_call_is_never_resubmitted`, `e2e/test_application_actions.py::test_retry_after_retryable_rejection_applies` (same row) |
| 6 | Pause, resume and stop per resume | ✅ | `e2e/test_api.py::test_lifecycle_endpoints`, `e2e/test_scheduler.py::test_pause_resume_stop_are_independent_per_resume` (stopped persists across restart) |
| 7 | Status reflects verified outcomes, not attempts | ✅ | `e2e/test_job_search_workflow.py::test_submission_outcomes` ("success" without a reference → NEEDS_RECONCILIATION), `unit/test_greenhouse.py::test_submit_response_interpretation` |
| 8 | Failed or unavailable sources never fabricate results | ✅ | `unit/test_job_sources.py::test_unconfigured_sources_report_limitation_not_results`, `e2e/test_job_search_workflow.py::test_nothing_searchable_fails_and_keeps_initial_pending`; Docker smoke run (no partner access → FAILED "no job source could be searched") |
| 9 | Recovery from process restarts without losing history | ✅ | `e2e/test_job_search_workflow.py::test_resume_from_checkpoint_after_crash`, `::test_postgres_checkpointer_resume` (real PostgreSQL checkpointer, new connection), `e2e/test_scheduler.py::test_stale_execution_is_resumed_or_failed` |
| 10 | Automated submission respects platform restrictions and uses only verified info | ✅ | Only registered, permitted `SubmissionIntegration`s are used (Greenhouse requires an employer-issued key). `unit/test_answers.py` (sensitive answers only from approved answers; grounded generation), `e2e/test_application_actions.py::test_greenhouse_through_the_workflow` |
| 11 | Test suite passes; setup instructions work in a clean environment | ✅ | Gate (10/10). Clean copy plus README steps: all tests pass. `docker compose up --build`: migrations 0001→0003, API healthy, upload → preferences → activate over HTTP, then the scheduler container ran the initial search. No errors in logs; the token never logged. |

## §12 Test cases

| Case | Test(s) |
|---|---|
| 1 First run searches 30 days | `e2e/test_job_search_workflow.py::test_initial_run_end_to_end`, `unit/test_date_filter.py::TestSearchWindow::test_first_run_searches_previous_30_days` |
| 2 Later runs search 24 hours | `e2e/test_job_search_workflow.py::test_recurring_run_after_restart`, `unit/test_date_filter.py::TestSearchWindow::test_subsequent_run_searches_previous_24_hours` |
| 3 59% is skipped | `unit/test_job_matcher.py::TestThreshold::test_59_is_skipped` (also `test_just_below_60_never_rounds_up`) |
| 4 Exactly 60% is eligible | `unit/test_job_matcher.py::TestThreshold::test_exactly_60_is_eligible`, `::test_exact_decimal_arithmetic` |
| 5 85% with a hard disqualifier is not submitted | `unit/test_job_matcher.py::TestHardConstraintsOverrideScore::test_85_with_hard_disqualifier_is_not_applied` |
| 6 Missing publication date is never auto-applied | `unit/test_date_filter.py::TestPublicationDateFilter::test_missing_date_never_eligible`, `e2e/test_job_search_workflow.py::test_initial_run_end_to_end` |
| 7 Cross-source duplicates are deduplicated | `unit/test_deduplication.py::TestDeduplicate::test_same_job_on_three_sources_is_merged`, `integration/test_job_discovery.py::TestDiscovery::test_cross_source_duplicates_are_merged_and_dates_filtered` |
| 8 Previously applied jobs are not resubmitted | `e2e/test_job_search_workflow.py::test_recurring_run_after_restart`, `integration/test_job_discovery.py::TestJobPostingRepository::test_previously_applied_jobs_filtered_per_resume` |
| 9 Different resumes can apply to the same job | `e2e/test_job_search_workflow.py::test_different_resumes_apply_to_same_job_independently` |
| 10 Pausing or stopping one resume doesn't affect another | `e2e/test_job_search_workflow.py::test_paused_resume_does_not_affect_another`, `e2e/test_scheduler.py::test_pause_resume_stop_are_independent_per_resume` |
| 11 A stop before submission prevents it | `e2e/test_job_search_workflow.py::test_stop_before_submission_prevents_submission`, `::test_stop_check_is_immediately_before_submit` |
| 12 A restart doesn't repeat the initial search | `e2e/test_scheduler.py::test_restart_does_not_repeat_initial_search`, `e2e/test_job_search_workflow.py::test_recurring_run_after_restart` |
| 13 A timeout never causes an unsafe duplicate | `e2e/test_job_search_workflow.py::test_slow_submission_times_out_without_resubmitting`, `::test_crash_during_platform_call_is_never_resubmitted` |
| 14 A failed source doesn't fail the search | `e2e/test_job_search_workflow.py::test_failed_source_does_not_fail_search`, `integration/test_job_discovery.py::TestDiscovery::test_failed_source_does_not_fail_search` |
| 15 Missing answers route to review | `e2e/test_job_search_workflow.py::test_missing_answers_route_to_review`, `unit/test_answers.py::test_sensitive_questions_only_from_approved_answers` |
| 16 Scheduler restart and missed executions | `e2e/test_scheduler.py::test_restart_does_not_repeat_initial_search`, `::test_late_start_within_grace_runs_scheduled_once`, `::test_outage_beyond_grace_runs_one_catch_up`, `::test_outage_beyond_grace_skip_policy` |
| 17 Concurrent executions don't duplicate attempts | `e2e/test_job_search_workflow.py::test_concurrent_executions_do_not_duplicate`, `e2e/test_scheduler.py::test_lease_blocks_second_instance_and_reentry`, `integration/test_resume_service.py::TestDatabaseSafeguards::test_only_one_open_submission_attempt` |
| 18 Authentication prevents cross-user access | `e2e/test_api.py::test_users_cannot_access_each_others_data` (every resource returns 404 to another user), `::test_unauthenticated_requests_are_rejected`, `unit/test_auth.py` |

## §15 Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Source code | `src/job_agent/` |
| 2 | Dependencies and environment config | `pyproject.toml` (extras: `openai`, `vertex`, `anthropic`, `otel`, `dev`); `.env.example` (every setting documented, enforced by a test) |
| 3 | DB models and migrations | `persistence/orm.py`; `migrations/versions/0001–0003` |
| 4 | LangChain prompts and structured-output schemas | `services/resume_extractors.py`, `services/job_matcher.py`, `services/answers.py`; `models/` |
| 5 | LangGraph workflows and state | `workflows/` (search graph, per-job and application subgraphs, typed state, runner) |
| 6 | Job-source adapter interfaces and implementations | `job_sources/` (LinkedIn, Indeed, ZipRecruiter, mock client) |
| 7 | Scheduler with per-resume lifecycle | `scheduler/` (`python -m job_agent.scheduler`) |
| 8 | Submission, verification and reconciliation | `services/application_service.py`, `services/application_actions.py`, `integrations/greenhouse.py` |
| 9 | FastAPI endpoints and schemas | `api/`, `main.py` (all §9 endpoints plus file replacement and approved answers) |
| 10 | Unit, integration and e2e tests | `tests/unit`, `tests/integration`, `tests/e2e` |
| 11 | Docker and Compose | `Dockerfile` (non-root, healthcheck), `docker-compose.yml` (db, migrate, api, scheduler), `.dockerignore` |
| 12 | README: setup, architecture, configuration, API examples | `README.md` |
| 13 | Architecture and sequence diagrams | `docs/architecture.md` (components, data model, state machines, graphs; sequences for initial search, recurring search, matching, submission, recovery) |
| 14 | Integrations needing credentials or approval | `docs/integrations.md` |

## Known limitations (deliberate, documented)

1. **No permitted job-seeker-side automated submission channel exists** on LinkedIn, Indeed or ZipRecruiter, and their search APIs need partner approval. Greenhouse submission works only with an employer-issued key. So in a typical setup, qualifying jobs become **prepared manual applications**, and the system never bypasses platform controls.
2. **No external integration has been verified against a live service:** the job sources, Greenhouse, and the OpenAI, Vertex and Anthropic providers. Each is tested against simulated responses only.
3. **The rate limiter is in-process.** For several API instances, add a shared limiter at the gateway.
4. **Separate API and scheduler processes:** lifecycle changes reach the scheduler within `SCHEDULER_RECONCILE_SECONDS`, or immediately with `API_RUN_SCHEDULER=true`.
5. **Synchronous SQLAlchemy inside async nodes.** It's correct, with short transactions and no session spanning an `await`, but it isn't fully non-blocking.
