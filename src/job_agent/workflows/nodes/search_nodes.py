"""Resume-level search nodes: load, validate, window, search, normalise, dedupe, filter,
fan out, summarise."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa
from langgraph.graph import END
from langgraph.types import Send
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from job_agent.job_sources.base import JobSourceAdapter
from job_agent.models.job import JobPosting, SourceSearchResult, SourceStatus
from job_agent.models.resume import JobSearchPreferences, WorkflowStatus
from job_agent.observability import metrics
from job_agent.observability.logging import get_logger
from job_agent.persistence.orm import (
    ExecutionStatus,
    Resume,
    ResumeVersion,
    WorkflowExecution,
)
from job_agent.repositories.jobs import JobPostingRepository
from job_agent.repositories.resumes import WorkflowRepository
from job_agent.services.date_filter import (
    SearchWindow,
    WindowTrigger,
    determine_search_window,
    filter_by_publication_date,
)
from job_agent.services.deduplication import deduplicate
from job_agent.services.job_discovery import NoSearchCriteria, build_search_criteria
from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.state import (
    HaltReason,
    JobState,
    OutcomeKind,
    SearchState,
    dump_job,
    load_job,
)

log = get_logger(__name__)

SearchNode = Callable[[SearchState], Awaitable[dict[str, Any]]]


def _halt(reason: HaltReason) -> dict[str, Any]:
    return {"halt_reason": reason.value}


def halted(state: SearchState) -> bool:
    return bool(state.get("halt_reason"))


class SearchNodes:
    def __init__(self, deps: WorkflowDeps) -> None:
        self.d = deps

    # -------------------------------------------------------------- set-up

    def load_resume(self, state: SearchState) -> dict[str, Any]:
        with self.d.session() as s:
            resume = s.get(Resume, uuid.UUID(state["resume_id"]))
            if resume is None or resume.deleted_at is not None:
                return _halt(HaltReason.RESUME_NOT_FOUND)
            version = s.scalar(
                sa.select(ResumeVersion).where(
                    ResumeVersion.resume_id == resume.id,
                    ResumeVersion.version == resume.current_version,
                )
            )
            assert version is not None
            return {
                "user_id": str(resume.user_id),
                "resume_version": resume.current_version,
                "profile": version.profile,
                "preferences": resume.preferences,
            }

    def validate_resume_status(self, state: SearchState) -> dict[str, Any]:
        if not self.d.workflow_active(state["resume_id"]):
            return _halt(HaltReason.NOT_ACTIVE)
        try:
            build_search_criteria(JobSearchPreferences.model_validate(state["preferences"]))
        except NoSearchCriteria:
            return _halt(HaltReason.NO_SEARCH_CRITERIA)
        return {}

    def determine_search_window(self, state: SearchState) -> dict[str, Any]:
        now = self.d.clock()
        execution_id = uuid.UUID(state["execution_id"])
        with self.d.session() as s:
            existing = s.get(WorkflowExecution, execution_id)
            if existing is not None:  # resumed execution: keep its original window
                return {
                    "window_start": existing.search_window_start.isoformat(),
                    "window_end": existing.search_window_end.isoformat(),
                    "trigger": existing.summary.get("window_trigger")
                    if existing.summary
                    else existing.trigger,
                    "started_at": existing.started_at.isoformat(),
                    "execution_created": True,
                }
            wf = WorkflowRepository(s).get_by_resume(uuid.UUID(state["resume_id"]))
            assert wf is not None
            window = determine_search_window(
                initial_search_completed=wf.initial_search_completed,
                last_window_end=wf.last_window_end,
                now=now,
                settings=self.d.settings,
            )
            requested = state.get("requested_trigger") or (
                "MANUAL" if state.get("manual") else None
            )
            trigger = (
                requested
                if requested and window.trigger is not WindowTrigger.INITIAL
                else window.trigger.value
            )
            execution = WorkflowExecution(
                id=execution_id,
                resume_id=wf.resume_id,
                workflow_id=wf.id,
                resume_version=state["resume_version"],
                trigger=trigger,
                search_window_start=window.start,
                search_window_end=window.end,
                started_at=now,
                status=ExecutionStatus.RUNNING,
                thread_id=str(execution_id),
                summary={"window_trigger": window.trigger.value},
            )
            try:
                with s.begin_nested():
                    s.add(execution)
            except IntegrityError:  # unique index: one RUNNING execution per resume
                return _halt(HaltReason.ALREADY_RUNNING)
        log.info(
            "execution_started",
            execution_id=state["execution_id"],
            resume_id=state["resume_id"],
            trigger=trigger,
            window_hours=window.hours,
        )
        return {
            "window_start": window.start.isoformat(),
            "window_end": window.end.isoformat(),
            "trigger": window.trigger.value,
            "started_at": now.isoformat(),
            "execution_created": True,
        }

    # ------------------------------------------------------------- searching

    def search_node(self, adapter: JobSourceAdapter) -> SearchNode:
        async def search(state: SearchState) -> dict[str, Any]:
            criteria = build_search_criteria(
                JobSearchPreferences.model_validate(state["preferences"])
            )
            posted_after = datetime.fromisoformat(state["window_start"])
            async with self.d.limits(state["execution_id"]).search:
                try:
                    result = await adapter.search_jobs(criteria, posted_after)
                except Exception as exc:  # isolate adapter bugs to this source
                    log.error(
                        "job_source_adapter_crashed",
                        source=adapter.source.value,
                        error=type(exc).__name__,
                    )
                    result = SourceSearchResult(
                        source=adapter.source,
                        status=SourceStatus.FAILED,
                        error=f"adapter error: {type(exc).__name__}",
                    )
            attrs = {"source": adapter.source.value}
            metrics.jobs_discovered.add(len(result.postings), attrs)
            metrics.postings_rejected.add(result.rejected_count, attrs)
            metrics.source_searches.add(1, {**attrs, "status": result.status.value})
            return {"source_results": [result.model_dump(mode="json")]}

        search.__name__ = f"search_{adapter.source.value}"
        return search

    def normalize_job_postings(self, state: SearchState) -> dict[str, Any]:
        postings: list[dict[str, Any]] = []
        for result in state.get("source_results", []):
            for raw in result.get("postings", []):
                try:  # adapters normalise; re-validate at the trust boundary
                    postings.append(JobPosting.model_validate(raw).model_dump(mode="json"))
                except ValidationError:
                    log.warning("posting_failed_validation", source=result.get("source"))
        return {"postings": postings}

    def deduplicate_jobs(self, state: SearchState) -> dict[str, Any]:
        postings = state.get("postings", [])
        jobs = deduplicate([JobPosting.model_validate(p) for p in postings])
        metrics.duplicates_folded.add(len(postings) - len(jobs))
        return {"jobs": [dump_job(j) for j in jobs]}

    def filter_by_publication_date(self, state: SearchState) -> dict[str, Any]:
        window = SearchWindow(
            datetime.fromisoformat(state["window_start"]),
            datetime.fromisoformat(state["window_end"]),
            WindowTrigger(state["trigger"]),
        )
        result = filter_by_publication_date(
            [load_job(j) for j in state.get("jobs", [])],
            window,
            now=self.d.clock(),
            missing_date_policy=self.d.settings.missing_publication_date_policy,
            future_tolerance=timedelta(seconds=self.d.settings.future_date_tolerance_seconds),
        )
        # Review-bound jobs are still matched (the user sees a score) but can't auto-apply.
        kept = [dump_job(j) for j in result.eligible] + [
            dump_job(j, date_review_reason=reason.value) for j, reason in result.needs_review
        ]
        counts = result.counts()
        for outcome, n in counts.items():
            metrics.date_filter_results.add(n, {"outcome": outcome})
        return {"to_process": kept, "date_filter_counts": counts}

    def filter_by_user_preferences(self, state: SearchState) -> dict[str, Any]:
        """Persist postings and drop jobs this resume already matched or applied to.

        User preferences and hard constraints are evaluated per job by the matcher so
        that every evaluated job gets a score and a recorded reason.
        """
        now = self.d.clock()
        jobs = [load_job(j) for j in state.get("to_process", [])]
        reasons = {
            j["canonical_key"]: j.get("date_review_reason") for j in state.get("to_process", [])
        }
        with self.d.session() as s:
            repo = JobPostingRepository(s)
            new, tracked = repo.partition_untracked(uuid.UUID(state["resume_id"]), jobs)
            out = []
            for job in new:
                original_key = job.canonical_key
                record = repo.persist_group(job, now=now)
                data = dump_job(job, date_review_reason=reasons.get(original_key))
                data["job_posting_id"] = str(record.id)
                out.append(data)
        return {"to_process": out, "already_tracked": len(tracked)}

    def fan_out(self, state: SearchState) -> list[Send] | str:
        jobs = state.get("to_process", [])
        if not jobs:
            return "generate_run_summary"
        base: JobState = {
            "execution_id": state["execution_id"],
            "execution_started_at": state["started_at"],
            "user_id": state["user_id"],
            "resume_id": state["resume_id"],
            "resume_version": state["resume_version"],
            "profile": state["profile"],
            "preferences": state["preferences"],
        }
        return [Send("process_job", {**base, "job": job}) for job in jobs]

    # --------------------------------------------------------------- summary

    def generate_run_summary(self, state: SearchState) -> dict[str, Any]:
        now = self.d.clock()
        threshold = self.d.settings.job_match_threshold
        sources = [SourceSearchResult.model_validate(r) for r in state.get("source_results", [])]
        outcomes = state.get("job_outcomes", [])
        by_status = Counter(o["status"] for o in outcomes)
        ok = any(r.status is SourceStatus.OK for r in sources)
        stopped = not self.d.workflow_active(state["resume_id"]) or any(
            o.get("stopped") for o in outcomes
        )
        errors = by_status[OutcomeKind.ERROR.value]

        if stopped:
            status = ExecutionStatus.CANCELLED
        elif not ok:
            status = ExecutionStatus.FAILED
        elif errors or any(r.status is SourceStatus.FAILED for r in sources):
            status = ExecutionStatus.PARTIAL
        else:
            status = ExecutionStatus.SUCCEEDED

        eligible = [o for o in outcomes if (o.get("score") or 0) >= threshold]
        skipped = [o for o in outcomes if o["status"] == OutcomeKind.SKIPPED.value]
        summary: dict[str, Any] = {
            "execution_id": state["execution_id"],
            "resume_id": state["resume_id"],
            "status": status,
            "trigger": state["trigger"],
            "started_at": state["started_at"],
            "completed_at": now.isoformat(),
            "window_start": state["window_start"],
            "window_end": state["window_end"],
            "window_trigger": state["trigger"],
            "sources": {
                r.source.value: {
                    "status": r.status.value,
                    "discovered": len(r.postings),
                    "error": r.error,
                    "limitation": r.limitation,
                    "manual_search_url": str(r.manual_search_url) if r.manual_search_url else None,
                }
                for r in sources
            },
            "discovered_by_source": {r.source.value: len(r.postings) for r in sources},
            "total_discovered": sum(len(r.postings) for r in sources),
            "unique_after_dedup": len(state.get("jobs", [])),
            "date_filter": state.get("date_filter_counts", {}),
            "already_tracked": state.get("already_tracked", 0),
            "evaluated": len(outcomes),
            "at_or_above_threshold": len(eligible),
            "skipped": {
                "count": len(skipped),
                "reasons": dict(
                    Counter((o.get("reason") or "unspecified")[:120] for o in skipped).most_common(
                        20
                    )
                ),
            },
            "applied": by_status[OutcomeKind.APPLIED.value],
            "pending_review": by_status[OutcomeKind.PENDING_REVIEW.value],
            "manual_action_required": by_status[OutcomeKind.APPLICATION_UNAVAILABLE.value],
            "failed": by_status[OutcomeKind.FAILED.value],
            "needs_reconciliation": by_status[OutcomeKind.NEEDS_RECONCILIATION.value],
            "duplicates": by_status[OutcomeKind.DUPLICATE.value],
            "errors": errors,
            "eligible_jobs": [
                {
                    k: o.get(k)
                    for k in (
                        "title",
                        "company",
                        "canonical_url",
                        "application_url",
                        "score",
                        "status",
                        "application_id",
                        "reason",
                    )
                }
                for o in eligible
            ],
            "application_ids": [o["application_id"] for o in outcomes if o.get("application_id")],
        }

        with self.d.session() as s:
            execution = s.get(WorkflowExecution, uuid.UUID(state["execution_id"]))
            assert execution is not None
            execution.completed_at = now
            execution.status = status
            execution.discovered_count = summary["total_discovered"]
            execution.unique_count = summary["unique_after_dedup"]
            execution.matched_count = len(eligible)
            execution.submitted_count = sum(1 for o in outcomes if o.get("submitted"))
            execution.applied_count = summary["applied"]
            execution.skipped_count = len(skipped)
            execution.pending_review_count = (
                summary["pending_review"] + summary["manual_action_required"]
            )
            execution.failed_count = summary["failed"] + summary["needs_reconciliation"] + errors
            execution.summary = summary
            if status is ExecutionStatus.FAILED:
                execution.error = "no job source could be searched"
            wf = WorkflowRepository(s).get_by_resume(uuid.UUID(state["resume_id"]))
            assert wf is not None
            wf.last_execution_status = status
            wf.last_execution_at = now
            # Coverage only advances when something was actually searched and the run
            # was not cut short; the initial 30-day flag is set only after the outcome
            # has been recorded (same transaction).
            if ok and not stopped:
                wf.last_window_end = datetime.fromisoformat(state["window_end"])
                if state["trigger"] == WindowTrigger.INITIAL.value:
                    WorkflowRepository(s).mark_initial_search_completed(wf, now)
            if wf.status is not WorkflowStatus.ACTIVE:
                wf.next_run_at = None
        self.d.release(state["execution_id"])
        metrics.executions_finished.add(1, {"status": str(status), "trigger": state["trigger"]})
        log.info(
            "execution_completed",
            execution_id=state["execution_id"],
            status=status,
            applied=summary["applied"],
            evaluated=len(outcomes),
        )
        return {"status": status, "summary": summary}

    def finish_halted(self, state: SearchState) -> dict[str, Any]:
        log.info(
            "execution_halted",
            execution_id=state["execution_id"],
            resume_id=state["resume_id"],
            reason=state.get("halt_reason"),
        )
        self.d.release(state["execution_id"])
        return {
            "status": ExecutionStatus.SKIPPED,
            "summary": {"halt_reason": state.get("halt_reason")},
        }


def route_after(next_node: str) -> Callable[[SearchState], str]:
    def route(state: SearchState) -> str:
        return "finish_halted" if halted(state) else next_node

    return route


__all__ = ["END", "SearchNodes", "route_after"]
