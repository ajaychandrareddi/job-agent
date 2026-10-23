"""Resume-level search graph.

load_resume -> validate_resume_status -> determine_search_window
  -> search_<source> (parallel, one node per configured source)
  -> normalize_job_postings -> deduplicate_jobs -> filter_by_publication_date
  -> filter_by_user_preferences -> process_job x N (Send, isolated per job)
  -> generate_run_summary (deferred: waits for every job)
Any halt (missing resume, not ACTIVE, no criteria, overlapping run) -> finish_halted.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from job_agent.observability.logging import get_logger
from job_agent.workflows.application_graph import build_job_graph
from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.nodes.job_nodes import build_outcome
from job_agent.workflows.nodes.search_nodes import SearchNodes, route_after
from job_agent.workflows.state import JobState, SearchInput, SearchState

log = get_logger(__name__)


def build_job_search_graph(deps: WorkflowDeps) -> StateGraph[Any, Any, Any, Any]:
    nodes = SearchNodes(deps)
    job_graph = build_job_graph(deps)

    async def process_job(state: JobState) -> dict[str, Any]:
        # One job's failure must not fail the run: contain it and report it.
        try:
            out: dict[str, Any] = await job_graph.ainvoke(state)
            return {"job_outcomes": out.get("job_outcomes", [])}
        except Exception as exc:
            log.error(
                "job_processing_failed",
                execution_id=state.get("execution_id"),
                error=type(exc).__name__,
            )
            return {"job_outcomes": [build_outcome(state, error=f"{type(exc).__name__}")]}

    g = StateGraph(SearchState, input_schema=SearchInput)
    g.add_node("load_resume", nodes.load_resume)
    g.add_node("validate_resume_status", nodes.validate_resume_status)
    g.add_node("determine_search_window", nodes.determine_search_window)
    g.add_node("normalize_job_postings", nodes.normalize_job_postings)
    g.add_node("deduplicate_jobs", nodes.deduplicate_jobs)
    g.add_node("filter_by_publication_date", nodes.filter_by_publication_date)
    g.add_node("filter_by_user_preferences", nodes.filter_by_user_preferences)
    g.add_node("process_job", process_job, input_schema=JobState)
    g.add_node("generate_run_summary", nodes.generate_run_summary, defer=True)
    g.add_node("finish_halted", nodes.finish_halted)

    g.add_edge(START, "load_resume")
    g.add_conditional_edges(
        "load_resume",
        route_after("validate_resume_status"),
        ["validate_resume_status", "finish_halted"],
    )
    g.add_conditional_edges(
        "validate_resume_status",
        route_after("determine_search_window"),
        ["determine_search_window", "finish_halted"],
    )

    search_nodes = []
    for adapter in deps.adapters:
        name = f"search_{adapter.source.value}"
        g.add_node(name, cast(Any, nodes.search_node(adapter)))
        search_nodes.append(name)

    def after_window(state: SearchState) -> list[str] | str:
        if state.get("halt_reason"):
            return "finish_halted"
        return search_nodes or "normalize_job_postings"

    g.add_conditional_edges(
        "determine_search_window",
        after_window,
        [*search_nodes, "normalize_job_postings", "finish_halted"],
    )
    if search_nodes:
        g.add_edge(search_nodes, "normalize_job_postings")
    g.add_edge("normalize_job_postings", "deduplicate_jobs")
    g.add_edge("deduplicate_jobs", "filter_by_publication_date")
    g.add_edge("filter_by_publication_date", "filter_by_user_preferences")
    g.add_conditional_edges(
        "filter_by_user_preferences", nodes.fan_out, ["process_job", "generate_run_summary"]
    )
    g.add_edge("process_job", "generate_run_summary")
    g.add_edge("generate_run_summary", END)
    g.add_edge("finish_halted", END)
    return g
