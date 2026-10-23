"""Per-job subgraph (match -> threshold -> duplicate check) and the application subgraph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.nodes.application_nodes import ApplicationNodes
from job_agent.workflows.nodes.job_nodes import JobNodes
from job_agent.workflows.state import JobOutput, JobRoute, JobState

APPLICATION_STEPS = (
    "verify_resume_active",
    "confirm_eligibility",
    "validate_application_method",
    "prepare_application",
    "validate_application_answers",
    "submit_application",
    "verify_submission",
    "persist_application_result",
)


def _continue_or_end(next_node: str) -> Callable[[JobState], str]:
    def route(state: JobState) -> str:
        return END if state.get("route") == JobRoute.HALT.value else next_node

    return route


def build_application_graph(deps: WorkflowDeps) -> CompiledStateGraph[Any, Any, Any, Any]:
    nodes = ApplicationNodes(deps)
    g = StateGraph(JobState)
    for name in APPLICATION_STEPS:
        g.add_node(name, getattr(nodes, name))
    g.add_edge(START, APPLICATION_STEPS[0])
    for current, nxt in zip(APPLICATION_STEPS, APPLICATION_STEPS[1:], strict=False):
        g.add_conditional_edges(current, _continue_or_end(nxt), [nxt, END])
    g.add_edge(APPLICATION_STEPS[-1], END)
    return g.compile(name="application")


def build_job_graph(deps: WorkflowDeps) -> CompiledStateGraph[Any, Any, Any, Any]:
    nodes = JobNodes(deps)
    g = StateGraph(JobState, output_schema=JobOutput)
    g.add_node("calculate_job_match", nodes.calculate_job_match)
    g.add_node("apply_match_threshold", nodes.apply_match_threshold)
    g.add_node("check_existing_applications", nodes.check_existing_applications)
    g.add_node("application", build_application_graph(deps))
    g.add_node("record_outcome", nodes.record_outcome)
    g.add_edge(START, "calculate_job_match")
    g.add_edge("calculate_job_match", "apply_match_threshold")
    g.add_edge("apply_match_threshold", "check_existing_applications")
    g.add_conditional_edges(
        "check_existing_applications",
        lambda s: "application" if s.get("route") == JobRoute.APPLY.value else "record_outcome",
        ["application", "record_outcome"],
    )
    g.add_edge("application", "record_outcome")
    g.add_edge("record_outcome", END)
    return g.compile(name="process_job")
