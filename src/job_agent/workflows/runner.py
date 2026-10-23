"""Run and resume resume-level executions with persistent checkpointing."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from job_agent.observability.metrics import checkpoint_recoveries, tracer
from job_agent.workflows.deps import WorkflowDeps
from job_agent.workflows.job_search_graph import build_job_search_graph


@dataclass
class RunResult:
    execution_id: uuid.UUID
    status: str | None
    halt_reason: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def postgres_checkpointer(database_url: str) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """LangGraph's PostgreSQL checkpointer (creates its tables on first use)."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        yield saver


class WorkflowRunner:
    """One graph per process. Each execution is its own checkpoint thread."""

    def __init__(
        self, deps: WorkflowDeps, checkpointer: BaseCheckpointSaver[Any] | None = None
    ) -> None:
        self.deps = deps
        self.graph = build_job_search_graph(deps).compile(
            checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
            name="job_search",
        )

    @staticmethod
    def _config(execution_id: uuid.UUID) -> RunnableConfig:
        return {"configurable": {"thread_id": str(execution_id)}, "recursion_limit": 100}

    async def run(
        self,
        resume_id: uuid.UUID,
        *,
        manual: bool = False,
        execution_id: uuid.UUID | None = None,
        requested_trigger: str | None = None,
    ) -> RunResult:
        execution_id = execution_id or uuid.uuid4()
        with tracer.start_as_current_span(
            "job_search_execution",
            attributes={"execution.id": str(execution_id), "resume.id": str(resume_id)},
        ):
            out = await self.graph.ainvoke(
                {
                    "execution_id": str(execution_id),
                    "resume_id": str(resume_id),
                    "manual": manual,
                    "requested_trigger": requested_trigger,
                },
                self._config(execution_id),
            )
        return self._result(execution_id, out)

    async def resume(self, execution_id: uuid.UUID) -> RunResult:
        """Continue an interrupted execution from its last checkpoint.

        Completed steps are not repeated; an interrupted submission is never resent
        (it is moved to NEEDS_RECONCILIATION by the submit step's open-attempt check).
        """
        config = self._config(execution_id)
        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            raise LookupError(f"no checkpoint for execution {execution_id}")
        if not snapshot.next:
            return self._result(execution_id, snapshot.values)
        # Concurrency primitives belong to the event loop of the interrupted run.
        self.deps.release(str(execution_id))
        checkpoint_recoveries.add(1)
        with tracer.start_as_current_span(
            "job_search_execution_resume", attributes={"execution.id": str(execution_id)}
        ):
            out = await self.graph.ainvoke(None, config)
        return self._result(execution_id, out)

    @staticmethod
    def _result(execution_id: uuid.UUID, out: dict[str, Any]) -> RunResult:
        return RunResult(
            execution_id=execution_id,
            status=out.get("status"),
            halt_reason=out.get("halt_reason"),
            summary=out.get("summary") or {},
        )
