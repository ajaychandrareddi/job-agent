"""FastAPI application factory: ``uvicorn job_agent.main:create_app --factory``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import sqlalchemy as sa
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from job_agent.api.container import AppContainer, build_container
from job_agent.api.routes import answers, applications, resumes
from job_agent.config import Settings, get_settings
from job_agent.observability.logging import configure_logging, get_logger
from job_agent.scheduler.scheduler_service import SchedulerService
from job_agent.services.errors import Conflict, DomainError, InvalidInput, NotFound

log = get_logger(__name__)

_STATUS = {
    NotFound: status.HTTP_404_NOT_FOUND,
    Conflict: status.HTTP_409_CONFLICT,
    InvalidInput: status.HTTP_422_UNPROCESSABLE_CONTENT,
}


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response


def create_app(
    settings: Settings | None = None, *, container: AppContainer | None = None
) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level, json=settings.log_json)
    c = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if settings.api_run_scheduler and c.scheduler is None:
                from job_agent.workflows.runner import WorkflowRunner, postgres_checkpointer

                saver = await stack.enter_async_context(
                    postgres_checkpointer(settings.database_url)
                )
                c.scheduler = SchedulerService(
                    session_factory=c.factory,
                    settings=settings,
                    runner=WorkflowRunner(c.deps, checkpointer=saver),
                    file_store=c.file_store,
                    reconcile_submissions=c.actions.reconcile_due,
                )
                await c.scheduler.start()
                stack.push_async_callback(c.scheduler.shutdown)
            yield
        if c.owns_engine:
            c.engine.dispose()

    app = FastAPI(
        title="Job Agent API",
        version="1.0.0",
        lifespan=lifespan,
        description="Per-resume job search, matching and application workflows. "
        "Authenticate with `Authorization: Bearer <token>`.",
    )
    app.state.container = c
    app.add_middleware(SecurityHeaders)

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError) -> JSONResponse:
        code = next(
            (v for k, v in _STATUS.items() if isinstance(exc, k)), status.HTTP_400_BAD_REQUEST
        )
        return JSONResponse(status_code=code, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", path=request.url.path, error=type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    def readyz() -> dict[str, str]:
        with c.engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return {"status": "ready"}

    app.include_router(resumes.router)
    app.include_router(applications.router)
    app.include_router(answers.router)
    return app
