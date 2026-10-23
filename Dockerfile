FROM python:3.12-slim

# LLM provider SDKs to bundle (switch providers with LLM_PROVIDER, no rebuild needed).
ARG EXTRAS="openai,vertex,anthropic,otel"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN useradd --create-home --uid 10001 app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[${EXTRAS}]"

COPY alembic.ini ./
COPY migrations ./migrations

RUN mkdir -p /app/var/resumes && chown -R app:app /app/var
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "job_agent.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
