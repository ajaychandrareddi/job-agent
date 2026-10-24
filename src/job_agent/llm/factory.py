"""Provider-agnostic chat-model construction.

Workflows depend only on ``BaseChatModel``; switching provider is a config change.
Provider packages are optional extras and imported lazily.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from job_agent.config import LLMProvider, Settings

DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.VERTEX: "gemini-2.0-flash",
    LLMProvider.ANTHROPIC: "claude-sonnet-5-5",
}


class LLMConfigurationError(RuntimeError):
    pass


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    if provider is LLMProvider.FAKE:
        raise LLMConfigurationError(
            "LLM_PROVIDER=fake has no chat model; use the heuristic extractor or inject a model"
        )
    model = settings.llm_model or DEFAULT_MODELS[provider]
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    common: dict[str, Any] = {"temperature": settings.llm_temperature}

    try:
        if provider is LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model, api_key=api_key, timeout=settings.llm_timeout_seconds, **common
            )
        if provider is LLMProvider.VERTEX:
            from langchain_google_vertexai import ChatVertexAI

            return ChatVertexAI(
                model=model,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                **common,
            )
        if provider is LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            # Current Claude models reject `temperature` with a 400, so it is never sent.
            return ChatAnthropic(
                model_name=model, api_key=api_key, timeout=settings.llm_timeout_seconds
            )
    except ImportError as exc:
        raise LLMConfigurationError(
            f"provider '{provider}' requires the optional extra: "
            f"pip install 'job-agent[{provider}]'"
        ) from exc
    raise LLMConfigurationError(f"unsupported LLM provider: {provider}")
