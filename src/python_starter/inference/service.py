"""Application service for model inference."""

from __future__ import annotations

from python_starter.inference.backend import InferenceBackend
from python_starter.inference.types import (
    BackendHTTPResponse,
    InferenceRequest,
    InferenceResult,
    OpenAIChatPayload,
)


class InferenceService:
    """Delegate inference requests to the configured backend."""

    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend

    @property
    def backend_name(self) -> str:
        """Return the configured backend name."""
        return self._backend.name

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._backend.model_name

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate a complete text response."""
        return await self._backend.generate(request)

    async def chat_completions(self, payload: OpenAIChatPayload) -> BackendHTTPResponse:
        """Handle an OpenAI-compatible chat completion request."""
        return await self._backend.chat_completions(payload)

    async def aclose(self) -> None:
        """Close the configured backend."""
        await self._backend.aclose()
