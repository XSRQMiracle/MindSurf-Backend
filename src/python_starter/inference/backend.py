"""Inference backend protocol."""

from __future__ import annotations

from typing import Protocol

from python_starter.inference.types import (
    BackendHTTPResponse,
    InferenceRequest,
    InferenceResult,
    OpenAIChatPayload,
)


class InferenceBackend(Protocol):
    """Interface implemented by all inference backends."""

    @property
    def name(self) -> str:
        """Return the backend name."""
        ...

    @property
    def model_name(self) -> str:
        """Return the served model name."""
        ...

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate a complete text response."""
        ...

    async def chat_completions(self, payload: OpenAIChatPayload) -> BackendHTTPResponse:
        """Handle an OpenAI-compatible chat completion request."""
        ...

    async def aclose(self) -> None:
        """Release resources owned by the backend."""
        ...
