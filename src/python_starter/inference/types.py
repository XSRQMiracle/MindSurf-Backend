"""Shared inference request and response types."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

FinishReason = Literal["stop", "length"]
OpenAIChatPayload = dict[str, Any]


@dataclass(frozen=True)
class PyTorchModelConfig:
    """Configuration required to load the native PyTorch backend."""

    checkpoint_path: str
    tokenizer_path: str
    model_name: str
    model_config_path: str | None = None
    device: str = "auto"


@dataclass(frozen=True)
class InferenceRequest:
    """Backend-neutral text generation request."""

    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class InferenceResult:
    """Normalized result returned by text generation backends."""

    input_prompt: str
    generated_text: str
    backend_name: str
    model_name: str
    input_tokens: int
    generated_tokens: int
    generation_time: float
    stop_reason: FinishReason


@dataclass(frozen=True)
class BackendHTTPResponse:
    """Transport-neutral HTTP response produced by an inference backend."""

    status_code: int
    content_type: str
    body: bytes | None = None
    stream: AsyncGenerator[bytes, None] | None = None


class InferenceBackendUnavailableError(RuntimeError):
    """Raised when the selected inference backend cannot be reached."""


class InferenceRequestError(ValueError):
    """Raised when a backend rejects a normalized inference request."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
