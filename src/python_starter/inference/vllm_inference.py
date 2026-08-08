"""vLLM inference backend using its OpenAI-compatible HTTP API."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx
import orjson

from python_starter.inference.types import (
    BackendHTTPResponse,
    FinishReason,
    InferenceBackendUnavailableError,
    InferenceRequest,
    InferenceRequestError,
    InferenceResult,
    OpenAIChatPayload,
)
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


class VLLMInference:
    """Inference backend backed by a remote vLLM OpenAI server."""

    name = "vllm"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        )

    @property
    def model_name(self) -> str:
        """Return the vLLM served model name."""
        return self._model_name

    async def chat_completions(self, payload: OpenAIChatPayload) -> BackendHTTPResponse:
        """Forward an OpenAI chat completion request without changing its payload."""
        is_streaming = payload.get("stream") is True
        upstream_request = self._client.build_request(
            "POST",
            "/v1/chat/completions",
            json=payload,
        )

        try:
            upstream = await self._client.send(upstream_request, stream=is_streaming)
        except httpx.RequestError as exc:
            logger.warning("vllm_unavailable", error=str(exc))
            raise InferenceBackendUnavailableError("vLLM unavailable") from exc

        content_type = upstream.headers.get("content-type", "application/json")
        if upstream.status_code >= 400 or not is_streaming:
            body = await upstream.aread()
            await upstream.aclose()
            return BackendHTTPResponse(
                status_code=upstream.status_code,
                content_type=content_type,
                body=body,
            )

        async def forward_stream() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            except httpx.RequestError as exc:
                logger.warning("vllm_stream_interrupted", error=str(exc))
            finally:
                await upstream.aclose()

        return BackendHTTPResponse(
            status_code=upstream.status_code,
            content_type=content_type,
            stream=forward_stream(),
        )

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate text through vLLM and normalize the OpenAI response."""
        payload: OpenAIChatPayload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.seed is not None:
            payload["seed"] = request.seed

        started = time.perf_counter()
        response = await self.chat_completions(payload)
        if response.body is None:
            raise InferenceRequestError(502, "vLLM returned an empty response")
        if response.status_code >= 400:
            raise InferenceRequestError(
                response.status_code,
                _extract_error_detail(response.body),
            )

        data = _load_object(response.body)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InferenceRequestError(502, "vLLM response has no completion choice")
        choice = cast(dict[str, Any], choices[0])
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise InferenceRequestError(502, "vLLM response has no generated text")

        usage_value = data.get("usage")
        usage = usage_value if isinstance(usage_value, dict) else {}
        finish_reason: FinishReason = (
            "length" if choice.get("finish_reason") == "length" else "stop"
        )
        return InferenceResult(
            input_prompt=request.prompt,
            generated_text=cast(str, message["content"]),
            backend_name=self.name,
            model_name=str(data.get("model", self.model_name)),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            generated_tokens=int(usage.get("completion_tokens", 0)),
            generation_time=time.perf_counter() - started,
            stop_reason=finish_reason,
        )

    async def aclose(self) -> None:
        """Close the owned HTTP client."""
        if self._owns_client:
            await self._client.aclose()


def _load_object(body: bytes) -> dict[str, Any]:
    try:
        value = orjson.loads(body)
    except orjson.JSONDecodeError as exc:
        raise InferenceRequestError(502, "vLLM returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise InferenceRequestError(502, "vLLM returned invalid JSON")
    return cast(dict[str, Any], value)


def _extract_error_detail(body: bytes) -> str:
    try:
        data = _load_object(body)
    except InferenceRequestError:
        return body.decode(errors="replace") or "vLLM request failed"
    error = data.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return cast(str, error["message"])
    if isinstance(data.get("detail"), str):
        return cast(str, data["detail"])
    return "vLLM request failed"
