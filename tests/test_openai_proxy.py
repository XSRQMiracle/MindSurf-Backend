"""Tests for the OpenAI-compatible inference endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from python_starter.api.dependencies import get_inference_service
from python_starter.api.routers import openai
from python_starter.inference.service import InferenceService
from python_starter.inference.vllm_inference import VLLMInference


class ChunkedSSEStream(httpx.AsyncByteStream):
    """Small asynchronous stream used to emulate vLLM SSE output."""

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _app_with_service(service: InferenceService) -> FastAPI:
    app = FastAPI()
    app.include_router(openai.router, prefix="/v1")

    async def override_service() -> InferenceService:
        return service

    app.dependency_overrides[get_inference_service] = override_service
    return app


@pytest.mark.asyncio
async def test_chat_completions_forwards_vllm_stream() -> None:
    stream = ChunkedSSEStream(
        (
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b"data: [DONE]\n\n",
        )
    )

    def handle_upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            stream=stream,
        )

    upstream_client = AsyncClient(
        base_url="http://vllm:8001",
        transport=httpx.MockTransport(handle_upstream),
    )
    service = InferenceService(
        VLLMInference("http://vllm:8001", "minimind", client=upstream_client)
    )
    app = _app_with_service(service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "minimind",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    await upstream_client.aclose()
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert stream.closed is True


@pytest.mark.asyncio
async def test_chat_completions_preserves_upstream_error() -> None:
    def handle_upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=400,
            headers={"Content-Type": "application/json"},
            json={"error": {"message": "unknown model"}},
        )

    upstream_client = AsyncClient(
        base_url="http://vllm:8001",
        transport=httpx.MockTransport(handle_upstream),
    )
    service = InferenceService(
        VLLMInference("http://vllm:8001", "minimind", client=upstream_client)
    )
    app = _app_with_service(service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": [], "stream": True},
        )

    await upstream_client.aclose()
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "unknown model"}}


@pytest.mark.asyncio
async def test_chat_completions_forwards_non_streaming_response() -> None:
    response_body = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    def handle_upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json=response_body)

    upstream_client = AsyncClient(
        base_url="http://vllm:8001",
        transport=httpx.MockTransport(handle_upstream),
    )
    service = InferenceService(
        VLLMInference("http://vllm:8001", "minimind", client=upstream_client)
    )
    app = _app_with_service(service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "minimind", "messages": [], "stream": False},
        )

    await upstream_client.aclose()
    assert response.status_code == 200
    assert response.json() == response_body
