"""OpenAI-compatible proxy endpoints backed by a vLLM server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


async def _buffer_response(upstream: httpx.Response) -> Response:
    """Read and close a non-streaming upstream response."""
    content = await upstream.aread()
    await upstream.aclose()
    return Response(
        content=content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    payload: dict[str, Any],
) -> Response:
    """Proxy OpenAI-compatible chat completions to the configured vLLM server."""
    client: httpx.AsyncClient | None = getattr(request.app.state, "vllm_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM client is not initialized",
        )

    upstream_request = client.build_request(
        "POST",
        "/v1/chat/completions",
        json=payload,
    )
    is_streaming = payload.get("stream") is True

    try:
        upstream = await client.send(upstream_request, stream=is_streaming)
    except httpx.RequestError as exc:
        logger.warning("vllm_unavailable", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="vLLM unavailable",
        ) from exc

    if upstream.status_code >= status.HTTP_400_BAD_REQUEST or not is_streaming:
        return await _buffer_response(upstream)

    async def forward_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                if await request.is_disconnected():
                    break
                yield chunk
        except httpx.RequestError as exc:
            logger.warning("vllm_stream_interrupted", error=str(exc))
        finally:
            await upstream.aclose()

    return StreamingResponse(
        forward_stream(),
        status_code=upstream.status_code,
        headers={
            "Content-Type": upstream.headers.get("content-type", "text/event-stream"),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
