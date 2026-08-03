"""OpenAI-compatible inference endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from python_starter.api.dependencies import InferenceServiceDep
from python_starter.inference.types import InferenceBackendUnavailableError
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    payload: dict[str, Any],
    service: InferenceServiceDep,
) -> Response:
    """Handle OpenAI-compatible chat completions with the selected backend."""
    try:
        backend_response = await service.chat_completions(payload)
    except InferenceBackendUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if backend_response.body is not None:
        return Response(
            content=backend_response.body,
            status_code=backend_response.status_code,
            media_type=backend_response.content_type,
        )

    upstream_stream = backend_response.stream
    if upstream_stream is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Inference backend returned an empty response",
        )

    async def forward_stream() -> AsyncIterator[bytes]:
        async with aclosing(upstream_stream):
            async for chunk in upstream_stream:
                if await request.is_disconnected():
                    break
                yield chunk

    return StreamingResponse(
        forward_stream(),
        status_code=backend_response.status_code,
        headers={
            "Content-Type": backend_response.content_type,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
