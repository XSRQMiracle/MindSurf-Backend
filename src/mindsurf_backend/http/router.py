"""Health probes and direct diagnostic endpoints for the Omni pipeline."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from mindsurf_backend.config import AppSettings
from mindsurf_backend.omni import OmniAdapter, OmniNotConfiguredError
from mindsurf_backend.omni.adapter import OmniStage
from mindsurf_omni.contract import (
    AUDIO_ENCODING,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    SpeechRequest,
    TranscriptionResponse,
)
from mindsurf_omni.service.audio import to_wav
from mindsurf_omni.service.engine import GenerationSettings, SpeechEngine


def register_http_api(
    app: FastAPI,
    settings: AppSettings,
    omni_adapter: OmniAdapter,
) -> None:
    """Register health probes and model diagnostic endpoints."""
    router = APIRouter()

    def require_stages(*stages: OmniStage) -> SpeechEngine:
        try:
            return omni_adapter.require_stages(*stages)
        except OmniNotConfiguredError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error

    @router.get("/health/live")
    async def live() -> dict[str, str]:
        """Report that the API process is alive, independent of model readiness."""
        return {
            "status": "live",
            "name": settings.app_name,
            "version": settings.app_version,
        }

    @router.get("/health/ready")
    async def ready() -> JSONResponse:
        """Report whether all three cascade stages can serve the full product flow."""
        stage_status = omni_adapter.stage_status
        body: dict[str, Any] = {
            "status": "ready" if omni_adapter.ready else "unavailable",
            "engine_available": omni_adapter.available,
            "stages": stage_status,
        }
        response_status = status.HTTP_200_OK
        if not omni_adapter.ready:
            response_status = status.HTTP_503_SERVICE_UNAVAILABLE
            try:
                omni_adapter.require_stages("transcriber", "generator", "synthesiser")
            except OmniNotConfiguredError as error:
                body["reason"] = str(error)
        return JSONResponse(body, status_code=response_status)

    @router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
    async def transcribe(request: Request) -> TranscriptionResponse:
        """Transcribe raw mono PCM16 audio sampled at 16 kHz."""
        engine = require_stages("transcriber")
        pcm = await request.body()
        if not pcm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="request body carried no audio",
            )
        text, language = await engine.transcribe(pcm, INPUT_SAMPLE_RATE)
        return TranscriptionResponse(
            text=text,
            language=language,
            duration_seconds=len(pcm) / 2 / INPUT_SAMPLE_RATE,
        )

    @router.post("/v1/chat/completions", response_model=None)
    async def chat(body: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
        """Run the text generator using an OpenAI-compatible request shape."""
        engine = require_stages("generator")
        generation = GenerationSettings(
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
        )
        messages = [{"role": message.role, "content": message.content} for message in body.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not body.stream:
            content = "".join([delta async for delta in engine.complete(messages, generation)])
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[
                    ChatChoice(
                        message=ChatMessage(role="assistant", content=content),
                        finish_reason="stop",
                    )
                ],
            )

        async def stream_completion() -> AsyncIterator[str]:
            async for delta in engine.complete(messages, generation):
                yield _sse_event(
                    completion_id=completion_id,
                    created=created,
                    model=body.model,
                    delta={"content": delta},
                )
            yield _sse_event(
                completion_id=completion_id,
                created=created,
                model=body.model,
                delta={},
                finish_reason="stop",
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_completion(), media_type="text/event-stream")

    @router.post("/v1/audio/speech")
    async def speech(body: SpeechRequest) -> StreamingResponse:
        """Synthesize text as raw streaming PCM or a buffered WAV container."""
        engine = require_stages("synthesiser")
        generation = GenerationSettings(voice=body.voice, emotion=body.emotion)
        speaking = engine.speak(body.input, generation)

        async def stream_audio() -> AsyncIterator[bytes]:
            if body.response_format == "wav":
                pcm = bytearray()
                async for chunk in speaking:
                    pcm.extend(chunk.pcm)
                yield to_wav(bytes(pcm), OUTPUT_SAMPLE_RATE)
                return
            async for chunk in speaking:
                if chunk.pcm:
                    yield chunk.pcm

        media_type = "audio/wav" if body.response_format == "wav" else "application/octet-stream"
        return StreamingResponse(
            stream_audio(),
            media_type=media_type,
            headers={
                "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                "X-Encoding": AUDIO_ENCODING,
            },
        )

    app.include_router(router)


def _sse_event(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> str:
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
