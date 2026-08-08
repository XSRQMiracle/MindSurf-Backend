"""Request lifecycle tests for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from mindsurf_backend.voice.lifecycle import (
    ActiveVoiceRequest,
    RequestState,
    validate_request_start,
)
from mindsurf_backend.voice.protocol import ErrorCode, ProtocolError


def test_request_lifecycle_reaches_generating() -> None:
    request = _active_request()

    request.begin_recording()
    commit = request.commit_input(
        {
            "last_sequence": None,
            "frame_count": 0,
            "sample_count": 0,
            "duration_ms": 0,
        }
    )

    assert commit.frame_count == 0
    assert request.state is RequestState.GENERATING


@pytest.mark.asyncio
async def test_request_completion_reaches_done() -> None:
    request = _generating_request()

    await request.complete()

    assert request.state is RequestState.DONE


@pytest.mark.asyncio
async def test_request_cancel_stops_consumer_and_closes_generation() -> None:
    request = _generating_request()
    generation = _BlockingGeneration()
    request.attach_generation(generation)
    task = asyncio.create_task(_consume(generation))
    request.attach_generation_task(task)
    await generation.started.wait()

    await request.cancel()

    assert request.state is RequestState.CANCELLED
    assert task.cancelled()
    assert generation.closed is True


def test_request_rejects_commit_outside_recording() -> None:
    request = _active_request()

    with pytest.raises(ProtocolError) as error:
        request.commit_input(
            {
                "last_sequence": None,
                "frame_count": 0,
                "sample_count": 0,
                "duration_ms": 0,
            }
        )

    assert error.value.code is ErrorCode.REQUEST_STATE_ERROR
    assert error.value.fatal is True


def test_request_start_validates_options_and_falls_back_voice() -> None:
    payload = _request_start_payload()
    payload["response"] = {"text": True, "audio": True, "voice": "unknown"}

    validated = validate_request_start(payload)

    assert validated.response.voice == "default"
    assert validated.conversation_id is not None


def _active_request() -> ActiveVoiceRequest:
    return ActiveVoiceRequest(
        uuid.uuid4(),
        validate_request_start(_request_start_payload()),
        max_recording_ms=60_000,
    )


def _generating_request() -> ActiveVoiceRequest:
    request = _active_request()
    request.begin_recording()
    request.commit_input(
        {
            "last_sequence": None,
            "frame_count": 0,
            "sample_count": 0,
            "duration_ms": 0,
        }
    )
    return request


def _request_start_payload() -> dict[str, Any]:
    return {
        "mode": "assistant",
        "language": "zh-CN",
        "conversation_id": str(uuid.uuid4()),
        "selection": {
            "asr": "sensevoice-small",
            "llm": "mindsurf-omni",
            "tts": "tts-default",
            "output_audio": "pcm16-24k-mono",
        },
        "input_audio": {
            "encoding": "pcm_s16le",
            "sample_rate": 16_000,
            "channels": 1,
            "frame_duration_ms": 20,
        },
        "response": {"text": True, "audio": True, "voice": "default"},
    }


class _BlockingGeneration:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    def __aiter__(self) -> _BlockingGeneration:
        return self

    async def __anext__(self) -> object:
        self.started.set()
        await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


async def _consume(generation: _BlockingGeneration) -> None:
    async for _ in generation:
        pass
