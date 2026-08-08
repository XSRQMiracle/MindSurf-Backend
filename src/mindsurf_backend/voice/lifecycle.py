"""Request lifecycle primitives for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mindsurf_backend.voice.audio import InputAudioStream, InputCommit
from mindsurf_backend.voice.protocol import ErrorCode, ProtocolError

_LANGUAGE_PATTERN = re.compile(r"^(?:auto|[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$")
_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

SUPPORTED_SELECTIONS = {
    "asr": "sensevoice-small",
    "llm": "mindsurf-omni",
    "tts": "tts-default",
    "output_audio": "pcm16-24k-mono",
}

DEFAULT_VOICE = "default"

CancelReason = Literal[
    "user_cancelled",
    "user_interrupted",
    "client_timeout",
    "audio_backpressure",
    "app_shutdown",
]


class RequestState(StrEnum):
    """States owned by one accepted voice request."""

    ACCEPTED = "accepted"
    RECORDING = "recording"
    GENERATING = "generating"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


class InferenceSelection(BaseModel):
    """Inference candidates selected from the server hello response."""

    model_config = ConfigDict(extra="ignore", strict=True)

    asr: str
    llm: str | None = None
    tts: str | None = None
    output_audio: str | None = None


class RequestInputAudio(BaseModel):
    """Fixed version 1 microphone format."""

    model_config = ConfigDict(extra="ignore", strict=True)

    encoding: Literal["pcm_s16le"]
    sample_rate: Literal[16000]
    channels: Literal[1]
    frame_duration_ms: int = Field(gt=0, le=100, strict=True)


class RequestedResponse(BaseModel):
    """Requested assistant outputs."""

    model_config = ConfigDict(extra="ignore", strict=True)

    text: bool
    audio: bool
    voice: str = Field(min_length=1, max_length=128)


class RequestStartPayload(BaseModel):
    """Validated request.start payload."""

    model_config = ConfigDict(extra="ignore", strict=True)

    mode: Literal["dictation", "assistant"]
    language: str = Field(min_length=2, max_length=35)
    conversation_id: uuid.UUID | None = None
    selection: InferenceSelection
    input_audio: RequestInputAudio
    response: RequestedResponse

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        """Accept `auto` or a bounded BCP 47 language tag."""
        if _LANGUAGE_PATTERN.fullmatch(value) is None:
            raise ValueError("language must be `auto` or a BCP 47 tag")
        return value

    @field_validator("conversation_id", mode="before")
    @classmethod
    def validate_conversation_id(cls, value: object) -> object:
        """Apply the protocol's canonical UUID spelling to conversation IDs."""
        if value is None or isinstance(value, uuid.UUID):
            return value
        if not isinstance(value, str) or _UUID_PATTERN.fullmatch(value) is None:
            raise ValueError("conversation_id must use lowercase canonical UUID form")
        return uuid.UUID(value)

    @model_validator(mode="after")
    def validate_mode_outputs(self) -> RequestStartPayload:
        """Enforce the output combinations defined by each request mode."""
        if self.mode == "assistant" and not self.response.text:
            raise ValueError("assistant mode requires a text response")
        if self.mode == "dictation" and self.response.audio:
            raise ValueError("dictation mode cannot request audio output")
        if self.mode == "assistant" and self.selection.llm is None:
            raise ValueError("assistant mode requires an LLM selection")
        if self.response.audio and (
            self.selection.tts is None or self.selection.output_audio is None
        ):
            raise ValueError("audio responses require TTS and output audio selections")
        return self


class RequestCancelPayload(BaseModel):
    """Validated request.cancel payload."""

    model_config = ConfigDict(extra="ignore", strict=True)

    reason: CancelReason


def validate_request_start(payload: dict[str, Any]) -> RequestStartPayload:
    """Validate request.start and reject candidates not advertised by this server."""
    try:
        request = RequestStartPayload.model_validate(payload)
    except ValidationError as error:
        raise ProtocolError(
            ErrorCode.INVALID_MESSAGE,
            "request.start payload is invalid",
            stage="input",
        ) from error
    required = {"asr"}
    if request.mode == "assistant":
        required.add("llm")
    if request.response.audio:
        required.update(("tts", "output_audio"))
    selection = request.selection.model_dump()
    for name, value in selection.items():
        if name in required and value != SUPPORTED_SELECTIONS[name]:
            raise ProtocolError(
                ErrorCode.UNSUPPORTED_INFERENCE_OPTION,
                f"unsupported {name} selection {value!r}",
                stage="input",
                fatal=True,
            )
        if name not in required and value is not None and value != SUPPORTED_SELECTIONS[name]:
            raise ProtocolError(
                ErrorCode.UNSUPPORTED_INFERENCE_OPTION,
                f"unsupported {name} selection {value!r}",
                stage="input",
                fatal=True,
            )
    if request.response.voice != DEFAULT_VOICE:
        response = request.response.model_copy(update={"voice": DEFAULT_VOICE})
        request = request.model_copy(update={"response": response})
    return request


def validate_request_cancel(payload: dict[str, Any]) -> RequestCancelPayload:
    """Validate request.cancel."""
    try:
        return RequestCancelPayload.model_validate(payload)
    except ValidationError as error:
        raise ProtocolError(
            ErrorCode.INVALID_MESSAGE,
            "request.cancel payload is invalid",
            stage="input",
        ) from error


class ActiveVoiceRequest:
    """Own one request's state, input audio, and live generation resources."""

    def __init__(
        self,
        request_id: uuid.UUID,
        payload: RequestStartPayload,
        *,
        max_recording_ms: int,
    ) -> None:
        self.request_id = request_id
        self.payload = payload
        self.state = RequestState.ACCEPTED
        self.input_audio = InputAudioStream(request_id, max_recording_ms=max_recording_ms)
        self._generations: list[AsyncIterator[Any]] = []
        self._generation_task: asyncio.Task[None] | None = None

    @property
    def is_terminal(self) -> bool:
        """Return whether no further request events may be accepted."""
        return self.state in {RequestState.DONE, RequestState.CANCELLED, RequestState.FAILED}

    def begin_recording(self) -> None:
        """Move an accepted request into its audio-receiving state."""
        self._transition(RequestState.ACCEPTED, RequestState.RECORDING)

    def append_audio(self, data: bytes) -> None:
        """Accept one binary input frame while recording."""
        self._require(RequestState.RECORDING)
        self.input_audio.append(data)

    def commit_input(self, payload: dict[str, Any]) -> InputCommit:
        """Validate commit statistics and enter generation."""
        self._require(RequestState.RECORDING)
        commit = self.input_audio.validate_commit(payload)
        self.state = RequestState.GENERATING
        return commit

    def attach_generation(self, generation: AsyncIterator[Any]) -> None:
        """Attach the Omni iterator that must be closed on termination."""
        self._require(RequestState.GENERATING)
        if any(item is generation for item in self._generations):
            raise RuntimeError("generation is already attached")
        self._generations.append(generation)

    def detach_generation(self, generation: AsyncIterator[Any]) -> None:
        """Forget a fully consumed iterator without disturbing a replacement."""
        self._generations = [item for item in self._generations if item is not generation]

    def attach_generation_task(self, task: asyncio.Task[None]) -> None:
        """Attach the task consuming the Omni iterator."""
        self._require(RequestState.GENERATING)
        if self._generation_task is not None:
            raise RuntimeError("generation task is already attached")
        self._generation_task = task

    async def complete(self) -> None:
        """Move a fully generated request into its success terminal state."""
        self.mark_done()
        await self.close_generation()

    def mark_done(self) -> None:
        """Claim successful completion before asynchronous cleanup begins."""
        self._transition(RequestState.GENERATING, RequestState.DONE)

    async def cancel(self) -> None:
        """Stop live generation and enter the cancellation terminal state."""
        if not self.is_terminal:
            self.state = RequestState.CANCELLED
        await self.close_generation()

    async def fail(self) -> None:
        """Stop live generation and enter the failure terminal state."""
        if not self.is_terminal:
            self.state = RequestState.FAILED
        await self.close_generation()

    async def close_generation(self) -> None:
        """Cancel the consumer and close its async iterator exactly once."""
        current_task = asyncio.current_task()
        task = self._generation_task
        self._generation_task = None
        if task is not None and task is not current_task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        generations = self._generations
        self._generations = []
        for generation in reversed(generations):
            close = getattr(generation, "aclose", None)
            if close is not None:
                with contextlib.suppress(RuntimeError):
                    await close()

    def accepted_payload(self) -> dict[str, Any]:
        """Return the negotiated request.accepted payload without conversation state."""
        return {
            "mode": self.payload.mode,
            "language": self.payload.language,
            "selection": self.payload.selection.model_dump(mode="json"),
            "voice": self.payload.response.voice,
            "max_recording_ms": self.input_audio.max_recording_ms,
        }

    def _require(self, expected: RequestState) -> None:
        if self.state is not expected:
            raise ProtocolError(
                ErrorCode.REQUEST_STATE_ERROR,
                f"request is {self.state.value}, expected {expected.value}",
                stage="input",
                fatal=True,
            )

    def _transition(self, expected: RequestState, target: RequestState) -> None:
        self._require(expected)
        self.state = target
