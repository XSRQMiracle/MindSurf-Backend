"""Control-message primitives for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import time
import uuid
from enum import IntEnum, StrEnum
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

PROTOCOL_VERSION = 1
VOICE_SUBPROTOCOL = "mindsurf.voice.v1"
PIPELINE = "cascade"
MAX_JSON_BYTES = 65_536
MAX_BINARY_BYTES = 65_536
MAX_JSON_DEPTH = 16
MAX_RECORDING_MS = 60_000
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000

_UUID_PATTERN = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_MESSAGE_TYPE_PATTERN = "^[a-z][a-z0-9]*(?:\\.[a-z][a-z0-9]*)*$"


class CloseCode(IntEnum):
    """WebSocket close codes used by the protocol."""

    NORMAL = 1000
    GOING_AWAY = 1001
    PROTOCOL_ERROR = 1002
    MESSAGE_TOO_BIG = 1009
    INTERNAL_ERROR = 1011
    HANDSHAKE_TIMEOUT = 4001
    VERSION_MISMATCH = 4002
    AUTHENTICATION_FAILED = 4003
    ORIGIN_REJECTED = 4004
    SERVER_BUSY = 4008


class ErrorCode(StrEnum):
    """Stable protocol error codes available to clients."""

    INVALID_JSON = "invalid_json"
    INVALID_MESSAGE = "invalid_message"
    UNSUPPORTED_MESSAGE_TYPE = "unsupported_message_type"
    PROTOCOL_VERSION_MISMATCH = "protocol_version_mismatch"
    HANDSHAKE_REQUIRED = "handshake_required"
    HANDSHAKE_TIMEOUT = "handshake_timeout"
    REQUEST_ALREADY_ACTIVE = "request_already_active"
    REQUEST_NOT_FOUND = "request_not_found"
    REQUEST_STATE_ERROR = "request_state_error"
    UNSUPPORTED_INFERENCE_OPTION = "unsupported_inference_option"
    UNSUPPORTED_AUDIO_KIND = "unsupported_audio_kind"
    UNSUPPORTED_AUDIO_FORMAT = "unsupported_audio_format"
    INVALID_AUDIO_FRAME = "invalid_audio_frame"
    AUDIO_SEQUENCE_ERROR = "audio_sequence_error"
    AUDIO_COMMIT_MISMATCH = "audio_commit_mismatch"
    AUDIO_TOO_SHORT = "audio_too_short"
    AUDIO_TOO_LONG = "audio_too_long"
    ASR_EMPTY = "asr_empty"
    ASR_FAILED = "asr_failed"
    LLM_FAILED = "llm_failed"
    TTS_FAILED = "tts_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    INTERNAL_ERROR = "internal_error"


ErrorStage = Literal["session", "input", "asr", "llm", "tts", "protocol"]


class ProtocolError(ValueError):
    """A protocol error that can be translated into a stable client response."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        stage: ErrorStage = "protocol",
        fatal: bool = False,
        close_code: CloseCode | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.fatal = fatal
        self.close_code = close_code


class ControlEnvelope(BaseModel):
    """Common JSON envelope for every control message."""

    model_config = ConfigDict(extra="ignore", strict=True)

    v: Literal[1]
    type: str = Field(pattern=_MESSAGE_TYPE_PATTERN, min_length=3, max_length=128)
    event_id: uuid.UUID
    request_id: uuid.UUID | None
    sent_at_ms: StrictInt = Field(ge=0)
    payload: dict[str, Any]

    @field_validator("event_id", "request_id", mode="before")
    @classmethod
    def validate_lowercase_uuid(cls, value: object) -> object:
        """Reject UUID spellings the desktop protocol does not emit."""
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        if not isinstance(value, str):
            raise ValueError("UUID fields must be strings")
        import re

        if re.fullmatch(_UUID_PATTERN, value) is None:
            raise ValueError("UUID fields must use lowercase canonical form")
        return uuid.UUID(value)


class ProtocolErrorPayload(BaseModel):
    """Stable error payload returned for protocol and request failures."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: str
    message: str
    stage: ErrorStage
    recoverable: bool
    fatal: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ClientIdentity(BaseModel):
    """Desktop-client identity sent during handshake."""

    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    platform: Literal["windows", "macos", "linux"]
    arch: str = Field(min_length=1, max_length=64)


class InputAudioCapability(BaseModel):
    """Supported microphone format advertised by a client."""

    model_config = ConfigDict(extra="ignore", strict=True)

    encoding: Literal["pcm_s16le"]
    sample_rate: Literal[16000]
    channels: Literal[1]


class OutputAudioCapability(BaseModel):
    """Supported playback formats advertised by a client."""

    model_config = ConfigDict(extra="ignore", strict=True)

    encoding: Literal["pcm_s16le"]
    sample_rates: list[Literal[16000, 24000]] = Field(min_length=1)
    channels: Literal[1]


class ClientHelloPayload(BaseModel):
    """Validated capabilities from the initial client handshake."""

    model_config = ConfigDict(extra="ignore", strict=True)

    client: ClientIdentity
    protocol_versions: list[StrictInt] = Field(min_length=1)
    pipelines: list[str] = Field(min_length=1)
    input_audio: list[InputAudioCapability] = Field(min_length=1)
    output_audio: list[OutputAudioCapability] = Field(min_length=1)


def parse_control_message(data: str | bytes, *, max_bytes: int = MAX_JSON_BYTES) -> ControlEnvelope:
    """Parse and validate one JSON control message."""
    encoded = data.encode("utf-8") if isinstance(data, str) else data
    if len(encoded) > max_bytes:
        raise ProtocolError(
            ErrorCode.INVALID_MESSAGE,
            "JSON message exceeds the negotiated size limit",
            close_code=CloseCode.MESSAGE_TOO_BIG,
        )
    try:
        value = orjson.loads(encoded)
    except (orjson.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolError(ErrorCode.INVALID_JSON, "message is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError(ErrorCode.INVALID_MESSAGE, "message must be an object")
    _validate_json_depth(value)
    try:
        return ControlEnvelope.model_validate(value)
    except ValidationError as error:
        raise ProtocolError(ErrorCode.INVALID_MESSAGE, "message envelope is invalid") from error


def validate_client_hello(envelope: ControlEnvelope) -> ClientHelloPayload:
    """Validate the initial handshake and negotiated media formats."""
    if envelope.type != "client.hello" or envelope.request_id is not None:
        raise ProtocolError(
            ErrorCode.HANDSHAKE_REQUIRED,
            "client.hello must be the first message",
            stage="session",
            fatal=True,
            close_code=CloseCode.PROTOCOL_ERROR,
        )
    try:
        hello = ClientHelloPayload.model_validate(envelope.payload)
    except ValidationError as error:
        raise ProtocolError(
            ErrorCode.INVALID_MESSAGE,
            "client.hello payload is invalid",
            stage="session",
            fatal=True,
            close_code=CloseCode.PROTOCOL_ERROR,
        ) from error
    if PROTOCOL_VERSION not in hello.protocol_versions:
        raise ProtocolError(
            ErrorCode.PROTOCOL_VERSION_MISMATCH,
            "client and server have no common protocol version",
            stage="session",
            fatal=True,
            close_code=CloseCode.VERSION_MISMATCH,
        )
    if PIPELINE not in hello.pipelines:
        raise ProtocolError(
            ErrorCode.PROTOCOL_VERSION_MISMATCH,
            "client does not support the cascade pipeline",
            stage="session",
            fatal=True,
            close_code=CloseCode.VERSION_MISMATCH,
        )
    supports_output = any(OUTPUT_SAMPLE_RATE in item.sample_rates for item in hello.output_audio)
    if not supports_output:
        raise ProtocolError(
            ErrorCode.UNSUPPORTED_AUDIO_FORMAT,
            "client does not support PCM16 24 kHz mono output",
            stage="session",
            fatal=True,
            close_code=CloseCode.PROTOCOL_ERROR,
        )
    return hello


def create_envelope(
    message_type: str,
    request_id: uuid.UUID | None,
    payload: dict[str, Any],
) -> ControlEnvelope:
    """Create a server control message with fresh identity and timestamp."""
    return ControlEnvelope(
        v=1,
        type=message_type,
        event_id=uuid.uuid4(),
        request_id=request_id,
        sent_at_ms=int(time.time() * 1000),
        payload=payload,
    )


def create_error_envelope(
    violation: ProtocolError,
    request_id: uuid.UUID | None = None,
) -> ControlEnvelope:
    """Translate a protocol violation into the documented error shape."""
    payload = ProtocolErrorPayload(
        code=violation.code.value,
        message=str(violation),
        stage=violation.stage,
        recoverable=not violation.fatal,
        fatal=violation.fatal,
    )
    return create_envelope("error", request_id, payload.model_dump(mode="json"))


def create_server_hello_payload(
    *,
    heartbeat_interval_ms: int,
    heartbeat_timeout_ms: int,
    max_recording_ms: int = MAX_RECORDING_MS,
    max_json_bytes: int = MAX_JSON_BYTES,
    max_binary_bytes: int = MAX_BINARY_BYTES,
    streaming_text: bool = False,
    streaming_audio: bool = False,
) -> dict[str, Any]:
    """Return the protocol capabilities currently implemented by the backend."""
    return {
        "session_id": str(uuid.uuid4()),
        "protocol_version": PROTOCOL_VERSION,
        "pipeline": PIPELINE,
        "limits": {
            "max_recording_ms": max_recording_ms,
            "max_json_bytes": max_json_bytes,
            "max_binary_bytes": max_binary_bytes,
        },
        "features": {
            "streaming_asr": False,
            "streaming_text": streaming_text,
            "streaming_audio": streaming_audio,
            "cancellation": True,
        },
        "inference_options": {
            "defaults": {
                "asr": "sensevoice-small",
                "llm": "mindsurf-omni",
                "tts": "tts-default",
                "output_audio": "pcm16-24k-mono",
            },
            "asr": [
                {
                    "id": "sensevoice-small",
                    "name": "SenseVoice Small",
                    "description": "MindSurf speech recognition",
                }
            ],
            "llm": [
                {
                    "id": "mindsurf-omni",
                    "name": "MindSurf Omni",
                    "description": "MindSurf assistant model",
                }
            ],
            "tts": [
                {
                    "id": "tts-default",
                    "name": "Default TTS",
                    "description": "Configured speech synthesiser",
                }
            ],
            "output_audio": [
                {
                    "id": "pcm16-24k-mono",
                    "name": "PCM 24 kHz Mono",
                    "description": "Uncompressed streaming speech",
                    "encoding": "pcm_s16le",
                    "sample_rate": OUTPUT_SAMPLE_RATE,
                    "channels": 1,
                }
            ],
        },
        "heartbeat": {
            "interval_ms": heartbeat_interval_ms,
            "timeout_ms": heartbeat_timeout_ms,
        },
    }


def envelope_json(envelope: ControlEnvelope) -> dict[str, Any]:
    """Convert an envelope to JSON-compatible values for WebSocket transmission."""
    return envelope.model_dump(mode="json")


def _validate_json_depth(value: object, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ProtocolError(ErrorCode.INVALID_MESSAGE, "JSON message is nested too deeply")
    if isinstance(value, dict):
        for item in value.values():
            _validate_json_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_depth(item, depth + 1)
