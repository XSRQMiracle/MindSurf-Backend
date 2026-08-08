"""Binary PCM frame primitives for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from mindsurf_backend.voice.protocol import (
    INPUT_SAMPLE_RATE,
    MAX_BINARY_BYTES,
    MAX_RECORDING_MS,
    PROTOCOL_VERSION,
    ErrorCode,
    ProtocolError,
)

AUDIO_MAGIC = b"MSVA"
AUDIO_HEADER_LENGTH = 48
_AUDIO_HEADER = struct.Struct("!4sBBHHHIQII16s")
_MAX_U32 = 0xFFFF_FFFF
_MAX_U64 = 0xFFFF_FFFF_FFFF_FFFF


class AudioKind(IntEnum):
    """Binary audio directions defined by protocol version 1."""

    INPUT_PCM = 0x01
    OUTPUT_PCM = 0x02


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One validated PCM16 audio frame."""

    kind: AudioKind
    sequence: int
    timestamp_us: int
    request_id: uuid.UUID
    pcm: bytes

    @property
    def sample_count(self) -> int:
        """Return the number of mono PCM16 samples in the frame."""
        return len(self.pcm) // 2


class InputCommit(BaseModel):
    """Client statistics supplied when microphone input is committed."""

    model_config = ConfigDict(extra="ignore", strict=True)

    last_sequence: StrictInt | None = Field(ge=0)
    frame_count: StrictInt = Field(ge=0)
    sample_count: StrictInt = Field(ge=0)
    duration_ms: StrictInt = Field(ge=0)


def encode_audio_frame(frame: AudioFrame) -> bytes:
    """Encode one validated frame using the fixed 48-byte network-order header."""
    _validate_numeric_range(frame.sequence, _MAX_U32, "audio sequence")
    _validate_numeric_range(frame.timestamp_us, _MAX_U64, "audio timestamp")
    _validate_pcm(frame.pcm)
    total_length = AUDIO_HEADER_LENGTH + len(frame.pcm)
    if total_length > MAX_BINARY_BYTES:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "audio frame exceeds the negotiated size limit",
            stage="input",
            fatal=True,
        )
    header = _AUDIO_HEADER.pack(
        AUDIO_MAGIC,
        PROTOCOL_VERSION,
        int(frame.kind),
        0,
        AUDIO_HEADER_LENGTH,
        0,
        frame.sequence,
        frame.timestamp_us,
        len(frame.pcm),
        0,
        frame.request_id.bytes,
    )
    return header + frame.pcm


def decode_audio_frame(
    data: bytes,
    *,
    expected_kind: AudioKind | None = None,
    expected_request_id: uuid.UUID | None = None,
    expected_sequence: int | None = None,
    max_bytes: int = MAX_BINARY_BYTES,
) -> AudioFrame:
    """Decode and validate one complete WebSocket binary message."""
    if len(data) > max_bytes:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "audio frame exceeds the negotiated size limit",
            stage="input",
            fatal=True,
        )
    if len(data) < AUDIO_HEADER_LENGTH:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "audio frame is shorter than its header",
            stage="input",
            fatal=True,
        )
    (
        magic,
        version,
        raw_kind,
        flags,
        header_length,
        reserved_1,
        sequence,
        timestamp_us,
        payload_length,
        reserved_2,
        request_bytes,
    ) = _AUDIO_HEADER.unpack_from(data)
    if magic != AUDIO_MAGIC or version != PROTOCOL_VERSION or header_length != AUDIO_HEADER_LENGTH:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "audio frame header is invalid",
            stage="input",
            fatal=True,
        )
    if flags != 0 or reserved_1 != 0 or reserved_2 != 0:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "reserved audio header fields must be zero",
            stage="input",
            fatal=True,
        )
    if len(data) != header_length + payload_length:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "audio payload length does not match its header",
            stage="input",
            fatal=True,
        )
    try:
        kind = AudioKind(raw_kind)
    except ValueError as error:
        raise ProtocolError(
            ErrorCode.UNSUPPORTED_AUDIO_KIND,
            f"audio kind {raw_kind} is not supported",
            stage="input",
            fatal=True,
        ) from error
    if expected_kind is not None and kind is not expected_kind:
        raise ProtocolError(
            ErrorCode.UNSUPPORTED_AUDIO_KIND,
            f"expected {expected_kind.name}, received {kind.name}",
            stage="input",
            fatal=True,
        )
    request_id = uuid.UUID(bytes=request_bytes)
    if expected_request_id is not None and request_id != expected_request_id:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "audio request ID does not match the active request",
            stage="input",
            fatal=True,
        )
    if expected_sequence is not None and sequence != expected_sequence:
        raise ProtocolError(
            ErrorCode.AUDIO_SEQUENCE_ERROR,
            f"expected audio sequence {expected_sequence}, received {sequence}",
            stage="input",
            fatal=True,
        )
    pcm = data[header_length:]
    _validate_pcm(pcm)
    return AudioFrame(
        kind=kind,
        sequence=sequence,
        timestamp_us=timestamp_us,
        request_id=request_id,
        pcm=pcm,
    )


@dataclass(slots=True)
class InputAudioStream:
    """Validate and accumulate one request's strictly ordered microphone stream."""

    request_id: uuid.UUID
    max_recording_ms: int = MAX_RECORDING_MS
    _parts: list[bytes] = field(default_factory=list)
    _frame_count: int = 0
    _sample_count: int = 0

    @property
    def frame_count(self) -> int:
        """Return the number of accepted input frames."""
        return self._frame_count

    @property
    def sample_count(self) -> int:
        """Return the number of accepted mono PCM samples."""
        return self._sample_count

    @property
    def last_sequence(self) -> int | None:
        """Return the last accepted sequence, or None for an empty stream."""
        return self._frame_count - 1 if self._frame_count else None

    @property
    def duration_ms(self) -> int:
        """Return the rounded duration represented by accepted samples."""
        return (self._sample_count * 1000 + INPUT_SAMPLE_RATE // 2) // INPUT_SAMPLE_RATE

    def append(self, data: bytes) -> AudioFrame:
        """Validate and append the next INPUT_PCM frame."""
        expected_timestamp = round(self._sample_count * 1_000_000 / INPUT_SAMPLE_RATE)
        frame = decode_audio_frame(
            data,
            expected_kind=AudioKind.INPUT_PCM,
            expected_request_id=self.request_id,
            expected_sequence=self._frame_count,
        )
        if frame.timestamp_us != expected_timestamp:
            raise ProtocolError(
                ErrorCode.INVALID_AUDIO_FRAME,
                f"expected audio timestamp {expected_timestamp}, received {frame.timestamp_us}",
                stage="input",
                fatal=True,
            )
        next_samples = self._sample_count + frame.sample_count
        if next_samples * 1000 > self.max_recording_ms * INPUT_SAMPLE_RATE:
            raise ProtocolError(
                ErrorCode.AUDIO_TOO_LONG,
                "recording exceeds the configured duration limit",
                stage="input",
                fatal=True,
            )
        self._parts.append(frame.pcm)
        self._frame_count += 1
        self._sample_count = next_samples
        return frame

    def validate_commit(self, payload: dict[str, Any]) -> InputCommit:
        """Validate client commit statistics against accepted audio."""
        try:
            commit = InputCommit.model_validate(payload)
        except ValidationError as error:
            raise ProtocolError(
                ErrorCode.INVALID_MESSAGE,
                "input.commit payload is invalid",
                stage="input",
            ) from error
        matches = (
            commit.last_sequence == self.last_sequence
            and commit.frame_count == self.frame_count
            and commit.sample_count == self.sample_count
            and commit.duration_ms == self.duration_ms
        )
        if not matches:
            raise ProtocolError(
                ErrorCode.AUDIO_COMMIT_MISMATCH,
                "input.commit statistics do not match received audio",
                stage="input",
                fatal=True,
            )
        return commit

    def pcm(self) -> bytes:
        """Return the validated PCM stream as one byte string."""
        return b"".join(self._parts)


def _validate_pcm(pcm: bytes) -> None:
    if not pcm or len(pcm) % 2 != 0:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            "PCM payload must contain complete 16-bit samples",
            stage="input",
            fatal=True,
        )


def _validate_numeric_range(value: int, maximum: int, name: str) -> None:
    if not isinstance(value, int) or not 0 <= value <= maximum:
        raise ProtocolError(
            ErrorCode.INVALID_AUDIO_FRAME,
            f"{name} is outside its unsigned integer range",
            stage="input",
            fatal=True,
        )
