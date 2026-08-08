"""Binary-frame tests for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import uuid

import pytest

from mindsurf_backend.voice.audio import (
    AUDIO_HEADER_LENGTH,
    AudioFrame,
    AudioKind,
    InputAudioStream,
    decode_audio_frame,
    encode_audio_frame,
)
from mindsurf_backend.voice.protocol import ErrorCode, ProtocolError


def test_audio_frame_matches_fixed_header_layout() -> None:
    request_id = uuid.uuid4()
    frame = AudioFrame(
        kind=AudioKind.INPUT_PCM,
        sequence=7,
        timestamp_us=140_000,
        request_id=request_id,
        pcm=b"\x01\x00\xff\xff",
    )

    encoded = encode_audio_frame(frame)
    decoded = decode_audio_frame(encoded)

    assert len(encoded) == AUDIO_HEADER_LENGTH + 4
    assert encoded[:6] == b"MSVA\x01\x01"
    assert int.from_bytes(encoded[8:10], "big") == AUDIO_HEADER_LENGTH
    assert int.from_bytes(encoded[12:16], "big") == 7
    assert int.from_bytes(encoded[16:24], "big") == 140_000
    assert int.from_bytes(encoded[24:28], "big") == 4
    assert encoded[32:48] == request_id.bytes
    assert decoded == frame


def test_audio_frame_rejects_nonzero_reserved_fields() -> None:
    encoded = bytearray(
        encode_audio_frame(
            AudioFrame(
                kind=AudioKind.INPUT_PCM,
                sequence=0,
                timestamp_us=0,
                request_id=uuid.uuid4(),
                pcm=b"\x00\x00",
            )
        )
    )
    encoded[6:8] = (1).to_bytes(2, "big")

    with pytest.raises(ProtocolError) as error:
        decode_audio_frame(bytes(encoded))

    assert error.value.code is ErrorCode.INVALID_AUDIO_FRAME


def test_input_stream_validates_sequence_timestamp_and_commit() -> None:
    request_id = uuid.uuid4()
    stream = InputAudioStream(request_id)
    first_pcm = b"\x00\x00" * 320
    second_pcm = b"\x01\x00" * 160

    stream.append(_input_frame(request_id, 0, 0, first_pcm))
    stream.append(_input_frame(request_id, 1, 20_000, second_pcm))
    commit = stream.validate_commit(
        {
            "last_sequence": 1,
            "frame_count": 2,
            "sample_count": 480,
            "duration_ms": 30,
        }
    )

    assert commit.sample_count == 480
    assert stream.pcm() == first_pcm + second_pcm


def test_input_stream_rejects_sequence_gap() -> None:
    request_id = uuid.uuid4()
    stream = InputAudioStream(request_id)

    with pytest.raises(ProtocolError) as error:
        stream.append(_input_frame(request_id, 1, 0, b"\x00\x00"))

    assert error.value.code is ErrorCode.AUDIO_SEQUENCE_ERROR


def test_input_stream_rejects_commit_mismatch() -> None:
    request_id = uuid.uuid4()
    stream = InputAudioStream(request_id)
    stream.append(_input_frame(request_id, 0, 0, b"\x00\x00" * 320))

    with pytest.raises(ProtocolError) as error:
        stream.validate_commit(
            {
                "last_sequence": 0,
                "frame_count": 1,
                "sample_count": 319,
                "duration_ms": 20,
            }
        )

    assert error.value.code is ErrorCode.AUDIO_COMMIT_MISMATCH


def _input_frame(
    request_id: uuid.UUID,
    sequence: int,
    timestamp_us: int,
    pcm: bytes,
) -> bytes:
    return encode_audio_frame(
        AudioFrame(
            kind=AudioKind.INPUT_PCM,
            sequence=sequence,
            timestamp_us=timestamp_us,
            request_id=request_id,
            pcm=pcm,
        )
    )
