"""MindSurf Voice Protocol application layer."""

from __future__ import annotations

from mindsurf_backend.voice.audio import (
    AUDIO_HEADER_LENGTH,
    AudioFrame,
    AudioKind,
    InputAudioStream,
    decode_audio_frame,
    encode_audio_frame,
)
from mindsurf_backend.voice.protocol import (
    PROTOCOL_VERSION,
    VOICE_SUBPROTOCOL,
    ControlEnvelope,
    ProtocolError,
    create_envelope,
    parse_control_message,
)

__all__ = [
    "AUDIO_HEADER_LENGTH",
    "PROTOCOL_VERSION",
    "VOICE_SUBPROTOCOL",
    "AudioFrame",
    "AudioKind",
    "ControlEnvelope",
    "InputAudioStream",
    "ProtocolError",
    "create_envelope",
    "decode_audio_frame",
    "encode_audio_frame",
    "parse_control_message",
]
