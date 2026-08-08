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
from mindsurf_backend.voice.lifecycle import (
    ActiveVoiceRequest,
    RequestStartPayload,
    RequestState,
    validate_request_cancel,
    validate_request_start,
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
    "ActiveVoiceRequest",
    "AudioFrame",
    "AudioKind",
    "ControlEnvelope",
    "InputAudioStream",
    "ProtocolError",
    "RequestStartPayload",
    "RequestState",
    "create_envelope",
    "decode_audio_frame",
    "encode_audio_frame",
    "parse_control_message",
    "validate_request_cancel",
    "validate_request_start",
]
