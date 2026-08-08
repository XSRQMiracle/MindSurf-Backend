"""Control-message tests for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import uuid

import pytest

from mindsurf_backend.voice.protocol import (
    ErrorCode,
    ProtocolError,
    create_envelope,
    create_server_hello_payload,
    parse_control_message,
    validate_client_hello,
)


def test_control_envelope_round_trip() -> None:
    request_id = uuid.uuid4()
    envelope = create_envelope("request.start", request_id, {"mode": "assistant"})

    parsed = parse_control_message(envelope.model_dump_json())

    assert parsed.type == "request.start"
    assert parsed.request_id == request_id
    assert parsed.payload == {"mode": "assistant"}


def test_control_message_rejects_oversized_json() -> None:
    with pytest.raises(ProtocolError) as error:
        parse_control_message(b"{}" * 40_000)

    assert error.value.code is ErrorCode.INVALID_MESSAGE
    assert error.value.close_code == 1009


def test_control_message_rejects_noncanonical_uuid() -> None:
    raw = (
        '{"v":1,"type":"session.pong","event_id":"'
        + str(uuid.uuid4()).upper()
        + '","request_id":null,"sent_at_ms":1,"payload":{}}'
    )

    with pytest.raises(ProtocolError) as error:
        parse_control_message(raw)

    assert error.value.code is ErrorCode.INVALID_MESSAGE


def test_client_hello_negotiates_protocol_and_audio() -> None:
    hello = create_envelope("client.hello", None, _client_hello_payload())

    parsed = validate_client_hello(hello)

    assert parsed.client.name == "mindsurf-voice-ai"
    assert parsed.protocol_versions == [1]


def test_server_hello_defaults_exist_in_candidate_lists() -> None:
    payload = create_server_hello_payload(
        heartbeat_interval_ms=15_000,
        heartbeat_timeout_ms=10_000,
    )
    options = payload["inference_options"]
    defaults = options["defaults"]

    for name in ("asr", "llm", "tts", "output_audio"):
        assert defaults[name] in {item["id"] for item in options[name]}


def _client_hello_payload() -> dict[str, object]:
    return {
        "client": {
            "name": "mindsurf-voice-ai",
            "version": "0.1.0",
            "platform": "macos",
            "arch": "aarch64",
        },
        "protocol_versions": [1],
        "pipelines": ["cascade"],
        "input_audio": [{"encoding": "pcm_s16le", "sample_rate": 16_000, "channels": 1}],
        "output_audio": [
            {
                "encoding": "pcm_s16le",
                "sample_rates": [16_000, 24_000],
                "channels": 1,
            }
        ],
    }
