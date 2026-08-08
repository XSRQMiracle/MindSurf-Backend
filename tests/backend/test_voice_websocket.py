"""WebSocket handshake tests for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from mindsurf_backend.app import create_app
from mindsurf_backend.config import AppSettings
from mindsurf_backend.voice.audio import AudioFrame, AudioKind, encode_audio_frame
from mindsurf_backend.voice.protocol import VOICE_SUBPROTOCOL


def test_websocket_negotiates_subprotocol_and_server_hello() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        websocket.send_json(_envelope("client.hello", None, _client_hello_payload()))
        hello = websocket.receive_json()

        assert websocket.accepted_subprotocol == VOICE_SUBPROTOCOL
        assert hello["type"] == "server.hello"
        assert hello["payload"]["protocol_version"] == 1
        assert hello["payload"]["pipeline"] == "cascade"
        assert hello["payload"]["features"] == {
            "streaming_asr": False,
            "streaming_text": False,
            "streaming_audio": False,
            "cancellation": True,
        }


def test_websocket_requires_client_hello_first() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        websocket.send_json(_envelope("request.start", uuid.uuid4(), {}))
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["payload"]["code"] == "handshake_required"
        assert error["payload"]["fatal"] is True
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
        assert closed.value.code == 1002


def test_websocket_allows_handshake_retry_after_invalid_json() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        websocket.send_text("not json")
        error = websocket.receive_json()
        websocket.send_json(_envelope("client.hello", None, _client_hello_payload()))
        hello = websocket.receive_json()

        assert error["payload"]["code"] == "invalid_json"
        assert error["payload"]["fatal"] is False
        assert hello["type"] == "server.hello"


def test_websocket_closes_when_subprotocol_is_missing() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws") as websocket,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        websocket.receive_json()

    assert closed.value.code == 1002


def test_websocket_accepts_audio_and_commits_input() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        websocket.send_json(_envelope("client.hello", None, _client_hello_payload()))
        websocket.receive_json()
        request_id = uuid.uuid4()
        websocket.send_json(_envelope("request.start", request_id, _request_start_payload()))
        accepted = websocket.receive_json()
        websocket.send_bytes(_input_frame(request_id, 0, 0, b"\x00\x00" * 320))
        websocket.send_json(
            _envelope(
                "input.commit",
                request_id,
                {
                    "last_sequence": 0,
                    "frame_count": 1,
                    "sample_count": 320,
                    "duration_ms": 20,
                },
            )
        )
        committed = websocket.receive_json()

        assert accepted["type"] == "request.accepted"
        assert accepted["request_id"] == str(request_id)
        assert accepted["payload"]["voice"] == "default"
        assert "conversation_id" not in accepted["payload"]
        assert committed["type"] == "input.committed"
        assert committed["payload"] == {"accepted_duration_ms": 20}


def test_websocket_rejects_parallel_request_without_stopping_active_request() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        _handshake(websocket)
        first_request_id = uuid.uuid4()
        second_request_id = uuid.uuid4()
        websocket.send_json(_envelope("request.start", first_request_id, _request_start_payload()))
        websocket.receive_json()
        websocket.send_json(_envelope("request.start", second_request_id, _request_start_payload()))
        error = websocket.receive_json()
        websocket.send_json(
            _envelope(
                "request.cancel",
                first_request_id,
                {"reason": "user_interrupted"},
            )
        )
        cancelled = websocket.receive_json()

        assert error["request_id"] == str(second_request_id)
        assert error["payload"]["code"] == "request_already_active"
        assert error["payload"]["fatal"] is True
        assert cancelled["type"] == "request.cancelled"
        assert cancelled["request_id"] == str(first_request_id)


def test_websocket_cancel_is_idempotent_and_allows_next_request() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        _handshake(websocket)
        first_request_id = uuid.uuid4()
        websocket.send_json(_envelope("request.start", first_request_id, _request_start_payload()))
        websocket.receive_json()
        cancel = _envelope(
            "request.cancel",
            first_request_id,
            {"reason": "user_cancelled"},
        )
        websocket.send_json(cancel)
        first_cancelled = websocket.receive_json()
        websocket.send_json(
            _envelope(
                "request.cancel",
                first_request_id,
                {"reason": "user_interrupted"},
            )
        )
        repeated_cancelled = websocket.receive_json()
        websocket.send_bytes(_input_frame(first_request_id, 0, 0, b"\x00\x00"))
        second_request_id = uuid.uuid4()
        websocket.send_json(_envelope("request.start", second_request_id, _request_start_payload()))
        accepted = websocket.receive_json()

        assert first_cancelled["payload"] == {"reason": "user_cancelled"}
        assert repeated_cancelled["payload"] == {"reason": "user_cancelled"}
        assert accepted["type"] == "request.accepted"
        assert accepted["request_id"] == str(second_request_id)


def test_websocket_fatal_audio_error_terminates_request() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        _handshake(websocket)
        request_id = uuid.uuid4()
        websocket.send_json(_envelope("request.start", request_id, _request_start_payload()))
        websocket.receive_json()
        websocket.send_bytes(_input_frame(request_id, 1, 0, b"\x00\x00"))
        error = websocket.receive_json()
        next_request_id = uuid.uuid4()
        websocket.send_json(_envelope("request.start", next_request_id, _request_start_payload()))
        accepted = websocket.receive_json()

        assert error["payload"]["code"] == "audio_sequence_error"
        assert error["payload"]["fatal"] is True
        assert accepted["type"] == "request.accepted"
        assert accepted["request_id"] == str(next_request_id)


def test_websocket_heartbeat_accepts_matching_pong() -> None:
    settings = AppSettings(
        heartbeat_interval_ms=10,
        heartbeat_timeout_ms=100,
        _env_file=None,
    )
    app = create_app(settings)

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        websocket.send_json(_envelope("client.hello", None, _client_hello_payload()))
        websocket.receive_json()
        ping = websocket.receive_json()
        websocket.send_json(_envelope("session.pong", None, ping["payload"]))

        assert ping["type"] == "session.ping"
        assert isinstance(ping["payload"]["nonce"], str)


def _envelope(
    message_type: str,
    request_id: uuid.UUID | None,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "v": 1,
        "type": message_type,
        "event_id": str(uuid.uuid4()),
        "request_id": str(request_id) if request_id is not None else None,
        "sent_at_ms": 1,
        "payload": payload,
    }


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


def _request_start_payload() -> dict[str, object]:
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
        "response": {"text": True, "audio": True, "voice": "requested-voice"},
    }


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


def _handshake(websocket: WebSocketTestSession) -> None:
    websocket.send_json(_envelope("client.hello", None, _client_hello_payload()))
    websocket.receive_json()
