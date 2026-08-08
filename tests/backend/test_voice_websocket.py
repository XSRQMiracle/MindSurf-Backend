"""WebSocket handshake tests for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mindsurf_backend.app import create_app
from mindsurf_backend.config import AppSettings
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
            "cancellation": False,
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


def test_websocket_replies_to_unimplemented_request_with_stable_error() -> None:
    app = create_app(AppSettings(_env_file=None))

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice/ws", subprotocols=[VOICE_SUBPROTOCOL]) as websocket,
    ):
        websocket.send_json(_envelope("client.hello", None, _client_hello_payload()))
        websocket.receive_json()
        request_id = uuid.uuid4()
        websocket.send_json(_envelope("request.start", request_id, {}))
        error = websocket.receive_json()

        assert error["request_id"] == str(request_id)
        assert error["payload"]["code"] == "unsupported_message_type"
        assert error["payload"]["recoverable"] is True


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
