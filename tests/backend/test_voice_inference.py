"""End-to-end Omni inference mapping for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from mindsurf_backend.app import create_app
from mindsurf_backend.config import AppSettings
from mindsurf_backend.omni import OmniAdapter
from mindsurf_backend.voice.audio import (
    AudioFrame,
    AudioKind,
    decode_audio_frame,
    encode_audio_frame,
)
from mindsurf_backend.voice.protocol import VOICE_SUBPROTOCOL
from mindsurf_omni.service.engine import GenerationSettings
from tests.backend.fakes import FakeSpeechEngine


def test_dictation_emits_asr_final_and_request_done() -> None:
    engine = FakeSpeechEngine(transcript="今天开会", deltas=())

    with _connected(engine) as websocket:
        request_id = uuid.uuid4()
        _commit_request(websocket, request_id, mode="dictation", audio=False)
        committed = websocket.receive_json()
        asr_final = websocket.receive_json()
        done = websocket.receive_json()

        assert committed["type"] == "input.committed"
        assert asr_final["type"] == "asr.final"
        assert asr_final["payload"]["text"] == "今天开会"
        assert done["type"] == "request.done"


def test_text_assistant_streams_deltas_and_finishes() -> None:
    engine = FakeSpeechEngine(deltas=("第一段", "，第二段。"))

    with _connected(engine) as websocket:
        request_id = uuid.uuid4()
        _commit_request(websocket, request_id, mode="assistant", audio=False)
        message_types = [websocket.receive_json()["type"] for _ in range(6)]

        assert message_types == [
            "input.committed",
            "asr.final",
            "assistant.text.delta",
            "assistant.text.delta",
            "assistant.text.done",
            "request.done",
        ]


def test_audio_assistant_streams_text_and_pcm_frames() -> None:
    engine = FakeSpeechEngine(deltas=("你好。", "世界"))

    with _connected(engine) as websocket:
        request_id = uuid.uuid4()
        _commit_request(websocket, request_id, mode="assistant", audio=True)
        assert websocket.receive_json()["type"] == "input.committed"
        assert websocket.receive_json()["type"] == "asr.final"
        assert websocket.receive_json()["type"] == "assistant.text.delta"
        assert websocket.receive_json()["type"] == "output.audio.start"
        first_frame = decode_audio_frame(websocket.receive_bytes())
        assert websocket.receive_json()["type"] == "assistant.text.delta"
        second_frame = decode_audio_frame(websocket.receive_bytes())
        assert websocket.receive_json()["type"] == "assistant.text.done"
        audio_done = websocket.receive_json()
        request_done = websocket.receive_json()

        assert first_frame.kind is AudioKind.OUTPUT_PCM
        assert first_frame.request_id == request_id
        assert first_frame.sequence == 0
        assert second_frame.sequence == 1
        assert audio_done["type"] == "output.audio.done"
        assert audio_done["payload"]["chunk_count"] == 2
        assert request_done["type"] == "request.done"
        assert engine.spoken_texts == ["你好。", "世界"]


def test_cancel_closes_live_omni_generation() -> None:
    engine = _BlockingSpeechEngine()

    with _connected(engine) as websocket:
        request_id = uuid.uuid4()
        _commit_request(websocket, request_id, mode="assistant", audio=False)
        assert websocket.receive_json()["type"] == "input.committed"
        assert websocket.receive_json()["type"] == "asr.final"
        assert websocket.receive_json()["type"] == "assistant.text.delta"
        websocket.send_json(
            _envelope(
                "request.cancel",
                request_id,
                {"reason": "user_interrupted"},
            )
        )
        cancelled = websocket.receive_json()

        assert cancelled["type"] == "request.cancelled"
        assert engine.generation_closed is True


class _BlockingSpeechEngine(FakeSpeechEngine):
    def __init__(self) -> None:
        super().__init__(deltas=())
        self.generation_closed = False

    def complete(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
    ) -> AsyncIterator[str]:
        async def generate() -> AsyncIterator[str]:
            try:
                yield "正在处理"
                await asyncio.Event().wait()
            finally:
                self.generation_closed = True

        return generate()


class _ConnectedVoiceSession:
    def __init__(self, engine: FakeSpeechEngine) -> None:
        app = create_app(AppSettings(_env_file=None), OmniAdapter(engine))
        self._client = TestClient(app)
        self._websocket_context: Any = None

    def __enter__(self) -> WebSocketTestSession:
        self._client.__enter__()
        self._websocket_context = self._client.websocket_connect(
            "/v1/voice/ws",
            subprotocols=[VOICE_SUBPROTOCOL],
        )
        websocket = self._websocket_context.__enter__()
        websocket.send_json(_envelope("client.hello", None, _client_hello_payload()))
        hello = websocket.receive_json()
        assert hello["payload"]["features"]["streaming_text"] is True
        assert hello["payload"]["features"]["streaming_audio"] is True
        return cast(WebSocketTestSession, websocket)

    def __exit__(self, *args: object) -> None:
        self._websocket_context.__exit__(*args)
        self._client.__exit__(*args)


def _connected(engine: FakeSpeechEngine) -> _ConnectedVoiceSession:
    return _ConnectedVoiceSession(engine)


def _commit_request(
    websocket: WebSocketTestSession,
    request_id: uuid.UUID,
    *,
    mode: str,
    audio: bool,
) -> None:
    websocket.send_json(_envelope("request.start", request_id, _request_payload(mode, audio)))
    assert websocket.receive_json()["type"] == "request.accepted"
    websocket.send_bytes(
        encode_audio_frame(
            AudioFrame(
                kind=AudioKind.INPUT_PCM,
                sequence=0,
                timestamp_us=0,
                request_id=request_id,
                pcm=b"\x00\x00" * 320,
            )
        )
    )
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


def _request_payload(mode: str, audio: bool) -> dict[str, object]:
    assistant = mode == "assistant"
    return {
        "mode": mode,
        "language": "zh-CN",
        "conversation_id": None,
        "selection": {
            "asr": "sensevoice-small",
            "llm": "mindsurf-omni" if assistant else None,
            "tts": "tts-default" if audio else None,
            "output_audio": "pcm16-24k-mono" if audio else None,
        },
        "input_audio": {
            "encoding": "pcm_s16le",
            "sample_rate": 16_000,
            "channels": 1,
            "frame_duration_ms": 20,
        },
        "response": {"text": True, "audio": audio, "voice": "default"},
    }


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
