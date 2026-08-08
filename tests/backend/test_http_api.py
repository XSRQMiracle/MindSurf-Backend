"""HTTP integration tests for health probes and direct Omni diagnostics."""

from __future__ import annotations

import struct

import pytest
from fastapi.testclient import TestClient

from mindsurf_backend.app import create_app
from mindsurf_backend.config import AppSettings
from mindsurf_backend.omni import OmniAdapter
from tests.backend.fakes import FakeSpeechEngine


def test_liveness_does_not_depend_on_model_configuration() -> None:
    settings = AppSettings(app_name="MindSurf Demo", app_version="1.2.3", _env_file=None)
    client = TestClient(create_app(settings, OmniAdapter(configuration_error="weights missing")))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "live",
        "name": "MindSurf Demo",
        "version": "1.2.3",
    }


def test_readiness_explains_why_an_engine_is_unavailable() -> None:
    client = _client(OmniAdapter(configuration_error="checkpoint is missing"))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "engine_available": False,
        "stages": {
            "transcriber": False,
            "generator": False,
            "synthesiser": False,
        },
        "reason": "checkpoint is missing",
    }


def test_readiness_requires_the_complete_cascade() -> None:
    engine = FakeSpeechEngine(unwired=("synthesiser",))
    client = _client(OmniAdapter(engine))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["stages"] == {
        "transcriber": True,
        "generator": True,
        "synthesiser": False,
    }
    assert "synthesiser" in response.json()["reason"]


def test_readiness_accepts_a_complete_fake_engine() -> None:
    response = _client(OmniAdapter(FakeSpeechEngine())).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "engine_available": True,
        "stages": {
            "transcriber": True,
            "generator": True,
            "synthesiser": True,
        },
    }


def test_chat_completion_and_streaming_sse_use_the_generator() -> None:
    client = _client(OmniAdapter(FakeSpeechEngine(deltas=("你好", "，世界。"))))
    request = {"messages": [{"role": "user", "content": "你好"}]}

    completion = client.post("/v1/chat/completions", json=request)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={**request, "stream": True},
    ) as streaming:
        lines = [line for line in streaming.iter_lines() if line.startswith("data: ")]

    assert completion.status_code == 200
    assert completion.json()["choices"][0]["message"]["content"] == "你好，世界。"
    assert streaming.status_code == 200
    assert streaming.headers["content-type"].startswith("text/event-stream")
    assert '"content": "你好"' in lines[0]
    assert lines[-1] == "data: [DONE]"


def test_raw_pcm_transcription_reports_language_and_duration() -> None:
    client = _client(OmniAdapter(FakeSpeechEngine(transcript="测试音频", language="zh-CN")))

    response = client.post("/v1/audio/transcriptions", content=b"\x00\x00" * 16_000)

    assert response.status_code == 200
    assert response.json()["text"] == "测试音频"
    assert response.json()["language"] == "zh-CN"
    assert response.json()["duration_seconds"] == pytest.approx(1.0)


def test_empty_transcription_input_is_rejected() -> None:
    response = _client(OmniAdapter(FakeSpeechEngine())).post(
        "/v1/audio/transcriptions", content=b""
    )

    assert response.status_code == 400


def test_speech_supports_raw_pcm_and_wav() -> None:
    engine = FakeSpeechEngine(speech_pcm=b"\x01\x00" * 24)
    client = _client(OmniAdapter(engine))

    pcm = client.post(
        "/v1/audio/speech",
        json={"input": "你好", "response_format": "pcm"},
    )
    wav = client.post("/v1/audio/speech", json={"input": "你好"})

    assert pcm.status_code == 200
    assert pcm.content == engine.speech_pcm
    assert pcm.headers["x-sample-rate"] == "24000"
    assert pcm.headers["x-encoding"] == "pcm_s16le"
    assert wav.status_code == 200
    assert wav.headers["content-type"].startswith("audio/wav")
    assert wav.content[:4] == b"RIFF"
    assert struct.unpack("<I", wav.content[24:28])[0] == 24_000
    assert wav.content[44:] == engine.speech_pcm


def test_each_debug_endpoint_only_requires_its_own_stage() -> None:
    engine = FakeSpeechEngine(deltas=("文字可用",), unwired=("transcriber", "synthesiser"))
    client = _client(OmniAdapter(engine))

    chat = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    transcription = client.post("/v1/audio/transcriptions", content=b"\x00\x00")
    speech = client.post("/v1/audio/speech", json={"input": "你好"})

    assert chat.status_code == 200
    assert transcription.status_code == 503
    assert "transcriber" in transcription.json()["detail"]
    assert speech.status_code == 503
    assert "synthesiser" in speech.json()["detail"]


def _client(adapter: OmniAdapter) -> TestClient:
    return TestClient(create_app(AppSettings(_env_file=None), adapter))
