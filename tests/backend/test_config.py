"""Configuration tests for the rewritten backend."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mindsurf_backend.config import AppSettings


def test_backend_environment_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINDSURF_BACKEND_PORT", "9100")
    monkeypatch.setenv("MINDSURF_BACKEND_VOICE_WS_PATH", "/voice")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9100
    assert settings.voice_ws_path == "/voice"


def test_voice_websocket_path_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="voice_ws_path"):
        AppSettings(voice_ws_path="v1/voice/ws", _env_file=None)
