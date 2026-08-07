"""Application-boundary tests for the rewritten backend."""

from __future__ import annotations

from fastapi import FastAPI

from mindsurf_backend.app import create_app
from mindsurf_backend.config import AppSettings
from mindsurf_backend.omni import OmniAdapter
from python_starter.api.main import create_app as create_legacy_app
from python_starter.infrastructure.config import Settings as LegacySettings


def test_new_backend_starts_without_loading_a_model() -> None:
    settings = AppSettings(
        app_name="MindSurf Test Backend",
        app_version="9.9.9",
        _env_file=None,
    )

    app = create_app(settings)

    assert app.title == "MindSurf Test Backend"
    assert app.version == "9.9.9"
    assert app.state.settings is settings
    assert isinstance(app.state.omni_adapter, OmniAdapter)
    assert app.state.omni_adapter.available is False


def test_legacy_application_remains_available() -> None:
    legacy_app = create_legacy_app(LegacySettings(ENV="test", _env_file=None))

    assert isinstance(legacy_app, FastAPI)
