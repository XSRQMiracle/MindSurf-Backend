"""FastAPI application entry point for the MindSurf Voice AI backend."""

from __future__ import annotations

from fastapi import FastAPI

from mindsurf_backend.config import AppSettings, get_settings
from mindsurf_backend.http import register_http_api
from mindsurf_backend.omni import OmniAdapter
from mindsurf_backend.voice.router import register_voice_protocol


def create_app(
    settings: AppSettings | None = None,
    omni_adapter: OmniAdapter | None = None,
) -> FastAPI:
    """Create the standalone Voice AI backend application."""
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Backend for the MindSurf Voice AI desktop client",
        docs_url=None if resolved_settings.is_production else "/docs",
        redoc_url=None if resolved_settings.is_production else "/redoc",
    )
    app.state.settings = resolved_settings
    app.state.omni_adapter = omni_adapter or OmniAdapter.from_environment()
    register_http_api(app, resolved_settings, app.state.omni_adapter)
    register_voice_protocol(app, resolved_settings, app.state.omni_adapter)
    return app


app = create_app()
