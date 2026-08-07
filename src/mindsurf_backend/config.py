"""Application settings for the MindSurf Voice AI backend."""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Configuration owned by the backend application layer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MINDSURF_BACKEND_",
        extra="ignore",
    )

    environment: str = "development"
    debug: bool = False
    app_name: str = "MindSurf Voice AI Backend"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    voice_ws_path: str = "/v1/voice/ws"

    @field_validator("voice_ws_path")
    @classmethod
    def validate_voice_ws_path(cls, value: str) -> str:
        """Require an absolute WebSocket route path."""
        if not value.startswith("/"):
            raise ValueError("voice_ws_path must start with '/'")
        return value

    @property
    def is_production(self) -> bool:
        """Return whether production-only application behavior should apply."""
        return self.environment.casefold() == "production"


@lru_cache
def get_settings() -> AppSettings:
    """Load and cache backend application settings."""
    return AppSettings()
