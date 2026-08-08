"""Application settings for the MindSurf Voice AI backend."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
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
    handshake_timeout_ms: int = Field(default=3_000, gt=0)
    heartbeat_interval_ms: int = Field(default=15_000, gt=0)
    heartbeat_timeout_ms: int = Field(default=10_000, gt=0)
    max_recording_ms: int = Field(default=60_000, gt=0)
    max_json_bytes: int = Field(default=65_536, ge=1024, le=65_536)
    max_binary_bytes: int = Field(default=65_536, ge=1024, le=65_536)

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
