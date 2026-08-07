"""Application-owned access point for the Omni speech engine."""

from __future__ import annotations

from mindsurf_omni.service.engine import SpeechEngine


class OmniNotConfiguredError(RuntimeError):
    """Raised when application code requests an Omni engine before startup wiring."""


class OmniAdapter:
    """Hold the configured Omni engine without owning model implementation details."""

    def __init__(self, engine: SpeechEngine | None = None) -> None:
        self._engine = engine

    @property
    def available(self) -> bool:
        """Return whether an Omni engine is attached."""
        return self._engine is not None

    @property
    def engine(self) -> SpeechEngine:
        """Return the attached engine or a precise configuration error."""
        if self._engine is None:
            raise OmniNotConfiguredError("Omni engine has not been configured")
        return self._engine

    def attach(self, engine: SpeechEngine) -> None:
        """Attach the engine built during application startup."""
        self._engine = engine
