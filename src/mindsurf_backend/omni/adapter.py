"""Application-owned access point for the Omni speech engine."""

from __future__ import annotations

from collections.abc import Mapping

from mindsurf_omni.service.config import ConfigurationError, Settings
from mindsurf_omni.service.engine import SpeechEngine
from mindsurf_omni.service.factory import build


class OmniNotConfiguredError(RuntimeError):
    """Raised when application code requests an Omni engine before startup wiring."""


class OmniAdapter:
    """Hold the configured Omni engine without owning model implementation details."""

    def __init__(
        self,
        engine: SpeechEngine | None = None,
        *,
        configuration_error: str | None = None,
    ) -> None:
        self._engine: SpeechEngine | None = None
        self._configuration_error = configuration_error
        if engine is not None:
            self.attach(engine)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> OmniAdapter:
        """Build the configured cascade engine without making startup fail closed."""
        source = dict(environment) if environment is not None else None
        try:
            engine = build(Settings.from_environment(source))
        except (ConfigurationError, ValueError) as error:
            return cls(configuration_error=str(error))
        return cls(engine)

    @property
    def available(self) -> bool:
        """Return whether an Omni engine is attached."""
        return self._engine is not None

    @property
    def streaming_text_available(self) -> bool:
        """Return whether voice input can reach the text generator."""
        return self._supports_stages("transcriber", "generator")

    @property
    def streaming_audio_available(self) -> bool:
        """Return whether voice input can reach text and speech output."""
        return self._supports_stages("transcriber", "generator", "synthesiser")

    @property
    def unavailable_reason(self) -> str:
        """Return an actionable reason when no compatible engine is attached."""
        return self._configuration_error or (
            "no Omni speech engine is configured; set MINDSURF_ENGINE=cascade and "
            "configure its checkpoint, ASR, and TTS components"
        )

    @property
    def engine(self) -> SpeechEngine:
        """Return the attached engine or a precise configuration error."""
        if self._engine is None:
            raise OmniNotConfiguredError(self.unavailable_reason)
        return self._engine

    def require(self, *, assistant: bool, audio: bool) -> SpeechEngine:
        """Return an engine whose stages cover the requested response mode."""
        engine = self.engine
        required = ["transcriber"]
        if assistant:
            required.append("generator")
        if audio:
            required.append("synthesiser")
        missing = [stage for stage in required if stage in self._unwired_stages()]
        if missing:
            raise OmniNotConfiguredError(
                "the configured Omni cascade cannot serve this request because these stages "
                f"are not wired: {', '.join(missing)}"
            )
        return engine

    def attach(self, engine: SpeechEngine) -> None:
        """Attach the engine built during application startup."""
        description = engine.describe()
        if description.path != "cascade":
            self._engine = None
            self._configuration_error = (
                "MindSurf Voice Protocol version 1 requires the cascade Omni path; "
                f"configured path is {description.path!r}"
            )
            return
        self._engine = engine
        self._configuration_error = None

    def _supports_stages(self, *stages: str) -> bool:
        return self._engine is not None and not set(stages).intersection(self._unwired_stages())

    def _unwired_stages(self) -> set[str]:
        return set(getattr(self._engine, "unwired", ()))
