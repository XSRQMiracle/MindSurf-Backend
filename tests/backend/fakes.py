"""Lightweight Omni engines used by backend protocol tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from mindsurf_omni.contract import TokenSpec
from mindsurf_omni.service.config import token_spec
from mindsurf_omni.service.engine import (
    EngineDescription,
    GenerationSettings,
    SpeechChunk,
    SpeechEngine,
)


class FakeSpeechEngine(SpeechEngine):
    """Deterministic cascade engine with no model or package dependencies."""

    def __init__(
        self,
        *,
        transcript: str = "测试输入",
        language: str | None = "zh-CN",
        deltas: tuple[str, ...] = ("你好。", "世界"),
        speech_pcm: bytes = b"\x01\x00" * 240,
        path: str = "cascade",
    ) -> None:
        self.transcript = transcript
        self.language = language
        self.deltas = deltas
        self.speech_pcm = speech_pcm
        self.path = path
        self.spoken_texts: list[str] = []

    def describe(self) -> EngineDescription:
        return EngineDescription(path=self.path)  # type: ignore[arg-type]

    def token_spec(self) -> TokenSpec:
        return token_spec()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return self.transcript, self.language

    def complete(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
    ) -> AsyncIterator[str]:
        async def generate() -> AsyncIterator[str]:
            for delta in self.deltas:
                yield delta

        return generate()

    async def speak(
        self,
        text: str,
        settings: GenerationSettings,
    ) -> AsyncIterator[SpeechChunk]:
        self.spoken_texts.append(text)
        yield SpeechChunk(pcm=self.speech_pcm, text=text, is_final=True)

    async def respond(
        self,
        pcm: bytes,
        sample_rate: int,
        settings: GenerationSettings,
    ) -> AsyncIterator[SpeechChunk]:
        async for chunk in self.speak("".join(self.deltas), settings):
            yield chunk
