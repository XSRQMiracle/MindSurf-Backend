"""Map one committed voice request onto the configured Omni cascade engine."""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from mindsurf_backend.voice.audio import AudioFrame, AudioKind, encode_audio_frame
from mindsurf_backend.voice.lifecycle import ActiveVoiceRequest
from mindsurf_backend.voice.protocol import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    ErrorCode,
    ErrorStage,
    ProtocolError,
)
from mindsurf_omni.service.config import ConfigurationError
from mindsurf_omni.service.engine import (
    GenerationSettings,
    SpeechChunk,
    SpeechEngine,
    split_first_utterance,
)

ControlSender = Callable[[str, uuid.UUID, dict[str, Any]], Awaitable[None]]
BinarySender = Callable[[bytes], Awaitable[None]]
ErrorSender = Callable[[ProtocolError, uuid.UUID], Awaitable[None]]

_PREFERRED_OUTPUT_PAYLOAD_BYTES = 3_840


class OmniRequestRunner:
    """Execute ASR, text generation, and optional TTS for one request."""

    def __init__(
        self,
        engine: SpeechEngine,
        request: ActiveVoiceRequest,
        *,
        send_control: ControlSender,
        send_binary: BinarySender,
        send_error: ErrorSender,
        max_binary_bytes: int,
    ) -> None:
        self._engine = engine
        self._request = request
        self._send_control = send_control
        self._send_binary = send_binary
        self._send_error = send_error
        payload_limit = max_binary_bytes - 48
        self._output_payload_bytes = min(_PREFERRED_OUTPUT_PAYLOAD_BYTES, payload_limit)
        self._output_payload_bytes -= self._output_payload_bytes % 2
        self._started = 0.0
        self._asr_final_ms = 0
        self._llm_first_token_ms = 0
        self._audio_first_chunk_ms = 0
        self._audio_sequence = 0
        self._audio_sample_count = 0
        self._audio_started = False
        self._audio_failed = False

    async def run(self) -> dict[str, Any]:
        """Run the selected cascade and return the request.done payload."""
        self._started = time.perf_counter()
        transcript, detected_language = await self._transcribe()
        self._asr_final_ms = self._elapsed_ms()
        await self._send_control(
            "asr.final",
            self._request.request_id,
            {
                "text": transcript,
                "language": self._resolved_language(detected_language),
                "confidence": None,
                "duration_ms": self._request.input_audio.duration_ms,
            },
        )

        if self._request.payload.mode == "assistant":
            await self._generate_assistant(transcript)

        return {
            "result": "success",
            "timing_ms": {
                "input_duration": self._request.input_audio.duration_ms,
                "asr_final_after_commit": self._asr_final_ms,
                "llm_first_token_after_commit": self._llm_first_token_ms,
                "audio_first_chunk_after_commit": self._audio_first_chunk_ms,
                "server_total_after_commit": self._elapsed_ms(),
            },
        }

    async def _transcribe(self) -> tuple[str, str | None]:
        try:
            text, language = await self._engine.transcribe(
                self._request.input_audio.pcm(),
                INPUT_SAMPLE_RATE,
            )
        except ConfigurationError as error:
            raise self._model_unavailable(error, "asr") from error
        except Exception as error:
            raise ProtocolError(
                ErrorCode.ASR_FAILED,
                "Omni speech recognition failed",
                stage="asr",
                fatal=True,
            ) from error
        text = text.strip()
        if not text:
            raise ProtocolError(
                ErrorCode.ASR_EMPTY,
                "Omni did not recognise any usable speech",
                stage="asr",
                fatal=True,
            )
        return text, language

    async def _generate_assistant(self, transcript: str) -> None:
        settings = GenerationSettings(voice=self._request.payload.response.voice)
        try:
            generation = self._engine.complete(
                [{"role": "user", "content": transcript}],
                settings,
            )
        except ConfigurationError as error:
            raise self._model_unavailable(error, "llm") from error
        except Exception as error:
            raise ProtocolError(
                ErrorCode.LLM_FAILED,
                "Omni text generation could not start",
                stage="llm",
                fatal=True,
            ) from error

        self._request.attach_generation(generation)
        text_parts: list[str] = []
        pending_speech = ""
        sequence = 0
        try:
            try:
                async for delta in generation:
                    if not delta:
                        continue
                    if not isinstance(delta, str):
                        raise TypeError("Omni text generation yielded a non-string delta")
                    if sequence == 0:
                        self._llm_first_token_ms = self._elapsed_ms()
                    await self._send_control(
                        "assistant.text.delta",
                        self._request.request_id,
                        {"sequence": sequence, "delta": delta},
                    )
                    sequence += 1
                    text_parts.append(delta)
                    if self._request.payload.response.audio and not self._audio_failed:
                        pending_speech += delta
                        pending_speech = await self._speak_complete_clauses(
                            pending_speech,
                            settings,
                        )
            except ConfigurationError as error:
                raise self._model_unavailable(error, "llm") from error
            except ProtocolError:
                raise
            except Exception as error:
                raise ProtocolError(
                    ErrorCode.LLM_FAILED,
                    "Omni text generation failed",
                    stage="llm",
                    fatal=True,
                ) from error
        finally:
            self._request.detach_generation(generation)
            await _close_iterator(generation)

        if not text_parts:
            raise ProtocolError(
                ErrorCode.LLM_FAILED,
                "Omni produced an empty assistant response",
                stage="llm",
                fatal=True,
            )

        if self._request.payload.response.audio and pending_speech.strip():
            await self._speak(pending_speech.strip(), settings)

        full_text = "".join(text_parts)
        await self._send_control(
            "assistant.text.done",
            self._request.request_id,
            {
                "text": full_text,
                "last_sequence": sequence - 1,
                "finish_reason": "stop",
                "usage": None,
            },
        )
        if self._request.payload.response.audio:
            await self._finish_audio()

    async def _speak_complete_clauses(
        self,
        pending: str,
        settings: GenerationSettings,
    ) -> str:
        while not self._audio_failed:
            clause = split_first_utterance(pending)
            if clause is None:
                break
            await self._speak(clause, settings)
            pending = pending[len(clause) :]
        return pending

    async def _speak(self, text: str, settings: GenerationSettings) -> None:
        if self._audio_failed or not text:
            return
        try:
            speaking = self._engine.speak(text, settings)
        except ConfigurationError as error:
            raise self._model_unavailable(error, "tts") from error
        except Exception as error:
            await self._report_tts_failure(error)
            return

        self._request.attach_generation(speaking)
        try:
            try:
                async for chunk in speaking:
                    await self._send_speech_chunk(chunk)
            except ConfigurationError as error:
                raise self._model_unavailable(error, "tts") from error
            except ProtocolError:
                raise
            except Exception as error:
                await self._report_tts_failure(error)
        finally:
            self._request.detach_generation(speaking)
            await _close_iterator(speaking)

    async def _send_speech_chunk(self, chunk: SpeechChunk) -> None:
        pcm = chunk.pcm
        if not pcm:
            return
        if len(pcm) % 2 != 0:
            await self._report_tts_failure(ValueError("TTS returned incomplete PCM16 samples"))
            return
        if not self._audio_started:
            await self._send_control(
                "output.audio.start",
                self._request.request_id,
                {
                    "encoding": "pcm_s16le",
                    "sample_rate": OUTPUT_SAMPLE_RATE,
                    "channels": 1,
                    "voice": self._request.payload.response.voice,
                },
            )
            self._audio_started = True

        for offset in range(0, len(pcm), self._output_payload_bytes):
            part = pcm[offset : offset + self._output_payload_bytes]
            timestamp_us = round(self._audio_sample_count * 1_000_000 / OUTPUT_SAMPLE_RATE)
            frame = AudioFrame(
                kind=AudioKind.OUTPUT_PCM,
                sequence=self._audio_sequence,
                timestamp_us=timestamp_us,
                request_id=self._request.request_id,
                pcm=part,
            )
            await self._send_binary(encode_audio_frame(frame))
            if self._audio_sequence == 0:
                self._audio_first_chunk_ms = self._elapsed_ms()
            self._audio_sequence += 1
            self._audio_sample_count += len(part) // 2

    async def _finish_audio(self) -> None:
        if self._audio_started:
            await self._send_control(
                "output.audio.done",
                self._request.request_id,
                {
                    "last_sequence": self._audio_sequence - 1,
                    "chunk_count": self._audio_sequence,
                    "sample_count": self._audio_sample_count,
                    "duration_ms": round(self._audio_sample_count * 1_000 / OUTPUT_SAMPLE_RATE),
                },
            )
            return
        if not self._audio_failed:
            await self._report_tts_failure(RuntimeError("TTS produced no audio"))

    async def _report_tts_failure(self, _error: Exception) -> None:
        if self._audio_failed:
            return
        self._audio_failed = True
        violation = ProtocolError(
            ErrorCode.TTS_FAILED,
            "Omni speech synthesis failed",
            stage="tts",
        )
        await self._send_error(violation, self._request.request_id)

    def _resolved_language(self, detected: str | None) -> str:
        if detected:
            return detected
        requested = self._request.payload.language
        return "und" if requested == "auto" else requested

    def _elapsed_ms(self) -> int:
        return round((time.perf_counter() - self._started) * 1_000)

    @staticmethod
    def _model_unavailable(error: Exception, stage: ErrorStage) -> ProtocolError:
        return ProtocolError(
            ErrorCode.MODEL_UNAVAILABLE,
            str(error),
            stage=stage,
            fatal=True,
        )


async def _close_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        with contextlib.suppress(RuntimeError):
            await close()
