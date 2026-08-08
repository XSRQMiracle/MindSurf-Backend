"""WebSocket session and request handling for MindSurf Voice Protocol version 1."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from mindsurf_backend.config import AppSettings
from mindsurf_backend.omni import OmniAdapter, OmniNotConfiguredError
from mindsurf_backend.voice.audio import decode_audio_frame
from mindsurf_backend.voice.inference import OmniRequestRunner
from mindsurf_backend.voice.lifecycle import (
    ActiveVoiceRequest,
    CancelReason,
    RequestState,
    validate_request_cancel,
    validate_request_start,
)
from mindsurf_backend.voice.protocol import (
    CloseCode,
    ControlEnvelope,
    ErrorCode,
    ProtocolError,
    create_envelope,
    create_error_envelope,
    create_server_hello_payload,
    envelope_json,
    parse_control_message,
    validate_client_hello,
)
from mindsurf_omni.service.engine import SpeechEngine


@dataclass(frozen=True, slots=True)
class _TerminalRequest:
    state: RequestState
    cancel_reason: CancelReason | None = None


class VoiceProtocolSession:
    """Own one negotiated WebSocket connection and its heartbeat state."""

    def __init__(
        self,
        websocket: WebSocket,
        settings: AppSettings,
        omni_adapter: OmniAdapter,
    ) -> None:
        self._websocket = websocket
        self._settings = settings
        self._omni_adapter = omni_adapter
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._event_ids: set[uuid.UUID] = set()
        self._last_activity = time.monotonic()
        self._pending_pong: str | None = None
        self._pong_received = asyncio.Event()
        self._active_request: ActiveVoiceRequest | None = None
        self._active_engine: SpeechEngine | None = None
        self._terminal_requests: dict[uuid.UUID, _TerminalRequest] = {}

    async def run(self) -> None:
        """Perform the handshake and serve session-level protocol messages."""
        if not await self._handshake():
            return
        heartbeat = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._receive_loop()
        finally:
            await self._release_active_request()
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _handshake(self) -> bool:
        deadline = time.monotonic() + self._settings.handshake_timeout_ms / 1000
        while not self._closed:
            try:
                message = await asyncio.wait_for(
                    self._websocket.receive(),
                    timeout=max(0, deadline - time.monotonic()),
                )
            except TimeoutError:
                violation = ProtocolError(
                    ErrorCode.HANDSHAKE_TIMEOUT,
                    "client.hello was not received before the handshake timeout",
                    stage="session",
                    fatal=True,
                    close_code=CloseCode.HANDSHAKE_TIMEOUT,
                )
                await self._fail(violation)
                return False
            if message["type"] == "websocket.disconnect":
                self._closed = True
                return False
            text = message.get("text")
            if text is None:
                violation = ProtocolError(
                    ErrorCode.HANDSHAKE_REQUIRED,
                    "client.hello must be the first message",
                    stage="session",
                    fatal=True,
                    close_code=CloseCode.PROTOCOL_ERROR,
                )
                await self._fail(violation)
                return False
            try:
                envelope = parse_control_message(text, max_bytes=self._settings.max_json_bytes)
                validate_client_hello(envelope)
            except ProtocolError as violation:
                await self._fail(violation)
                if violation.fatal or violation.close_code is not None:
                    return False
                continue
            self._remember_event(envelope)
            self._last_activity = time.monotonic()
            payload = create_server_hello_payload(
                heartbeat_interval_ms=self._settings.heartbeat_interval_ms,
                heartbeat_timeout_ms=self._settings.heartbeat_timeout_ms,
                max_recording_ms=self._settings.max_recording_ms,
                max_json_bytes=self._settings.max_json_bytes,
                max_binary_bytes=self._settings.max_binary_bytes,
                streaming_text=self._omni_adapter.streaming_text_available,
                streaming_audio=self._omni_adapter.streaming_audio_available,
            )
            await self._send(create_envelope("server.hello", None, payload))
            return True
        return False

    async def _receive_loop(self) -> None:
        while not self._closed:
            try:
                message = await self._websocket.receive()
            except WebSocketDisconnect:
                self._closed = True
                return
            if message["type"] == "websocket.disconnect":
                self._closed = True
                return
            text = message.get("text")
            if text is not None:
                await self._handle_control(text)
                continue
            binary = message.get("bytes")
            if binary is not None:
                await self._handle_binary(binary)

    async def _handle_control(self, text: str) -> None:
        try:
            envelope = parse_control_message(text, max_bytes=self._settings.max_json_bytes)
        except ProtocolError as violation:
            await self._fail(violation)
            return
        self._last_activity = time.monotonic()
        if envelope.event_id in self._event_ids:
            return
        self._remember_event(envelope)
        try:
            if envelope.type == "session.pong":
                await self._handle_pong(envelope)
                return
            if envelope.type == "request.start":
                await self._handle_request_start(envelope)
                return
            if envelope.type == "input.commit":
                await self._handle_input_commit(envelope)
                return
            if envelope.type == "request.cancel":
                await self._handle_request_cancel(envelope)
                return
            if envelope.type == "error":
                return
            raise ProtocolError(
                ErrorCode.UNSUPPORTED_MESSAGE_TYPE,
                f"message type {envelope.type!r} is not implemented",
                stage="protocol",
            )
        except ProtocolError as violation:
            await self._handle_request_violation(violation, envelope.request_id)

    async def _handle_binary(self, data: bytes) -> None:
        self._last_activity = time.monotonic()
        try:
            frame = decode_audio_frame(data, max_bytes=self._settings.max_binary_bytes)
        except ProtocolError as violation:
            request_id = (
                self._active_request.request_id if self._active_request is not None else None
            )
            await self._handle_request_violation(violation, request_id)
            return
        if frame.request_id in self._terminal_requests:
            return
        request = self._active_request
        if request is None or frame.request_id != request.request_id:
            not_found = ProtocolError(
                ErrorCode.REQUEST_NOT_FOUND,
                "audio does not belong to the active request",
                stage="input",
            )
            await self._send(create_error_envelope(not_found, frame.request_id))
            return
        try:
            request.append_audio(data)
        except ProtocolError as violation:
            await self._handle_request_violation(violation, request.request_id)

    async def _handle_request_start(self, envelope: ControlEnvelope) -> None:
        request_id = self._require_request_id(envelope)
        if request_id in self._terminal_requests:
            raise ProtocolError(
                ErrorCode.REQUEST_STATE_ERROR,
                "request ID has already reached a terminal state and cannot be reused",
                stage="input",
                fatal=True,
            )
        if self._active_request is not None:
            raise ProtocolError(
                ErrorCode.REQUEST_ALREADY_ACTIVE,
                "the connection already has an active request",
                stage="input",
                fatal=True,
            )
        payload = validate_request_start(envelope.payload)
        try:
            engine = self._omni_adapter.require(
                assistant=payload.mode == "assistant",
                audio=payload.response.audio,
            )
        except OmniNotConfiguredError as error:
            raise ProtocolError(
                ErrorCode.MODEL_UNAVAILABLE,
                str(error),
                stage="protocol",
                fatal=True,
            ) from error
        request = ActiveVoiceRequest(
            request_id,
            payload,
            max_recording_ms=self._settings.max_recording_ms,
        )
        self._active_request = request
        self._active_engine = engine
        await self._send(
            create_envelope("request.accepted", request_id, request.accepted_payload())
        )
        request.begin_recording()

    async def _handle_input_commit(self, envelope: ControlEnvelope) -> None:
        request_id = self._require_request_id(envelope)
        if request_id in self._terminal_requests:
            return
        request = self._require_active_request(request_id)
        request.commit_input(envelope.payload)
        await self._send(
            create_envelope(
                "input.committed",
                request_id,
                {"accepted_duration_ms": request.input_audio.duration_ms},
            )
        )
        engine = self._active_engine
        if engine is None:
            raise ProtocolError(
                ErrorCode.MODEL_UNAVAILABLE,
                self._omni_adapter.unavailable_reason,
                stage="protocol",
                fatal=True,
            )
        task = asyncio.create_task(
            self._run_inference(request, engine),
            name=f"voice-inference-{request_id}",
        )
        request.attach_generation_task(task)

    async def _handle_request_cancel(self, envelope: ControlEnvelope) -> None:
        request_id = self._require_request_id(envelope)
        payload = validate_request_cancel(envelope.payload)
        terminal = self._terminal_requests.get(request_id)
        if terminal is not None:
            if terminal.state is RequestState.CANCELLED and terminal.cancel_reason is not None:
                await self._send(
                    create_envelope(
                        "request.cancelled",
                        request_id,
                        {"reason": terminal.cancel_reason},
                    )
                )
            return
        request = self._require_active_request(request_id)
        if request.is_terminal:
            return
        await request.cancel()
        self._remember_terminal(
            request_id,
            _TerminalRequest(RequestState.CANCELLED, payload.reason),
        )
        self._active_request = None
        self._active_engine = None
        await self._send(
            create_envelope("request.cancelled", request_id, {"reason": payload.reason})
        )

    async def _handle_pong(self, envelope: ControlEnvelope) -> None:
        nonce = envelope.payload.get("nonce")
        if envelope.request_id is not None or not isinstance(nonce, str):
            violation = ProtocolError(
                ErrorCode.INVALID_MESSAGE,
                "session.pong must carry a session-level nonce",
                stage="session",
            )
            await self._send(create_error_envelope(violation))
            return
        if self._pending_pong is None or nonce != self._pending_pong:
            violation = ProtocolError(
                ErrorCode.INVALID_MESSAGE,
                "session.pong nonce does not match the active heartbeat",
                stage="session",
            )
            await self._send(create_error_envelope(violation))
            return
        self._pending_pong = None
        self._pong_received.set()

    async def _heartbeat_loop(self) -> None:
        interval = self._settings.heartbeat_interval_ms / 1000
        timeout = self._settings.heartbeat_timeout_ms / 1000
        while not self._closed:
            await asyncio.sleep(interval)
            if self._closed or time.monotonic() - self._last_activity < interval:
                continue
            nonce = str(uuid.uuid4())
            self._pending_pong = nonce
            self._pong_received.clear()
            await self._send(create_envelope("session.ping", None, {"nonce": nonce}))
            try:
                await asyncio.wait_for(self._pong_received.wait(), timeout=timeout)
            except TimeoutError:
                await self._close(CloseCode.GOING_AWAY, "heartbeat timeout")
                return

    async def _fail(self, violation: ProtocolError) -> None:
        await self._send(create_error_envelope(violation))
        if violation.close_code is not None:
            await self._close(violation.close_code, violation.code.value)

    async def _handle_request_violation(
        self,
        violation: ProtocolError,
        request_id: uuid.UUID | None,
    ) -> None:
        await self._send(create_error_envelope(violation, request_id))
        if violation.fatal and request_id is not None:
            request = self._active_request
            if request is not None and request.request_id == request_id:
                await request.fail()
                self._active_request = None
                self._active_engine = None
                self._remember_terminal(request_id, _TerminalRequest(RequestState.FAILED))
            elif request_id not in self._terminal_requests:
                self._remember_terminal(request_id, _TerminalRequest(RequestState.FAILED))
        if violation.close_code is not None:
            await self._close(violation.close_code, violation.code.value)

    async def _send(self, envelope: ControlEnvelope) -> None:
        if self._closed:
            return
        async with self._send_lock:
            await self._websocket.send_json(envelope_json(envelope))

    async def _send_control(
        self,
        message_type: str,
        request_id: uuid.UUID,
        payload: dict[str, Any],
    ) -> None:
        await self._send(create_envelope(message_type, request_id, payload))

    async def _send_binary(self, data: bytes) -> None:
        if self._closed:
            return
        async with self._send_lock:
            await self._websocket.send_bytes(data)

    async def _send_inference_error(
        self,
        violation: ProtocolError,
        request_id: uuid.UUID,
    ) -> None:
        await self._send(create_error_envelope(violation, request_id))

    async def _run_inference(
        self,
        request: ActiveVoiceRequest,
        engine: SpeechEngine,
    ) -> None:
        runner = OmniRequestRunner(
            engine,
            request,
            send_control=self._send_control,
            send_binary=self._send_binary,
            send_error=self._send_inference_error,
            max_binary_bytes=self._settings.max_binary_bytes,
        )
        try:
            done_payload = await runner.run()
        except asyncio.CancelledError:
            raise
        except ProtocolError as violation:
            await self._handle_request_violation(violation, request.request_id)
            return
        except Exception:
            internal_violation = ProtocolError(
                ErrorCode.INTERNAL_ERROR,
                "Omni inference failed unexpectedly",
                stage="protocol",
                fatal=True,
            )
            await self._handle_request_violation(internal_violation, request.request_id)
            return

        if self._active_request is not request or request.state is not RequestState.GENERATING:
            return
        request.mark_done()
        self._active_request = None
        self._active_engine = None
        self._remember_terminal(request.request_id, _TerminalRequest(RequestState.DONE))
        await request.close_generation()
        await self._send_control("request.done", request.request_id, done_payload)

    async def _close(self, code: CloseCode, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(RuntimeError):
            await self._websocket.close(code=int(code), reason=reason[:123])

    async def _release_active_request(self) -> None:
        request = self._active_request
        self._active_request = None
        self._active_engine = None
        if request is not None:
            await request.cancel()

    def _require_request_id(self, envelope: ControlEnvelope) -> uuid.UUID:
        if envelope.request_id is None:
            raise ProtocolError(
                ErrorCode.INVALID_MESSAGE,
                f"{envelope.type} must carry a request ID",
                stage="input",
            )
        return envelope.request_id

    def _require_active_request(self, request_id: uuid.UUID) -> ActiveVoiceRequest:
        request = self._active_request
        if request is None or request.request_id != request_id:
            raise ProtocolError(
                ErrorCode.REQUEST_NOT_FOUND,
                "request ID does not match the active request",
                stage="input",
            )
        return request

    def _remember_terminal(
        self,
        request_id: uuid.UUID,
        terminal: _TerminalRequest,
    ) -> None:
        self._terminal_requests[request_id] = terminal
        if len(self._terminal_requests) > 1024:
            oldest_request_id = next(iter(self._terminal_requests))
            del self._terminal_requests[oldest_request_id]

    def _remember_event(self, envelope: ControlEnvelope) -> None:
        self._event_ids.add(envelope.event_id)
        if len(self._event_ids) > 1024:
            self._event_ids = {envelope.event_id}
