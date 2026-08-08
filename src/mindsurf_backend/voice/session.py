"""WebSocket session handling for the MindSurf Voice Protocol handshake."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from mindsurf_backend.config import AppSettings
from mindsurf_backend.voice.audio import decode_audio_frame
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


class VoiceProtocolSession:
    """Own one negotiated WebSocket connection and its heartbeat state."""

    def __init__(self, websocket: WebSocket, settings: AppSettings) -> None:
        self._websocket = websocket
        self._settings = settings
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._event_ids: set[uuid.UUID] = set()
        self._last_activity = time.monotonic()
        self._pending_pong: str | None = None
        self._pong_received = asyncio.Event()

    async def run(self) -> None:
        """Perform the handshake and serve session-level protocol messages."""
        if not await self._handshake():
            return
        heartbeat = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._receive_loop()
        finally:
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
        if envelope.type == "session.pong":
            await self._handle_pong(envelope)
            return
        if envelope.type == "error":
            return
        unsupported = ProtocolError(
            ErrorCode.UNSUPPORTED_MESSAGE_TYPE,
            f"message type {envelope.type!r} is not implemented yet",
            stage="protocol",
        )
        await self._send(create_error_envelope(unsupported, envelope.request_id))

    async def _handle_binary(self, data: bytes) -> None:
        self._last_activity = time.monotonic()
        try:
            frame = decode_audio_frame(data, max_bytes=self._settings.max_binary_bytes)
        except ProtocolError as violation:
            await self._send(create_error_envelope(violation))
            return
        not_found = ProtocolError(
            ErrorCode.REQUEST_NOT_FOUND,
            "audio cannot be accepted before request.start",
            stage="input",
        )
        await self._send(create_error_envelope(not_found, frame.request_id))

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

    async def _send(self, envelope: ControlEnvelope) -> None:
        if self._closed:
            return
        async with self._send_lock:
            await self._websocket.send_json(envelope_json(envelope))

    async def _close(self, code: CloseCode, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(RuntimeError):
            await self._websocket.close(code=int(code), reason=reason[:123])

    def _remember_event(self, envelope: ControlEnvelope) -> None:
        self._event_ids.add(envelope.event_id)
        if len(self._event_ids) > 1024:
            self._event_ids = {envelope.event_id}
