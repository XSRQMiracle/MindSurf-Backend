"""FastAPI route registration for MindSurf Voice Protocol version 1."""

from __future__ import annotations

from fastapi import FastAPI, WebSocket

from mindsurf_backend.config import AppSettings
from mindsurf_backend.voice.protocol import VOICE_SUBPROTOCOL, CloseCode
from mindsurf_backend.voice.session import VoiceProtocolSession


def register_voice_protocol(app: FastAPI, settings: AppSettings) -> None:
    """Register the configured Voice WebSocket endpoint on an application."""

    async def voice_websocket(websocket: WebSocket) -> None:
        offered = websocket.scope.get("subprotocols", [])
        if VOICE_SUBPROTOCOL not in offered:
            await websocket.accept()
            await websocket.close(code=int(CloseCode.PROTOCOL_ERROR), reason="subprotocol required")
            return
        await websocket.accept(subprotocol=VOICE_SUBPROTOCOL)
        await VoiceProtocolSession(websocket, settings).run()

    app.add_api_websocket_route(
        settings.voice_ws_path,
        voice_websocket,
        name="voice-protocol-v1",
    )
