# MindSurf Omni

Omni is packaged as `mindsurf_omni` alongside the existing `python_starter`
package. The existing package name and text inference paths remain unchanged.

## Runtime layout

```text
MindSurf/
├── src/mindsurf_omni/              # contract, native/cascade engines, API
├── models/mindsurf-omni/           # ignored model assets
│   ├── sft_merge_768.pth
│   └── tokenizer/
└── vendor/minimind-o/              # ignored, pinned upstream checkout
    └── model/
        ├── model_omni.py
        ├── SenseVoiceSmall/
        ├── mimi/
        └── campplus/
```

The checkpoint and upstream assets are intentionally not committed. Record the
checkpoint SHA-256 and MiniMind-O commit in the deployment manifest.

## Environment

```bash
export MINDSURF_ENGINE=cascade
export MINDSURF_DEVICE=cuda
export MINDSURF_WEIGHTS="$PWD/models/mindsurf-omni"
export MINDSURF_THINKER="$PWD/models/mindsurf-omni/sft_merge_768.pth"
export MINDSURF_TOKENIZER="$PWD/models/mindsurf-omni/tokenizer"
export MINIMIND_O_ROOT="$PWD/vendor/minimind-o"
export MINDSURF_ASR="$PWD/vendor/minimind-o/model/SenseVoiceSmall"
export MINDSURF_CODEC="$PWD/vendor/minimind-o/model/mimi"
export MINDSURF_SPEAKER="$PWD/vendor/minimind-o/model/campplus"
export MINDSURF_TTS=edge
```

Install the project in its virtual environment with the required runtime
extras. Use `omni-tts-local` instead of `omni-tts` for VoxCPM.

```bash
uv sync --extra dev --extra omni-asr --extra omni-tts
```

Start the standalone Omni API with one worker so a GPU model is loaded once:

```bash
uv run uvicorn --factory mindsurf_omni.service.app:create_app \
  --host 0.0.0.0 --port 8001 --workers 1
```

The service exposes OpenAI-compatible chat, transcription, speech, model
metadata, token metadata, and realtime WebSocket endpoints. The protocol source
of truth is `src/mindsurf_omni/contract.py`.

## API contract

The standalone service exposes these endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/audio/transcriptions` | PCM16 speech-to-text |
| `POST /v1/chat/completions` | Text completion, optionally streamed as SSE |
| `POST /v1/audio/speech` | Text-to-speech |
| `GET /v1/models` | Active path, components, and licence |
| `GET /v1/voices` | Available voices |
| `GET /v1/token-spec` | Machine-readable text and audio token layout |
| `GET /v1/licence` | Complete model and asset licence chain |
| `WS /v1/realtime` | Streaming speech sessions |

Clients should import event names, token IDs, sample rates, and stable response
fields from `mindsurf_omni.contract` rather than duplicating them. This keeps
the HTTP/WebSocket adapters and model engine aligned when the protocol grows.
