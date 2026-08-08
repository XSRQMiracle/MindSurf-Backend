# MindSurf 后端重写

新的应用入口位于 `src/mindsurf_backend/`，负责适配 MindSurf Voice AI 桌面客户端。
`src/mindsurf_omni/` 保持为模型推理层，`src/python_starter/` 及其数据库、Redis、Celery、
MiniMind 和 vLLM 实验代码继续保留，不删除也不作为本次重写的修改目标。

当前已经建立应用边界和 `mindsurf.voice.v1` 协议基础层，包括控制信封、固定 48 字节
音频帧、握手、心跳、关闭码和标准错误消息。请求层支持单活跃请求、音频上传、输入提交、
取消和终态去重；断连、取消或 fatal 请求错误都会关闭已挂接的 Omni async generator。
`conversation_id` 当前只校验和接收，不持久化。后端启动时会从 `MINDSURF_*` 环境变量
构建 cascade `SpeechEngine`；dictation 输出 `asr.final`，assistant 流式输出文字，并按请求
输出 24 kHz PCM 音频。未配置或缺少组件时应用仍能启动，握手能力为 `false`，请求返回
`model_unavailable`。

开发启动命令：

```bash
uv run uvicorn mindsurf_backend.app:app --reload
```

应用配置使用 `MINDSURF_BACKEND_` 前缀，例如：

```bash
MINDSURF_BACKEND_PORT=9000 \
MINDSURF_BACKEND_VOICE_WS_PATH=/v1/voice/ws \
uv run uvicorn mindsurf_backend.app:app
```

训练交付目录已经包含正式 checkpoint 和 tokenizer，但 SenseVoice、MiniMind-O 模型代码及
TTS 运行时需要单独准备。使用当前本地交付结果的最小 cascade 配置如下：

```bash
export MINDSURF_ENGINE=cascade
export MINDSURF_DEVICE=cpu
export MINDSURF_WEIGHTS=/Users/linzihao/Developer/mindsurf-omni-main/models/mindsurf-omni
export MINDSURF_THINKER=/Users/linzihao/Developer/mindsurf-omni-main/models/mindsurf-omni/sft_merge_768.pth
export MINDSURF_TOKENIZER=/Users/linzihao/Developer/mindsurf-omni-main/models/mindsurf-omni/tokenizer
export MINIMIND_O_ROOT=/path/to/minimind-o
export MINDSURF_ASR=/path/to/minimind-o/model/SenseVoiceSmall
export MINDSURF_CODEC=/path/to/minimind-o/model/mimi
export MINDSURF_SPEAKER=/path/to/minimind-o/model/campplus
export MINDSURF_TTS=edge
uv run uvicorn mindsurf_backend.app:app --host 127.0.0.1 --port 8000 --workers 1
```

依赖必须安装在项目虚拟环境中。Edge TTS 使用
`uv sync --extra dev --extra omni-asr --extra omni-tts`；本地 VoxCPM 改用
`--extra omni-tts-local` 和 `MINDSURF_TTS=voxcpm`。GPU 部署时把 device 改为 `cuda`，
Apple Silicon 本地验证可尝试 `mps`；第一版固定单 worker，避免重复加载模型权重。
