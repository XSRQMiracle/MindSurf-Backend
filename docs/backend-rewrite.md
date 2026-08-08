# MindSurf 后端重写

新的应用入口位于 `src/mindsurf_backend/`，负责适配 MindSurf Voice AI 桌面客户端。
`src/mindsurf_omni/` 保持为模型推理层，`src/python_starter/` 及其数据库、Redis、Celery、
MiniMind 和 vLLM 实验代码继续保留，不删除也不作为本次重写的修改目标。

当前已经建立应用边界和 `mindsurf.voice.v1` 协议基础层，包括控制信封、固定 48 字节
音频帧、握手、心跳、关闭码和标准错误消息。业务请求状态机与真实 Omni 加载仍未实现，
因此 `server.hello.features` 会如实把流式推理能力声明为 `false`。

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
