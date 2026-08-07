# MindSurf 后端重写

新的应用入口位于 `src/mindsurf_backend/`，负责适配 MindSurf Voice AI 桌面客户端。
`src/mindsurf_omni/` 保持为模型推理层，`src/python_starter/` 及其数据库、Redis、Celery、
MiniMind 和 vLLM 实验代码继续保留，不删除也不作为本次重写的修改目标。

当前阶段只建立应用边界，还没有实现 Voice WebSocket 协议或加载真实模型。

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
