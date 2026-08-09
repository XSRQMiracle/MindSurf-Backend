# MindSurf Backend

MindSurf Backend 是 MindSurf 项目的后端服务仓库，负责提供后端 API、语音交互能力以及模型推理层的接入。



具体功能说明、开发环境配置、启动方式和部署流程将在后续补充。

## 项目结构

```text
.
├── src/
│   ├── mindsurf_backend/    # 后端应用、HTTP API 与语音交互协议
│   │   ├── http/            # HTTP 路由
│   │   ├── omni/            # 模型推理层适配
│   │   └── voice/           # 实时语音交互
│   └── mindsurf_omni/       # Omni 模型推理服务
├── tests/
│   ├── backend/             # 后端测试
│   └── omni/                # Omni 推理层测试
├── docker/                  # 后端容器配置
├── docs/                    # 后端相关文档
├── .env.omni.example        # Omni 环境变量示例
├── pyproject.toml           # 项目及依赖配置
└── uv.lock                  # 依赖锁定文件
```

## TODO
- [ ] 完善 README 文档
- [ ] 完善后端 API 文档
- [ ] 完善docker配置文件