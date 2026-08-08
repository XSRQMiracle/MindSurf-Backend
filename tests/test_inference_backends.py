"""Tests for unified inference backends and service delegation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import orjson
import pytest
import torch
from httpx import AsyncClient, MockTransport, Request, Response
from pydantic import ValidationError

from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.inference import pytorch_inference
from python_starter.inference.factory import create_inference_service
from python_starter.inference.pytorch_inference import PyTorchInference
from python_starter.inference.service import InferenceService
from python_starter.inference.types import (
    BackendHTTPResponse,
    InferenceRequest,
    InferenceResult,
    OpenAIChatPayload,
    PyTorchModelConfig,
)
from python_starter.inference.vllm_inference import VLLMInference
from python_starter.infrastructure.config import Settings


class StubBackend:
    """Minimal backend used to verify service delegation."""

    name = "stub"
    model_name = "stub-model"

    def __init__(self) -> None:
        self.closed = False

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            input_prompt=request.prompt,
            generated_text="generated",
            backend_name=self.name,
            model_name=self.model_name,
            input_tokens=1,
            generated_tokens=1,
            generation_time=0.1,
            stop_reason="stop",
        )

    async def chat_completions(self, payload: OpenAIChatPayload) -> BackendHTTPResponse:
        return BackendHTTPResponse(200, "application/json", body=orjson.dumps(payload))

    async def aclose(self) -> None:
        self.closed = True


class FakeTokenizer:
    """Tokenizer double for native backend tests."""

    eos_token_id = 9

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> list[int]:
        tokens = [1, 2]
        return tokens[:max_length] if truncation and max_length is not None else tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "answer" if token_ids else ""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


class FakeModel:
    """Model double that appends two completion tokens."""

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        completion = torch.tensor([[3, 4]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, completion], dim=1)


@pytest.mark.asyncio
async def test_inference_service_delegates_and_closes() -> None:
    backend = StubBackend()
    service = InferenceService(backend)

    result = await service.generate(InferenceRequest(prompt="hello"))
    response = await service.chat_completions({"model": "stub-model"})
    await service.aclose()

    assert service.backend_name == "stub"
    assert service.model_name == "stub-model"
    assert result.generated_text == "generated"
    assert response.status_code == 200
    assert backend.closed is True


@pytest.mark.asyncio
async def test_factory_uses_vllm_by_default() -> None:
    settings = Settings(
        INFERENCE_BACKEND="vllm",
        VLLM_BASE_URL="http://vllm:8001",
        VLLM_MODEL="minimind",
    )

    service = await create_inference_service(settings)
    assert service.backend_name == "vllm"
    assert service.model_name == "minimind"
    await service.aclose()


def test_settings_reject_unknown_inference_backend() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"INFERENCE_BACKEND": "unknown"})


@pytest.mark.asyncio
async def test_vllm_generate_normalizes_openai_response() -> None:
    def handle_upstream(request: Request) -> Response:
        return Response(
            200,
            json={
                "model": "minimind",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "generated"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    client = AsyncClient(
        base_url="http://vllm:8001",
        transport=MockTransport(handle_upstream),
    )
    backend = VLLMInference("http://vllm:8001", "minimind", client=client)

    result = await backend.generate(InferenceRequest(prompt="hello", max_tokens=8))

    await client.aclose()
    assert result.generated_text == "generated"
    assert result.input_tokens == 3
    assert result.generated_tokens == 2


@pytest.fixture
def pytorch_backend(monkeypatch: pytest.MonkeyPatch) -> PyTorchInference:
    model_config = ModelConfig(
        vocab_size=16,
        n_embed=8,
        n_layer=1,
        n_head=1,
        max_seq_len=16,
        hidden_dim=16,
    )
    monkeypatch.setattr(
        pytorch_inference,
        "_load_model",
        lambda config, device: (cast(TransformerLM, FakeModel()), model_config),
    )
    monkeypatch.setattr(
        pytorch_inference,
        "load_tokenizer",
        lambda path: cast(Any, FakeTokenizer()),
    )
    return PyTorchInference(
        PyTorchModelConfig(
            checkpoint_path="unused.pt",
            tokenizer_path="unused-tokenizer",
            model_name="minimind",
            device="cpu",
        )
    )


@pytest.mark.asyncio
async def test_pytorch_backend_generates_complete_response(
    pytorch_backend: PyTorchInference,
) -> None:
    result = await pytorch_backend.generate(
        InferenceRequest(prompt="hello", max_tokens=4, seed=7)
    )

    assert result.generated_text == "answer"
    assert result.input_tokens == 2
    assert result.generated_tokens == 2
    assert result.backend_name == "pytorch"


@pytest.mark.asyncio
async def test_pytorch_backend_rejects_streaming(
    pytorch_backend: PyTorchInference,
) -> None:
    response = await pytorch_backend.chat_completions(
        {
            "model": "minimind",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    )

    assert response.status_code == 400
    assert response.body is not None
    assert orjson.loads(response.body)["error"]["message"] == (
        "PyTorch backend does not support streaming"
    )


@pytest.mark.asyncio
async def test_pytorch_backend_returns_openai_non_streaming_response(
    pytorch_backend: PyTorchInference,
) -> None:
    response = await pytorch_backend.chat_completions(
        {
            "model": "minimind",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "max_tokens": 4,
        }
    )

    assert response.status_code == 200
    assert response.body is not None
    data = orjson.loads(response.body)
    assert data["choices"][0]["message"]["content"] == "answer"
    assert data["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }


def test_load_model_accepts_compiled_checkpoint_prefix(tmp_path: Path) -> None:
    model_config = ModelConfig(
        vocab_size=16,
        n_embed=8,
        n_layer=1,
        n_head=1,
        max_seq_len=16,
        hidden_dim=16,
    )
    source_model = TransformerLM(model_config)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_config": model_config.to_dict(),
            "model_state_dict": {
                f"_orig_mod.{name}": tensor for name, tensor in source_model.state_dict().items()
            },
        },
        checkpoint_path,
    )

    loaded_model, loaded_config = pytorch_inference._load_model(
        PyTorchModelConfig(
            checkpoint_path=str(checkpoint_path),
            tokenizer_path="unused",
            model_name="test",
        ),
        torch.device("cpu"),
    )

    assert loaded_config.to_dict() == model_config.to_dict()
    assert loaded_model.training is False
