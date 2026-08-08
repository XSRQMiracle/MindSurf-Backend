"""Native PyTorch inference backend."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, cast, overload

import orjson
import torch
from omegaconf import OmegaConf

from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.tokenizer import decode_tokens, encode_text, load_tokenizer
from python_starter.core.utils import get_device, set_seed
from python_starter.inference.types import (
    BackendHTTPResponse,
    InferenceRequest,
    InferenceRequestError,
    InferenceResult,
    OpenAIChatPayload,
    PyTorchModelConfig,
)

_SUPPORTED_CHAT_FIELDS = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "n",
}


class PyTorchInference:
    """Run non-streaming inference with the native TransformerLM."""

    name = "pytorch"

    def __init__(self, config: PyTorchModelConfig) -> None:
        self._config = config
        self._device = get_device(config.device)
        self._model, self._model_config = _load_model(config, self._device)
        self._tokenizer = load_tokenizer(config.tokenizer_path)
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        """Return the configured local model name."""
        return self._config.model_name

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate a complete response without blocking the event loop."""
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: InferenceRequest) -> InferenceResult:
        if request.max_tokens <= 0:
            raise InferenceRequestError(422, "max_tokens must be positive")
        if request.max_tokens >= self._model_config.max_seq_len:
            raise InferenceRequestError(422, "max_tokens exceeds the model context length")
        if request.temperature < 0:
            raise InferenceRequestError(422, "temperature must be non-negative")
        if not 0 < request.top_p <= 1:
            raise InferenceRequestError(422, "top_p must be in (0, 1]")

        max_prompt_tokens = self._model_config.max_seq_len - request.max_tokens
        input_tokens = encode_text(
            self._tokenizer,
            request.prompt,
            max_length=max_prompt_tokens,
        )
        input_tensor = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=self._device,
        )
        if request.seed is not None:
            set_seed(request.seed)

        started = time.perf_counter()
        with torch.inference_mode():
            output_tokens = self._model.generate(
                input_tensor,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        completion_ids = [
            int(token_id) for token_id in output_tokens[0, len(input_tokens) :].tolist()
        ]
        generated_text = decode_tokens(self._tokenizer, completion_ids)
        generated_text, stopped_by_string = _apply_stop_strings(generated_text, request.stop)
        if stopped_by_string:
            completion_ids = encode_text(
                self._tokenizer,
                generated_text,
                add_special_tokens=False,
            )
        stopped_by_eos = bool(
            completion_ids
            and self._tokenizer.eos_token_id is not None
            and completion_ids[-1] == self._tokenizer.eos_token_id
        )

        return InferenceResult(
            input_prompt=request.prompt,
            generated_text=generated_text,
            backend_name=self.name,
            model_name=self.model_name,
            input_tokens=len(input_tokens),
            generated_tokens=len(completion_ids),
            generation_time=time.perf_counter() - started,
            stop_reason="stop" if stopped_by_string or stopped_by_eos else "length",
        )

    async def chat_completions(self, payload: OpenAIChatPayload) -> BackendHTTPResponse:
        """Return a non-streaming OpenAI-compatible chat completion."""
        try:
            request = self._parse_chat_request(payload)
            result = await self.generate(request)
        except InferenceRequestError as exc:
            return _error_response(exc.status_code, exc.detail)

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.generated_text},
                    "finish_reason": result.stop_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.generated_tokens,
                "total_tokens": result.input_tokens + result.generated_tokens,
            },
        }
        return BackendHTTPResponse(
            status_code=200,
            content_type="application/json",
            body=orjson.dumps(response),
        )

    def _parse_chat_request(self, payload: OpenAIChatPayload) -> InferenceRequest:
        if payload.get("stream") is True:
            raise InferenceRequestError(400, "PyTorch backend does not support streaming")
        unsupported = set(payload) - _SUPPORTED_CHAT_FIELDS
        if unsupported:
            fields = ", ".join(sorted(unsupported))
            raise InferenceRequestError(422, f"Unsupported PyTorch fields: {fields}")
        if payload.get("model") != self.model_name:
            raise InferenceRequestError(400, f"Unknown model: {payload.get('model')}")
        if payload.get("n", 1) != 1:
            raise InferenceRequestError(422, "PyTorch backend only supports n=1")

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise InferenceRequestError(422, "messages must be a non-empty list")
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise InferenceRequestError(422, "Each message must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise InferenceRequestError(422, "PyTorch messages must contain text content")
            normalized_messages.append({"role": cast(str, role), "content": content})

        prompt_value = self._tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt_value, str):
            raise InferenceRequestError(500, "Tokenizer did not render a text prompt")

        max_tokens = _number(payload.get("max_tokens", 128), "max_tokens", int)
        temperature = _number(payload.get("temperature", 0.7), "temperature", float)
        top_p = _number(payload.get("top_p", 0.9), "top_p", float)
        stop = _parse_stop(payload.get("stop"))
        return InferenceRequest(
            prompt=prompt_value,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=_optional_int(payload.get("seed")),
        )

    async def aclose(self) -> None:
        """Release backend resources."""


def _load_model(
    config: PyTorchModelConfig,
    device: torch.device,
) -> tuple[TransformerLM, ModelConfig]:
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a mapping")

    raw_config = checkpoint.get("model_config")
    if isinstance(raw_config, dict):
        model_config = ModelConfig.from_dict(cast(dict[str, Any], raw_config))
    elif config.model_config_path is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(config.model_config_path), resolve=True)
        if not isinstance(loaded, dict):
            raise ValueError("Model config must contain a mapping")
        model_config = ModelConfig.from_dict(cast(dict[str, Any], loaded))
    else:
        raise ValueError("Checkpoint has no model_config and no fallback was configured")

    raw_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint has no valid model_state_dict")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("State dict must map names to tensors")
        state[name.removeprefix("_orig_mod.")] = tensor

    model = TransformerLM(model_config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, model_config


def _apply_stop_strings(text: str, stop: tuple[str, ...]) -> tuple[str, bool]:
    indexes = [index for value in stop if (index := text.find(value)) >= 0]
    if not indexes:
        return text, False
    return text[: min(indexes)], True


def _parse_stop(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(cast(list[str], value))
    raise InferenceRequestError(422, "stop must be a string or a list of strings")


@overload
def _number(value: object, name: str, target: type[int]) -> int: ...


@overload
def _number(value: object, name: str, target: type[float]) -> float: ...


def _number(
    value: object,
    name: str,
    target: type[int] | type[float],
) -> int | float:
    if target is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise InferenceRequestError(422, f"{name} must be an integer")
        return value
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise InferenceRequestError(422, f"{name} must be a number")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise InferenceRequestError(422, "seed must be an integer")
    return value


def _error_response(status_code: int, detail: str) -> BackendHTTPResponse:
    return BackendHTTPResponse(
        status_code=status_code,
        content_type="application/json",
        body=orjson.dumps(
            {
                "error": {
                    "message": detail,
                    "type": "invalid_request_error",
                    "code": None,
                }
            }
        ),
    )
