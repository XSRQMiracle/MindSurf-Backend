"""Factory for configured inference services."""

from __future__ import annotations

import asyncio

from python_starter.inference.pytorch_inference import PyTorchInference
from python_starter.inference.service import InferenceService
from python_starter.inference.types import PyTorchModelConfig
from python_starter.inference.vllm_inference import VLLMInference
from python_starter.infrastructure.config import Settings


async def create_inference_service(settings: Settings) -> InferenceService:
    """Create the inference service selected by application settings."""
    if settings.inference_backend == "vllm":
        return InferenceService(
            VLLMInference(
                base_url=settings.vllm_base_url,
                model_name=settings.vllm_model,
                api_key=settings.vllm_api_key,
            )
        )
    if settings.inference_backend == "pytorch":
        config = PyTorchModelConfig(
            checkpoint_path=settings.pytorch_checkpoint_path,
            tokenizer_path=settings.pytorch_tokenizer_path,
            model_name=settings.pytorch_model_name,
            model_config_path=settings.pytorch_model_config_path,
            device=settings.pytorch_device,
        )
        backend = await asyncio.to_thread(PyTorchInference, config)
        return InferenceService(backend)
    raise ValueError(f"Unsupported inference backend: {settings.inference_backend}")
