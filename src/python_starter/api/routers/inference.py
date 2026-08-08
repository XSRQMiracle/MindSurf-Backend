"""Compatibility endpoints for plain-text model inference."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from python_starter.api.dependencies import InferenceServiceDep
from python_starter.api.schemas.models import (
    InferenceRequest as APIInferenceRequest,
)
from python_starter.api.schemas.models import (
    InferenceResponse,
)
from python_starter.inference.types import (
    InferenceBackendUnavailableError,
    InferenceRequest,
    InferenceRequestError,
)
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("", response_model=InferenceResponse)
async def run_inference(
    request: APIInferenceRequest,
    service: InferenceServiceDep,
) -> InferenceResponse:
    """Generate text through the configured inference backend."""
    logger.info(
        "inference_request",
        backend=service.backend_name,
        input_length=len(request.text),
        max_length=request.max_length,
        temperature=request.temperature,
    )

    try:
        result = await service.generate(
            InferenceRequest(
                prompt=request.text,
                max_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        )
    except InferenceBackendUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except InferenceRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    logger.info(
        "inference_complete",
        backend=result.backend_name,
        output_length=len(result.generated_text),
        elapsed_ms=result.generation_time * 1000,
    )
    return InferenceResponse(
        text=result.generated_text,
        input_tokens=result.input_tokens,
        output_tokens=result.generated_tokens,
        generation_time_ms=result.generation_time * 1000,
    )


@router.get("/models")
async def list_available_models(service: InferenceServiceDep) -> dict[str, object]:
    """Return the model served by the configured inference backend."""
    return {
        "models": [
            {
                "name": service.model_name,
                "backend": service.backend_name,
            }
        ]
    }
