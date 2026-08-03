"""Unified model inference backends and service layer."""

from python_starter.inference.service import InferenceService
from python_starter.inference.types import InferenceRequest, InferenceResult

__all__ = ["InferenceRequest", "InferenceResult", "InferenceService"]
