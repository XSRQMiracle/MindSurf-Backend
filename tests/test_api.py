"""API endpoint tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from python_starter.api.schemas.models import ExperimentStatus


@pytest.mark.asyncio
async def test_health_endpoint(api_client: AsyncClient) -> None:
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_readiness_probe(api_client: AsyncClient) -> None:
    response = await api_client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "services" in data


@pytest.mark.asyncio
async def test_create_experiment(api_client: AsyncClient) -> None:
    payload = {"name": "test-exp", "description": "A test experiment"}
    response = await api_client.post("/experiments", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-exp"
    assert data["status"] == ExperimentStatus.CREATED.value


@pytest.mark.asyncio
async def test_list_experiments(api_client: AsyncClient) -> None:
    # Create an experiment first
    await api_client.post("/experiments", json={"name": "list-test"})

    response = await api_client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_inference_uses_configured_backend(api_client: AsyncClient) -> None:
    payload = {"text": "Hello world", "max_length": 32}
    response = await api_client.post("/inference", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Generated: Hello world"
    assert data["input_tokens"] == 2
    assert data["output_tokens"] == 3


@pytest.mark.asyncio
async def test_inference_models_uses_configured_backend(api_client: AsyncClient) -> None:
    response = await api_client.get("/inference/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [{"name": "test-model", "backend": "fake"}]
    }
