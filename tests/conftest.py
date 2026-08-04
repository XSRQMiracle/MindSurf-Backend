"""Pytest fixtures and configuration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from python_starter.api.dependencies import get_db, get_inference_service
from python_starter.api.main import create_app
from python_starter.inference.service import InferenceService
from python_starter.inference.types import (
    BackendHTTPResponse,
    InferenceRequest,
    InferenceResult,
    OpenAIChatPayload,
)
from python_starter.infrastructure.config import Settings, get_settings
from python_starter.infrastructure.database import Base


class FakeInferenceBackend:
    """Deterministic inference backend for API tests."""

    name = "fake"
    model_name = "test-model"

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            input_prompt=request.prompt,
            generated_text=f"Generated: {request.prompt}",
            backend_name=self.name,
            model_name=self.model_name,
            input_tokens=2,
            generated_tokens=3,
            generation_time=0.01,
            stop_reason="stop",
        )

    async def chat_completions(self, payload: OpenAIChatPayload) -> BackendHTTPResponse:
        return BackendHTTPResponse(status_code=200, content_type="application/json", body=b"{}")

    async def aclose(self) -> None:
        return None


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Return test settings with in-memory database."""
    return Settings(
        ENV="test",
        DEBUG=True,
        POSTGRES_URL="sqlite+aiosqlite:///:memory:",
        REDIS_URL="redis://localhost:6379/15",
        SECRET_KEY="test-secret-key-32-characters-long",
    )


@pytest_asyncio.fixture
async def db_session(test_settings: Settings) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    engine = create_async_engine(test_settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def fake_redis() -> AsyncGenerator[fakeredis.aioredis.FakeRedis, None]:
    """Provide a fake Redis instance for testing."""
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield redis
    await redis.flushall()
    await cast(Any, redis).aclose()


@pytest_asyncio.fixture
async def api_client(
    test_settings: Settings,
    db_session: AsyncSession,
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an HTTP client for API testing with injected dependencies."""
    app = create_app(test_settings)
    inference_service = InferenceService(FakeInferenceBackend())

    # Override dependencies for testing
    async def override_get_settings() -> Settings:
        return test_settings

    async def override_inference_service() -> InferenceService:
        return inference_service

    async def override_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_settings] = override_get_settings
    app.dependency_overrides[get_inference_service] = override_inference_service
    app.dependency_overrides[get_db] = override_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
