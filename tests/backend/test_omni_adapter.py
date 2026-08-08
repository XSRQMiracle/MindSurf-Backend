"""Tests for the application-to-Omni integration boundary."""

from __future__ import annotations

import pytest

from mindsurf_backend.omni import OmniAdapter, OmniNotConfiguredError
from mindsurf_omni.service.config import ConfigurationError
from tests.backend.fakes import FakeSpeechEngine


def test_unconfigured_adapter_reports_unavailable() -> None:
    adapter = OmniAdapter()

    assert adapter.available is False
    with pytest.raises(OmniNotConfiguredError, match="no Omni speech engine is configured"):
        _ = adapter.engine


def test_adapter_builds_cascade_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = FakeSpeechEngine()
    monkeypatch.setattr("mindsurf_backend.omni.adapter.build", lambda _: engine)

    adapter = OmniAdapter.from_environment({})

    assert adapter.available is True
    assert adapter.engine is engine


def test_adapter_keeps_configuration_error_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: object) -> None:
        raise ConfigurationError("checkpoint is missing")

    monkeypatch.setattr("mindsurf_backend.omni.adapter.build", fail)

    adapter = OmniAdapter.from_environment({})

    assert adapter.available is False
    with pytest.raises(OmniNotConfiguredError, match="checkpoint is missing"):
        _ = adapter.engine


def test_adapter_rejects_non_cascade_engine() -> None:
    adapter = OmniAdapter(FakeSpeechEngine(path="native"))

    assert adapter.available is False
    with pytest.raises(OmniNotConfiguredError, match="requires the cascade"):
        _ = adapter.engine


def test_adapter_reports_and_requires_individual_stages() -> None:
    engine = FakeSpeechEngine(unwired=("transcriber",))
    adapter = OmniAdapter(engine)

    assert adapter.ready is False
    assert adapter.stage_status == {
        "transcriber": False,
        "generator": True,
        "synthesiser": True,
    }
    assert adapter.require_stages("generator") is engine
    with pytest.raises(OmniNotConfiguredError, match="transcriber"):
        adapter.require_stages("transcriber")
