"""Tests for the application-to-Omni integration boundary."""

from __future__ import annotations

import pytest

from mindsurf_backend.omni import OmniAdapter, OmniNotConfiguredError


def test_unconfigured_adapter_reports_unavailable() -> None:
    adapter = OmniAdapter()

    assert adapter.available is False
    with pytest.raises(OmniNotConfiguredError, match="has not been configured"):
        _ = adapter.engine
