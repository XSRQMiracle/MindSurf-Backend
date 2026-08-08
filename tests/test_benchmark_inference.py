"""Tests for inference benchmark helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from scripts.benchmark_inference import (
    BackendReport,
    BenchmarkMetrics,
    encode_prompts,
    load_prompts,
    parse_batch_sizes,
    validate_prompt_lengths,
)


def test_parse_batch_sizes() -> None:
    assert parse_batch_sizes("1, 8,32") == (1, 8, 32)

    with pytest.raises(argparse.ArgumentTypeError):
        parse_batch_sizes("1,0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_batch_sizes("4,4")


def test_load_prompts(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.txt"
    prompt_path.write_text("first\n\nsecond\n", encoding="utf-8")

    prompts = load_prompts(prompt_path)

    assert prompts == ("first", "second")


def test_encode_prompts_and_validate_lengths() -> None:
    class FakeTokenizer:
        def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens
            return [1, len(prompt), 2]

    encoded = encode_prompts(FakeTokenizer(), ("a", "abcd"))

    assert encoded == ((1, 1, 2), (1, 4, 2))
    validate_prompt_lengths(encoded, max_new_tokens=5, max_model_len=8)

    with pytest.raises(ValueError, match="exceeds"):
        validate_prompt_lengths(encoded, max_new_tokens=6, max_model_len=8)


def test_backend_report_round_trip() -> None:
    metric = BenchmarkMetrics(
        backend="pytorch",
        batch_size=8,
        repeats=2,
        elapsed_seconds=4.0,
        prompt_tokens=80,
        output_tokens=160,
    )
    report = BackendReport(backend="pytorch", load_seconds=1.5, metrics=(metric,))

    restored = BackendReport.from_dict(report.to_dict())

    assert restored == report
    assert metric.average_batch_latency_ms == 2000.0
    assert metric.requests_per_second == 4.0
    assert metric.output_tokens_per_second == 40.0
