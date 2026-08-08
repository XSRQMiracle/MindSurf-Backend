"""Tests for MindSurf checkpoint conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from scripts.convert_model import (
    build_qwen3_weight_name_map,
    build_weight_name_map,
    convert_checkpoint_to_llama,
    convert_checkpoint_to_qwen3,
    load_checkpoint_state_dict,
    max_logit_difference,
    resolve_conversion_target,
)
from transformers import AutoModelForCausalLM, LlamaForCausalLM, Qwen3ForCausalLM

from python_starter.core.model import ModelConfig, TransformerLM


def test_convert_checkpoint_to_llama(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=100,
        n_embed=64,
        n_layer=2,
        n_head=4,
        max_seq_len=32,
        hidden_dim=128,
        tie_weights=True,
        rope_theta=10_000.0,
        qk_norm=False,
    )
    source_model = TransformerLM(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_dir = tmp_path / "llama"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    convert_checkpoint_to_llama(
        checkpoint_path,
        output_dir,
        config,
        dtype=torch.float32,
    )

    converted_model = LlamaForCausalLM.from_pretrained(output_dir)
    converted_state = converted_model.state_dict()
    for source_name, target_name in build_weight_name_map(config.n_layer).items():
        torch.testing.assert_close(
            source_model.state_dict()[source_name],
            converted_state[target_name],
        )

    assert converted_model.config.hidden_size == config.n_embed
    assert converted_model.config.intermediate_size == config.hidden_dim
    assert converted_model.config.num_hidden_layers == config.n_layer
    assert converted_model.config.num_key_value_heads == config.n_head
    assert converted_model.config.rope_theta == config.rope_theta


def test_convert_checkpoint_to_qwen3_round_trip(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=100,
        n_embed=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        max_seq_len=32,
        hidden_dim=128,
        tie_weights=True,
        rope_theta=1_000_000.0,
        qk_norm=True,
    )
    source_model = TransformerLM(config).eval()
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_dir = tmp_path / "qwen3"
    release = {"license": "test-only"}
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": source_model.state_dict(),
            "release": release,
        },
        checkpoint_path,
    )

    convert_checkpoint_to_qwen3(
        checkpoint_path,
        output_dir,
        dtype=torch.float32,
        tolerance=0.0,
    )

    converted_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.float32,
    )
    assert isinstance(converted_model, Qwen3ForCausalLM)
    converted_state = converted_model.state_dict()
    for source_name, target_name in build_qwen3_weight_name_map(config.n_layer).items():
        torch.testing.assert_close(
            source_model.state_dict()[source_name],
            converted_state[target_name],
        )

    assert max_logit_difference(source_model, converted_model) == 0.0
    assert converted_model.config.model_type == "qwen3"
    assert converted_model.config.num_key_value_heads == config.n_kv_head
    assert converted_model.config.rope_theta == config.rope_theta
    assert converted_model.config.sliding_window is None
    assert json.loads((output_dir / "mindsurf_release.json").read_text()) == release


def test_resolve_conversion_target_checks_qk_norm() -> None:
    qwen_config = ModelConfig(n_layer=1, qk_norm=True)
    qwen_state = {"blocks.0.attn.q_norm.weight": torch.ones(1)}
    qwen_state["blocks.0.attn.k_norm.weight"] = torch.ones(1)
    assert resolve_conversion_target("auto", qwen_config, qwen_state) == "qwen3"

    llama_config = ModelConfig(n_layer=1, qk_norm=False)
    assert resolve_conversion_target("auto", llama_config, {}) == "llama"

    with pytest.raises(ValueError, match="requires target qwen3"):
        resolve_conversion_target("llama", qwen_config, qwen_state)


def test_load_checkpoint_state_dict_removes_compile_prefix(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "compiled.pt"
    expected = {"token_embed.weight": torch.ones(2, 2)}
    compiled = {f"_orig_mod.{name}": tensor for name, tensor in expected.items()}
    torch.save({"model_state_dict": compiled}, checkpoint_path)

    loaded = load_checkpoint_state_dict(checkpoint_path)

    assert loaded.keys() == expected.keys()
    torch.testing.assert_close(loaded["token_embed.weight"], expected["token_embed.weight"])
