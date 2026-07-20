"""Tests for MindSurf-to-Llama checkpoint conversion."""

from __future__ import annotations

from pathlib import Path

import torch
from scripts.convert_model import (
    build_weight_name_map,
    convert_checkpoint_to_llama,
    load_checkpoint_state_dict,
)
from transformers import LlamaForCausalLM

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


def test_load_checkpoint_state_dict_removes_compile_prefix(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "compiled.pt"
    expected = {"token_embed.weight": torch.ones(2, 2)}
    compiled = {f"_orig_mod.{name}": tensor for name, tensor in expected.items()}
    torch.save({"model_state_dict": compiled}, checkpoint_path)

    loaded = load_checkpoint_state_dict(checkpoint_path)

    assert loaded.keys() == expected.keys()
    torch.testing.assert_close(loaded["token_embed.weight"], expected["token_embed.weight"])
