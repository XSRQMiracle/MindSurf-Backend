"""Convert a MindSurf checkpoint to a Hugging Face Llama model.

Usage:
    uv run scripts/convert_model.py \
        --checkpoint models/checkpoints/final_model.pt \
        --model-config configs/model/minimind.yaml \
        --output-dir models/mindsurf-llama \
        --tokenizer path/to/tokenizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.model import ModelConfig

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a MindSurf model configuration from a model YAML file."""
    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Model config must be a mapping: {path}")

    if "model" in raw_config:
        raw_config = raw_config["model"]
    if not isinstance(raw_config, dict):
        raise ValueError(f"Model config must contain a mapping: {path}")

    config_values = cast(dict[str, Any], raw_config)
    return ModelConfig(**config_values)


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load weights from a Trainer checkpoint or a raw state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")

    raw_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint field 'model_state_dict' must be a mapping")

    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("State dict must map string names to tensors")
        state_dict[name] = tensor

    compile_prefix = "_orig_mod."
    if state_dict and all(name.startswith(compile_prefix) for name in state_dict):
        state_dict = {
            name.removeprefix(compile_prefix): tensor for name, tensor in state_dict.items()
        }

    return state_dict


def build_llama_config(
    model_config: ModelConfig,
    dtype: torch.dtype,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> LlamaConfig:
    """Translate a MindSurf model configuration to an equivalent Llama configuration."""
    if model_config.n_embed % model_config.n_head != 0:
        raise ValueError("n_embed must be divisible by n_head")

    config_kwargs: dict[str, Any] = {
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.n_embed,
        "intermediate_size": model_config.hidden_dim,
        "num_hidden_layers": model_config.n_layer,
        "num_attention_heads": model_config.n_head,
        "num_key_value_heads": model_config.n_head,
        "max_position_embeddings": model_config.max_seq_len,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": model_config.dropout,
        "mlp_bias": False,
        "tie_word_embeddings": model_config.tie_weights,
        "use_cache": True,
        "bos_token_id": tokenizer.bos_token_id if tokenizer is not None else None,
        "eos_token_id": tokenizer.eos_token_id if tokenizer is not None else None,
        "pad_token_id": tokenizer.pad_token_id if tokenizer is not None else None,
    }

    transformers_major_version = int(transformers.__version__.split(".", maxsplit=1)[0])
    if transformers_major_version >= 5:
        config_kwargs["dtype"] = dtype
        config_kwargs["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": 10000.0,
        }
    else:
        config_kwargs["torch_dtype"] = dtype
        config_kwargs["rope_theta"] = 10000.0

    return LlamaConfig(**config_kwargs)


def build_weight_name_map(num_hidden_layers: int) -> dict[str, str]:
    """Map MindSurf state-dict names to Llama state-dict names."""
    name_map = {
        "token_embed.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for layer_index in range(num_hidden_layers):
        source = f"blocks.{layer_index}"
        target = f"model.layers.{layer_index}"
        name_map.update(
            {
                f"{source}.attn.q_proj.weight": f"{target}.self_attn.q_proj.weight",
                f"{source}.attn.k_proj.weight": f"{target}.self_attn.k_proj.weight",
                f"{source}.attn.v_proj.weight": f"{target}.self_attn.v_proj.weight",
                f"{source}.attn.o_proj.weight": f"{target}.self_attn.o_proj.weight",
                f"{source}.ffn.w1.weight": f"{target}.mlp.gate_proj.weight",
                f"{source}.ffn.w2.weight": f"{target}.mlp.up_proj.weight",
                f"{source}.ffn.w3.weight": f"{target}.mlp.down_proj.weight",
                f"{source}.attn_norm.weight": f"{target}.input_layernorm.weight",
                f"{source}.ffn_norm.weight": f"{target}.post_attention_layernorm.weight",
            }
        )

    return name_map


def convert_state_dict(
    source_state: dict[str, torch.Tensor],
    target_model: LlamaForCausalLM,
) -> dict[str, torch.Tensor]:
    """Rename and validate all MindSurf tensors for a Llama model."""
    name_map = build_weight_name_map(target_model.config.num_hidden_layers)
    expected_source_names = set(name_map)
    actual_source_names = set(source_state)

    missing_source = expected_source_names - actual_source_names
    unexpected_source = actual_source_names - expected_source_names
    if missing_source or unexpected_source:
        raise ValueError(
            "Source state dict does not match ModelConfig: "
            f"missing={sorted(missing_source)}, unexpected={sorted(unexpected_source)}"
        )

    converted_state = {
        target_name: source_state[source_name] for source_name, target_name in name_map.items()
    }
    target_state = target_model.state_dict()

    missing_target = set(target_state) - set(converted_state)
    unexpected_target = set(converted_state) - set(target_state)
    if missing_target or unexpected_target:
        raise ValueError(
            "Converted state dict does not match Llama: "
            f"missing={sorted(missing_target)}, unexpected={sorted(unexpected_target)}"
        )

    for name, tensor in converted_state.items():
        expected_shape = target_state[name].shape
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {name}: source={tuple(tensor.shape)}, "
                f"target={tuple(expected_shape)}"
            )

    return converted_state


def convert_checkpoint_to_llama(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig,
    *,
    tokenizer_path: str | Path | None = None,
    dtype: torch.dtype = torch.float16,
    safe_serialization: bool = True,
    trust_remote_code: bool = False,
) -> Path:
    """Convert and save a MindSurf checkpoint as a Hugging Face Llama model."""
    tokenizer = None
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        if len(tokenizer) != model_config.vocab_size:
            raise ValueError(
                "Tokenizer size must match model vocab_size: "
                f"tokenizer={len(tokenizer)}, model={model_config.vocab_size}"
            )

    source_state = load_checkpoint_state_dict(checkpoint_path)
    llama_config = build_llama_config(model_config, dtype, tokenizer)
    llama_model_factory = cast(Any, LlamaForCausalLM)
    llama_model = cast(LlamaForCausalLM, llama_model_factory(llama_config))
    cast(Any, llama_model).to(dtype=dtype)
    converted_state = convert_state_dict(source_state, llama_model)
    llama_model.load_state_dict(converted_state, strict=True)
    cast(Any, llama_model).eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    llama_model.save_pretrained(output_path, safe_serialization=safe_serialization)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a MindSurf checkpoint to Hugging Face Llama format"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the MindSurf checkpoint")
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to configs/model/*.yaml used to train the model",
    )
    parser.add_argument("--output-dir", required=True, help="Output Hugging Face model directory")
    parser.add_argument("--tokenizer", help="Tokenizer name or local path to save with the model")
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="float16",
        help="Output parameter dtype (default: float16)",
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Save pytorch_model.bin instead of model.safetensors",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom tokenizer code when loading --tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion from the command line."""
    args = parse_args()
    model_config = load_model_config(args.model_config)
    output_path = convert_checkpoint_to_llama(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_config=model_config,
        tokenizer_path=args.tokenizer,
        dtype=DTYPES[args.dtype],
        safe_serialization=not args.no_safe_serialization,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Converted model saved to: {output_path}")


if __name__ == "__main__":
    main()
