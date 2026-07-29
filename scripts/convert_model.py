"""Convert a MindSurf checkpoint to a Hugging Face model.

Usage:
    uv run scripts/convert_model.py \
        --checkpoint models/checkpoints/test/final_model.pt \
        --output-dir models/transformers/mindsurf-qwen3 \
        --tokenizer models/tokenizers/minimind_tokenizer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.model import ModelConfig, TransformerLM

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


def _normalize_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint has no valid model_state_dict")

    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("State dict must map string names to tensors")
        state[name.removeprefix("_orig_mod.")] = tensor
    return state


def load_checkpoint(
    path: str | Path,
    fallback_config: ModelConfig | None = None,
) -> tuple[ModelConfig, dict[str, torch.Tensor], dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError("Checkpoint must contain a mapping")

    raw_config = blob.get("model_config")
    if isinstance(raw_config, dict):
        model_config = ModelConfig.from_dict(raw_config)
    elif fallback_config is not None:
        model_config = fallback_config
    else:
        raise ValueError("Checkpoint has no model_config and no fallback was provided")

    state = _normalize_state_dict(blob.get("model_state_dict", blob))
    return model_config, state, blob


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load weights from a Trainer checkpoint or a raw state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")

    return _normalize_state_dict(checkpoint.get("model_state_dict", checkpoint))


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
            "rope_theta": model_config.rope_theta,
        }
    else:
        config_kwargs["torch_dtype"] = dtype
        config_kwargs["rope_theta"] = model_config.rope_theta

    return LlamaConfig(**config_kwargs)


def build_qwen3_config(
    config: ModelConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Qwen3Config:
    kwargs: dict[str, Any] = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.n_embed,
        "intermediate_size": config.hidden_dim,
        "num_hidden_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "num_key_value_heads": config.n_kv_head,
        "head_dim": config.n_embed // config.n_head,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": config.tie_weights,
        "attention_bias": False,
        "attention_dropout": config.dropout,
        "hidden_act": "silu",
        "use_cache": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": config.n_layer,
        "bos_token_id": tokenizer.bos_token_id if tokenizer else None,
        "eos_token_id": tokenizer.eos_token_id if tokenizer else None,
        "pad_token_id": tokenizer.pad_token_id if tokenizer else None,
    }

    major = int(transformers.__version__.split(".", 1)[0])
    if major >= 5:
        kwargs["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": config.rope_theta,
        }
    else:
        kwargs["rope_theta"] = config.rope_theta
        kwargs["rope_scaling"] = None

    return Qwen3Config(**kwargs)


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


def build_qwen3_weight_name_map(num_layers: int) -> dict[str, str]:
    """Map MindSurf state-dict names to Qwen3 state-dict names."""
    mapping = {
        "token_embed.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for index in range(num_layers):
        source = f"blocks.{index}"
        target = f"model.layers.{index}"

        mapping.update(
            {
                f"{source}.attn_norm.weight": f"{target}.input_layernorm.weight",
                f"{source}.ffn_norm.weight": f"{target}.post_attention_layernorm.weight",
                f"{source}.attn.q_proj.weight": f"{target}.self_attn.q_proj.weight",
                f"{source}.attn.k_proj.weight": f"{target}.self_attn.k_proj.weight",
                f"{source}.attn.v_proj.weight": f"{target}.self_attn.v_proj.weight",
                f"{source}.attn.o_proj.weight": f"{target}.self_attn.o_proj.weight",
                f"{source}.attn.q_norm.weight": f"{target}.self_attn.q_norm.weight",
                f"{source}.attn.k_norm.weight": f"{target}.self_attn.k_norm.weight",
                f"{source}.ffn.w1.weight": f"{target}.mlp.gate_proj.weight",
                f"{source}.ffn.w2.weight": f"{target}.mlp.up_proj.weight",
                f"{source}.ffn.w3.weight": f"{target}.mlp.down_proj.weight",
            }
        )

    return mapping


def probe_ids(config: ModelConfig) -> torch.Tensor:
    length = min(12, config.max_seq_len)
    step = max(1, config.vocab_size // length)

    ascending = [(index * step) % config.vocab_size for index in range(length)]
    descending = [
        (config.vocab_size - 1 - index * step) % config.vocab_size for index in range(length)
    ]
    return torch.tensor([ascending, descending], dtype=torch.long)


def max_logit_difference(
    source: TransformerLM,
    target: Qwen3ForCausalLM,
) -> float:
    input_ids = probe_ids(source.config)

    with torch.inference_mode():
        source_logits, _ = source(input_ids)
        target_logits = target(input_ids).logits

    return float((source_logits.float() - target_logits.float()).abs().max())


def _rename_and_validate_state_dict(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    name_map: dict[str, str],
    target_name: str,
) -> dict[str, torch.Tensor]:
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
        target_key: source_state[source_key] for source_key, target_key in name_map.items()
    }
    missing_target = set(target_state) - set(converted_state)
    unexpected_target = set(converted_state) - set(target_state)
    if missing_target or unexpected_target:
        raise ValueError(
            f"Converted state dict does not match {target_name}: "
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


def convert_state_dict(
    source_state: dict[str, torch.Tensor],
    target_model: LlamaForCausalLM,
) -> dict[str, torch.Tensor]:
    """Rename and validate all MindSurf tensors for a Llama model."""
    return _rename_and_validate_state_dict(
        source_state,
        target_model.state_dict(),
        build_weight_name_map(target_model.config.num_hidden_layers),
        "Llama",
    )


def convert_qwen3_state_dict(
    source_state: dict[str, torch.Tensor],
    target_model: Qwen3ForCausalLM,
) -> dict[str, torch.Tensor]:
    """Rename and validate all MindSurf tensors for a Qwen3 model."""
    return _rename_and_validate_state_dict(
        source_state,
        target_model.state_dict(),
        build_qwen3_weight_name_map(target_model.config.num_hidden_layers),
        "Qwen3",
    )


def convert_checkpoint_to_llama(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig | None = None,
    *,
    tokenizer_path: str | Path | None = None,
    dtype: torch.dtype = torch.float16,
    safe_serialization: bool = True,
    trust_remote_code: bool = False,
) -> Path:
    """Convert and save a MindSurf checkpoint as a Hugging Face Llama model."""
    resolved_config, source_state, _ = load_checkpoint(checkpoint_path, model_config)
    if resolved_config.qk_norm:
        raise ValueError("Llama export cannot represent QK normalization; use --target qwen3")

    tokenizer = None
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        if len(tokenizer) != resolved_config.vocab_size:
            raise ValueError(
                "Tokenizer size must match model vocab_size: "
                f"tokenizer={len(tokenizer)}, model={resolved_config.vocab_size}"
            )

    llama_config = build_llama_config(resolved_config, dtype, tokenizer)
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


def _write_legacy_rope_theta(output_dir: Path, rope_theta: float) -> None:
    config_path = output_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.setdefault("rope_theta", rope_theta)
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_exported_qwen3(output_dir: Path) -> Qwen3ForCausalLM:
    major = int(transformers.__version__.split(".", 1)[0])
    load_kwargs: dict[str, Any] = (
        {"dtype": torch.float32} if major >= 5 else {"torch_dtype": torch.float32}
    )

    model = AutoModelForCausalLM.from_pretrained(output_dir, **load_kwargs)
    if not isinstance(model, Qwen3ForCausalLM):
        raise TypeError(f"Expected Qwen3ForCausalLM after reload, got {type(model).__name__}")
    return model.eval()


def convert_checkpoint_to_qwen3(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    fallback_config: ModelConfig | None = None,
    tokenizer_path: str | Path | None = None,
    dtype: torch.dtype = torch.float32,
    tolerance: float = 0.0,
    safe_serialization: bool = True,
    trust_remote_code: bool = False,
) -> Path:
    """Convert, validate, save, and reload a MindSurf checkpoint as Qwen3."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    config, source_state, blob = load_checkpoint(
        checkpoint_path,
        fallback_config,
    )

    if not config.qk_norm:
        raise ValueError("Qwen3 export requires qk_norm=True")

    tokenizer = None
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        if len(tokenizer) != config.vocab_size:
            raise ValueError(f"Tokenizer size {len(tokenizer)} != {config.vocab_size}")

    source_model = TransformerLM(config)
    source_model.load_state_dict(source_state, strict=True)
    source_model.eval().float()

    target_model = Qwen3ForCausalLM(build_qwen3_config(config, tokenizer))

    converted_state = convert_qwen3_state_dict(source_state, target_model)
    target_model.load_state_dict(converted_state, strict=True)
    target_model.eval().float()

    difference = max_logit_difference(source_model, target_model)
    if difference > tolerance:
        raise ValueError(f"Qwen3 logits differ by {difference:.6e}; tolerance={tolerance:.6e}")

    target_model.to(dtype=dtype)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_model.save_pretrained(output, safe_serialization=safe_serialization)

    if tokenizer is not None:
        tokenizer.save_pretrained(output)

    _write_legacy_rope_theta(output, config.rope_theta)

    del target_model
    reloaded_model = _load_exported_qwen3(output)
    reloaded_difference = max_logit_difference(source_model, reloaded_model)
    if reloaded_difference > tolerance:
        raise ValueError(
            f"Reloaded Qwen3 logits differ by {reloaded_difference:.6e}; tolerance={tolerance:.6e}"
        )

    release = blob.get("release")
    if isinstance(release, dict):
        (output / "mindsurf_release.json").write_text(
            json.dumps(
                release,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a MindSurf checkpoint to Hugging Face Qwen3 or Llama format"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the MindSurf checkpoint")
    parser.add_argument(
        "--model-config",
        help="Fallback configs/model/*.yaml for checkpoints without embedded model_config",
    )
    parser.add_argument("--output-dir", required=True, help="Output Hugging Face model directory")
    parser.add_argument("--tokenizer", help="Tokenizer name or local path to save with the model")
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        help="Output dtype (default: float32 for Qwen3, float16 for Llama)",
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
    parser.add_argument(
        "--target",
        choices=("auto", "qwen3", "llama"),
        default="auto",
        help="Target model format for conversion (default: auto)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed logit difference when converting to Qwen3 (default: 0.0)",
    )
    return parser.parse_args()


def checkpoint_uses_qk_norm(
    state: dict[str, torch.Tensor],
) -> bool:
    return any(name.endswith(".attn.q_norm.weight") for name in state)


def resolve_conversion_target(
    requested_target: str,
    model_config: ModelConfig,
    state: dict[str, torch.Tensor],
) -> str:
    """Resolve auto conversion and reject config/checkpoint architecture mismatches."""
    has_q_norm = checkpoint_uses_qk_norm(state)
    has_k_norm = any(name.endswith(".attn.k_norm.weight") for name in state)
    if has_q_norm != has_k_norm:
        raise ValueError("Checkpoint contains only one of QK normalization weights")
    if model_config.qk_norm != has_q_norm:
        raise ValueError(
            "ModelConfig qk_norm does not match checkpoint weights: "
            f"config={model_config.qk_norm}, checkpoint={has_q_norm}"
        )

    resolved_target = "qwen3" if has_q_norm else "llama"
    if requested_target != "auto" and requested_target != resolved_target:
        raise ValueError(f"Checkpoint requires target {resolved_target}, not {requested_target}")
    return resolved_target


def main() -> None:
    """Run checkpoint conversion from the command line."""
    args = parse_args()
    fallback_config = load_model_config(args.model_config) if args.model_config else None
    model_config, state, _ = load_checkpoint(args.checkpoint, fallback_config)
    target = resolve_conversion_target(args.target, model_config, state)
    dtype = (
        DTYPES[args.dtype]
        if args.dtype
        else (torch.float32 if target == "qwen3" else torch.float16)
    )

    if target == "qwen3":
        output_path = convert_checkpoint_to_qwen3(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            fallback_config=model_config,
            tokenizer_path=args.tokenizer,
            dtype=dtype,
            tolerance=args.tolerance,
            safe_serialization=not args.no_safe_serialization,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        output_path = convert_checkpoint_to_llama(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            model_config=model_config,
            tokenizer_path=args.tokenizer,
            dtype=dtype,
            safe_serialization=not args.no_safe_serialization,
            trust_remote_code=args.trust_remote_code,
        )
    print(f"Converted model saved to: {output_path}")


if __name__ == "__main__":
    main()
