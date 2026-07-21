"""Dataset parsing tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase

from python_starter.core.dataset import PretrainDataset, SFTDataset, collate_fn


class _FakeTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "test-template"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        max_length: int | None = None,
        padding: str | None = None,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        token_ids = [ord(character) for character in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        if padding == "max_length" and max_length is not None:
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        return {"input_ids": token_ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: Any,
    ) -> str:
        del tokenize, add_generation_prompt, tools
        return "".join(
            f"{self.bos_token}{message['role']}\n{message['content']}{self.eos_token}\n"
            for message in messages
        )


def _tokenizer() -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, _FakeTokenizer())


def test_pretrain_dataset_reads_jsonl_and_shifts_labels(tmp_path: Path) -> None:
    data_path = tmp_path / "pretrain.jsonl"
    data_path.write_text(json.dumps({"text": "abc"}), encoding="utf-8")

    sample = PretrainDataset(data_path, _tokenizer(), max_length=6)[0]

    torch.testing.assert_close(sample["input_ids"], torch.tensor([1, 97, 98, 99, 2, 0]))
    torch.testing.assert_close(sample["labels"], torch.tensor([97, 98, 99, 2, -100, -100]))


def test_sft_dataset_masks_non_assistant_tokens(tmp_path: Path) -> None:
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    sample = SFTDataset(data_path, _tokenizer(), max_length=64)[0]

    assert sample["input_ids"].shape == sample["labels"].shape == (64,)
    assert (sample["labels"] == -100).any()
    assert (sample["labels"] != -100).any()


def test_collate_fn_stacks_samples() -> None:
    batch = [
        {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([2, 3])},
        {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([5, 6])},
    ]

    result = collate_fn(batch)

    torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2], [4, 5]]))
    torch.testing.assert_close(result["labels"], torch.tensor([[2, 3], [5, 6]]))
