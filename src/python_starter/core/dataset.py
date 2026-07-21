"""JSONL datasets for language-model pretraining and fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PretrainDataset(Dataset[dict[str, torch.Tensor]]):
    """Pretraining dataset containing one ``{"text": ...}`` object per line."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        del stride  # JSONL records are independent samples.
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = cast(
            HFDataset,
            load_dataset("json", data_files=str(data_path), split="train"),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        text = sample.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Pretrain sample {index} must contain non-empty text")
        if self.tokenizer.bos_token_id is None or self.tokenizer.eos_token_id is None:
            raise ValueError("Pretrain tokenizer must define BOS and EOS tokens")

        content_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )["input_ids"]
        token_ids = [self.tokenizer.bos_token_id, *content_ids, self.tokenizer.eos_token_id]
        token_ids += [self._pad_token_id] * (self.max_length + 1 - len(token_ids))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        labels[labels == self._pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

    @property
    def _pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    """SFT dataset containing one ``{"conversations": [...]}`` object per line."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if tokenizer.chat_template is None:
            raise ValueError("SFT tokenizer must define a chat_template")
        if tokenizer.bos_token is None or tokenizer.eos_token is None:
            raise ValueError("SFT tokenizer must define BOS and EOS tokens")

        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features(
            {
                "conversations": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                        "reasoning_content": Value("string"),
                        "tools": Value("string"),
                        "tool_calls": Value("string"),
                    }
                ]
            }
        )
        self.samples = cast(
            HFDataset,
            load_dataset(
                "json",
                data_files=str(data_path),
                split="train",
                features=features,
            ),
        )
        self.assistant_start_ids = self.tokenizer(
            f"{tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        )["input_ids"]
        self.message_end_ids = self.tokenizer(
            f"{tokenizer.eos_token}\n",
            add_special_tokens=False,
        )["input_ids"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        conversations = sample.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"SFT sample {index} must contain conversations")

        prompt = self._create_chat_prompt(conversations)
        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length + 1,
            padding="max_length",
        )["input_ids"]
        labels = self._generate_labels(input_ids)
        if all(label == -100 for label in labels):
            raise ValueError(f"SFT sample {index} has no trainable assistant response")

        return {
            "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:], dtype=torch.long),
        }

    def _create_chat_prompt(self, conversations: list[Any]) -> str:
        messages: list[dict[str, Any]] = []
        tools = None
        for raw_message in conversations:
            message = dict(raw_message)
            raw_tools = message.pop("tools", None)
            if message.get("role") == "system" and raw_tools:
                tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, str) and tool_calls:
                message["tool_calls"] = json.loads(tool_calls)
            messages.append(message)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=cast(Any, tools),
        )
        if not isinstance(prompt, str):
            raise TypeError("chat template must render a string")
        return prompt

    def _generate_labels(self, input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        index = 0

        while index < len(input_ids):
            if input_ids[index : index + len(self.assistant_start_ids)] != self.assistant_start_ids:
                index += 1
                continue

            start = index + len(self.assistant_start_ids)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(self.message_end_ids)] == self.message_end_ids:
                    end += len(self.message_end_ids)
                    break
                end += 1

            labels[start:end] = input_ids[start:end]
            index = end

        return labels


# Backward-compatible name used by existing scripts.
TextDataset = PretrainDataset


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack fixed-length samples into a batch."""
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
    }
