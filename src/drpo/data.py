from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_prompt_file(path: str | Path, limit: int | None = None) -> list[str]:
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No prompts found in {path}.")
    return prompts


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} must be an object.")
            rows.append(value)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def _image_paths(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        paths = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        paths = value
    else:
        raise TypeError("Expected an image path or a list of image paths.")
    paths = [path for path in paths if path]
    if not paths:
        raise ValueError("Expected at least one image path.")
    return paths


@dataclass(frozen=True)
class Batch:
    prompts: list[str]
    input_ids: torch.Tensor
    chosen: list[torch.Tensor] | None = None
    rejected: list[torch.Tensor] | None = None


class PreferenceDataset(Dataset):
    """JSONL dataset for both offline preference pairs and online prompts."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        mode: str,
        image_size: int,
        max_samples: int | None = None,
        seed: int | None = None,
        empty_prompt_probability: float = 0.0,
    ) -> None:
        if mode not in {"offline", "online", "offline_distance"}:
            raise ValueError(f"Unsupported mode: {mode}.")
        self.path = Path(jsonl_path)
        self.base_dir = self.path.resolve().parent
        self.tokenizer = tokenizer
        self.mode = mode
        self.empty_prompt_probability = empty_prompt_probability
        self.rows = read_jsonl(self.path)
        if max_samples is not None:
            rng = random.Random(seed)
            rng.shuffle(self.rows)
            self.rows = self.rows[:max_samples]
        self.empty_input_ids = self._tokenize("")
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(
            [prompt],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

    def _read_image_tensor(self, path: str) -> torch.Tensor:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.base_dir / resolved
        with Image.open(resolved) as image:
            return self.image_transform(image.convert("RGB"))

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        prompt = str(row.get("prompt", ""))
        input_ids = self.empty_input_ids.clone() if random.random() < self.empty_prompt_probability else self._tokenize(prompt)
        item: dict[str, object] = {"prompt": prompt, "input_ids": input_ids}
        if self.mode in {"offline", "offline_distance"}:
            item["chosen"] = torch.stack([self._read_image_tensor(path) for path in _image_paths(row["chosen"])])
            item["rejected"] = torch.stack([self._read_image_tensor(path) for path in _image_paths(row["rejected"])])
        return item


class PromptDataset(Dataset):
    """Prompt-only dataset for online DrPO candidate ranking."""

    def __init__(self, prompt_file: str | Path, tokenizer, max_samples: int | None = None, seed: int | None = None) -> None:
        self.path = Path(prompt_file)
        self.tokenizer = tokenizer
        self.rows = self._load_rows(max_samples, seed=seed)

    def _load_rows(self, max_samples: int | None, *, seed: int | None) -> list[dict[str, Any]]:
        if self.path.suffix.lower() != ".jsonl":
            rows = [{"prompt": prompt} for prompt in load_prompt_file(self.path, max_samples)]
            if seed is not None:
                random.Random(seed).shuffle(rows)
            return rows
        rows = read_jsonl(self.path)
        if max_samples is not None:
            rows = rows[:max_samples]
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            normalized_rows.append({"prompt": prompt})
        if not normalized_rows:
            raise ValueError(f"No prompts found in {self.path}.")
        if seed is not None:
            random.Random(seed).shuffle(normalized_rows)
        return normalized_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        prompt = str(row["prompt"])
        input_ids = self.tokenizer(
            [prompt],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"prompt": prompt, "input_ids": input_ids}


def collate_preference_batch(items: list[dict[str, object]]) -> Batch:
    chosen = [item["chosen"].float() for item in items] if "chosen" in items[0] else None
    rejected = [item["rejected"].float() for item in items] if "rejected" in items[0] else None
    return Batch(
        prompts=[str(item["prompt"]) for item in items],
        input_ids=torch.stack([item["input_ids"] for item in items]),
        chosen=chosen,
        rejected=rejected,
    )
