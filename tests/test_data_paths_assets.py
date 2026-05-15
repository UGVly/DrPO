from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.assets import check_assets
from drpo.data import PreferenceDataset, PromptDataset, collate_preference_batch, load_prompt_file, read_jsonl
from drpo.paths import require_local_path, resolve_path


class FakeTokenizer:
    model_max_length = 6

    def __call__(self, prompts, *, max_length, padding, truncation, return_tensors):
        del padding, truncation, return_tensors
        ids = []
        for prompt in prompts:
            values = [min(ord(char), 255) for char in prompt[:max_length]]
            values.extend([0] * (max_length - len(values)))
            ids.append(values)
        return type("Tokenized", (), {"input_ids": torch.tensor(ids, dtype=torch.long)})


class DataPathAssetTest(unittest.TestCase):
    def test_prompt_and_jsonl_loaders_skip_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_file = root / "prompts.txt"
            prompt_file.write_text("\nfirst\n\nsecond\n", encoding="utf-8")
            self.assertEqual(load_prompt_file(prompt_file), ["first", "second"])
            self.assertEqual(load_prompt_file(prompt_file, limit=1), ["first"])

            jsonl = root / "rows.jsonl"
            jsonl.write_text(json.dumps({"prompt": "x"}) + "\n\n", encoding="utf-8")
            self.assertEqual(read_jsonl(jsonl), [{"prompt": "x"}])

    def test_prompt_dataset_and_preference_collate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("RGB", (16, 16), "red").save(root / "images" / "chosen.png")
            Image.new("RGB", (16, 16), "blue").save(root / "images" / "rejected.png")

            prompt_file = root / "prompts.txt"
            prompt_file.write_text("a red square\n", encoding="utf-8")
            prompt_dataset = PromptDataset(prompt_file, FakeTokenizer())
            self.assertEqual(prompt_dataset[0]["prompt"], "a red square")

            rows = root / "pairs.jsonl"
            rows.write_text(
                json.dumps(
                    {
                        "prompt": "a red square",
                        "chosen": "images/chosen.png",
                        "rejected": "images/rejected.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = PreferenceDataset(rows, FakeTokenizer(), "offline", image_size=16)
            batch = collate_preference_batch([dataset[0]])
            self.assertEqual(batch.prompts, ["a red square"])
            self.assertEqual(tuple(batch.input_ids.shape), (1, 6))
            self.assertEqual(tuple(batch.chosen[0].shape), (1, 3, 16, 16))
            self.assertEqual(tuple(batch.rejected[0].shape), (1, 3, 16, 16))

    def test_prompt_dataset_loads_geneval_jsonl_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_file = root / "geneval_prompts.jsonl"
            prompt_file.write_text(
                json.dumps(
                    {
                        "tag": "single_object",
                        "include": [{"class": "cat", "count": 1}],
                        "prompt": "a photo of a cat",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = PromptDataset(prompt_file, FakeTokenizer())
            self.assertTrue(dataset.has_geneval_metadata)
            item = dataset[0]
            self.assertEqual(item["prompt"], "a photo of a cat")
            self.assertEqual(item["geneval_metadata"]["tag"], "single_object")
            batch = collate_preference_batch([item])
            self.assertEqual(batch.geneval_metadata[0]["prompt"], "a photo of a cat")

    def test_local_path_validation_and_asset_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_file = root / "asset.bin"
            local_file.write_text("x", encoding="utf-8")
            self.assertEqual(require_local_path(local_file, description="asset", must_be_file=True), local_file)
            self.assertEqual(resolve_path("asset.bin", root), local_file.resolve())
            with self.assertRaises(FileNotFoundError):
                require_local_path(root / "missing.bin", description="missing", must_be_file=True)

            status = dict((asset.name, exists) for asset, exists in check_assets(root))
            self.assertFalse(any(status.values()))


if __name__ == "__main__":
    unittest.main()
