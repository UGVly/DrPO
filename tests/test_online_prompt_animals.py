from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.data import PromptDataset, collate_preference_batch, load_prompt_file


class FakeTokenizer:
    model_max_length = 8

    def __call__(self, prompts, *, max_length, padding, truncation, return_tensors):
        del padding, truncation, return_tensors
        rows = []
        for prompt in prompts:
            values = [min(ord(char), 255) for char in prompt[:max_length]]
            values.extend([0] * (max_length - len(values)))
            rows.append(values)
        return type("Tokenized", (), {"input_ids": torch.tensor(rows, dtype=torch.long)})


class OnlineAnimalPromptTest(unittest.TestCase):
    def setUp(self):
        self.prompt_file = ROOT / "data" / "prompts" / "simple_animals.txt"
        self.wrapper = ROOT / "scripts" / "drpo" / "online" / "prompt_dataset" / "simple_animals.sh"
        self.aes_wrapper = ROOT / "scripts" / "drpo" / "online" / "prompt_dataset" / "simple_animals_aes.sh"

    def test_simple_animals_prompt_file_exists(self):
        self.assertTrue(self.prompt_file.is_file(), f"missing animal prompt file: {self.prompt_file}")

    def test_simple_animals_prompt_file_has_expected_size(self):
        prompts = load_prompt_file(self.prompt_file)
        self.assertEqual(len(prompts), 45)
        self.assertEqual(len(set(prompts)), 45)
        self.assertTrue(all(prompt and prompt == prompt.strip() for prompt in prompts))

    def test_prompt_loader_limit_keeps_original_order(self):
        prompts = load_prompt_file(self.prompt_file, limit=3)
        self.assertEqual(prompts, ["cat", "dog", "horse"])

    def test_prompt_dataset_tokenizes_animal_prompts(self):
        dataset = PromptDataset(self.prompt_file, FakeTokenizer(), max_samples=5)
        self.assertEqual(len(dataset), 5)
        item = dataset[0]
        self.assertEqual(item["prompt"], "cat")
        self.assertEqual(tuple(item["input_ids"].shape), (8,))

    def test_prompt_dataset_seed_shuffles_deterministically(self):
        dataset_a = PromptDataset(self.prompt_file, FakeTokenizer(), max_samples=5, seed=123)
        dataset_b = PromptDataset(self.prompt_file, FakeTokenizer(), max_samples=5, seed=123)
        dataset_c = PromptDataset(self.prompt_file, FakeTokenizer(), max_samples=5, seed=456)
        prompts_a = [dataset_a[i]["prompt"] for i in range(len(dataset_a))]
        prompts_b = [dataset_b[i]["prompt"] for i in range(len(dataset_b))]
        prompts_c = [dataset_c[i]["prompt"] for i in range(len(dataset_c))]
        self.assertEqual(prompts_a, prompts_b)
        self.assertNotEqual(prompts_a, prompts_c)

    def test_prompt_dataset_collates_as_online_batch(self):
        dataset = PromptDataset(self.prompt_file, FakeTokenizer(), max_samples=2)
        batch = collate_preference_batch([dataset[0], dataset[1]])
        self.assertEqual(batch.prompts, ["cat", "dog"])
        self.assertEqual(tuple(batch.input_ids.shape), (2, 8))
        self.assertIsNone(batch.chosen)
        self.assertIsNone(batch.rejected)

    def test_simple_animals_wrapper_points_to_prompt_file(self):
        text = self.wrapper.read_text(encoding="utf-8")
        self.assertIn("data/prompts/simple_animals.txt", text)
        self.assertIn("outputs/drpo/online/prompt_dataset/simple_animals", text)
        self.assertIn("accelerate launch", text)
        self.assertIn("src/drpo/training/sdturbo_lora.py", text)

    def test_simple_animals_aes_wrapper_selects_aes_reward(self):
        text = self.aes_wrapper.read_text(encoding="utf-8")
        self.assertIn("CHOICE_MODEL=\"${CHOICE_MODEL:-aes}\"", text)
        self.assertIn("outputs/drpo/online/prompt_dataset/simple_animals_aes", text)
        self.assertIn("accelerate launch", text)
        self.assertIn("src/drpo/training/sdturbo_lora.py", text)

    def test_prompt_dataset_wrappers_have_valid_bash_syntax(self):
        for path in (self.wrapper, self.aes_wrapper):
            result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
