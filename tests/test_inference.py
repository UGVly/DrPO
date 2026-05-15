from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.common import (
    build_tasks,
    checkpoint_output_dir,
    latent_seed_for,
    manifest_rows,
    output_dir_needs_resample,
    parse_int_list,
    read_prompts,
)
from inference.discover import discover_checkpoints


class InferenceUtilityTest(unittest.TestCase):
    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("42,43, 44"), (42, 43, 44))

    def test_prompt_tasks_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "prompts.txt"
            prompt_path.write_text("first\nsecond\n", encoding="utf-8")
            prompts = read_prompts(prompt_path)
            tasks = build_tasks(Path(tmp) / "samples", prompts, (42, 43))
            self.assertEqual(len(tasks), 4)
            self.assertEqual(tasks[0].prompt_id, 0)
            self.assertEqual(tasks[0].latent_seed, latent_seed_for(0, 42))
            self.assertEqual(tasks[-1].image_path.name, "000001.png")

    def test_discover_and_output_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "outputs" / "dpo" / "online" / "latent" / "run1" / "checkpoint-100"
            lora = checkpoint / "unet_lora"
            lora.mkdir(parents=True)
            (lora / "adapter_model.safetensors").write_bytes(b"fake")
            self.assertEqual(discover_checkpoints(root / "outputs"), [checkpoint])
            output = checkpoint_output_dir(root / "samples", checkpoint, outputs_dir=root / "outputs")
            self.assertEqual(output, root / "samples" / "sd-turbo-lora" / "dpo" / "online" / "latent" / "run1" / "checkpoint-100")

    def test_old_manifest_requests_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "manifest.jsonl").write_text('{"model_type":"sd-turbo-baseline"}\n', encoding="utf-8")
            self.assertTrue(output_dir_needs_resample(output_dir))

            prompt_path = output_dir / "prompts.txt"
            prompt_path.write_text("first\n", encoding="utf-8")
            rows = manifest_rows(build_tasks(output_dir, read_prompts(prompt_path), (42,)), model_type="sd-turbo-baseline")
            (output_dir / "manifest.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            self.assertFalse(output_dir_needs_resample(output_dir))


if __name__ == "__main__":
    unittest.main()
