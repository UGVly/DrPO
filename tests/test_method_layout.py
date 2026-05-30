from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class MethodLayoutTest(unittest.TestCase):
    def test_method_packages_exist(self):
        for method in ("sdxl_draft", "sdxl_drpo", "sdxl_grpo"):
            path = ROOT / "src" / "drpo" / "methods" / method / "__init__.py"
            self.assertTrue(path.is_file(), path)

    def test_train_scripts_exist(self):
        for script in ("draft", "dpo", "grpo", "sdxl_turbo_draft", "sdxl_turbo_drpo_mae", "sdxl_turbo_drpo_teacher", "sdxl_turbo_grpo", "spo", "vggflow"):
            path = ROOT / "scripts" / "train" / f"{script}.sh"
            self.assertTrue(path.is_file(), path)
        self.assertFalse((ROOT / "scripts" / "train" / "neighbor_grpo.sh").exists())
        self.assertFalse((ROOT / "scripts" / "train" / "drpo.sh").exists())
        self.assertFalse((ROOT / "scripts" / "train" / "drpo_full.sh").exists())

    def test_baseline_implementations_are_outside_src(self):
        expected = {
            "draft": {"train_lora.py"},
            "dpo": {"train_lora.py"},
            "grpo": {"train_lora.py", "trainer.py", "losses.py"},
            "spo": {"train_lora.py", "losses.py"},
            "vggflow": {"train_lora.py", "reward_gradient.py"},
        }
        self.assertFalse((ROOT / "baselines" / "neighbor_grpo").exists())
        for method, filenames in expected.items():
            method_dir = ROOT / "baselines" / method
            self.assertTrue(method_dir.is_dir(), method_dir)
            actual = {path.name for path in method_dir.glob("*.py") if path.name != "__init__.py"}
            self.assertEqual(actual, filenames)

    def test_train_scripts_point_to_canonical_trainers(self):
        expected = {
            "dpo": "baselines/dpo/train_lora.py",
            "draft": "baselines/draft/train_lora.py",
            "grpo": "baselines/grpo/train_lora.py",
            "spo": "baselines/spo/train_lora.py",
            "vggflow": "baselines/vggflow/train_lora.py",
            "sdxl_turbo_draft": "src/drpo/methods/sdxl_draft/trainer.py",
            "sdxl_turbo_drpo_mae": "src/drpo/methods/sdxl_drpo/trainer.py",
            "sdxl_turbo_drpo_teacher": "src/drpo/methods/sdxl_drpo/trainer.py",
            "sdxl_turbo_grpo": "src/drpo/methods/sdxl_grpo/trainer.py",
        }
        for script, needle in expected.items():
            text = (ROOT / "scripts" / "train" / f"{script}.sh").read_text(encoding="utf-8")
            self.assertIn(needle, text)

    def test_open_clip_is_not_vendored(self):
        self.assertFalse((ROOT / "src" / "open_clip").exists())

    def test_baseline_trainers_are_lora_only(self):
        for path in sorted((ROOT / "baselines").glob("*/train_lora.py")):
            text = path.read_text(encoding="utf-8")
            trainer = path.with_name("trainer.py")
            if trainer.is_file() and "argparse" not in text:
                text = trainer.read_text(encoding="utf-8")
            self.assertRegex(text, re.compile(r'--use_lora", action=(argparse\.)?BooleanOptionalAction|--use_lora", action="store_true", default=True'))
            self.assertNotIn("--no_use_lora", text)

    def test_canonical_trainer_specs_resolve(self):
        for module in (
            "drpo.methods.sdxl_draft.trainer",
            "drpo.methods.sdxl_drpo.trainer",
            "drpo.methods.sdxl_grpo.trainer",
        ):
            self.assertIsNotNone(importlib.util.find_spec(module), module)

    def test_legacy_sdturbo_training_stack_is_removed(self):
        self.assertFalse((ROOT / "src" / "drpo" / "training" / "__init__.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "training" / "trainer.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "training" / "sdturbo_lora.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "training" / "sdturbo_full.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "config.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "shared.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "dino_features.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "methods" / "_baseline_loader.py").exists())
        for method in ("draft", "dpo", "grpo", "neighbor_grpo", "spo", "vggflow"):
            self.assertFalse((ROOT / "src" / "drpo" / "methods" / method / "__init__.py").exists())
            self.assertFalse((ROOT / "src" / "drpo" / "methods" / method / "trainer.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "methods" / "drpo" / "__init__.py").exists())
        self.assertFalse((ROOT / "src" / "drpo" / "methods" / "drpo" / "trainer.py").exists())


if __name__ == "__main__":
    unittest.main()
