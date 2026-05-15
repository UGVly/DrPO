from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScriptCleanlinessTest(unittest.TestCase):
    def test_shell_scripts_are_direct_fixed_recipes(self):
        forbidden = re.compile(
            r"\$\{|SCRIPT_DIR|PROJECT_ROOT|TIMESTAMP|TRAIN_SCRIPT|TRAIN_MODULE|BASH_SOURCE|"
            r"dirname|date \+%Y%m%d|exec bash|bash .*\.sh|"
            r"^\s*(source|\.)\s+|^\s*(function\s+\w+|\w+\s*\(\)\s*\{)|cmd=\(",
            re.MULTILINE,
        )
        offenders = []
        for path in sorted((ROOT / "scripts").rglob("*.sh")):
            text = path.read_text(encoding="utf-8")
            if forbidden.search(text):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_train_wrappers_do_not_call_other_shell_scripts(self):
        offenders = []
        nested = re.compile(r"\b(?:bash|sh)\s+[^\n]*\.sh|\./[^\n]*\.sh")
        for path in sorted((ROOT / "scripts" / "train").glob("*.sh")):
            text = path.read_text(encoding="utf-8")
            if nested.search(text):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_sdxl_teacher_recipe_uses_fixed_h200_defaults(self):
        text = (ROOT / "scripts/train/sdxl_turbo_drpo_teacher.sh").read_text(encoding="utf-8")
        expected = [
            "--num_processes 8",
            "--feature_extractor teacher_unet",
            "--teacher_feature_layers down_blocks.2,mid_block,up_blocks.0",
            "--teacher_feature_noise 0.1",
            "--batchsize_gen 24",
            "--num_pos_images 12",
            "--num_neg_images 12",
            "--max_train_steps 5000",
            "--gradient_accumulation_steps 4",
            "--learning_rate 1e-5",
            "--drifting_pos_weight 3000.0",
            "--drifting_neg_weight 3000.0",
            "--drifting_ref_weight 3000.0",
            "--drifting_ref_neg_weight 3000.0",
        ]
        for needle in expected:
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
