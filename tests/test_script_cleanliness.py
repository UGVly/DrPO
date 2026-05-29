from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScriptCleanlinessTest(unittest.TestCase):
    def test_shell_scripts_have_valid_bash_syntax(self):
        for path in sorted((ROOT / "scripts").rglob("*.sh")):
            result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, f"{path.relative_to(ROOT)}\n{result.stderr}")

    def test_shell_scripts_are_portable(self):
        forbidden = (
            "/" + "datapool/",
            "/home/",
            "/Users/",
        )
        offenders: list[str] = []
        for path in sorted((ROOT / "scripts").rglob("*.sh")):
            text = path.read_text(encoding="utf-8")
            if any(needle in text for needle in forbidden):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_train_wrappers_resolve_project_root(self):
        offenders: list[str] = []
        for path in sorted((ROOT / "scripts" / "train").glob("*.sh")):
            text = path.read_text(encoding="utf-8")
            if "PROJECT_ROOT" not in text or 'cd "$PROJECT_ROOT"' not in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_sdxl_teacher_recipe_uses_fixed_defaults(self):
        text = (ROOT / "scripts/train/sdxl_turbo_drpo_teacher.sh").read_text(encoding="utf-8")
        expected = [
            "--feature_extractor teacher_unet",
            "--teacher_feature_layers down_blocks.2,mid_block,up_blocks.0",
            "--teacher_feature_noise 0.1",
            "--batchsize_gen 16",
            "--num_pos_images 8",
            "--num_neg_images 8",
            "--max_train_steps 5000",
            "--learning_rate 1e-5",
        ]
        for needle in expected:
            self.assertIn(needle, text)

    def test_sdxl_drpo_default_wrappers_do_not_enable_extra_regularizers(self):
        disabled_defaults = [
            "--ref_model_l2_weight",
            "--feature_diversity_weight",
            "--feature_diversity_margin_scale",
            "--vgg_anchor_weight",
        ]
        for relative in (
            "scripts/train/sdxl_turbo_drpo_mae.sh",
            "scripts/train/sdxl_turbo_drpo_teacher.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            for needle in disabled_defaults:
                self.assertNotIn(needle, text, relative)


if __name__ == "__main__":
    unittest.main()
