from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GenevalLayoutTest(unittest.TestCase):
    def test_geneval_train_wrappers_exist(self):
        for relative in (
            "scripts/drpo/train_drpo_geneval_sdturbo_lora.sh",
            "scripts/train/drpo-geneval.sh",
            "scripts/geneval_score_worker.py",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_geneval_shell_wrappers_have_valid_bash_syntax(self):
        for relative in (
            "scripts/drpo/train_drpo_geneval_sdturbo_lora.sh",
            "scripts/train/drpo-geneval.sh",
        ):
            result = subprocess.run(["bash", "-n", str(ROOT / relative)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_geneval_defaults_use_vit_l14_clip(self):
        utils_text = (ROOT / "src" / "utils" / "geneval_utils.py").read_text(encoding="utf-8")
        worker_text = (ROOT / "scripts" / "geneval_score_worker.py").read_text(encoding="utf-8")
        wrapper_text = (ROOT / "scripts" / "drpo" / "train_drpo_geneval_sdturbo_lora.sh").read_text(encoding="utf-8")
        self.assertIn('clip_model", "ViT-L-14"', utils_text)
        self.assertIn('clip_model", "ViT-L-14"', worker_text)
        self.assertNotIn('GENEVAL_OPTIONS="${GENEVAL_OPTIONS:-backend=modern}"', wrapper_text)


if __name__ == "__main__":
    unittest.main()
