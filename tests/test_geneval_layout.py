from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GenevalLayoutTest(unittest.TestCase):
    def test_geneval_score_worker_exists_without_legacy_train_wrapper(self):
        self.assertTrue((ROOT / "scripts/geneval_score_worker.py").is_file())
        self.assertFalse((ROOT / "scripts/train/drpo-geneval.sh").exists())

    def test_geneval_defaults_use_vit_l14_clip(self):
        utils_text = (ROOT / "src" / "utils" / "geneval_utils.py").read_text(encoding="utf-8")
        worker_text = (ROOT / "scripts" / "geneval_score_worker.py").read_text(encoding="utf-8")
        self.assertIn('clip_model", "ViT-L-14"', utils_text)
        self.assertIn('clip_model", "ViT-L-14"', worker_text)


if __name__ == "__main__":
    unittest.main()
