import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.features import DirectLatentFeatureExtractor
from drpo.utils.tensors import add_rank_selection_stats, select_disjoint_pref_indices, select_top_fraction_indices


class TrainingSelectionConfigTest(unittest.TestCase):
    def test_train_wrappers_use_fixed_core_budget(self):
        expected = {
            "scripts/train/draft.sh": ("--num_processes 4", "--gradient_accumulation_steps 8", "--max_train_steps 1000"),
            "scripts/train/dpo.sh": ("--num_processes 4", "--gradient_accumulation_steps 8", "--max_train_steps 1000"),
            "scripts/train/grpo.sh": ("--num_processes 8", "--gradient_accumulation_steps 4", "--max_train_steps 5000"),
            "scripts/train/sdxl_turbo_drpo_mae.sh": ("--num_processes 8", "--gradient_accumulation_steps 8", "--max_train_steps 5000"),
            "scripts/train/sdxl_turbo_drpo_teacher.sh": ("--num_processes 4", "--gradient_accumulation_steps 8", "--max_train_steps 5000"),
        }
        for relative, budget in expected.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            for flag in budget:
                self.assertIn(flag, text, relative)
            self.assertIn("--lr_warmup_steps 0", text, relative)
            self.assertIn("PROJECT_ROOT", text, relative)
            for disallowed in ("${MAX_TRAIN_STEPS", "${BATCHSIZE_GEN", "${LEARNING_RATE", "${RUN_NAME"):
                self.assertNotIn(disallowed, text, relative)

    def test_select_top_fraction_keeps_all_for_one_point_zero(self):
        scores = torch.tensor([0.3, 0.9, 0.1, 0.7])
        indices = select_top_fraction_indices(scores, 1.0)
        self.assertEqual(set(indices.tolist()), {0, 1, 2, 3})

    def test_select_top_fraction_keeps_ceil_for_point_nine(self):
        scores = torch.tensor([0.3, 0.9, 0.1, 0.7, 0.5])
        indices = select_top_fraction_indices(scores, 0.9)
        self.assertEqual(len(indices), 5)
        self.assertEqual(indices[0].item(), 1)

    def test_select_top_fraction_rejects_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "1D"):
            select_top_fraction_indices(torch.ones(2, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "fraction"):
            select_top_fraction_indices(torch.ones(2), -0.1)

    def test_select_disjoint_pref_indices_uses_equal_positive_and_negative_counts(self):
        scores = torch.tensor([0.1, 0.8, 0.3, 0.7, 0.2, 0.6])
        best, worst, feature = select_disjoint_pref_indices(scores, num_pos=2, num_neg=2, feature_top_fraction=1.0)
        self.assertEqual(set(best.tolist()), {1, 3})
        self.assertEqual(set(worst.tolist()), {0, 4})
        self.assertEqual(set(feature.tolist()), set(range(6)))
        self.assertTrue(set(best.tolist()).isdisjoint(set(worst.tolist())))

    def test_rank_selection_stats_report_counts_and_gap(self):
        scores = torch.tensor([0.2, 0.9, 0.1, 0.8])
        best, worst, feature = select_disjoint_pref_indices(scores, num_pos=1, num_neg=1, feature_top_fraction=0.9)
        info = add_rank_selection_stats({}, scores, best, worst, feature, prefix="online")
        self.assertAlmostEqual(info["online_best_score"].item(), 0.9)
        self.assertAlmostEqual(info["online_worst_score"].item(), 0.1)
        self.assertEqual(info["online_selected_pos_count"].item(), 1.0)
        self.assertEqual(info["online_selected_neg_count"].item(), 1.0)
        self.assertGreater(info["online_selected_score_gap"].item(), 0.0)

    def test_direct_latent_feature_extractor_exposes_default_feature_keys(self):
        extractor = DirectLatentFeatureExtractor(patch_pool_sizes=(2, 4), include_spatial_features=True)
        latents = torch.randn(2, 4, 8, 8)
        features = extractor.vector_features(latents, keys=("latent", "latent_mean", "latent_std", "latent_mean_2"))
        self.assertEqual(tuple(features["latent"].shape), (2, 64, 4))
        self.assertEqual(tuple(features["latent_mean"].shape), (2, 1, 4))
        self.assertEqual(tuple(features["latent_mean_2"].shape), (2, 16, 4))


if __name__ == "__main__":
    unittest.main()
