from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.config import DrPOConfig, parse_config
from drpo.features import DirectLatentFeatureExtractor
from drpo.training.trainer import _validate_config, compute_offline_distance_scores
from drpo.utils.tensors import add_rank_selection_stats, select_disjoint_pref_indices, select_top_fraction_indices


def minimal_config(**overrides) -> DrPOConfig:
    values = dict(
        pretrained_model_name_or_path="models/sd-turbo",
        output_dir="outputs/test",
        train_mode="online",
        pairs_jsonl="data/pairs.jsonl",
        prompt_file="data/prompts.txt",
        drifting_mae_path="drifting/mae.pth",
    )
    values.update(overrides)
    return DrPOConfig(**values)


class TrainingSelectionConfigTest(unittest.TestCase):
    def test_parser_defaults_match_current_online_training_budget(self):
        config = parse_config([])
        self.assertEqual(config.max_train_steps, 1000)
        self.assertEqual(config.batchsize_gen, 24)
        self.assertEqual(config.lr_warmup_steps, 0)
        self.assertEqual(config.online_feature_top_fraction, 1.0)
        self.assertEqual(config.num_pos_images, config.num_neg_images)

    def test_train_wrappers_use_fixed_core_budget(self):
        for relative in (
            "scripts/train/drpo.sh",
            "scripts/train/draft.sh",
            "scripts/train/dpo.sh",
            "scripts/train/grpo.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("--num_processes 4", text, relative)
            self.assertIn("--gradient_accumulation_steps 8", text, relative)
            self.assertIn("--max_train_steps 1000", text, relative)
            self.assertIn("--lr_warmup_steps 0", text, relative)
            self.assertIn("PROJECT_ROOT", text, relative)
            for disallowed in ("${MAX_TRAIN_STEPS", "${BATCHSIZE_GEN", "${LEARNING_RATE", "${RUN_NAME"):
                self.assertNotIn(disallowed, text, relative)

    def test_validate_accepts_default_online_anchor_counts(self):
        _validate_config(minimal_config(batchsize_gen=24, num_pos_images=8, num_neg_images=8))

    def test_validate_rejects_invalid_online_feature_fraction(self):
        with self.assertRaisesRegex(ValueError, "online_feature_top_fraction"):
            _validate_config(minimal_config(online_feature_top_fraction=0.0))
        with self.assertRaisesRegex(ValueError, "online_feature_top_fraction"):
            _validate_config(minimal_config(online_feature_top_fraction=1.01))

    def test_validate_rejects_non_positive_anchor_counts(self):
        with self.assertRaisesRegex(ValueError, "num_pos_images"):
            _validate_config(minimal_config(num_pos_images=0))
        with self.assertRaisesRegex(ValueError, "num_pos_images"):
            _validate_config(minimal_config(num_neg_images=0))

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

    def test_offline_distance_separate_zscore_produces_finite_scores(self):
        scores, info = compute_offline_distance_scores(
            cand_feat_dict={"latent": torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]], [[3.0, 3.0]]])},
            pos_feat_dict={"latent": torch.tensor([[[0.0, 0.0]]])},
            neg_feat_dict={"latent": torch.tensor([[[3.0, 3.0]]])},
            feature_keys=("latent",),
            score_mode="separate_zscore",
            normalize="zscore",
            aggregation="mean",
            ref_reduction="mean",
        )
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertIn("offline_distance_score_std", info)

    def test_offline_distance_cosine_softmax_diff_reports_cosine_terms(self):
        scores, info = compute_offline_distance_scores(
            cand_feat_dict={"latent": torch.eye(3).view(3, 1, 3)},
            pos_feat_dict={"latent": torch.tensor([[[1.0, 0.0, 0.0]]])},
            neg_feat_dict={"latent": torch.tensor([[[0.0, 0.0, 1.0]]])},
            feature_keys=("latent",),
            score_mode="cosine_softmax_diff",
            normalize="none",
            aggregation="sum",
            ref_reduction="mean",
        )
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertIn("offline_distance_cos_pos_mean", info)
        self.assertGreater(scores[0].item(), scores[2].item())

    def test_direct_latent_feature_extractor_exposes_default_feature_keys(self):
        extractor = DirectLatentFeatureExtractor(patch_pool_sizes=(2, 4), include_spatial_features=True)
        latents = torch.randn(2, 4, 8, 8)
        features = extractor.vector_features(latents, keys=("latent", "latent_mean", "latent_std", "latent_mean_2"))
        self.assertEqual(tuple(features["latent"].shape), (2, 64, 4))
        self.assertEqual(tuple(features["latent_mean"].shape), (2, 1, 4))
        self.assertEqual(tuple(features["latent_mean_2"].shape), (2, 16, 4))


if __name__ == "__main__":
    unittest.main()
