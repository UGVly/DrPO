from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.config import DrPOConfig
from drpo.training.trainer import (
    _compute_prompt_terms,
    _make_tracker_safe,
    _sample_geneval_pref_indices,
    _validate_config,
    compute_offline_distance_scores,
)


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


class TrainerConfigTest(unittest.TestCase):
    def test_online_anchor_count_must_fit_generated_batch(self):
        config = minimal_config(batchsize_gen=4, num_pos_images=3, num_neg_images=2)
        with self.assertRaisesRegex(ValueError, "num_pos_images"):
            _validate_config(config)

    def test_geneval_can_span_multiple_rollout_rounds(self):
        config = minimal_config(
            choice_model="geneval",
            batchsize_gen=4,
            num_pos_images=4,
            num_neg_images=4,
            geneval_max_rollout_rounds=2,
        )
        _validate_config(config)

    def test_geneval_uses_min_positive_requirement_for_validation(self):
        config = minimal_config(
            choice_model="geneval",
            batchsize_gen=8,
            geneval_max_rollout_rounds=3,
            num_pos_images_min=8,
            num_pos_images=16,
            num_neg_images=16,
        )
        _validate_config(config)

    def test_online_feature_top_count_must_fit_after_negative_anchors(self):
        config = minimal_config(
            choice_model="geneval",
            batchsize_gen=8,
            geneval_max_rollout_rounds=4,
            num_pos_images_min=8,
            num_pos_images=16,
            num_neg_images=16,
            online_feature_top_count=32,
        )
        with self.assertRaisesRegex(ValueError, "online_feature_top_count"):
            _validate_config(config)

    def test_geneval_sampling_draws_from_positive_and_negative_pools(self):
        torch.manual_seed(0)
        raw_scores = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        best_idx, worst_idx, feature_top_idx = _sample_geneval_pref_indices(
            raw_scores,
            num_pos_min=2,
            num_pos=2,
            num_neg=2,
            feature_top_count=2,
            feature_top_fraction=0.5,
        )
        self.assertTrue(torch.all(raw_scores.index_select(0, best_idx) >= 0.5))
        self.assertTrue(torch.all(raw_scores.index_select(0, worst_idx) < 0.5))
        self.assertFalse(set(feature_top_idx.tolist()) & set(worst_idx.tolist()))

    def test_geneval_sampling_caps_positive_count_and_feature_count(self):
        torch.manual_seed(0)
        raw_scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        best_idx, worst_idx, feature_top_idx = _sample_geneval_pref_indices(
            raw_scores,
            num_pos_min=2,
            num_pos=3,
            num_neg=2,
            feature_top_count=2,
            feature_top_fraction=1.0,
        )
        self.assertEqual(best_idx.numel(), 3)
        self.assertEqual(worst_idx.numel(), 2)
        self.assertEqual(feature_top_idx.numel(), 2)

    def test_geneval_sampling_requires_exact_feature_count_when_configured(self):
        raw_scores = torch.tensor([1.0, 1.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "optimization candidates"):
            _sample_geneval_pref_indices(
                raw_scores,
                num_pos_min=2,
                num_pos=2,
                num_neg=2,
                feature_top_count=3,
                feature_top_fraction=1.0,
            )

    def test_resolution_must_be_latent_compatible(self):
        config = minimal_config(resolution=510)
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            _validate_config(config)

    def test_offline_distance_top_fraction_must_be_valid(self):
        config = minimal_config(train_mode="offline_distance", offline_distance_top_fraction=0.0)
        with self.assertRaisesRegex(ValueError, "offline_distance_top_fraction"):
            _validate_config(config)

    def test_loss_accepts_anchor_sets_with_different_counts(self):
        config = minimal_config()
        feature_keys = ("latent",)
        generated_features = {"latent": torch.randn(2, 4, 2, 2).flatten(1).unsqueeze(1)}
        reference_features = {"latent": torch.randn(2, 4, 2, 2).flatten(1).unsqueeze(1)}
        positive_features = {"latent": torch.randn(1, 4, 2, 2).flatten(1).unsqueeze(1)}
        negative_features = {"latent": torch.randn(1, 4, 2, 2).flatten(1).unsqueeze(1)}
        metrics = _compute_prompt_terms(
            feature_keys,
            generated_features,
            reference_features,
            positive_features,
            negative_features,
            config=config,
        )
        self.assertTrue(metrics["loss"].isfinite())
        self.assertIn("pref_loss", metrics)

    def test_compute_offline_distance_scores_prefers_closer_positive_and_farther_negative(self):
        scores, info = compute_offline_distance_scores(
            cand_feat_dict={
                "latent": torch.tensor(
                    [
                        [[0.0, 0.0]],
                        [[2.0, 2.0]],
                        [[4.0, 4.0]],
                    ]
                )
            },
            pos_feat_dict={"latent": torch.tensor([[[0.0, 0.0]]])},
            neg_feat_dict={"latent": torch.tensor([[[4.0, 4.0]]])},
            feature_keys=("latent",),
            score_mode="margin",
            normalize="none",
            aggregation="mean",
            ref_reduction="mean",
        )
        self.assertGreater(scores[0].item(), scores[1].item())
        self.assertGreater(scores[1].item(), scores[2].item())
        self.assertIn("offline_distance_margin_mean", info)

    def test_generation_target_timestep_must_not_exceed_generation_timestep(self):
        config = minimal_config(generation_timestep=10, generation_target_timestep=11)
        with self.assertRaisesRegex(ValueError, "generation_target_timestep"):
            _validate_config(config)

    def test_drifting_kernel_must_be_known(self):
        config = minimal_config(drifting_kernel="unknown")
        with self.assertRaisesRegex(ValueError, "drifting_kernel"):
            _validate_config(config)

    def test_geneval_requires_online_mode(self):
        with self.assertRaisesRegex(ValueError, "train_mode=online"):
            _validate_config(minimal_config(choice_model="geneval", train_mode="offline"))

    def test_geneval_allows_periodic_eval(self):
        _validate_config(minimal_config(choice_model="geneval", eval_every_steps=10))

    def test_tracker_config_serializes_non_scalar_values(self):
        self.assertEqual(_make_tracker_safe(("a", 1, True)), '["a", 1, true]')
        self.assertEqual(_make_tracker_safe({"k": (1, 2)}), '{"k": [1, 2]}')


if __name__ == "__main__":
    unittest.main()
