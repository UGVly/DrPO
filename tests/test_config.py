from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.config import parse_config


class ConfigTest(unittest.TestCase):
    def test_csv_arguments_are_parsed(self):
        config = parse_config(
            [
                "--lora_target_modules",
                "to_q,to_v",
                "--drifting_pref_r_list",
                "0.1,0.2",
                "--drifting_feature_patch_sizes",
                "2,4",
                "--prompt_file",
                "prompts.txt",
                "--choice_models",
                "clip,pickscore",
                "--choice_model_weights",
                "0.25,0.75",
                "--lr_scheduler",
                "constant",
                "--lr_warmup_steps",
                "12",
                "--drifting_kernel",
                "rbf",
            ]
        )
        self.assertEqual(config.lora_target_modules, ("to_q", "to_v"))
        self.assertEqual(config.drifting_pref_r_list, (0.1, 0.2))
        self.assertEqual(config.drifting_feature_patch_sizes, (2, 4))
        self.assertEqual(config.prompt_file, "prompts.txt")
        self.assertEqual(config.choice_models, ("clip", "pickscore"))
        self.assertEqual(config.choice_model_weights, (0.25, 0.75))
        self.assertEqual(config.lr_scheduler, "constant")
        self.assertEqual(config.lr_warmup_steps, 12)
        self.assertEqual(config.drifting_kernel, "rbf")
        self.assertTrue(config.use_lora)

    def test_lora_can_be_disabled(self):
        config = parse_config(["--no_use_lora"])
        self.assertFalse(config.use_lora)

    def test_lora_and_full_entrypoints_force_training_mode(self):
        self.assertTrue(parse_config(["--no_use_lora"], force_use_lora=True).use_lora)
        self.assertFalse(parse_config(["--use_lora"], force_use_lora=False).use_lora)

    def test_resume_checkpoint_argument_is_parsed(self):
        config = parse_config(["--resume_from_checkpoint", "outputs/run/checkpoint-100"])
        self.assertEqual(config.resume_from_checkpoint, "outputs/run/checkpoint-100")

    def test_dino_feature_extractor_uses_dino_default_keys(self):
        config = parse_config(["--drifting_feature_extractor", "dino"])
        self.assertEqual(config.drifting_feature_extractor, "dino")
        self.assertEqual(config.drifting_feature_key, "layer12_patch_mean")
        self.assertIn("layer12_patch_mean", config.drifting_feature_keys)
        self.assertIn("layer12_cls", config.drifting_feature_keys)
        self.assertNotIn("layer12_patch_patch2_mean", config.drifting_feature_keys)
        self.assertNotIn("layer4_mean", config.drifting_feature_keys)

    def test_dino_feature_keys_can_be_overridden(self):
        config = parse_config(
            [
                "--drifting_feature_extractor",
                "dino",
                "--drifting_feature_key",
                "layer12_patch_mean",
                "--drifting_feature_keys",
                "layer12_patch_mean,layer12_cls",
            ]
        )
        self.assertEqual(config.drifting_feature_key, "layer12_patch_mean")
        self.assertEqual(config.drifting_feature_keys, ("layer12_patch_mean", "layer12_cls"))

    def test_latent_feature_extractor_uses_latent_default_keys(self):
        config = parse_config(["--drifting_feature_extractor", "latent"])
        self.assertEqual(config.drifting_feature_extractor, "latent")
        self.assertEqual(config.drifting_feature_key, "latent")
        self.assertEqual(
            config.drifting_feature_keys,
            ("latent", "latent_mean", "latent_std", "latent_mean_2", "latent_std_2", "latent_mean_4", "latent_std_4"),
        )

    def test_offline_distance_arguments_are_parsed(self):
        config = parse_config(
            [
                "--train_mode",
                "offline_distance",
                "--offline_distance_top_fraction",
                "0.25",
                "--offline_distance_score_normalize",
                "none",
                "--offline_distance_score_aggregation",
                "sum",
                "--offline_distance_ref_reduction",
                "min",
                "--offline_latent_encode_mode",
                "sample",
                "--offline_distance_score_mode",
                "separate_zscore",
                "--offline_distance_use_data_anchors",
            ]
        )
        self.assertEqual(config.train_mode, "offline_distance")
        self.assertEqual(config.offline_distance_top_fraction, 0.25)
        self.assertEqual(config.offline_distance_score_mode, "separate_zscore")
        self.assertEqual(config.offline_distance_score_normalize, "none")
        self.assertEqual(config.offline_distance_score_aggregation, "sum")
        self.assertEqual(config.offline_distance_ref_reduction, "min")
        self.assertEqual(config.offline_latent_encode_mode, "sample")
        self.assertTrue(config.offline_distance_use_data_anchors)

    def test_geneval_arguments_are_parsed(self):
        config = parse_config(
            [
                "--choice_model",
                "geneval",
                "--prompt_file",
                "geneval.jsonl",
                "--geneval_repo",
                "/tmp/geneval",
                "--geneval_detector_path",
                "/tmp/geneval_detector",
                "--geneval_model_config",
                "/tmp/model.py",
                "--geneval_options",
                "backend=modern,threshold=0.5",
                "--geneval_max_rollout_rounds",
                "6",
                "--num_pos_images_min",
                "8",
                "--num_pos_images",
                "16",
                "--num_neg_images",
                "16",
                "--online_feature_top_count",
                "32",
            ]
        )
        self.assertEqual(config.choice_model, "geneval")
        self.assertEqual(config.prompt_file, "geneval.jsonl")
        self.assertEqual(config.geneval_repo, "/tmp/geneval")
        self.assertEqual(config.geneval_detector_path, "/tmp/geneval_detector")
        self.assertEqual(config.geneval_model_config, "/tmp/model.py")
        self.assertEqual(config.geneval_options, "backend=modern,threshold=0.5")
        self.assertEqual(config.geneval_max_rollout_rounds, 6)
        self.assertEqual(config.num_pos_images_min, 8)
        self.assertEqual(config.num_pos_images, 16)
        self.assertEqual(config.num_neg_images, 16)
        self.assertEqual(config.online_feature_top_count, 32)

if __name__ == "__main__":
    unittest.main()
