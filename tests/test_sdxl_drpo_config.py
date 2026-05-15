from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.methods.sdxl_drpo import trainer as sdxl_trainer


class _TupleBlock(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        out = x * self.scale
        return out, (out,)


class _TensorBlock(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class _FakeSDXLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([_TensorBlock(1.0), _TensorBlock(1.0), _TupleBlock(2.0)])
        self.mid_block = _TensorBlock(3.0)
        self.up_blocks = nn.ModuleList([_TensorBlock(4.0)])

    def forward(self, sample, timestep, *, encoder_hidden_states, added_cond_kwargs, return_dict=False):
        self.last_timestep = timestep
        self.last_encoder_hidden_states = encoder_hidden_states
        self.last_added_cond_kwargs = added_cond_kwargs
        hidden = sample
        hidden = self.down_blocks[0](hidden)
        hidden = self.down_blocks[1](hidden)
        hidden, _ = self.down_blocks[2](hidden)
        hidden = self.mid_block(hidden)
        hidden = self.up_blocks[0](hidden)
        return (hidden,)


class SDXLDrPOConfigTest(unittest.TestCase):
    def test_parser_defaults_use_three_k_drift_weights(self):
        config = sdxl_trainer.parse_config([])
        self.assertEqual(config.drifting_pos_weight, 3000.0)
        self.assertEqual(config.drifting_neg_weight, 3000.0)
        self.assertEqual(config.drifting_ref_weight, 3000.0)
        self.assertEqual(config.drifting_ref_neg_weight, 3000.0)

    def test_teacher_feature_parser_defaults(self):
        config = sdxl_trainer.parse_config(["--feature_extractor", "teacher_unet"])
        self.assertEqual(config.feature_extractor, "teacher_unet")
        self.assertEqual(config.teacher_feature_layers, ("down_blocks.2", "mid_block", "up_blocks.0"))
        self.assertEqual(config.teacher_feature_noise, 0.1)
        self.assertEqual(config.teacher_feature_timestep, 100)
        self.assertEqual(config.teacher_feature_pool_size, 4)

    def test_wrapper_passes_all_drift_weights(self):
        text = (ROOT / "scripts" / "train" / "sdxl_turbo_drpo_mae.sh").read_text(encoding="utf-8")
        self.assertIn('--drifting_pos_weight 3000.0', text)
        self.assertIn('--drifting_neg_weight 3000.0', text)
        self.assertIn('--drifting_ref_weight 3000.0', text)
        self.assertIn('--drifting_ref_neg_weight 3000.0', text)

    def test_teacher_wrapper_selects_teacher_unet_features(self):
        text = (ROOT / "scripts" / "train" / "sdxl_turbo_drpo_teacher.sh").read_text(encoding="utf-8")
        self.assertIn("--feature_extractor teacher_unet", text)
        self.assertIn('--teacher_feature_layers down_blocks.2,mid_block,up_blocks.0', text)
        self.assertIn('--teacher_feature_noise 0.1', text)
        self.assertIn('--teacher_feature_pool_size 4', text)
        self.assertIn('--learning_rate 1e-5', text)

    def test_teacher_unet_feature_extractor_keeps_generated_gradients(self):
        ref_unet = _FakeSDXLUNet()
        extractor = sdxl_trainer.SDXLTeacherUNetFeatureExtractor(
            ref_unet,
            feature_layers=("down_blocks.2", "mid_block", "up_blocks.0"),
            feature_noise=0.0,
            feature_timestep=100,
            pool_size=2,
        )
        latents = torch.randn(2, 4, 8, 8, requires_grad=True)
        prompt_embeds = torch.randn(2, 3, 6)
        pooled_prompt_embeds = torch.randn(2, 6)
        add_time_ids = torch.randn(2, 6)

        features = extractor.vector_features(
            latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
            require_grad=True,
        )
        self.assertEqual(set(features), {"down_blocks.2", "mid_block", "up_blocks.0"})
        self.assertEqual(tuple(features["down_blocks.2"].shape), (2, 1, 64))
        self.assertTrue(features["mid_block"].requires_grad)
        sum(value.sum() for value in features.values()).backward()
        self.assertIsNotNone(latents.grad)
        self.assertTrue(torch.isfinite(latents.grad).all())

    def test_prompt_terms_use_independent_reference_negative_weight(self):
        calls = []

        def fake_drift_loss(
            generated,
            positive,
            negative,
            *,
            positive_weight,
            negative_weight,
            radii,
            mask_negative_self=False,
            kernel="laplacian",
        ):
            del positive, negative, radii, kernel
            calls.append(
                {
                    "positive_weight": positive_weight,
                    "negative_weight": negative_weight,
                    "mask_negative_self": mask_negative_self,
                }
            )
            return generated.new_ones(generated.shape[0]), {}

        config = sdxl_trainer.SDXLDrPOConfig(
            pretrained_model_name_or_path="models/sdxl-turbo",
            output_dir="outputs/test",
            prompt_file="data/prompts.txt",
            mae_model_name_or_path="models/mae",
            drifting_pos_weight=11.0,
            drifting_neg_weight=22.0,
            drifting_ref_weight=33.0,
            drifting_ref_neg_weight=44.0,
        )
        features = {
            "feat": torch.tensor(
                [
                    [[0.0, 0.0, 0.0]],
                    [[1.0, 1.0, 1.0]],
                ]
            )
        }
        original = sdxl_trainer.drift_loss
        try:
            sdxl_trainer.drift_loss = fake_drift_loss
            metrics = sdxl_trainer._compute_prompt_terms(
                ("feat",),
                generated_features=features,
                reference_features=features,
                positive_features=features,
                negative_features=features,
                config=config,
            )
        finally:
            sdxl_trainer.drift_loss = original

        self.assertTrue(metrics["loss"].isfinite())
        self.assertEqual(calls[0]["positive_weight"], 11.0)
        self.assertEqual(calls[0]["negative_weight"], 22.0)
        self.assertFalse(calls[0]["mask_negative_self"])
        self.assertEqual(calls[1]["positive_weight"], 33.0)
        self.assertEqual(calls[1]["negative_weight"], 44.0)
        self.assertTrue(calls[1]["mask_negative_self"])


if __name__ == "__main__":
    unittest.main()
