from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.metrics import (
    ImageTensorDataset,
    batched,
    build_summary,
    discover_manifests,
    feature_cache_path,
    fid_from_features,
    grouped_diversity,
    image_batch_to_tensor,
    image_tensor_loader,
    manifest_metric_dir,
    num_batches,
    open_images,
    pairwise_cosine_distance_mean,
    read_jsonl,
    resolve_image_path,
    scalar_summary,
    write_jsonl,
    write_summary_csv,
)


class MetricsCoreIntegrationTest(unittest.TestCase):
    def _make_image(self, path: Path, color: str = "red") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 16), color).save(path)
        return path

    def _rows(self, tmp: Path) -> list[dict]:
        return [
            {"prompt": "first", "seed": 42, "image_path": str(self._make_image(tmp / "a.png", "red"))},
            {"prompt": "second", "seed": 42, "image_path": str(self._make_image(tmp / "b.png", "blue"))},
            {"prompt": "third", "seed": 43, "image_path": str(self._make_image(tmp / "c.png", "green"))},
        ]

    def test_jsonl_roundtrip_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [{"a": 1}, {"a": 2}])
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            self.assertEqual(read_jsonl(path), [{"a": 1}, {"a": 2}])

    def test_batched_and_num_batches_handle_partial_batches(self):
        rows = [{"i": i} for i in range(5)]
        self.assertEqual(num_batches(5, 2), 3)
        self.assertEqual(num_batches(0, 2), 0)
        self.assertEqual([len(batch) for batch in batched(rows, 2)], [2, 2, 1])

    def test_resolve_image_path_keeps_absolute_paths(self):
        absolute = Path("/tmp/drpo-test-image.png")
        self.assertEqual(resolve_image_path(absolute), absolute)

    def test_image_batch_to_tensor_resizes_and_normalizes(self):
        images = [Image.new("RGB", (10, 20), "red"), Image.new("RGB", (20, 10), "blue")]
        tensor = image_batch_to_tensor(images, size=12)
        self.assertEqual(tuple(tensor.shape), (2, 3, 12, 12))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_image_tensor_dataset_and_loader_preserve_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(Path(tmp))
            dataset = ImageTensorDataset(rows, size=16)
            self.assertEqual(tuple(dataset[0].shape), (3, 16, 16))
            loader = image_tensor_loader(rows, size=16, batch_size=2, device=torch.device("cpu"), num_workers=0, prefetch_factor=2)
            batches = list(loader)
            self.assertEqual([tuple(batch.shape) for batch in batches], [(2, 3, 16, 16), (1, 3, 16, 16)])

    def test_open_images_returns_rgb_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(Path(tmp))[:2]
            images = open_images(rows)
            try:
                self.assertEqual([image.mode for image in images], ["RGB", "RGB"])
            finally:
                for image in images:
                    image.close()

    def test_feature_cache_path_depends_on_manifest_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"a": 1}\n', encoding="utf-8")
            first = feature_cache_path(root / "cache", prefix="dino", manifest=manifest)
            manifest.write_text('{"a": 1, "b": 2}\n', encoding="utf-8")
            second = feature_cache_path(root / "cache", prefix="dino", manifest=manifest)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("dino-"))

    def test_pairwise_cosine_distance_and_grouped_diversity(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        rows = [{"seed": 1}, {"seed": 1}, {"seed": 2}]
        self.assertAlmostEqual(pairwise_cosine_distance_mean(features[:1]), 0.0)
        global_value, per_seed = grouped_diversity(rows, features)
        self.assertGreater(global_value, 0.0)
        self.assertIn("1", per_seed)
        self.assertIn("2", per_seed)
        self.assertEqual(per_seed["2"], 0.0)

    def test_fid_is_zero_for_identical_features_and_positive_for_shifted_features(self):
        ref = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        self.assertLess(fid_from_features(ref, ref), 1e-8)
        shifted = ref + 2.0
        self.assertGreater(fid_from_features(ref, shifted), 0.1)

    def test_scalar_and_reward_summary(self):
        summary = build_summary(
            [
                {"model_type": "x", "checkpoint_path": "ckpt", "pickscore": 1.0, "clip": 0.2, "aes": 5.0, "hpsv2": 0.1},
                {"model_type": "x", "checkpoint_path": "ckpt", "pickscore": 3.0, "clip": 0.4, "aes": 7.0, "hpsv2": 0.3, "imagereward": 2.0},
            ]
        )
        self.assertEqual(summary["num_images"], 2)
        self.assertAlmostEqual(summary["pickscore_mean"], 2.0)
        self.assertAlmostEqual(summary["imagereward_mean"], 2.0)
        self.assertEqual(scalar_summary([1.0])["std"], 0.0)

    def test_write_summary_csv_collects_nested_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics_dir = Path(tmp) / "metrics"
            first = metrics_dir / "a" / "summary.json"
            second = metrics_dir / "b" / "summary.json"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text(json.dumps({"pickscore_mean": 1.0}), encoding="utf-8")
            second.write_text(json.dumps({"clip_mean": 0.5}), encoding="utf-8")
            write_summary_csv(metrics_dir)
            text = (metrics_dir / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("pickscore_mean", text)
            self.assertIn("clip_mean", text)

    def test_manifest_metric_dir_preserves_samples_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "samples" / "sd-turbo-lora" / "method" / "run" / "manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n", encoding="utf-8")
            metric_dir = manifest_metric_dir(root / "samples", root / "metrics", manifest)
            self.assertEqual(metric_dir, root / "metrics" / "sd-turbo-lora" / "method" / "run")

    def test_discover_manifests_finds_baseline_and_lora_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples"
            for relative in (
                "sd-turbo-baseline/default/manifest.jsonl",
                "sd-turbo-lora/drpo/online/default/checkpoint-100/manifest.jsonl",
            ):
                path = samples / relative
                path.parent.mkdir(parents=True)
                path.write_text("{}\n", encoding="utf-8")
            manifests = [path.relative_to(samples).as_posix() for path in discover_manifests(samples)]
            self.assertEqual(
                manifests,
                [
                    "sd-turbo-baseline/default/manifest.jsonl",
                    "sd-turbo-lora/drpo/online/default/checkpoint-100/manifest.jsonl",
                ],
            )


if __name__ == "__main__":
    unittest.main()
