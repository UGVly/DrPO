from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.common import PromptRecord, build_tasks, sample_tasks
from inference.metrics import dino_image_features, evaluate_core, read_jsonl


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for real GPU smoke tests.")
class RealGPUModelSmokeTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _assert_cuda_ready(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA is required for real GPU smoke tests on wyd-5880.")
        torch.empty(1, device="cuda")

    def _write_image(self, path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(path)

    def _write_manifest(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_cuda_and_required_model_paths_exist(self):
        self._assert_cuda_ready()
        required = [
            ROOT / "models" / "sd-turbo" / "unet" / "diffusion_pytorch_model.safetensors",
            ROOT / "models" / "dinov2-base" / "model.safetensors",
            ROOT / "models" / "PickScore_v1" / "model.safetensors",
            ROOT / "models" / "CLIP-ViT-L-14" / "model.safetensors",
            ROOT / "models" / "Aesthetic" / "sac+logos+ava1-l14-linearMSE.pth",
            ROOT / "models" / "HPSv2" / "HPS_v2_compressed.pt",
            ROOT / "models" / "ImageReward" / "ImageReward.pt",
            ROOT / "models" / "ImageReward" / "med_config.json",
            ROOT / "models" / "bert-base-uncased" / "vocab.txt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        self.assertEqual(missing, [], "missing required real-model assets")

    def test_real_sdturbo_sampler_generates_one_image_on_cuda(self):
        self._assert_cuda_ready()
        with tempfile.TemporaryDirectory(prefix="drpo_sdturbo_smoke_") as tmp:
            output_dir = Path(tmp) / "samples"
            tasks = build_tasks(output_dir, [PromptRecord(prompt_id=0, prompt="a centered object on a plain background")], (42,))
            sample_tasks(
                tasks=tasks,
                pretrained_model_path=ROOT / "models" / "sd-turbo",
                device="cuda",
                dtype_name="fp16",
                batch_size=1,
                resolution=64,
                overwrite=True,
            )
            self.assertTrue(tasks[0].image_path.is_file())
            with Image.open(tasks[0].image_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (64, 64))

    def test_real_dino_features_run_on_cuda(self):
        self._assert_cuda_ready()
        with tempfile.TemporaryDirectory(prefix="drpo_dino_smoke_") as tmp:
            root = Path(tmp)
            rows = []
            for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
                image_path = root / f"image_{index}.png"
                self._write_image(image_path, color)
                rows.append({"prompt": f"prompt {index}", "seed": 42, "image_path": str(image_path)})
            features = dino_image_features(
                rows,
                device="cuda",
                batch_size=2,
                num_workers=0,
                model_path=ROOT / "models" / "dinov2-base",
            )
            self.assertEqual(features.shape[0], 2)
            self.assertTrue(torch.isfinite(features).all())

    def test_real_core_metrics_generates_scores_summary_and_fid_on_cuda(self):
        self._assert_cuda_ready()
        with tempfile.TemporaryDirectory(prefix="drpo_metrics_smoke_") as tmp:
            root = Path(tmp)
            samples_dir = root / "samples"
            metrics_dir = root / "metrics"
            baseline_dir = samples_dir / "sd-turbo-baseline" / "default"
            candidate_dir = samples_dir / "sd-turbo-lora" / "smoke" / "checkpoint-1"
            baseline_rows = []
            candidate_rows = []
            for index, seed in enumerate((42, 43)):
                base_image = baseline_dir / "images" / f"base_{index}.png"
                cand_image = candidate_dir / "images" / f"cand_{index}.png"
                self._write_image(base_image, (220, 40 + index * 20, 40))
                self._write_image(cand_image, (40, 80, 220 - index * 20))
                baseline_rows.append(
                    {
                        "model_type": "sd-turbo-baseline",
                        "checkpoint_path": None,
                        "prompt_id": index,
                        "prompt": "a simple studio object",
                        "seed": seed,
                        "latent_seed": seed * 1000000 + index,
                        "image_path": str(base_image),
                    }
                )
                candidate_rows.append(
                    {
                        "model_type": "sd-turbo-lora",
                        "checkpoint_path": "checkpoint-1",
                        "prompt_id": index,
                        "prompt": "a simple studio object",
                        "seed": seed,
                        "latent_seed": seed * 1000000 + index,
                        "image_path": str(cand_image),
                    }
                )
            baseline_manifest = baseline_dir / "manifest.jsonl"
            candidate_manifest = candidate_dir / "manifest.jsonl"
            self._write_manifest(baseline_manifest, baseline_rows)
            self._write_manifest(candidate_manifest, candidate_rows)
            args = SimpleNamespace(
                samples_dir=str(samples_dir),
                metrics_dir=str(metrics_dir),
                manifest=str(candidate_manifest),
                baseline_manifest=str(baseline_manifest),
                device="cuda",
                reward_batch_size=2,
                feature_batch_size=2,
                fid_batch_size=2,
                num_workers=0,
                prefetch_factor=2,
                cache_dir=str(metrics_dir / ".cache"),
                no_feature_cache=False,
                dino_model_path=str(ROOT / "models" / "dinov2-base"),
                dino_processor_path=None,
                dino_feature_key="layer12_patch_mean",
                force=True,
            )
            evaluate_core(args)
            summary_path = metrics_dir / "sd-turbo-lora" / "smoke" / "checkpoint-1" / "summary.json"
            scores_path = metrics_dir / "sd-turbo-lora" / "smoke" / "checkpoint-1" / "scores.jsonl"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(scores_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for key in ("pickscore_mean", "clip_mean", "aes_mean", "hpsv2_mean", "clip_diversity", "dino_diversity", "fid_vs_baseline"):
                self.assertIn(key, summary)
                self.assertTrue(isinstance(summary[key], (int, float)))
            rows = read_jsonl(scores_path)
            for key in ("pickscore", "clip", "aes", "hpsv2"):
                self.assertTrue(all(key in row for row in rows), key)
            self.assertTrue((metrics_dir / "summary.csv").is_file())

    def test_real_imagereward_scores_one_image_in_eval_env(self):
        self._assert_cuda_ready()
        uv_bin = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        self.assertTrue(Path(uv_bin).is_file(), f"uv not found for ImageReward eval env: {uv_bin}")
        code = r"""
import sys
import tempfile
from pathlib import Path
from PIL import Image
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from drpo.rewards import ImageRewardSelector
with tempfile.TemporaryDirectory(prefix="drpo_imagereward_smoke_") as tmp:
    path = Path(tmp) / "image.png"
    Image.new("RGB", (64, 64), (128, 64, 192)).save(path)
    selector = ImageRewardSelector("cuda")
    score = selector.score_paths([path], ["a simple studio object"])[0]
    assert isinstance(score, float)
    print(f"imagereward_score_ok={score:.6f}")
"""
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        result = subprocess.run(
            [uv_bin, "run", "--project", str(ROOT / "eval_envs" / "imagereward"), "python", "-c", code],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=900,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("imagereward_score_ok=", result.stdout)


if __name__ == "__main__":
    unittest.main()
