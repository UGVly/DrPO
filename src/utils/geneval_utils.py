# coding=utf-8
"""
GenEval-based reward for online preference selection.

Requires a local clone of https://github.com/djghosh13/geneval with its environment
(mmdetection, clip-benchmark, detector checkpoint). See geneval/evaluation/README flow.

Each training JSONL row must include ``geneval_metadata``: the same JSON object GenEval
writes into per-folder ``metadata.jsonl`` (fields such as ``tag``, ``prompt``, ``include``).
"""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from PIL import Image

from .reward_common import to_feature_tensor


def _parse_geneval_options(options_str: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not options_str or not options_str.strip():
        return out
    for part in options_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        try:
            out[key] = int(val)
        except ValueError:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def _load_evaluate_images_module(geneval_repo: str) -> Any:
    path = os.path.join(geneval_repo, "evaluation", "evaluate_images.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GenEval evaluation/evaluate_images.py not found under geneval_repo={geneval_repo!r}"
        )
    spec = importlib.util.spec_from_file_location("geneval_evaluate_images", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_worker_script_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "scripts" / "geneval_score_worker.py")


def _build_worker_command() -> List[str]:
    python_bin = os.environ.get("GENEVAL_PYTHON_BIN", "").strip()
    if python_bin:
        return [python_bin]
    conda_env = os.environ.get("GENEVAL_CONDA_ENV", "").strip()
    if conda_env:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]
    return []


def _encode_images_to_base64_payload(images: List[Image.Image]) -> List[str]:
    payloads: List[str] = []
    for image in images:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        payloads.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return payloads


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]


def _load_object_names(geneval_repo: str) -> List[str]:
    path = os.path.join(geneval_repo, "evaluation", "object_names.txt")
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    area = lambda box: max(float(box[2] - box[0] + 1), 0.0) * max(float(box[3] - box[1] + 1), 0.0)
    inter = [
        max(float(box_a[0]), float(box_b[0])),
        max(float(box_a[1]), float(box_b[1])),
        min(float(box_a[2]), float(box_b[2])),
        min(float(box_a[3]), float(box_b[3])),
    ]
    inter_area = area(inter)
    union_area = area(box_a) + area(box_b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _relative_position(
    obj_a: tuple[np.ndarray, None],
    obj_b: tuple[np.ndarray, None],
    position_threshold: float,
) -> set[str]:
    boxes = np.array([obj_a[0], obj_b[0]], dtype=np.float32)[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0.0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations: set[str] = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


class _ModernSelectorBackend:
    def __init__(self, device: torch.device, geneval_repo: str, geneval_options: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.geneval_repo = os.path.abspath(geneval_repo)
        self.options = _parse_geneval_options(geneval_options)
        self.detector_model_id = str(self.options.get("detector_model", "google/owlv2-base-patch16-ensemble"))
        self.clip_model_name = str(self.options.get("clip_model", "ViT-L-14"))
        self.clip_pretrained = str(self.options.get("clip_model_id", self.options.get("clip_pretrained", "openai")))
        self.threshold = float(self.options.get("threshold", 0.3))
        self.counting_threshold = float(self.options.get("counting_threshold", 0.9))
        self.max_objects = int(self.options.get("max_objects", 16))
        self.nms_threshold = float(self.options.get("max_overlap", 1.0))
        self.position_threshold = float(self.options.get("position_threshold", 0.1))
        self._clip_backend = "open_clip"

        self.classnames = _load_object_names(self.geneval_repo)
        self.class_prompts = [f"a photo of a {name}" for name in self.classnames]
        self.detector_processor = AutoProcessor.from_pretrained(self.detector_model_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_model_id).to(device).eval()
        if str(self.options.get("clip_backend", "")).strip().lower() in {"transformers", "hf"}:
            self._load_transformers_clip_fallback()
            self.color_classifier_cache: Dict[str, torch.Tensor] = {}
            return

        import open_clip

        clip_pretrained_arg = self._resolve_clip_pretrained_arg()
        try:
            if self.clip_pretrained.lower() == "openai" and os.path.isfile(clip_pretrained_arg):
                from open_clip.openai import load_openai_model
                from open_clip.transform import image_transform

                self.clip_model = load_openai_model(
                    clip_pretrained_arg,
                    precision="fp32",
                    device=str(device),
                )
                self.clip_transform = image_transform(
                    self.clip_model.visual.image_size,
                    is_train=False,
                    mean=getattr(self.clip_model.visual, "image_mean", None),
                    std=getattr(self.clip_model.visual, "image_std", None),
                )
            else:
                self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
                    self.clip_model_name,
                    pretrained=clip_pretrained_arg,
                    device=str(device),
                )
            self.clip_model.eval()
            self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        except Exception:
            self._load_transformers_clip_fallback()
        self.color_classifier_cache: Dict[str, torch.Tensor] = {}

    def _load_transformers_clip_fallback(self) -> None:
        from transformers import AutoProcessor, CLIPModel

        model_id = self._resolve_transformers_clip_model_id()
        self._clip_backend = "transformers"
        self.clip_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(self.device).eval()

    def _resolve_transformers_clip_model_id(self) -> str:
        override = self.options.get("clip_model_id")
        if override:
            return str(override)
        if self.clip_pretrained.lower() != "openai":
            return self.clip_pretrained
        mapping = {
            "ViT-B-32": "openai/clip-vit-base-patch32",
            "ViT-L-14": "openai/clip-vit-large-patch14",
        }
        return mapping.get(self.clip_model_name, self.clip_pretrained)

    def _resolve_clip_pretrained_arg(self) -> str:
        if os.path.isfile(self.clip_pretrained):
            return self.clip_pretrained
        if self.clip_pretrained.lower() != "openai":
            return self.clip_pretrained

        cache_root = os.environ.get("OPEN_CLIP_CACHE", "").strip()
        if not cache_root:
            cache_root = os.path.join(os.path.expanduser("~"), ".cache", "clip")
        local_path = os.path.join(cache_root, f"{self.clip_model_name}.pt")
        if os.path.isfile(local_path):
            return local_path
        return self.clip_pretrained

    def _build_color_classifier(self, classname: str) -> torch.Tensor:
        prompts = []
        for color in COLORS:
            prompts.append(
                [
                    f"a photo of a {color} {classname}",
                    f"a photo of a {color}-colored {classname}",
                    f"a photo of a {color} object",
                ]
            )

        with torch.no_grad():
            text_features = []
            for prompt_group in prompts:
                feats = self._as_feature_tensor(self._encode_text(prompt_group))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                pooled = feats.mean(dim=0)
                pooled = pooled / pooled.norm()
                text_features.append(pooled)
        return torch.stack(text_features, dim=0)

    def _encode_text(self, prompt_group: List[str]) -> torch.Tensor:
        if self._clip_backend == "transformers":
            tokens = self.clip_processor.tokenizer(
                prompt_group,
                padding=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            text_features = self.clip_model.get_text_features(**tokens)
            return self._as_feature_tensor(text_features)
        tokens = self.clip_tokenizer(prompt_group).to(self.device)
        text_features = self.clip_model.encode_text(tokens)
        return self._as_feature_tensor(text_features)

    def _encode_crops(self, crops: List[Image.Image]) -> torch.Tensor:
        if self._clip_backend == "transformers":
            inputs = self.clip_processor.image_processor(images=crops, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            return self._as_feature_tensor(image_features)
        batch = torch.stack([self.clip_transform(crop.convert("RGB")) for crop in crops], dim=0).to(self.device)
        image_features = self.clip_model.encode_image(batch)
        return self._as_feature_tensor(image_features)

    @staticmethod
    def _as_feature_tensor(features: Any) -> torch.Tensor:
        return to_feature_tensor(features)

    def _classify_colors(
        self,
        image: Image.Image,
        objects: List[tuple[np.ndarray, None]],
        classname: str,
    ) -> List[str]:
        if classname not in self.color_classifier_cache:
            self.color_classifier_cache[classname] = self._build_color_classifier(classname)
        classifier = self.color_classifier_cache[classname]

        crops = []
        width, height = image.size
        for box, _ in objects:
            x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
            x0 = max(0, min(x0, width - 1))
            y0 = max(0, min(y0, height - 1))
            x1 = max(x0 + 1, min(x1, width))
            y1 = max(y0 + 1, min(y1, height))
            crops.append(image.crop((x0, y0, x1, y1)).convert("RGB"))

        with torch.no_grad():
            image_features = self._encode_crops(crops)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ classifier.T
            indices = logits.argmax(dim=-1).tolist()
        return [COLORS[idx] for idx in indices]

    def _detect_objects(
        self,
        image: Image.Image,
        metadata: Mapping[str, Any],
    ) -> Dict[str, List[tuple[np.ndarray, None]]]:
        confidence_threshold = self.threshold if metadata["tag"] != "counting" else self.counting_threshold
        inputs = self.detector_processor(
            text=[self.class_prompts],
            images=image,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.detector(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        result = self.detector_processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=confidence_threshold,
            target_sizes=target_sizes,
            text_labels=[self.class_prompts],
        )[0]

        detected: Dict[str, List[tuple[np.ndarray, None]]] = {}
        per_class: Dict[str, List[np.ndarray]] = {}
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            classname = self.classnames[int(label)]
            arr = np.array([box[0].item(), box[1].item(), box[2].item(), box[3].item(), score.item()], dtype=np.float32)
            per_class.setdefault(classname, []).append(arr)

        for classname, boxes in per_class.items():
            boxes.sort(key=lambda item: float(item[4]), reverse=True)
            kept: List[tuple[np.ndarray, None]] = []
            for candidate in boxes:
                if len(kept) >= self.max_objects:
                    break
                if self.nms_threshold != 1.0 and any(
                    _compute_iou(candidate[:4], existing[0][:4]) >= self.nms_threshold for existing in kept
                ):
                    continue
                kept.append((candidate, None))
            if kept:
                detected[classname] = kept
        return detected

    def _evaluate_image(self, image: Image.Image, metadata: Mapping[str, Any]) -> bool:
        objects = self._detect_objects(image, metadata)
        matched_groups: List[List[tuple[np.ndarray, None]] | None] = []
        correct = True

        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[: req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
            else:
                if "color" in req:
                    colors = self._classify_colors(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group] or []:
                                if expected_rel not in _relative_position(obj, target_obj, self.position_threshold):
                                    correct = matched = False
                                    break
                            if not matched:
                                break
            matched_groups.append(found_objects if matched else None)

        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
        return correct

    def score(self, images: List[Image.Image], metadata: Mapping[str, Any]) -> List[float]:
        return [1.0 if self._evaluate_image(img.convert("RGB"), metadata) else 0.0 for img in images]


class _SubprocessSelectorBackend:
    def __init__(
        self,
        device: torch.device,
        geneval_repo: str,
        geneval_detector_path: str,
        geneval_model_config: str | None,
        geneval_options: str,
    ):
        cmd = _build_worker_command()
        if not cmd:
            raise RuntimeError(
                "GenEval subprocess backend requires GENEVAL_PYTHON_BIN or GENEVAL_CONDA_ENV."
            )
        cmd.extend(
            [
                "-u",
                _get_worker_script_path(),
                "--device",
                str(device),
                "--geneval_repo",
                os.path.abspath(geneval_repo),
                "--geneval_detector_path",
                os.path.abspath(geneval_detector_path),
                "--geneval_options",
                geneval_options,
            ]
        )
        if geneval_model_config:
            cmd.extend(["--geneval_model_config", geneval_model_config])

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)
        ready = self._read_message()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"GenEval worker failed to start: {ready}")

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            self._proc = None

    def _read_message(self) -> Dict[str, Any]:
        assert self._proc is not None
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            if self._proc.poll() is not None:
                stderr_text = ""
                if self._proc.stderr is not None:
                    stderr_text = self._proc.stderr.read().strip()
                raise RuntimeError(
                    "GenEval worker exited unexpectedly"
                    + (f": {stderr_text}" if stderr_text else ".")
                )

    def score(self, images: List[Image.Image], metadata: Mapping[str, Any]) -> List[float]:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("GenEval worker is not running.")
        payload = {"images_b64": _encode_images_to_base64_payload(list(images)), "metadata": dict(metadata)}
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        resp = self._read_message()
        if "error" in resp:
            raise RuntimeError(resp["error"])
        scores = resp.get("scores")
        if not isinstance(scores, list):
            raise RuntimeError(f"Malformed GenEval worker response: {resp}")
        return [float(score) for score in scores]


def _wire_geneval_globals(
    mod: Any,
    device: torch.device,
    detector_path: str,
    model_config: str | None,
    options: Mapping[str, Any],
) -> None:
    import mmdet

    def _resolve_default_model_config() -> str:
        package_dir = os.path.dirname(mmdet.__file__)
        candidates = [
            os.path.join(package_dir, ".mim", "configs", "mask2former", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"),
            os.path.join(package_dir, "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"),
        ]
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not locate Mask2Former config under installed mmdet package."
        )

    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        torch.cuda.set_device(device_index)
        mod.DEVICE = f"cuda:{device_index}"
    fake_args = argparse.Namespace(
        imagedir=".",
        outfile="unused.jsonl",
        model_config=model_config,
        model_path=detector_path,
        options=dict(options),
    )
    if fake_args.model_config is None:
        fake_args.model_config = _resolve_default_model_config()

    object_detector, clip_pack, classnames = mod.load_models(fake_args)
    mod.object_detector = object_detector
    mod.clip_model, mod.transform, mod.tokenizer = clip_pack
    mod.classnames = classnames
    mod.args = fake_args
    mod.COLOR_CLASSIFIERS = {}

    opt = fake_args.options
    mod.THRESHOLD = float(opt.get("threshold", 0.3))
    mod.COUNTING_THRESHOLD = float(opt.get("counting_threshold", 0.9))
    mod.MAX_OBJECTS = int(opt.get("max_objects", 16))
    mod.NMS_THRESHOLD = float(opt.get("max_overlap", 1.0))
    mod.POSITION_THRESHOLD = float(opt.get("position_threshold", 0.1))


class Selector:
    """
    Scores images with GenEval's ``evaluate_image`` (boolean correct -> 1.0 / 0.0).
    ``metadata`` must match GenEval's structured evaluation spec for the prompt.
    """

    def __init__(
        self,
        device: torch.device,
        geneval_repo: str,
        geneval_detector_path: str,
        geneval_model_config: str | None = None,
        geneval_options: str = "",
    ):
        self.device = device
        self._backend = None
        self._mod = None
        parsed_options = _parse_geneval_options(geneval_options)
        backend_name = str(parsed_options.get("backend", "")).strip().lower()
        if backend_name in {"modern", "owl", "owlv2"}:
            self._backend = _ModernSelectorBackend(
                device=device,
                geneval_repo=geneval_repo,
                geneval_options=geneval_options,
            )
            return

        force_subprocess = bool(_build_worker_command())
        if not force_subprocess:
            if not torch.cuda.is_available():
                raise RuntimeError("GenEval evaluate_images.py requires CUDA (see upstream assert).")
            self._mod = _load_evaluate_images_module(os.path.abspath(geneval_repo))
            _wire_geneval_globals(
                self._mod,
                device,
                os.path.abspath(geneval_detector_path),
                geneval_model_config,
                _parse_geneval_options(geneval_options),
            )
        else:
            self._backend = _SubprocessSelectorBackend(
                device=device,
                geneval_repo=geneval_repo,
                geneval_detector_path=geneval_detector_path,
                geneval_model_config=geneval_model_config,
                geneval_options=geneval_options,
            )

    def score(self, images: List[Image.Image], metadata: Mapping[str, Any]) -> List[float]:
        if not isinstance(metadata, Mapping):
            raise TypeError(f"geneval metadata must be a mapping, got {type(metadata)}")
        meta_dict: Dict[str, Any] = dict(metadata)
        if "tag" not in meta_dict:
            raise ValueError("geneval_metadata must include a 'tag' field (GenEval format).")

        if self._backend is not None:
            return self._backend.score(images, meta_dict)

        scores: List[float] = []
        with tempfile.TemporaryDirectory(prefix="geneval_score_") as tmp:
            for img in images:
                path = os.path.join(tmp, "g.png")
                img.convert("RGB").save(path)
                row = self._mod.evaluate_image(path, meta_dict)
                scores.append(1.0 if bool(row.get("correct")) else 0.0)
        return scores
