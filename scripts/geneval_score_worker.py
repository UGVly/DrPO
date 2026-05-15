#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--geneval_repo", required=True)
    parser.add_argument("--geneval_detector_path", required=True)
    parser.add_argument("--geneval_model_config", default=None)
    parser.add_argument("--geneval_options", default="")
    return parser.parse_args()


def parse_geneval_options(options_str: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not options_str.strip():
        return out
    for part in options_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def load_evaluate_images_module(geneval_repo: str) -> Any:
    path = os.path.join(geneval_repo, "evaluation", "evaluate_images.py")
    spec = importlib.util.spec_from_file_location("geneval_evaluate_images_worker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TransformersClipAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def encode_text(self, inputs):
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        return self.model.get_text_features(**inputs)

    def encode_image(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values.to(self.model.device))


class TransformersClipTransform:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"][0]


class TransformersClipTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


def wire_globals(
    mod: Any,
    *,
    device: str,
    geneval_repo: str,
    detector_path: str,
    model_config: str | None,
    options: dict[str, Any],
) -> None:
    import mmdet
    from mmdet.apis import init_detector

    def resolve_transformers_clip_model_id(clip_arch: str, options: dict[str, Any]) -> str:
        override = options.get("clip_model_id")
        if override:
            return str(override)
        mapping = {
            "ViT-B-32": "openai/clip-vit-base-patch32",
            "ViT-L-14": "openai/clip-vit-large-patch14",
        }
        return mapping.get(clip_arch, clip_arch)

    def resolve_default_model_config() -> str:
        package_dir = os.path.dirname(mmdet.__file__)
        candidates = [
            os.path.join(package_dir, ".mim", "configs", "mask2former", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"),
            os.path.join(package_dir, "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"),
        ]
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError("Could not locate Mask2Former config under installed mmdet package.")

    if device == "cuda":
        device = "cuda:0"
    torch.cuda.set_device(device)
    fake_args = argparse.Namespace(
        imagedir=".",
        outfile="unused.jsonl",
        model_config=model_config,
        model_path=detector_path,
        options=dict(options),
    )
    if fake_args.model_config is None:
        fake_args.model_config = resolve_default_model_config()
    model_name = fake_args.options.get("model", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco")
    object_detector = init_detector(
        fake_args.model_config,
        os.path.join(fake_args.model_path, f"{model_name}.pth"),
        device=device,
    )
    clip_arch = fake_args.options.get("clip_model", "ViT-L-14")
    clip_backend = str(fake_args.options.get("clip_backend", "open_clip"))
    clip_device = str(fake_args.options.get("clip_device", device))
    if clip_backend == "transformers":
        from transformers import AutoProcessor, CLIPModel

        model_id = resolve_transformers_clip_model_id(str(clip_arch), fake_args.options)
        processor = AutoProcessor.from_pretrained(model_id)
        clip_model = TransformersClipAdapter(CLIPModel.from_pretrained(model_id).to(clip_device).eval())
        transform = TransformersClipTransform(processor.image_processor)
        tokenizer = TransformersClipTokenizer(processor.tokenizer)
    else:
        import open_clip

        clip_model, _, transform = open_clip.create_model_and_transforms(
            clip_arch,
            pretrained="openai",
            device=clip_device,
        )
        tokenizer = open_clip.get_tokenizer(clip_arch)
    object_names_path = os.path.join(geneval_repo, "evaluation", "object_names.txt")
    with open(object_names_path, "r", encoding="utf-8") as cls_file:
        classnames = [line.strip() for line in cls_file if line.strip()]

    mod.object_detector = object_detector
    mod.clip_model, mod.transform, mod.tokenizer = clip_model, transform, tokenizer
    mod.classnames = classnames
    mod.args = fake_args
    mod.COLOR_CLASSIFIERS = {}
    mod.DEVICE = clip_device
    mod.CLIP_BACKEND = clip_backend

    opt = fake_args.options
    mod.THRESHOLD = float(opt.get("threshold", 0.3))
    mod.COUNTING_THRESHOLD = float(opt.get("counting_threshold", 0.9))
    mod.MAX_OBJECTS = int(opt.get("max_objects", 16))
    mod.NMS_THRESHOLD = float(opt.get("max_overlap", 1.0))
    mod.POSITION_THRESHOLD = float(opt.get("position_threshold", 0.1))


def decode_image_payloads(image_payloads: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for payload in image_payloads:
        image_bytes = base64.b64decode(payload.encode("ascii"))
        with Image.open(io.BytesIO(image_bytes)) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    return images


def evaluate_images_batch(mod: Any, images: list[Image.Image], metadata: dict[str, Any]) -> list[float]:
    detector_inputs = [np.asarray(image)[:, :, ::-1].copy() for image in images]
    results = mod.inference_detector(mod.object_detector, detector_inputs)
    scores: list[float] = []
    confidence_threshold = mod.THRESHOLD if metadata["tag"] != "counting" else mod.COUNTING_THRESHOLD
    for image, result in zip(images, results):
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        detected = {}
        for index, classname in enumerate(mod.classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
            ordering = ordering[:mod.MAX_OBJECTS].tolist()
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                ordering = [
                    obj for obj in ordering
                    if mod.NMS_THRESHOLD == 1 or mod.compute_iou(bbox[index][max_obj], bbox[index][obj]) < mod.NMS_THRESHOLD
                ]
            if not detected[classname]:
                del detected[classname]
        is_correct, _ = mod.evaluate(image, detected, metadata)
        scores.append(1.0 if bool(is_correct) else 0.0)
    return scores


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GenEval worker requires CUDA.")

    geneval_repo = os.path.abspath(args.geneval_repo)
    mod = load_evaluate_images_module(geneval_repo)
    wire_globals(
        mod,
        device=args.device,
        geneval_repo=geneval_repo,
        detector_path=os.path.abspath(args.geneval_detector_path),
        model_config=args.geneval_model_config,
        options=parse_geneval_options(args.geneval_options),
    )
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            metadata = req["metadata"]
            image_payloads = req.get("images_b64")
            if not isinstance(image_payloads, list):
                raise TypeError("Expected images_b64 to be a list of base64-encoded PNG payloads.")
            images = decode_image_payloads(image_payloads)
            scores = evaluate_images_batch(mod, images, metadata)
            print(json.dumps({"scores": scores}), flush=True)
        except Exception as exc:  # pragma: no cover - subprocess bridge
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
