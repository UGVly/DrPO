from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.geneval_utils import _encode_images_to_base64_payload


def _load_worker_module():
    path = ROOT / "scripts" / "geneval_score_worker.py"
    spec = importlib.util.spec_from_file_location("geneval_score_worker_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load worker module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GenevalWorkerPayloadTest(unittest.TestCase):
    def test_image_payload_roundtrip_preserves_rgb_pixels(self):
        worker = _load_worker_module()
        image = Image.new("RGB", (2, 2), (12, 34, 56))
        payloads = _encode_images_to_base64_payload([image])
        decoded = worker.decode_image_payloads(payloads)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].mode, "RGB")
        self.assertEqual(decoded[0].size, (2, 2))
        self.assertEqual(decoded[0].getpixel((0, 0)), (12, 34, 56))


if __name__ == "__main__":
    unittest.main()
