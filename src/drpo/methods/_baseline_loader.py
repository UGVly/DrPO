from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_baseline_module(method: str, filename: str = "train_lora.py") -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    path = root / "baselines" / method / filename
    spec = importlib.util.spec_from_file_location(f"_strong_drpo_baseline_{method}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module
