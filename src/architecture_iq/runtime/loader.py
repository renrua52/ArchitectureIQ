from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_file(path: Path, module_name: str) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_cached_sibling_modules() -> None:
    for name in ("model", "optimizer", "loss", "train"):
        sys.modules.pop(name, None)


def load_candidate_train(candidate_dir: Path) -> ModuleType:
    """Load train.py from a candidate folder; sibling model/loss/optimizer imports resolve."""
    candidate_dir = candidate_dir.resolve()
    train_file = candidate_dir / "train.py"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train.py in {candidate_dir}")

    path_str = str(candidate_dir)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        _clear_cached_sibling_modules()
        module_name = f"candidate_train_{candidate_dir.name}"
        return load_module_from_file(train_file, module_name)
    finally:
        if inserted:
            sys.path.remove(path_str)


def load_synthesize_module(synthesize_path: Path) -> ModuleType:
    """Load synthesize.py for a dataset instance."""
    synthesize_path = synthesize_path.resolve()
    module_name = f"dataset_synth_{synthesize_path.parent.name}"
    return load_module_from_file(synthesize_path, module_name)
