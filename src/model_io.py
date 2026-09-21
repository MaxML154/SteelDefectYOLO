"""
Resolve YOLO weights relative to the project root and models/ directory.

Author: MaxML154
Created: 2026-09-21
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

_OFFICIAL_WEIGHT = re.compile(
    r"^yolo(v8|11|12|26)[nslmx]([-.].+)?\.pt$",
    re.IGNORECASE,
)


def resolve_weight(model_arg: str, download: bool | None = None) -> Path:
    """Resolve a .pt/.onnx path.

    Search order: given path, project-root-relative path, models/<filename>.
    Official pretrained names (yolo26n.pt, ...) are downloaded into models/ if missing.
    """
    raw = Path(model_arg)
    name = raw.name
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(PROJECT_ROOT / raw)
        candidates.append(MODELS_DIR / name)

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    if download is None:
        download = _OFFICIAL_WEIGHT.match(name) is not None and raw.parent == Path(".")

    if download:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / name
        cached = _find_ultralytics_cache(name)
        if cached is not None:
            shutil.copy2(cached, dest)
            return dest.resolve()
        _download_official(dest)
        if dest.is_file():
            return dest.resolve()

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Model not found: {model_arg}\nSearched:\n  {searched}\n"
        f"Place the file in {MODELS_DIR} or pass an existing path."
    )


def _find_ultralytics_cache(name: str) -> Path | None:
    try:
        from ultralytics.utils import SETTINGS
        cached = Path(SETTINGS["weights_dir"]) / name
        if cached.is_file():
            return cached
    except Exception:
        pass

    extras = [
        Path.home() / ".cache" / "ultralytics" / name,
        Path.home() / ".config" / "Ultralytics" / name,
    ]
    for path in extras:
        if path.is_file():
            return path
    return None


def _download_official(dest: Path) -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    print(f"Downloading {dest.name} into {dest.parent} ...")
    attempt_download_asset(dest)
