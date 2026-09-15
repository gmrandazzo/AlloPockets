"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Author: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Model weight distribution hub, remote download manager, and cross-platform caching.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Dict, Optional, Union
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

GITHUB_REPO = "gmrandazzo/AlloPockets"
DEFAULT_RELEASE_TAG = "v1.0.0"
DEFAULT_BRANCH = "develop"

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "minimal_lgbm": {
        "description": "Regularized LightGBM (186 physicochemical, geometric & evolutionary features)",
        "size_kb": 344,
        "filename": "allopockets_minimal_lgbm.tar.gz",
        "default": True,
    },
    "minimal_xgboost": {
        "description": "Regularized XGBoost (186 physicochemical, geometric & evolutionary features)",
        "size_kb": 252,
        "filename": "allopockets_minimal_xgboost.tar.gz",
        "default": False,
    },
}

DEFAULT_MODEL = "minimal_lgbm"


def get_cache_dir() -> Path:
    """
    Return local AlloPockets cache directory.
    Linux / macOS: ~/.cache/allopockets (or $XDG_CACHE_HOME/allopockets)
    Windows: %USERPROFILE%\\.cache\\allopockets
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    cache_dir = base / "allopockets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available models with local cache status and paths."""
    models_info = {}
    cache_base = get_cache_dir() / "models"

    for name, meta in AVAILABLE_MODELS.items():
        cached_path = cache_base / name
        is_cached = (cached_path / "model.joblib").exists() and (
            cached_path / "features.json"
        ).exists()

        models_info[name] = {
            "description": meta["description"],
            "size_kb": meta["size_kb"],
            "default": meta.get("default", False),
            "cached": is_cached,
            "local_path": str(cached_path) if is_cached else None,
        }
    return models_info


def download_model(
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
    tag: str = DEFAULT_RELEASE_TAG,
    branch: str = DEFAULT_BRANCH,
) -> Path:
    """
    Download model weights and cache locally in ~/.cache/allopockets/models/<model_name>/.
    Attempts to download release tarball first, falling back to GitHub raw CDN.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(AVAILABLE_MODELS.keys())}"
        )

    target_dir = get_cache_dir() / "models" / model_name
    if (
        not force
        and (target_dir / "model.joblib").exists()
        and (target_dir / "features.json").exists()
    ):
        logger.info(f"Model '{model_name}' already present in cache: {target_dir}")
        return target_dir

    filename = AVAILABLE_MODELS[model_name]["filename"]
    release_url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{filename}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        download_success = False

        # Attempt 1: GitHub Release tarball
        logger.info(f"Downloading {model_name} from GitHub Release asset ({tag})...")
        try:
            resp = requests.get(release_url, stream=True, timeout=30)
            if resp.status_code == 200:
                tar_dest = tmp_path / filename
                total_size = int(resp.headers.get("content-length", 0))
                with (
                    open(tar_dest, "wb") as f,
                    tqdm(
                        desc=filename,
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar,
                ):
                    for chunk in resp.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)

                # Extract tarball safely
                with tarfile.open(tar_dest, "r:gz") as tar:
                    if hasattr(tarfile, "data_filter"):
                        tar.extractall(path=tmp_path, filter="data")  # nosec B202
                    else:
                        for member in tar.getmembers():
                            target_file = (tmp_path / member.name).resolve()
                            if not target_file.is_relative_to(tmp_path.resolve()):
                                raise RuntimeError(
                                    f"Unsafe path traversal in tar member: {member.name}"
                                )
                        tar.extractall(path=tmp_path)  # nosec B202
                download_success = True
            else:
                logger.info(
                    f"Release asset not found (HTTP {resp.status_code}), trying raw CDN fallback..."
                )
        except Exception as e:
            logger.warning(f"Release download failed: {e}. Trying raw CDN fallback...")

        # Attempt 2: GitHub Raw CDN fallback
        if not download_success:
            logger.info(f"Downloading model files directly from branch '{branch}' CDN...")
            raw_base = (
                f"https://raw.githubusercontent.com/{GITHUB_REPO}/{branch}/models/{model_name}"
            )
            raw_files = [
                "model.joblib",
                "features.json",
                "metadata.json",
                "metrics.json",
                "feature_importance.csv",
            ]
            for rfile in raw_files:
                file_url = f"{raw_base}/{rfile}"
                r = requests.get(file_url, stream=True, timeout=30)
                if r.status_code == 200:
                    with open(tmp_path / rfile, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                elif rfile in ("model.joblib", "features.json"):
                    raise RuntimeError(f"Failed to download required file {rfile} from {file_url}")

            download_success = True

        # Ensure required files exist
        if not (tmp_path / "model.joblib").exists() or not (tmp_path / "features.json").exists():
            raise RuntimeError(f"Corrupted download for '{model_name}': missing model.joblib")

        # Atomic move to cache
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for item in tmp_path.iterdir():
            if item.is_file() and not item.name.endswith(".tar.gz"):
                shutil.copy2(item, target_dir / item.name)

    logger.info(f"Successfully cached '{model_name}' to {target_dir}")
    return target_dir


def get_model_dir(model_name: str = DEFAULT_MODEL, auto_download: bool = True) -> Path:
    """
    Resolve model directory in order of precedence:
    1. Local cache (~/.cache/allopockets/models/<name>)
    2. Local git repository (models/<name>)
    3. Auto-download to cache from remote
    """
    # 1. Local user cache
    cached_path = get_cache_dir() / "models" / model_name
    if (cached_path / "model.joblib").exists() and (cached_path / "features.json").exists():
        return cached_path

    # 2. Local git repository (if running from source checkout)
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_repo_path = repo_root / "models" / model_name
    if (local_repo_path / "model.joblib").exists() and (local_repo_path / "features.json").exists():
        return local_repo_path

    # 3. Remote download
    if auto_download:
        return download_model(model_name=model_name)

    raise FileNotFoundError(
        f"Model '{model_name}' not found locally. Run 'allopockets download-model --model {model_name}' "
        f"or call download_model('{model_name}')."
    )


def load_model(model_name_or_path: Optional[Union[str, Path]] = None) -> Any:
    """
    Load a PocketClassifier model instance.
    Accepts either an explicit filesystem directory or an available model name (default: 'minimal_lgbm').
    """
    from allopockets.ml.models import PocketClassifier

    if model_name_or_path is not None:
        p = Path(model_name_or_path)
        if p.exists() and (p / "model.joblib").exists():
            return PocketClassifier.load(p)
        if str(model_name_or_path) in AVAILABLE_MODELS:
            model_dir = get_model_dir(str(model_name_or_path))
            return PocketClassifier.load(model_dir)
        raise FileNotFoundError(f"Model path or name '{model_name_or_path}' not recognized.")

    model_dir = get_model_dir(DEFAULT_MODEL)
    return PocketClassifier.load(model_dir)


def clear_cache(model_name: Optional[str] = None) -> None:
    """Clear cached models from ~/.cache/allopockets/models/."""
    cache_base = get_cache_dir() / "models"
    if model_name:
        target = cache_base / model_name
        if target.exists():
            shutil.rmtree(target)
            logger.info(f"Cleared cache for '{model_name}'.")
    else:
        if cache_base.exists():
            shutil.rmtree(cache_base)
            logger.info("Cleared all cached models.")
