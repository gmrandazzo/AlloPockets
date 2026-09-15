#!/usr/bin/env python3
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

Package trained model checkpoints into release distribution tarballs.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "dist" / "models"

DEFAULT_MODELS = ["minimal_lgbm", "minimal_xgboost"]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def package_model(model_dir: Path, output_tar: Path) -> str:
    """Package a model directory into a .tar.gz archive and return its SHA256."""
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Packaging {model_dir.name} -> {output_tar}...")

    with tarfile.open(output_tar, "w:gz") as tar:
        for item in sorted(model_dir.iterdir()):
            if item.is_file() and not item.name.startswith("."):
                # Store files directly under archive root
                tar.add(item, arcname=item.name)

    sha256 = compute_sha256(output_tar)
    size_kb = output_tar.stat().st_size / 1024
    logger.info(f"Created {output_tar.name} ({size_kb:.1f} KB, sha256={sha256[:16]}...)")
    return sha256


def main():
    parser = argparse.ArgumentParser(description="Package AlloPockets models for release.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="List of model directory names under models/ to package",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for packaged tarballs",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for name in args.models:
        src = MODELS_DIR / name
        if not src.exists():
            logger.warning(f"Model directory '{src}' does not exist, skipping.")
            continue
        if not (src / "model.joblib").exists():
            logger.warning(f"No 'model.joblib' found in '{src}', skipping.")
            continue

        tar_name = f"allopockets_{name}.tar.gz"
        out_tar = args.outdir / tar_name
        sha256 = package_model(src, out_tar)

        manifest[tar_name] = {
            "model_name": name,
            "filename": tar_name,
            "size_bytes": out_tar.stat().st_size,
            "sha256": sha256,
        }

    checksums_file = args.outdir / "checksums.json"
    with open(checksums_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved release checksums to {checksums_file}")


if __name__ == "__main__":
    main()
