"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Original Author: Francho Nerín Fonz <fnerin@bioacademy.gr>
Refactoring & Packaging: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
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

AlloPockets: Allosteric Pocket Prediction, Pathway Tracing, and 3D Visualization.
"""

try:
    from importlib.metadata import version

    __version__ = version("allopockets")
except Exception:
    __version__ = "unknown"


from allopockets.pockets import Pocket, run_fpocket, get_pockets_info
from allopockets.ml import (
    PocketClassifier,
    prepare_dataset,
    train_pipeline,
    load_model,
    download_model,
    get_cache_dir,
    list_available_models,
)
from allopockets.viz import inspect_pocket_cli, generate_3d_pocket_html
from allopockets.predict import (
    get_cif,
    get_clean_pdb,
    get_pockets,
    get_pocket,
    get_features,
    get_pockets_features,
    predict,
    get_pathways,
    view_pdb,
    view_pockets,
    view_pockets_pathways,
    Site,
)

__all__ = [
    "__version__",
    "get_cif",
    "get_clean_pdb",
    "get_pockets",
    "get_pocket",
    "get_features",
    "get_pockets_features",
    "predict",
    "get_pathways",
    "view_pdb",
    "view_pockets",
    "view_pockets_pathways",
    "Site",
    "Pocket",
    "run_fpocket",
    "get_pockets_info",
    "PocketClassifier",
    "prepare_dataset",
    "train_pipeline",
    "inspect_pocket_cli",
    "generate_3d_pocket_html",
    "load_model",
    "download_model",
    "get_cache_dir",
    "list_available_models",
]
