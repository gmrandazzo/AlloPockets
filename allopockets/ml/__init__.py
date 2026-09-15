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

Machine learning models, dataset preparation, and training pipeline.
"""

from allopockets.ml.config import ModelConfig, TrainConfig, DEFAULT_FEATURE_NAMES
from allopockets.ml.dataset import load_pocket_dataset, get_group_splits, impute_features
from allopockets.ml.models import PocketClassifier
from allopockets.ml.metrics import compute_classification_metrics, compute_topk_retrieval
from allopockets.ml.prepare import prepare_dataset
from allopockets.ml.train import train_pipeline
from allopockets.ml.hub import (
    get_cache_dir,
    get_model_dir,
    download_model,
    load_model,
    list_available_models,
    clear_cache,
)

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "DEFAULT_FEATURE_NAMES",
    "load_pocket_dataset",
    "get_group_splits",
    "impute_features",
    "PocketClassifier",
    "compute_classification_metrics",
    "compute_topk_retrieval",
    "prepare_dataset",
    "train_pipeline",
    "get_cache_dir",
    "get_model_dir",
    "download_model",
    "load_model",
    "list_available_models",
    "clear_cache",
]
