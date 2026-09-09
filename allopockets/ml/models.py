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

Unified Gradient Boost model wrapper supporting HistGradientBoosting, LightGBM, and XGBoost.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from allopockets.ml.config import ModelConfig, DEFAULT_FEATURE_NAMES

logger = logging.getLogger(__name__)


class PocketClassifier:
    """
    Transparent, reproducible Gradient Boost classifier for pocket ranking.
    Supports scikit-learn HistGradientBoosting, LightGBM, and XGBoost.
    """

    def __init__(
        self, config: Optional[ModelConfig] = None, feature_names: Optional[List[str]] = None
    ):
        self.config = config or ModelConfig()
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES
        self.model: Any = None
        self.model_backend: str = "hist_gradient_boost"
        self._init_model()

    def _init_model(self):
        mtype = self.config.model_type.lower()
        seed = self.config.seed

        if mtype in ("lightgbm", "lgb"):
            try:
                import lightgbm as lgb

                self.model = lgb.LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    num_leaves=self.config.num_leaves,
                    min_child_samples=self.config.min_child_samples,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    scale_pos_weight=self.config.scale_pos_weight or 1.0,
                    random_state=seed,
                    n_jobs=self.config.n_jobs,
                    verbose=-1,
                )
                self.model_backend = "lightgbm"
                return
            except ImportError:
                logger.warning(
                    "LightGBM not installed. Falling back to HistGradientBoostingClassifier."
                )

        if mtype in ("xgboost", "xgb"):
            try:
                import xgboost as xgb

                self.model = xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    scale_pos_weight=self.config.scale_pos_weight or 1.0,
                    random_state=seed,
                    n_jobs=self.config.n_jobs,
                    eval_metric="logloss",
                )
                self.model_backend = "xgboost"
                return
            except ImportError:
                logger.warning(
                    "XGBoost not installed. Falling back to HistGradientBoostingClassifier."
                )

        # Default scikit-learn HistGradientBoosting (Fast, built-in, no dependencies)
        from sklearn.ensemble import HistGradientBoostingClassifier

        class_weight = (
            "balanced"
            if self.config.scale_pos_weight is None
            else {0: 1.0, 1: float(self.config.scale_pos_weight)}
        )
        self.model = HistGradientBoostingClassifier(
            max_iter=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            max_leaf_nodes=self.config.num_leaves,
            min_samples_leaf=self.config.min_child_samples,
            class_weight=class_weight,
            random_state=seed,
        )
        self.model_backend = "hist_gradient_boost"

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit model with automatic feature selection alignment."""
        if isinstance(X, pd.DataFrame):
            # Align features with registered schema
            available = [c for c in self.feature_names if c in X.columns]
            if len(available) < len(self.feature_names):
                missing = set(self.feature_names) - set(available)
                logger.warning(f"Missing {len(missing)} features from schema, imputing with 0.0")
                X = X.copy()
                for c in missing:
                    X[c] = 0.0
            X_arr = X[self.feature_names].values
        else:
            X_arr = np.asarray(X)

        y_arr = np.asarray(y, dtype=int)

        # Handle class imbalance if not set
        if self.config.scale_pos_weight is None and self.model_backend in ("lightgbm", "xgboost"):
            n_pos = np.sum(y_arr == 1)
            n_neg = np.sum(y_arr == 0)
            if n_pos > 0:
                pos_weight = n_neg / n_pos
                self.model.set_params(scale_pos_weight=pos_weight)

        if sample_weight is not None:
            self.model.fit(X_arr, y_arr, sample_weight=sample_weight)
        else:
            self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_arr = self._prepare_input(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return probabilities of shape (N, 2)."""
        X_arr = self._prepare_input(X)
        return self.model.predict_proba(X_arr)

    def predict_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return positive class allosteric probability vector of shape (N,)."""
        probs = self.predict_proba(X)
        return probs[:, 1]

    def _prepare_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            for c in self.feature_names:
                if c not in X_copy.columns:
                    X_copy[c] = 0.0
            return X_copy[self.feature_names].values
        return np.asarray(X)

    def save(self, output_dir: Union[str, Path]):
        """Export trained model and metadata."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, out / "model.joblib")
        with open(out / "features.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)

        metadata = {
            "backend": self.model_backend,
            "n_features": len(self.feature_names),
            "config": self.config.__dict__,
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if hasattr(self.model, "feature_importances_"):
            try:
                fi_df = pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": self.model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                fi_df.to_csv(out / "feature_importance.csv", index=False)
            except Exception as e:
                logger.debug(f"Could not compute feature importances: {e}")

        logger.info(f"Saved model to {out}")

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "PocketClassifier":
        """Load trained model and registered features."""
        model_dir = Path(model_dir)
        model_file = model_dir / "model.joblib"
        feat_file = model_dir / "features.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(feat_file, "r") as f:
            feature_names = json.load(f)

        classifier = cls(feature_names=feature_names)
        classifier.model = joblib.load(model_file)
        if (model_dir / "metadata.json").exists():
            with open(model_dir / "metadata.json", "r") as f:
                meta = json.load(f)
                classifier.model_backend = meta.get("backend", "unknown")
        return classifier
