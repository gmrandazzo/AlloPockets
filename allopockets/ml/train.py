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

Reproducible Gradient Boost model training and evaluation engine.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

from allopockets.ml.config import ModelConfig
from allopockets.ml.dataset import get_group_splits, impute_features, load_pocket_dataset
from allopockets.ml.metrics import compute_classification_metrics, compute_topk_retrieval
from allopockets.ml.models import PocketClassifier

logger = logging.getLogger(__name__)


def train_pipeline(
    data_path: Union[str, Path],
    model_type: str = "hist_gradient_boost",
    seed: int = 42,
    n_splits: int = 5,
    output_dir: str = "models/lgbm_pocket_classifier",
    learning_rate: float = 0.03,
    n_estimators: int = 300,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    test_data_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Execute 5-fold Stratified Group K-Fold cross-validation and export production model.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pocket_dataset(data_path)
    logger.info(f"Loaded dataset with {len(df)} samples and {df.shape[1]} columns.")

    # Determine target and group columns
    target_col = (
        "Label_label"
        if "Label_label" in df.columns
        else [c for c in df.columns if "label" in c.lower()][0]
    )
    group_col = (
        "Pockets_pdb"
        if "Pockets_pdb" in df.columns
        else [c for c in df.columns if "pdb" in c.lower()][0]
    )

    # Exclude metadata columns from feature list
    exclude_cols = {
        target_col,
        group_col,
        "pdb",
        "pocket",
        "Pockets_pocket",
        "site_in_pocket",
        "pocket_in_site",
        "index",
        "level_0",
    }
    feature_names = [
        c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    logger.info(f"Using {len(feature_names)} numerical features for training.")

    # Cross-validation
    splits = get_group_splits(
        df, group_col=group_col, target_col=target_col, n_splits=n_splits, seed=seed
    )

    oof_probs = np.zeros(len(df))
    fold_metrics = []

    m_config = ModelConfig(
        model_type=model_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        seed=seed,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        train_sub = df.iloc[train_idx]
        val_sub = df.iloc[val_idx]

        train_imp, val_imp = impute_features(train_sub, val_sub, feature_cols=feature_names)
        assert val_imp is not None

        X_train = train_imp[feature_names].values
        y_train = train_imp[target_col].astype(int).values

        X_val = val_imp[feature_names].values
        y_val = val_imp[target_col].astype(int).values

        clf = PocketClassifier(config=m_config, feature_names=feature_names)
        clf.fit(X_train, y_train)

        val_probs = clf.predict_score(X_val)
        oof_probs[val_idx] = val_probs

        metrics = compute_classification_metrics(y_val, val_probs)
        fold_metrics.append(metrics)
        logger.info(
            f"Fold {fold_idx}/{n_splits} - MCC: {metrics['mcc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}"
        )

    # Compute mean and standard deviation across folds to prevent calibration shift artifacts
    mean_fold_metrics: Dict[str, float] = {}
    std_fold_metrics: Dict[str, float] = {}
    metric_keys = (
        "mcc",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "balanced_accuracy",
    )
    for k in metric_keys:
        vals = [
            float(m[k]) for m in fold_metrics if k in m and m[k] is not None and not np.isnan(m[k])
        ]
        if vals:
            mean_fold_metrics[k] = float(np.mean(vals))
            std_fold_metrics[k] = float(np.std(vals))

    # Overall out-of-fold metrics (pooled)
    oof_metrics = compute_classification_metrics(df[target_col].values, oof_probs)

    # Top-K Retrieval on out-of-fold predictions
    df_eval = df[[group_col, target_col]].copy()
    df_eval["score"] = oof_probs
    df_eval = df_eval.rename(columns={group_col: "pdb", target_col: "label"})
    retrieval_metrics = compute_topk_retrieval(
        df_eval, pdb_col="pdb", label_col="label", score_col="score"
    )

    summary: Dict[str, Any] = {
        "model_type": model_type,
        "n_samples": len(df),
        "n_features": len(feature_names),
        "mean_fold_classification_metrics": mean_fold_metrics,
        "std_fold_classification_metrics": std_fold_metrics,
        "oof_classification_metrics": oof_metrics,
        "oof_topk_retrieval": retrieval_metrics,
        "fold_metrics": fold_metrics,
    }

    # Final fit on entire dataset
    df_full_imp, _ = impute_features(df, feature_cols=feature_names)
    final_clf = PocketClassifier(config=m_config, feature_names=feature_names)
    final_clf.fit(df_full_imp[feature_names].values, df_full_imp[target_col].astype(int).values)
    final_clf.save(out_dir)

    # Optional evaluation on held-out test dataset
    if test_data_path and Path(test_data_path).exists():
        df_test = load_pocket_dataset(test_data_path)
        logger.info(f"Evaluating final model on held-out test set ({len(df_test)} samples)...")
        _, df_test_imp = impute_features(df, df_test, feature_cols=feature_names)
        assert df_test_imp is not None
        test_scores = final_clf.predict_score(df_test_imp[feature_names].values)

        test_target_col = (
            target_col
            if target_col in df_test.columns
            else [c for c in df_test.columns if "label" in c.lower()][0]
        )
        test_group_col = (
            group_col
            if group_col in df_test.columns
            else [c for c in df_test.columns if "pdb" in c.lower()][0]
        )

        test_cls_metrics = compute_classification_metrics(
            df_test_imp[test_target_col].astype(int).values, test_scores
        )
        df_test_eval = df_test[[test_group_col, test_target_col]].copy()
        df_test_eval["score"] = test_scores
        df_test_eval = df_test_eval.rename(
            columns={test_group_col: "pdb", test_target_col: "label"}
        )
        test_retrieval = compute_topk_retrieval(
            df_test_eval, pdb_col="pdb", label_col="label", score_col="score"
        )

        summary["test_metrics"] = {
            "n_samples": len(df_test),
            "classification_metrics": test_cls_metrics,
            "topk_retrieval": test_retrieval,
        }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Cross-Validation Results ===")
    logger.info(
        f"Mean Fold CV -> MCC: {mean_fold_metrics.get('mcc', 0.0):.4f} +/- {std_fold_metrics.get('mcc', 0.0):.4f} | "
        f"ROC-AUC: {mean_fold_metrics.get('roc_auc', 0.0):.4f} +/- {std_fold_metrics.get('roc_auc', 0.0):.4f} | "
        f"PR-AUC: {mean_fold_metrics.get('pr_auc', 0.0):.4f} +/- {std_fold_metrics.get('pr_auc', 0.0):.4f}"
    )
    logger.info(
        f"Pooled OOF   -> MCC: {oof_metrics['mcc']:.4f} | ROC-AUC: {oof_metrics['roc_auc']:.4f} | PR-AUC: {oof_metrics['pr_auc']:.4f}"
    )
    logger.info(
        f"Top-1 Accuracy: {retrieval_metrics['top_1_accuracy']*100:.1f}% | Top-3 Accuracy: {retrieval_metrics['top_3_accuracy']*100:.1f}%"
    )

    if test_data_path and Path(test_data_path).exists():
        logger.info("=== Held-Out Test Results ===")
        logger.info(
            f"Test MCC: {test_cls_metrics['mcc']:.4f} | Test ROC-AUC: {test_cls_metrics['roc_auc']:.4f} | Test PR-AUC: {test_cls_metrics['pr_auc']:.4f}"
        )
        logger.info(
            f"Test Top-1 Accuracy: {test_retrieval['top_1_accuracy']*100:.1f}% | Test Top-3 Accuracy: {test_retrieval['top_3_accuracy']*100:.1f}%"
        )

    return summary
