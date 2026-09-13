"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Author: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

Train and evaluate models on Route 1 (155 multi-modal features) for benchmark structures:
- LightGBM (regularized GBDT)
- XGBoost (regularized GBDT)
- AutoGluon TabularPredictor (Ensemble)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from allopockets.ml.dataset import impute_features, load_pocket_dataset
from allopockets.ml.metrics import compute_classification_metrics, compute_topk_retrieval
from allopockets.ml.train import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(
    df_eval: pd.DataFrame,
    pred_probs: np.ndarray,
    target_col: str = "Label_label",
    pdb_col: str = "Pockets_pdb",
) -> Dict[str, Any]:
    """Compute comprehensive ranking and classification metrics on predictions."""
    y_true = df_eval[target_col].astype(int).values
    probs = np.asarray(pred_probs, dtype=float)

    cls_metrics = compute_classification_metrics(y_true, probs)

    df_retrieval = pd.DataFrame(
        {
            "pdb": df_eval[pdb_col].values,
            "label": y_true,
            "score": probs,
        }
    )
    retrieval_metrics = compute_topk_retrieval(
        df_retrieval, pdb_col="pdb", label_col="label", score_col="score"
    )

    return {
        "roc_auc": cls_metrics["roc_auc"],
        "pr_auc": cls_metrics["pr_auc"],
        "mcc": cls_metrics["mcc"],
        "f1": cls_metrics["f1"],
        "top_1_accuracy": retrieval_metrics["top_1_accuracy"],
        "top_3_accuracy": retrieval_metrics["top_3_accuracy"],
        "top_5_accuracy": retrieval_metrics["top_5_accuracy"],
        "evaluated_pdbs": retrieval_metrics["evaluated_pdbs"],
    }


def train_autogluon_route1(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str = "models/route1_autogluon",
    time_limit: int = 600,
) -> Dict[str, Any]:
    """Train AutoGluon TabularPredictor on Route 1 features."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as e:
        logger.warning(f"AutoGluon not available in current environment: {e}")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    target_col = "Label_label"

    # Prepare datasets
    ag_train = train_df[feature_cols + [target_col]].copy()
    ag_test = test_df[feature_cols + [target_col]].copy()
    ag_train[target_col] = ag_train[target_col].astype(int)
    ag_test[target_col] = ag_test[target_col].astype(int)

    predictor = TabularPredictor(
        label=target_col,
        problem_type="binary",
        eval_metric="mcc",
        path=output_dir,
    )

    logger.info(f"Training AutoGluon TabularPredictor on {len(feature_cols)} features...")
    predictor.fit(
        train_data=ag_train,
        presets="medium_quality",
        time_limit=time_limit,
        verbosity=2,
    )

    test_probs = predictor.predict_proba(ag_test, as_multiclass=False)
    metrics = evaluate_predictions(test_df, test_probs.values)
    metrics["best_model"] = predictor.model_best
    metrics["model_names"] = predictor.model_names()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate Route 1 (155 features) benchmark models.")
    parser.add_argument(
        "--train-data",
        default="data/benchmark/curated_train_pockets_155feats.parquet",
        help="Path to 155-feature train parquet.",
    )
    parser.add_argument(
        "--test-data",
        default="data/benchmark/curated_test_pockets_155feats.parquet",
        help="Path to 155-feature test parquet.",
    )
    parser.add_argument(
        "--output-json",
        default="data/benchmark/route1_benchmark_results.json",
        help="Path to output results json.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="AutoGluon training time limit in seconds.",
    )
    args = parser.parse_args()

    train_df = load_pocket_dataset(args.train_data)
    test_df = load_pocket_dataset(args.test_data)

    target_col = "Label_label"
    group_col = "Pockets_pdb"
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
    feature_cols = [
        c for c in train_df.columns if c not in exclude_cols and np.issubdtype(train_df[c].dtype, np.number)
    ]

    logger.info(f"Loaded train ({len(train_df)} rows) and test ({len(test_df)} rows) with {len(feature_cols)} features.")

    all_results: Dict[str, Any] = {}
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json) as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    # 1. LightGBM
    logger.info("\n--- Training Regularized LightGBM on 155 Features ---")
    lgb_res = train_pipeline(
        data_path=args.train_data,
        test_data_path=args.test_data,
        model_type="lightgbm",
        output_dir="models/route1_lgbm",
        seed=42,
    )
    all_results["LightGBM"] = {
        "oof": lgb_res["oof_classification_metrics"],
        "oof_retrieval": lgb_res["oof_topk_retrieval"],
        "test": lgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": lgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    # 2. XGBoost
    logger.info("\n--- Training Regularized XGBoost on 155 Features ---")
    xgb_res = train_pipeline(
        data_path=args.train_data,
        test_data_path=args.test_data,
        model_type="xgboost",
        output_dir="models/route1_xgboost",
        seed=42,
    )
    all_results["XGBoost"] = {
        "oof": xgb_res["oof_classification_metrics"],
        "oof_retrieval": xgb_res["oof_topk_retrieval"],
        "test": xgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": xgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    # 3. AutoGluon
    logger.info("\n--- Training AutoGluon Ensemble on 155 Features ---")
    ag_res = train_autogluon_route1(
        train_df,
        test_df,
        feature_cols,
        output_dir="models/route1_autogluon",
        time_limit=args.time_limit,
    )
    if ag_res:
        all_results["AutoGluon"] = ag_res

    # Save structured results
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved benchmark results to {args.output_json}")

    # Print summary table
    print("\n" + "=" * 80)
    print("ROUTE 1 (155 MULTI-MODAL FEATURES) MODEL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<16} | {'Test ROC-AUC':<12} | {'Test PR-AUC':<12} | {'Test MCC':<10} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10}")
    print("-" * 80)

    for model_name, res in all_results.items():
        if model_name == "AutoGluon":
            roc = res.get("roc_auc", 0.0)
            pr = res.get("pr_auc", 0.0)
            mcc = res.get("mcc", 0.0)
            top1 = res.get("top_1_accuracy", 0.0) * 100
            top3 = res.get("top_3_accuracy", 0.0) * 100
        else:
            t_cls = res.get("test", {})
            t_ret = res.get("test_retrieval", {})
            roc = t_cls.get("roc_auc", 0.0)
            pr = t_cls.get("pr_auc", 0.0)
            mcc = t_cls.get("mcc", 0.0)
            top1 = t_ret.get("top_1_accuracy", 0.0) * 100
            top3 = t_ret.get("top_3_accuracy", 0.0) * 100

        print(f"{model_name:<16} | {roc:<12.4f} | {pr:<12.4f} | {mcc:<10.4f} | {top1:<9.1f}% | {top3:<9.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
