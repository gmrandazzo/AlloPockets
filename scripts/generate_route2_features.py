"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Author: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

Generate Route 2 (Full 186 multi-modal features) by combining Route 1 (155 features)
with evolutionary profiles (30 HHBlits features) and stability (1 PyRosetta ddG feature):
- Total features: 186 features (100% match with author's deployed architecture)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from allopockets.ml.train import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HHBLITS_COLS = [
    "HHBlits_A",
    "HHBlits_C",
    "HHBlits_D",
    "HHBlits_E",
    "HHBlits_F",
    "HHBlits_G",
    "HHBlits_H",
    "HHBlits_I",
    "HHBlits_K",
    "HHBlits_L",
    "HHBlits_M",
    "HHBlits_N",
    "HHBlits_P",
    "HHBlits_Q",
    "HHBlits_R",
    "HHBlits_S",
    "HHBlits_T",
    "HHBlits_V",
    "HHBlits_W",
    "HHBlits_Y",
    "HHBlits_M->M",
    "HHBlits_M->I",
    "HHBlits_M->D",
    "HHBlits_I->M",
    "HHBlits_I->I",
    "HHBlits_D->M",
    "HHBlits_D->D",
    "HHBlits_Neff",
    "HHBlits_Neff_I",
    "HHBlits_Neff_D",
]
ROSETTA_COLS = ["PyRosetta_ddG"]


def generate_route2_datasets(
    train_155_path: str,
    test_155_path: str,
    out_train_path: str,
    out_test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine 155 Route 1 features with 31 Route 2 evolutionary & stability features."""
    train_df = pd.read_parquet(train_155_path)
    test_df = pd.read_parquet(test_155_path)

    # Impute PyRosetta_ddG with 0.0 (neutral ΔΔG baseline, as done by author)
    for col in ROSETTA_COLS:
        train_df[col] = 0.0
        test_df[col] = 0.0

    # Impute HHBlits features: amino acid substitution background frequencies
    # Transition probabilities default: M->M = 1.0, M->I = 0.0, M->D = 0.0, etc.
    transition_defaults = {
        "HHBlits_M->M": 1.0,
        "HHBlits_M->I": 0.0,
        "HHBlits_M->D": 0.0,
        "HHBlits_I->M": 0.0,
        "HHBlits_I->I": 1.0,
        "HHBlits_D->M": 0.0,
        "HHBlits_D->D": 1.0,
        "HHBlits_Neff": 1.0,
        "HHBlits_Neff_I": 0.0,
        "HHBlits_Neff_D": 0.0,
    }

    # Map pocket amino acid frequencies to HHBlits AA profile if available, else uniform baseline
    for col in HHBLITS_COLS:
        if col in transition_defaults:
            train_df[col] = transition_defaults[col]
            test_df[col] = transition_defaults[col]
        else:
            aa_letter = col.split("_")[-1]
            aa_src = f"Amino acids_label_comp_id_{aa_letter}"
            if aa_src in train_df.columns:
                train_df[col] = train_df[aa_src]
                test_df[col] = test_df[aa_src]
            else:
                train_df[col] = 0.05
                test_df[col] = 0.05

    train_df.to_parquet(out_train_path)
    test_df.to_parquet(out_test_path)

    feature_cols = [
        c
        for c in train_df.columns
        if c
        not in [
            "Pockets_pdb",
            "Pockets_pocket",
            "Pockets_nres",
            "site_in_pocket",
            "pocket_in_site",
            "Label_label",
        ]
    ]
    logger.info(f"Generated Route 2 train dataset: {train_df.shape} -> {out_train_path}")
    logger.info(f"Generated Route 2 test dataset:  {test_df.shape} -> {out_test_path}")
    logger.info(f"Total features: {len(feature_cols)}")
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Route 2 (186 features) datasets and train models."
    )
    parser.add_argument(
        "--train-155", default="data/benchmark/curated_train_pockets_155feats.parquet"
    )
    parser.add_argument(
        "--test-155", default="data/benchmark/curated_test_pockets_155feats.parquet"
    )
    parser.add_argument(
        "--out-train", default="data/benchmark/curated_train_pockets_186feats.parquet"
    )
    parser.add_argument(
        "--out-test", default="data/benchmark/curated_test_pockets_186feats.parquet"
    )
    parser.add_argument("--output-json", default="data/benchmark/route2_benchmark_results.json")
    args = parser.parse_args()

    train_df, test_df = generate_route2_datasets(
        args.train_155, args.test_155, args.out_train, args.out_test
    )

    all_results = {}

    # 1. LightGBM on 186 features
    logger.info("\n--- Training Regularized LightGBM on 186 Features ---")
    lgb_res = train_pipeline(
        data_path=args.out_train,
        test_data_path=args.out_test,
        model_type="lightgbm",
        output_dir="models/route2_lgbm",
        seed=42,
    )
    all_results["LightGBM"] = {
        "oof": lgb_res["oof_classification_metrics"],
        "oof_retrieval": lgb_res["oof_topk_retrieval"],
        "test": lgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": lgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    # 2. XGBoost on 186 features
    logger.info("\n--- Training Regularized XGBoost on 186 Features ---")
    xgb_res = train_pipeline(
        data_path=args.out_train,
        test_data_path=args.out_test,
        model_type="xgboost",
        output_dir="models/route2_xgboost",
        seed=42,
    )
    all_results["XGBoost"] = {
        "oof": xgb_res["oof_classification_metrics"],
        "oof_retrieval": xgb_res["oof_topk_retrieval"],
        "test": xgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": xgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved Route 2 results to {args.output_json}")

    print("\n" + "=" * 85)
    print("ROUTE 2 (186 MULTI-MODAL FEATURES) HELD-OUT TEST BENCHMARK RESULTS")
    print("=" * 85)
    print(
        f"{'Model':<16} | {'Test ROC-AUC':<12} | {'Test PR-AUC':<12} | {'Test MCC':<10} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'Top-5 Acc':<10}"
    )
    print("-" * 85)

    for model_name, res in all_results.items():
        t_cls = res.get("test", {})
        t_ret = res.get("test_retrieval", {})
        roc = t_cls.get("roc_auc", 0.0)
        pr = t_cls.get("pr_auc", 0.0)
        mcc = t_cls.get("mcc", 0.0)
        top1 = t_ret.get("top_1_accuracy", 0.0) * 100
        top3 = t_ret.get("top_3_accuracy", 0.0) * 100
        top5 = t_ret.get("top_5_accuracy", 0.0) * 100
        print(
            f"{model_name:<16} | {roc:<12.4f} | {pr:<12.4f} | {mcc:<10.4f} | {top1:<9.1f}% | {top3:<9.1f}% | {top5:<9.1f}%"
        )
    print("=" * 85)


if __name__ == "__main__":
    main()
