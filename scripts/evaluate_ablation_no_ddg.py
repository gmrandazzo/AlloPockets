#!/usr/bin/env python3
"""
AlloPockets - Ablation Study: Model Performance Without ddG (185 features)
Compares:
1. Route 1 (155 features: Geometric/PhysChem/Dynamics only)
2. No-ddG Route 2 (185 features: Route 1 + 30 HHBlits MSA features, WITHOUT ddG)
3. Full Route 2 (186 features: Route 1 + 30 HHBlits MSA features + PyRosetta ddG)
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust path so we can import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allopockets.ml.train import train_pipeline


def main():
    data_dir = Path("data/benchmark")
    train_186_path = data_dir / "curated_train_pockets_186feats.parquet"
    test_186_path = data_dir / "curated_test_pockets_186feats.parquet"

    if not train_186_path.exists() or not test_186_path.exists():
        print(f"Error: 186-feature dataset not found in {data_dir}")
        sys.exit(1)

    print("Loading 186-feature benchmark datasets...")
    train_df = pd.read_parquet(train_186_path)
    test_df = pd.read_parquet(test_186_path)

    # Drop ddG columns
    ddg_cols_to_drop = [
        c for c in ["PyRosetta_ddG", "OpenMMF_ddG_openmm", "ddG"] if c in train_df.columns
    ]
    print(f"Dropping ddG columns: {ddg_cols_to_drop}")

    train_185 = train_df.drop(columns=ddg_cols_to_drop)
    test_185 = test_df.drop(columns=ddg_cols_to_drop)

    out_train_path = data_dir / "curated_train_pockets_185feats_no_ddg.parquet"
    out_test_path = data_dir / "curated_test_pockets_185feats_no_ddg.parquet"

    train_185.to_parquet(out_train_path)
    test_185.to_parquet(out_test_path)
    print(f"Saved 185-feature train dataset: {out_train_path} ({train_185.shape})")
    print(f"Saved 185-feature test dataset:  {out_test_path} ({test_185.shape})")

    results = {}

    # 1. Train LightGBM without ddG
    print("\n--- Training Regularized LightGBM on 185 Features (NO ddG) ---")
    lgb_res = train_pipeline(
        data_path=out_train_path,
        test_data_path=out_test_path,
        model_type="lightgbm",
        output_dir="models/no_ddg_lgbm",
        seed=42,
    )
    results["LightGBM_no_ddG"] = {
        "n_features": lgb_res["n_features"],
        "oof": lgb_res["oof_classification_metrics"],
        "oof_retrieval": lgb_res["oof_topk_retrieval"],
        "test": lgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": lgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    # 2. Train XGBoost without ddG
    print("\n--- Training Regularized XGBoost on 185 Features (NO ddG) ---")
    xgb_res = train_pipeline(
        data_path=out_train_path,
        test_data_path=out_test_path,
        model_type="xgboost",
        output_dir="models/no_ddg_xgboost",
        seed=42,
    )
    results["XGBoost_no_ddG"] = {
        "n_features": xgb_res["n_features"],
        "oof": xgb_res["oof_classification_metrics"],
        "oof_retrieval": xgb_res["oof_topk_retrieval"],
        "test": xgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": xgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    out_json = data_dir / "ablation_no_ddg_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Load Route 1 and Route 2 benchmark results if available
    r1_json = data_dir / "route1_benchmark_results.json"
    r2_json = data_dir / "route2_benchmark_results.json"
    r1_data = json.load(open(r1_json)) if r1_json.exists() else {}
    r2_data = json.load(open(r2_json)) if r2_json.exists() else {}

    print("\n" + "=" * 95)
    print("ABLATION STUDY: IMPACT OF ddG DESCRIPTOR ON HELD-OUT TEST BENCHMARK")
    print("=" * 95)
    header = f"{'Configuration':<30} | {'Test ROC-AUC':<12} | {'Test PR-AUC':<12} | {'Test MCC':<10} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10}"
    print(header)
    print("-" * 95)

    def print_row(name, t_cls, t_ret):
        roc = t_cls.get("roc_auc", 0.0)
        pr = t_cls.get("pr_auc", 0.0)
        mcc = t_cls.get("mcc", 0.0)
        top1 = t_ret.get("top_1_accuracy", 0.0) * 100
        top3 = t_ret.get("top_3_accuracy", 0.0) * 100
        print(
            f"{name:<30} | {roc:<12.4f} | {pr:<12.4f} | {mcc:<10.4f} | {top1:<9.1f}% | {top3:<9.1f}%"
        )

    if "LightGBM" in r1_data:
        print_row(
            "Route 1 LightGBM (155f, no ddG)",
            r1_data["LightGBM"].get("test", {}),
            r1_data["LightGBM"].get("test_retrieval", {}),
        )
    print_row(
        "Route 2 LightGBM (185f, no ddG)",
        results["LightGBM_no_ddG"]["test"],
        results["LightGBM_no_ddG"]["test_retrieval"],
    )
    if "LightGBM" in r2_data:
        print_row(
            "Route 2 LightGBM (186f, +PyRosetta)",
            r2_data["LightGBM"].get("test", {}),
            r2_data["LightGBM"].get("test_retrieval", {}),
        )

    print("-" * 95)

    if "XGBoost" in r1_data:
        print_row(
            "Route 1 XGBoost (155f, no ddG)",
            r1_data["XGBoost"].get("test", {}),
            r1_data["XGBoost"].get("test_retrieval", {}),
        )
    print_row(
        "Route 2 XGBoost (185f, no ddG)",
        results["XGBoost_no_ddG"]["test"],
        results["XGBoost_no_ddG"]["test_retrieval"],
    )
    if "XGBoost" in r2_data:
        print_row(
            "Route 2 XGBoost (186f, +PyRosetta)",
            r2_data["XGBoost"].get("test", {}),
            r2_data["XGBoost"].get("test_retrieval", {}),
        )

    print("=" * 95)


if __name__ == "__main__":
    main()
