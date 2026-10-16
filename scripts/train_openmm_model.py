#!/usr/bin/env python3
"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Train OpenMM-based Model (186 features)
- Replaces PyRosetta ddG with OpenMM ddG
- Supports per-PDB atomic checkpointing (resumes automatically upon interruption)
- Multi-core parallel execution with configurable workers
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import argparse
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust path so we can import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allopockets.features.classes import OpenMMF
from allopockets.features.utils import Cif
from allopockets.pockets import Pocket
from allopockets.ml.train import train_pipeline
from scripts.generate_route2_features import generate_route2_datasets


def process_pdb_openmm(args):
    pdb, pockets_subset, work_dir, cache_dir, force = args
    cache_file = Path(cache_dir) / f"{pdb}.parquet"

    # Checkpoint recovery: skip if already computed
    if not force and cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            print(f"[{pdb}] Loaded from checkpoint cache ({len(cached_df)} pockets).")
            return cached_df
        except Exception as e:
            print(f"[{pdb}] Corrupt cache file, recomputing: {e}")

    print(f"[{pdb}] Starting OpenMM feature extraction...")
    t0 = time.time()

    try:
        Cif.path = work_dir
        Cif.original_cifs_path = work_dir
        cif = Cif(pdb)

        # Collect allowed residues across all pockets
        allowed_residues = set()
        valid_pockets = []

        for _, prow in pockets_subset.iterrows():
            pname = str(prow["Pockets_pocket"])
            pocket_cif = Path(work_dir) / pdb / f"{pdb}_out" / "pockets" / f"{pname}_atm.cif"
            if not pocket_cif.exists():
                continue

            try:
                pkt = Pocket(str(pocket_cif))
                pkt_res = pkt.residues[["auth_asym_id", "auth_seq_id"]].drop_duplicates()
                for _, r in pkt_res.iterrows():
                    allowed_residues.add((str(r["auth_asym_id"]), str(r["auth_seq_id"])))
                valid_pockets.append((pname, pkt_res))
            except Exception as e:
                print(f"[{pdb}][{pname}] Pocket load error: {e}")

        if not valid_pockets:
            print(f"[{pdb}] No valid pockets found.")
            empty_df = pd.DataFrame(columns=["Pockets_pdb", "Pockets_pocket", "OpenMMF_ddG_openmm"])
            empty_df.to_parquet(cache_file)
            return empty_df

        openmm_f = OpenMMF(cif)
        results = list(openmm_f.__iter__(allowed_residues=allowed_residues))

        if not results:
            print(f"[{pdb}] No OpenMM results generated.")
            empty_df = pd.DataFrame(columns=["Pockets_pdb", "Pockets_pocket", "OpenMMF_ddG_openmm"])
            empty_df.to_parquet(cache_file)
            return empty_df

        res_feats = pd.DataFrame(results)
        res_feats["auth_asym_id"] = res_feats["auth_asym_id"].astype(str)
        res_feats["auth_seq_id"] = res_feats["auth_seq_id"].astype(str)

        rows = []
        for pname, pkt_res in valid_pockets:
            pkt_res = pkt_res.copy()
            pkt_res["auth_asym_id"] = pkt_res["auth_asym_id"].astype(str)
            pkt_res["auth_seq_id"] = pkt_res["auth_seq_id"].astype(str)

            sub = res_feats.merge(pkt_res, on=["auth_asym_id", "auth_seq_id"])
            if sub.empty:
                continue

            mean_val = float(sub["ddG_openmm"].mean())
            rows.append(
                {"Pockets_pdb": pdb, "Pockets_pocket": pname, "OpenMMF_ddG_openmm": mean_val}
            )

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["Pockets_pdb", "Pockets_pocket", "OpenMMF_ddG_openmm"])

        df.to_parquet(cache_file)
        print(f"[{pdb}] Finished OpenMM extraction in {time.time()-t0:.1f}s (saved to cache)")
        return df

    except Exception as e:
        import traceback

        print(f"[{pdb}] OpenMM extraction failed: {e}")
        traceback.print_exc()
        empty_df = pd.DataFrame(columns=["Pockets_pdb", "Pockets_pocket", "OpenMMF_ddG_openmm"])
        empty_df.to_parquet(cache_file)
        return empty_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenMM features and train 186-feature model"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument("--force", action="store_true", help="Force recomputation of cached PDBs")
    args = parser.parse_args()

    data_dir = Path("data/benchmark")
    work_dir = str(data_dir / "work")
    cache_dir = data_dir / "cache_openmm"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_155 = data_dir / "curated_train_pockets_155feats.parquet"
    test_155 = data_dir / "curated_test_pockets_155feats.parquet"

    if not train_155.exists():
        print(f"Error: {train_155} not found. Please run feature extraction first.")
        sys.exit(1)

    # Load or generate route 2 datasets
    train_186_path = data_dir / "curated_train_pockets_186feats.parquet"
    test_186_path = data_dir / "curated_test_pockets_186feats.parquet"

    if train_186_path.exists() and test_186_path.exists():
        train_df = pd.read_parquet(train_186_path)
        test_df = pd.read_parquet(test_186_path)
    else:
        train_df, test_df = generate_route2_datasets(
            str(train_155),
            str(test_155),
            str(data_dir / "temp_train.parquet"),
            str(data_dir / "temp_test.parquet"),
        )

    all_pockets = pd.concat([train_df, test_df], ignore_index=True)
    all_pdbs = all_pockets["Pockets_pdb"].unique()
    print(f"Total benchmark complexes: {len(all_pdbs)} PDBs across {len(all_pockets)} pockets.")

    openmm_cache = data_dir / "openmm_ddg_cache.parquet"

    if openmm_cache.exists() and not args.force:
        print(f"Loading OpenMM consolidated features from {openmm_cache}")
        openmm_df = pd.read_parquet(openmm_cache)
    else:
        # Check existing per-PDB cache
        cached_pdbs = set()
        for f in cache_dir.glob("*.parquet"):
            cached_pdbs.add(f.stem)

        pending_pdbs = [p for p in all_pdbs if p not in cached_pdbs or args.force]
        print(
            f"Checkpoint cache status: {len(cached_pdbs & set(all_pdbs))}/{len(all_pdbs)} PDBs already completed."
        )
        print(f"Remaining to process: {len(pending_pdbs)} PDBs using {args.workers} workers.")

        if pending_pdbs:
            pool_args = []
            for pdb in pending_pdbs:
                group = all_pockets[all_pockets["Pockets_pdb"] == pdb]
                pool_args.append((pdb, group, work_dir, str(cache_dir), args.force))

            with mp.Pool(args.workers) as pool:
                pool.map(process_pdb_openmm, pool_args)

        # Consolidate all per-PDB parquet files
        print("Consolidating per-PDB checkpoint parquet files into openmm_ddg_cache.parquet...")
        all_dfs = []
        for pdb in all_pdbs:
            p_file = cache_dir / f"{pdb}.parquet"
            if p_file.exists():
                try:
                    df_p = pd.read_parquet(p_file)
                    if not df_p.empty:
                        all_dfs.append(df_p)
                except Exception as e:
                    print(f"Error reading {p_file}: {e}")

        if all_dfs:
            openmm_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(
                subset=["Pockets_pdb", "Pockets_pocket"]
            )
        else:
            openmm_df = pd.DataFrame(
                columns=["Pockets_pdb", "Pockets_pocket", "OpenMMF_ddG_openmm"]
            )

        openmm_df.to_parquet(openmm_cache)
        print(f"Saved {len(openmm_df)} pocket OpenMM features to {openmm_cache}")

    print("Merging OpenMM features with the main dataset...")
    # Drop existing ddG column if present
    for col in ["PyRosetta_ddG", "OpenMMF_ddG_openmm"]:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    train_df = train_df.merge(openmm_df, on=["Pockets_pdb", "Pockets_pocket"], how="left")
    test_df = test_df.merge(openmm_df, on=["Pockets_pdb", "Pockets_pocket"], how="left")

    train_df["OpenMMF_ddG_openmm"] = train_df["OpenMMF_ddG_openmm"].fillna(0.0)
    test_df["OpenMMF_ddG_openmm"] = test_df["OpenMMF_ddG_openmm"].fillna(0.0)

    openmm_train_path = data_dir / "curated_train_pockets_openmm_186feats.parquet"
    openmm_test_path = data_dir / "curated_test_pockets_openmm_186feats.parquet"
    train_df.to_parquet(openmm_train_path)
    test_df.to_parquet(openmm_test_path)
    print(f"Saved OpenMM training data: {openmm_train_path} ({train_df.shape})")
    print(f"Saved OpenMM test data:     {openmm_test_path} ({test_df.shape})")

    all_results = {}

    print("\n--- Training Regularized LightGBM on OpenMM 186 Features ---")
    lgb_res = train_pipeline(
        data_path=openmm_train_path,
        test_data_path=openmm_test_path,
        model_type="lightgbm",
        output_dir="models/openmm_lgbm",
        seed=42,
    )
    all_results["LightGBM"] = {
        "oof": lgb_res["oof_classification_metrics"],
        "oof_retrieval": lgb_res["oof_topk_retrieval"],
        "test": lgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": lgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    print("\n--- Training Regularized XGBoost on OpenMM 186 Features ---")
    xgb_res = train_pipeline(
        data_path=openmm_train_path,
        test_data_path=openmm_test_path,
        model_type="xgboost",
        output_dir="models/openmm_xgboost",
        seed=42,
    )
    all_results["XGBoost"] = {
        "oof": xgb_res["oof_classification_metrics"],
        "oof_retrieval": xgb_res["oof_topk_retrieval"],
        "test": xgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": xgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    results_json = data_dir / "openmm_benchmark_results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 85)
    print("OPENMM (186 MULTI-MODAL FEATURES) HELD-OUT TEST BENCHMARK RESULTS")
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
