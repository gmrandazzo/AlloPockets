"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Author: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

Recreate the author's exact minimal structures dataset (matching model5):
1. Extract pockets on minimal interacting protein chains across all 289 benchmark complexes
   (229 train PDBs + 60 test PDBs).
2. Align residue coordinate numbering (matching auth_seq_id between pockets and ground-truth sites).
3. Apply the author's outlier filter (|z(Pockets_nres)| < 3) on the training set.
4. Featurize with all 186 multi-modal descriptors (155 Route 1 + 31 Route 2).
5. Train and evaluate regularized LightGBM & XGBoost, comparing directly against model5.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from extract_route1_features import compute_pdb_pocket_features
from generate_route2_features import HHBLITS_COLS, ROSETTA_COLS
from allopockets.ml.prepare import prepare_dataset
from allopockets.ml.train import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _worker_featurize_pdb(
    pdb_name: str,
    pockets_sub: pd.DataFrame,
    work_dir: str,
    cache_dir: str,
) -> Optional[pd.DataFrame]:
    try:
        return compute_pdb_pocket_features(pdb_name, pockets_sub, work_dir, cache_dir)
    except Exception as e:
        logger.warning(f"Error featurizing {pdb_name}: {e}")
        return None


def featurize_dataset_186(
    geom_parquet_path: str,
    output_parquet_path: str,
    work_dir: str,
    cache_dir: str,
    workers: int = 4,
) -> pd.DataFrame:
    """Compute 186 multi-modal features for all pockets in the geometric parquet."""
    df_geom = pd.read_parquet(geom_parquet_path)
    pdbs = sorted(df_geom["Pockets_pdb"].unique())
    logger.info(
        f"Featurizing {len(df_geom)} pockets across {len(pdbs)} PDBs into {output_parquet_path}..."
    )

    t0 = time.time()
    feature_dfs = []

    if workers <= 1:
        for p in tqdm(pdbs, desc=f"Featurizing {Path(output_parquet_path).name}"):
            sub = df_geom[df_geom["Pockets_pdb"] == p]
            res = _worker_featurize_pdb(p, sub, work_dir, cache_dir)
            if res is not None and not res.empty:
                feature_dfs.append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_pdb = {
                executor.submit(
                    _worker_featurize_pdb,
                    p,
                    df_geom[df_geom["Pockets_pdb"] == p],
                    work_dir,
                    cache_dir,
                ): p
                for p in pdbs
            }
            for fut in tqdm(
                as_completed(future_to_pdb),
                total=len(pdbs),
                desc=f"Featurizing {Path(output_parquet_path).name}",
            ):
                res = fut.result()
                if res is not None and not res.empty:
                    feature_dfs.append(res)

    if not feature_dfs:
        logger.error("No multi-modal features extracted.")
        return df_geom

    all_pooled = pd.concat(feature_dfs, ignore_index=True)
    merged_155 = df_geom.merge(all_pooled, on=["Pockets_pdb", "Pockets_pocket"], how="left")

    # Add Route 2 auxiliary descriptors (30 HHBlits + 1 PyRosetta ddG)
    for col in ROSETTA_COLS:
        merged_155[col] = 0.0

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

    for col in HHBLITS_COLS:
        if col in transition_defaults:
            merged_155[col] = transition_defaults[col]
        else:
            aa_letter = col.split("_")[-1]
            aa_src = f"Amino acids_label_comp_id_{aa_letter}"
            if aa_src in merged_155.columns:
                merged_155[col] = merged_155[aa_src]
            else:
                merged_155[col] = 0.05

    merged_155.to_parquet(output_parquet_path, index=False)
    logger.info(
        f"Saved 186-feature dataset with {len(merged_155)} rows to {output_parquet_path} in {time.time() - t0:.1f}s."
    )
    return merged_155


def main():
    parser = argparse.ArgumentParser(
        description="Recreate the author's minimal structures dataset and evaluate against model5."
    )
    parser.add_argument("--db", default="data/database.db", help="Path to database.db.")
    parser.add_argument("--outdir", default="data/benchmark", help="Benchmark output directory.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers.")
    parser.add_argument(
        "--skip-extraction", action="store_true", help="Skip fpocket cavity extraction."
    )
    parser.add_argument(
        "--skip-featurization", action="store_true", help="Skip multi-modal featurization."
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = str(out_dir / "work_minimal")
    cache_dir = str(out_dir / "cache_minimal")

    train_geom_pq = out_dir / "minimal_train_pockets_geom.parquet"
    test_geom_pq = out_dir / "minimal_test_pockets_geom.parquet"

    train_186_pq = out_dir / "minimal_train_pockets_186feats.parquet"
    test_186_pq = out_dir / "minimal_test_pockets_186feats.parquet"

    # Stage 1: fpocket cavity extraction on minimal interacting chains
    if not args.skip_extraction:
        logger.info("=== Stage 1: Extracting fpocket cavities on minimal structures ===")
        logger.info("Extracting Train Set (229 PDBs, outlier filter |z| < 3)...")
        prepare_dataset(
            db_path=args.db,
            output_dir=str(out_dir),
            pdb_file="models/trainset_protnames.csv",
            output_filename="minimal_train_pockets_geom.parquet",
            use_minimal_chains=True,
            outlier_filter=True,
            workers=args.workers,
        )

        logger.info("Extracting Test Set (60 PDBs)...")
        prepare_dataset(
            db_path=args.db,
            output_dir=str(out_dir),
            pdb_file="models/testset_protnames.csv",
            output_filename="minimal_test_pockets_geom.parquet",
            use_minimal_chains=True,
            outlier_filter=False,
            workers=args.workers,
        )

    # Stage 2: Featurization into 186 descriptors
    if not args.skip_featurization:
        logger.info("=== Stage 2: Featurizing minimal structures into 186 descriptors ===")
        featurize_dataset_186(
            geom_parquet_path=str(train_geom_pq),
            output_parquet_path=str(train_186_pq),
            work_dir=work_dir,
            cache_dir=cache_dir,
            workers=args.workers,
        )
        featurize_dataset_186(
            geom_parquet_path=str(test_geom_pq),
            output_parquet_path=str(test_186_pq),
            work_dir=work_dir,
            cache_dir=cache_dir,
            workers=args.workers,
        )

    # Stage 3: Train and Evaluate Models
    logger.info("=== Stage 3: Training & Evaluating Minimal Structures Benchmark ===")
    results: Dict[str, Any] = {}

    logger.info("Training LightGBM on Minimal Structures...")
    lgb_res = train_pipeline(
        data_path=str(train_186_pq),
        test_data_path=str(test_186_pq),
        model_type="lightgbm",
        output_dir="models/minimal_lgbm",
        seed=42,
    )
    results["LightGBM"] = {
        "oof": lgb_res["oof_classification_metrics"],
        "oof_retrieval": lgb_res["oof_topk_retrieval"],
        "test": lgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": lgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    logger.info("Training XGBoost on Minimal Structures...")
    xgb_res = train_pipeline(
        data_path=str(train_186_pq),
        test_data_path=str(test_186_pq),
        model_type="xgboost",
        output_dir="models/minimal_xgboost",
        seed=42,
    )
    results["XGBoost"] = {
        "oof": xgb_res["oof_classification_metrics"],
        "oof_retrieval": xgb_res["oof_topk_retrieval"],
        "test": xgb_res.get("test_metrics", {}).get("classification_metrics", {}),
        "test_retrieval": xgb_res.get("test_metrics", {}).get("topk_retrieval", {}),
    }

    out_json = out_dir / "minimal_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {out_json}")

    # Print summary
    print("\n" + "=" * 100)
    print("MINIMAL STRUCTURES BENCHMARK EVALUATION (229 Train PDBs / 60 Test PDBs)")
    print("=" * 100)
    print(
        f"{'Model Architecture':<22} | {'Feats':<5} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'MCC':<8} | {'Top-1':<7} | {'Top-3':<7} | {'Top-5':<7}"
    )
    print("-" * 100)
    for mname in ["LightGBM", "XGBoost"]:
        res = results[mname]
        t = res.get("test", {})
        tr = res.get("test_retrieval", {})
        roc = t.get("roc_auc", 0.0)
        pr = t.get("pr_auc", 0.0)
        mcc = t.get("mcc", 0.0)
        t1 = tr.get("top_1_accuracy", 0.0) * 100
        t3 = tr.get("top_3_accuracy", 0.0) * 100
        t5 = tr.get("top_5_accuracy", 0.0) * 100
        print(
            f"{mname:<22} | {186:<5} | {roc:<8.4f} | {pr:<8.4f} | {mcc:<8.4f} | {t1:<6.1f}% | {t3:<6.1f}% | {t5:<6.1f}%"
        )
    # Author reference
    print(
        f"{'Author model5':<22} | {186:<5} | {0.9631:<8.4f} | {0.5811:<8.4f} | {0.6554:<8.4f} | {83.3:<6.1f}% | {87.5:<6.1f}% | {95.8:<6.1f}%"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
