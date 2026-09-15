"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Authors: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

Extract Route 1 (155 multi-modal features) for benchmark structures:
- FPocket: 22 geometric cavity features (pre-computed in benchmark parquet)
- Amino Acids: 20 residue frequency one-hot features
- Graphein: 67 Meiler & Expasy physicochemical scales (local static lookups)
- FreeSASA: 9 total, polar, apolar, main/side-chain relative SASA features
- DSSP: 19 secondary structure one-hot & hydrogen-bond energy features
- ProDy: 5 dynamics & allosteric perturbation features (PRS, MechStiff, RMSF, ESSA)
- TransferEntropy: 1 vectorized Kirchhoff GNM transfer entropy feature
- Biopython: 5 half-sphere exposure, contact number, and depth features
- Melodia: 7 backbone curvature, torsion, arc-length, writhing, and propensity features
Total: 155 features.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prevent worker thread oversubscription across OpenBLAS/OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from Bio.SeqUtils import IUPACData

from allopockets.features.aa_scales import get_residues_scales_df
from allopockets.features.classes import (
    BiopythonF,
    DSSPF,
    FreeSASAF,
    MelodiaF,
    ProDyF,
    TransferEntropyF,
)
from allopockets.features.utils import Cif
from allopockets.pockets.pocket import Pocket

AA_LETTERS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "T",
    "V",
    "W",
    "Y",
    "S",
]
LETTER_MAP = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}


def extract_residue_features(pdb: str, work_dir: str) -> pd.DataFrame:
    """Extract all residue-level features for a PDB structure."""
    Cif.path = work_dir
    Cif.original_cifs_path = work_dir
    BiopythonF.dssp_path = shutil.which("mkdssp") or "mkdssp-4.4.0-linux-x64"

    cif = Cif(pdb)
    res = cif.residues[["auth_asym_id", "auth_seq_id", "label_comp_id"]].drop_duplicates()
    if "pdbx_PDB_ins_code" in cif.residues.columns:
        res["pdbx_PDB_ins_code"] = cif.residues["pdbx_PDB_ins_code"]
    else:
        res["pdbx_PDB_ins_code"] = "?"

    # 1. Amino acid 20 frequencies (one-hot)
    res["aa1"] = res["label_comp_id"].astype(str).str.upper().map(LETTER_MAP).fillna("X")
    aa_dummies = pd.get_dummies(res["aa1"], dtype=float)
    for aa in AA_LETTERS:
        if aa not in aa_dummies.columns:
            aa_dummies[aa] = 0.0
    aa_df = pd.concat(
        [
            res[["auth_asym_id", "auth_seq_id"]],
            aa_dummies[AA_LETTERS].add_prefix("Amino acids_label_comp_id_"),
        ],
        axis=1,
    )

    # 2. Graphein (67 Meiler + Expasy scales)
    graphein_raw = get_residues_scales_df(res)
    graphein_cols = [
        c
        for c in graphein_raw.columns
        if c not in ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]
    ]
    graphein_df = graphein_raw[["auth_asym_id", "auth_seq_id"] + graphein_cols].rename(
        columns={c: f"Graphein_{c}" for c in graphein_cols}
    )

    # 3. FreeSASA (9 features)
    try:
        fs = FreeSASAF(cif).freesasa()
        fs_cols = [
            c
            for c in fs.columns
            if c
            not in ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "relative-area_side-chain"]
        ]
        fs_df = fs[["auth_asym_id", "auth_seq_id"] + fs_cols].rename(
            columns={c: f"FreeSASA_{c}" for c in fs_cols}
        )
    except Exception as e:
        print(f"[{pdb}] FreeSASA error: {e}", file=sys.stderr)
        fs_df = pd.DataFrame(columns=["auth_asym_id", "auth_seq_id"])

    # 4. DSSP (19 features)
    try:
        ds = DSSPF(cif).dssp()
        ss_types = ["H", "B", "E", "G", "I", "T", "S", "-"]
        ss_dummies = pd.get_dummies(ds["secondary structure"], dtype=float)
        for st in ss_types:
            if st not in ss_dummies.columns:
                ss_dummies[st] = 0.0
        ss_df = ss_dummies[ss_types].add_prefix("DSSP_secondary structure_")
        num_dssp_cols = [
            "phi",
            "psi",
            "NH_O_1_relidx",
            "NH_O_1_energy",
            "O_NH_1_relidx",
            "O_NH_1_energy",
            "NH_O_2_relidx",
            "NH_O_2_energy",
            "O_NH_2_relidx",
            "O_NH_2_energy",
            "relative ASA",
        ]
        ds_num = ds[num_dssp_cols].replace("NA", np.nan).astype(float).add_prefix("DSSP_")
        dssp_df = pd.concat([ds[["auth_asym_id", "auth_seq_id"]], ds_num, ss_df], axis=1)
    except Exception as e:
        print(f"[{pdb}] DSSP error: {e}", file=sys.stderr)
        dssp_df = pd.DataFrame(columns=["auth_asym_id", "auth_seq_id"])

    # 5. ProDy (5 features)
    try:
        pro = ProDyF(cif)
        prs = pro.prs(n_modes=50)
        stiff = pro.mechstiff(n_modes=50)
        rmsf = pro.rmsf(n_modes=20)
        essa = pro.essa()
        prody_df = (
            prs[["auth_asym_id", "auth_seq_id", "prs_effectiveness", "prs_sensitivity"]]
            .merge(
                stiff[["auth_asym_id", "auth_seq_id", "mechstiff"]],
                on=["auth_asym_id", "auth_seq_id"],
                how="outer",
            )
            .merge(
                rmsf[["auth_asym_id", "auth_seq_id", "rmsf"]],
                on=["auth_asym_id", "auth_seq_id"],
                how="outer",
            )
            .merge(
                essa[["auth_asym_id", "auth_seq_id", "essa"]],
                on=["auth_asym_id", "auth_seq_id"],
                how="outer",
            )
            .rename(
                columns={
                    "prs_effectiveness": "ProDy_prs_effectiveness",
                    "prs_sensitivity": "ProDy_prs_sensitivity",
                    "mechstiff": "ProDy_mechstiff",
                    "rmsf": "ProDy_rmsf",
                    "essa": "ProDy_essa",
                }
            )
        )
    except Exception as e:
        print(f"[{pdb}] ProDy error: {e}", file=sys.stderr)
        prody_df = pd.DataFrame(columns=["auth_asym_id", "auth_seq_id"])

    # 6. TransferEntropy (1 feature)
    try:
        te_df = (
            TransferEntropyF(cif)
            .transfer_entropy()[["auth_asym_id", "auth_seq_id", "TE"]]
            .rename(columns={"TE": "TransferEntropy_TE"})
        )
    except Exception as e:
        print(f"[{pdb}] TransferEntropy error: {e}", file=sys.stderr)
        te_df = pd.DataFrame(columns=["auth_asym_id", "auth_seq_id"])

    # 7. Biopython (5 features)
    try:
        bio = BiopythonF(cif)
        cb = bio.exposureCB()[["auth_asym_id", "auth_seq_id", "EXP_HSE_B_U", "EXP_HSE_B_D"]]
        cn = bio.exposureCN()[["auth_asym_id", "auth_seq_id", "EXP_CN"]]
        rd = bio.residue_depth()[["auth_asym_id", "auth_seq_id", "EXP_RD", "EXP_RD_CA"]]
        biopy_df = (
            cb.merge(cn, on=["auth_asym_id", "auth_seq_id"], how="outer")
            .merge(rd, on=["auth_asym_id", "auth_seq_id"], how="outer")
            .rename(
                columns={
                    "EXP_HSE_B_U": "Biopython_EXP_HSE_B_U",
                    "EXP_HSE_B_D": "Biopython_EXP_HSE_B_D",
                    "EXP_CN": "Biopython_EXP_CN",
                    "EXP_RD": "Biopython_EXP_RD",
                    "EXP_RD_CA": "Biopython_EXP_RD_CA",
                }
            )
        )
    except Exception as e:
        print(f"[{pdb}] Biopython error: {e}", file=sys.stderr)
        biopy_df = pd.DataFrame(columns=["auth_asym_id", "auth_seq_id"])

    # 8. Melodia (7 features)
    try:
        mel = MelodiaF(cif).melodia()
        mel_cols = ["curvature", "torsion", "arc_len", "writhing", "phi", "psi", "propensity"]
        mel_df = mel[["auth_asym_id", "auth_seq_id"] + mel_cols].rename(
            columns={c: f"Melodia_{c}" for c in mel_cols}
        )
    except Exception as e:
        print(f"[{pdb}] Melodia error: {e}", file=sys.stderr)
        mel_df = pd.DataFrame(columns=["auth_asym_id", "auth_seq_id"])

    # Merge all parts on auth_asym_id and auth_seq_id
    merged = res[["auth_asym_id", "auth_seq_id"]].drop_duplicates()
    for part in [aa_df, graphein_df, fs_df, dssp_df, prody_df, te_df, biopy_df, mel_df]:
        if len(part) > 0:
            merged = merged.merge(
                part.drop_duplicates(subset=["auth_asym_id", "auth_seq_id"]),
                on=["auth_asym_id", "auth_seq_id"],
                how="left",
            )
    return merged


def compute_pdb_pocket_features(
    pdb: str, pockets_subset: pd.DataFrame, work_dir: str, cache_dir: str
) -> pd.DataFrame:
    """Compute and pool residue features for all pockets of a given PDB."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = Path(cache_dir) / f"{pdb}_155feats.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    t0 = time.time()
    res_feats = extract_residue_features(pdb, work_dir)

    rows = []
    for _, prow in pockets_subset.iterrows():
        pname = str(prow["Pockets_pocket"])
        pocket_cif = Path(work_dir) / pdb / f"{pdb}_out" / "pockets" / f"{pname}_atm.cif"
        if not pocket_cif.exists():
            continue
        try:
            pkt = Pocket(str(pocket_cif))
            pkt_res = pkt.residues[["auth_asym_id", "auth_seq_id"]].drop_duplicates()
            sub = res_feats.merge(pkt_res, on=["auth_asym_id", "auth_seq_id"])
            means = (
                sub.drop(columns=["auth_asym_id", "auth_seq_id"]).mean(numeric_only=True).to_dict()
            )
            record = {"Pockets_pdb": pdb, "Pockets_pocket": pname}
            record.update(means)
            rows.append(record)
        except Exception as e:
            print(f"[{pdb}][{pname}] Pocket pooling error: {e}", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.to_parquet(cache_file)
    print(f"[{pdb}] Processed {len(rows)} pockets in {time.time() - t0:.1f}s")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Extract Route 1 155 features for benchmark.")
    parser.add_argument("--data-dir", default="data/benchmark", help="Benchmark data directory.")
    parser.add_argument("--pdb", default=None, help="Process single PDB only for testing.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel worker processes."
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform single dry-run check.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    work_dir = str(data_dir / "work")
    cache_dir = str(data_dir / "cache")

    train_pq = data_dir / "curated_train_pockets.parquet"
    test_pq = data_dir / "curated_test_pockets.parquet"

    train_df = pd.read_parquet(train_pq)
    test_df = pd.read_parquet(test_pq)

    all_pockets = pd.concat([train_df, test_df], ignore_index=True)
    all_pdbs = sorted(all_pockets["Pockets_pdb"].unique())

    if args.pdb:
        all_pdbs = [args.pdb]

    print(
        f"Starting Route 1 feature extraction for {len(all_pdbs)} PDBs using {args.workers} workers..."
    )

    from concurrent.futures import ProcessPoolExecutor, as_completed

    all_results = []
    # If single worker or single PDB, run sequentially
    if args.workers <= 1 or len(all_pdbs) == 1:
        for i, pdb in enumerate(all_pdbs, 1):
            print(f"[{i}/{len(all_pdbs)}] Processing PDB: {pdb}")
            sub_pockets = all_pockets[all_pockets["Pockets_pdb"] == pdb]
            df_pdb = compute_pdb_pocket_features(pdb, sub_pockets, work_dir, cache_dir)
            if not df_pdb.empty:
                all_results.append(df_pdb)
            if args.dry_run:
                break
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for pdb in all_pdbs:
                sub_pockets = all_pockets[all_pockets["Pockets_pdb"] == pdb]
                fut = executor.submit(
                    compute_pdb_pocket_features, pdb, sub_pockets, work_dir, cache_dir
                )
                futures[fut] = pdb

            completed_count = 0
            for fut in as_completed(futures):
                pdb = futures[fut]
                completed_count += 1
                try:
                    df_pdb = fut.result()
                    if not df_pdb.empty:
                        all_results.append(df_pdb)
                    print(f"[{completed_count}/{len(all_pdbs)}] Finished PDB: {pdb}")
                except Exception as e:
                    print(
                        f"[{completed_count}/{len(all_pdbs)}] Error on PDB {pdb}: {e}",
                        file=sys.stderr,
                    )

    if args.dry_run or not all_results:
        print("Dry run completed successfully.")
        return

    pooled_df = pd.concat(all_results, ignore_index=True)

    # Merge with original train and test datasets
    train_155 = train_df.merge(pooled_df, on=["Pockets_pdb", "Pockets_pocket"], how="left")
    test_155 = test_df.merge(pooled_df, on=["Pockets_pdb", "Pockets_pocket"], how="left")

    out_train = data_dir / "curated_train_pockets_155feats.parquet"
    out_test = data_dir / "curated_test_pockets_155feats.parquet"

    train_155.to_parquet(out_train)
    test_155.to_parquet(out_test)

    feature_cols = [
        c
        for c in train_155.columns
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

    print("\n=== Extraction Complete ===")
    print(f"Train set: {train_155.shape} saved to {out_train}")
    print(f"Test set:  {test_155.shape} saved to {out_test}")
    print(f"Extracted feature count: {len(feature_cols)}")


if __name__ == "__main__":
    main()
