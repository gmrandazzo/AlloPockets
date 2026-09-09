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

Data preparation pipeline to generate training datasets directly from database.db.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from tqdm import tqdm

from allopockets.pockets.fpocket import run_fpocket
from allopockets.pockets.pocket import Pocket

logger = logging.getLogger(__name__)


def get_site_residue_keys(site_json: Dict) -> Set[Tuple[str, str]]:
    """
    Extract set of (chain, res_num) from the stored site JSON in database.db.
    """
    if not site_json:
        return set()

    chains = site_json.get("label_asym_id") or site_json.get("auth_asym_id") or []
    seqs = site_json.get("label_seq_id") or site_json.get("auth_seq_id") or []

    res_set = set()
    for c, s in zip(chains, seqs):
        res_set.add((str(c), str(s)))
    return res_set


def process_pdb_structure(
    entry_id: str,
    site_records: List[Dict],
    work_dir: Path,
    label_threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Process a single PDB structure:
    1. Fetch CIF and run fpocket.
    2. Extract pocket residues and geometry.
    3. Calculate overlap with ground-truth allosteric site(s).
    4. Return DataFrame of candidate pockets.
    """
    from allopockets.predict import get_cif, get_clean_pdb

    pdb = get_cif(pdb_id=entry_id, path=str(work_dir))
    clean_pdb = get_clean_pdb(pdb, protein_chains=None, path=str(work_dir))

    # Run fpocket
    pockets_df = run_fpocket(clean_pdb, path=str(work_dir))
    if pockets_df.empty:
        return pd.DataFrame()

    # Collect all site residue sets for this PDB
    all_site_residues = []
    for s_rec in site_records:
        s_keys = get_site_residue_keys(s_rec.get("site", {}))
        if s_keys:
            all_site_residues.append(s_keys)

    rows = []
    for _, row in pockets_df.iterrows():
        pname = row["pocket"]
        cif_path = work_dir / entry_id / f"{entry_id}_out" / "pockets" / f"{pname}_atm.cif"
        if not cif_path.exists():
            continue

        pkt = Pocket(str(cif_path))
        feats = pkt.feats

        # Extract contacting residues in pocket
        pocket_res_set = set()
        if hasattr(pkt, "residues") and not pkt.residues.empty:
            for _, r in pkt.residues.iterrows():
                cid = r.get("label_asym_id") or r.get("auth_asym_id", "")
                seqid = r.get("label_seq_id") or r.get("auth_seq_id", "")
                pocket_res_set.add((str(cid), str(seqid)))

        # Compute max overlap against all annotated sites for this PDB
        max_site_in_pocket = 0.0
        max_pocket_in_site = 0.0

        for s_keys in all_site_residues:
            if not s_keys:
                continue
            common = pocket_res_set.intersection(s_keys)
            site_in_pkt = len(common) / len(s_keys) if len(s_keys) > 0 else 0.0
            pkt_in_site = len(common) / len(pocket_res_set) if len(pocket_res_set) > 0 else 0.0
            if site_in_pkt > max_site_in_pocket:
                max_site_in_pocket = site_in_pkt
            if pkt_in_site > max_pocket_in_site:
                max_pocket_in_site = pkt_in_site

        label = 1 if max_site_in_pocket >= label_threshold else 0

        # Assemble row
        row_data = {
            "Pockets_pdb": entry_id,
            "Pockets_pocket": pname,
            "Pockets_nres": len(pocket_res_set),
            "site_in_pocket": max_site_in_pocket,
            "pocket_in_site": max_pocket_in_site,
            "Label_label": label,
        }
        # Add fpocket features with FPocket_ prefix
        for k, v in feats.items():
            row_data[f"FPocket_{k}"] = v

        rows.append(row_data)

    return pd.DataFrame(rows)


def prepare_dataset(
    db_path: str = "data/database.db",
    output_dir: str = "data",
    limit: Optional[int] = None,
    pdb_list: Optional[List[str]] = None,
    label_threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Main entrypoint to generate training data directly from database.db.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    resolved_db = Path(db_path)
    if not resolved_db.exists() and Path("database.db").exists():
        resolved_db = Path("database.db")

    out_path = Path(output_dir)
    cache_dir = out_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database.db
    conn = sqlite3.connect(str(resolved_db))
    cursor = conn.cursor()

    if pdb_list:
        placeholders = ",".join(["?"] * len(pdb_list))
        query = f"SELECT entry_id FROM pdb WHERE entry_id IN ({placeholders})"  # nosec B608
        cursor.execute(query, pdb_list)
    else:
        cursor.execute("SELECT entry_id FROM pdb")

    all_entries = [r[0] for r in cursor.fetchall()]
    if limit:
        all_entries = all_entries[:limit]

    logger.info(f"Preparing dataset for {len(all_entries)} structures from {db_path}...")

    results = []
    for entry_id in tqdm(all_entries, desc="Extracting PDB complexes"):
        cache_file = cache_dir / f"{entry_id}.parquet"
        if cache_file.exists():
            df_entry = pd.read_parquet(cache_file)
            results.append(df_entry)
            continue

        # Fetch site records for this PDB
        cursor.execute("SELECT site, modulator, info FROM site WHERE pdb_id = ?", (entry_id,))
        site_rows = cursor.fetchall()
        site_records = []
        for s_raw, m_raw, i_raw in site_rows:
            try:
                site_records.append(
                    {
                        "site": json.loads(s_raw) if isinstance(s_raw, str) else s_raw,
                        "modulator": json.loads(m_raw) if isinstance(m_raw, str) else m_raw,
                        "info": json.loads(i_raw) if isinstance(i_raw, str) else i_raw,
                    }
                )
            except Exception:
                continue

        try:
            df_entry = process_pdb_structure(
                entry_id=entry_id,
                site_records=site_records,
                work_dir=out_path / "work",
                label_threshold=label_threshold,
            )
            if not df_entry.empty:
                df_entry.to_parquet(cache_file, index=False)
                results.append(df_entry)
        except Exception as e:
            logger.warning(f"Failed to process {entry_id}: {e}")
            continue

    if not results:
        logger.error("No pockets extracted.")
        return pd.DataFrame()

    full_dataset = pd.concat(results, ignore_index=True)
    full_dataset.to_parquet(out_path / "pockets_dataset.parquet", index=False)
    full_dataset.to_pickle(out_path / "pockets_dataset.pkl")

    pos_count = (full_dataset["Label_label"] == 1).sum()
    neg_count = (full_dataset["Label_label"] == 0).sum()
    logger.info(
        f"Dataset ready: {len(full_dataset)} total pockets across {len(results)} PDBs. "
        f"Positives: {pos_count} ({pos_count/len(full_dataset)*100:.1f}%), Negatives: {neg_count}."
    )
    return full_dataset
