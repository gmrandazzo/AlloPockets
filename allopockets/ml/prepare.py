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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def get_site_residue_keys(
    site_json: Dict, info_json: Optional[Dict] = None
) -> Set[Tuple[str, str]]:
    """
    Extract set of (chain, auth_seq_num) from the stored site JSON in database.db.
    Matches author curation by filtering to protein-interacting chains and auth_seq_id.
    """
    if not site_json:
        return set()

    ic_chains = set()
    if info_json:
        for ic in info_json.get("interacting_chains_info", []):
            for c in ic.get("interacting_chains", {}).get("label_asym_id", []):
                ic_chains.add(str(c))

    chains = site_json.get("label_asym_id") or site_json.get("auth_asym_id") or []
    seqs = site_json.get("auth_seq_id") or site_json.get("label_seq_id") or []

    res_set = set()
    for c, s in zip(chains, seqs):
        if not ic_chains or str(c) in ic_chains:
            res_set.add((str(c), str(s)))
    return res_set


def process_pdb_structure(
    entry_id: str,
    site_records: List[Dict],
    work_dir: Path,
    label_threshold: float = 0.65,
    use_minimal_chains: bool = True,
) -> pd.DataFrame:
    """
    Process a single PDB structure:
    1. Fetch CIF and run fpocket (optionally restricted to interacting protein chains).
    2. Extract pocket residues and geometry.
    3. Calculate overlap with ground-truth allosteric site(s) using auth_seq_id alignment.
    4. Return DataFrame of candidate pockets.
    """
    from allopockets.predict import get_cif, get_clean_pdb

    interacting_chains = set()
    for s_rec in site_records:
        info = s_rec.get("info", {})
        for ic in info.get("interacting_chains_info", []):
            chains = ic.get("interacting_chains", {}).get("label_asym_id", [])
            interacting_chains.update(chains)
    chains_filter = (
        list(interacting_chains) if (use_minimal_chains and interacting_chains) else None
    )

    pdb = get_cif(pdb_id=entry_id, path=str(work_dir))
    clean_pdb = get_clean_pdb(pdb, protein_chains=chains_filter, path=str(work_dir))

    # Run fpocket
    pockets_df = run_fpocket(clean_pdb, path=str(work_dir))
    if pockets_df.empty:
        return pd.DataFrame()

    # Collect all site residue sets for this PDB
    all_site_residues = []
    for s_rec in site_records:
        s_keys = get_site_residue_keys(s_rec.get("site", {}), s_rec.get("info", {}))
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

        # Extract contacting residues in pocket matching auth_seq_id
        pocket_res_set_auth = set()
        pocket_res_set_label = set()
        nres = 0
        if hasattr(pkt, "residues") and not pkt.residues.empty:
            res = pkt.residues
            nres = len(res)
            c_label = res["label_asym_id"].astype(str)
            c_auth = res.get("auth_asym_id", res["label_asym_id"]).astype(str)
            s_auth = res.get("auth_seq_id", res["label_seq_id"]).astype(str)
            s_label = res["label_seq_id"].astype(str)
            for cl, ca, sa, sl in zip(c_label, c_auth, s_auth, s_label):
                pocket_res_set_auth.add((cl, sa))
                pocket_res_set_auth.add((ca, sa))
                pocket_res_set_label.add((cl, sl))

        # Compute max overlap against all annotated sites for this PDB
        max_site_in_pocket = 0.0
        max_pocket_in_site = 0.0

        for s_keys in all_site_residues:
            if not s_keys:
                continue
            common_auth = pocket_res_set_auth.intersection(s_keys)
            common_label = pocket_res_set_label.intersection(s_keys)
            common = common_auth if len(common_auth) >= len(common_label) else common_label

            site_in_pkt = len(common) / len(s_keys) if len(s_keys) > 0 else 0.0
            pkt_in_site = len(common) / nres if nres > 0 else 0.0
            if site_in_pkt > max_site_in_pocket:
                max_site_in_pocket = site_in_pkt
            if pkt_in_site > max_pocket_in_site:
                max_pocket_in_site = pkt_in_site

        label = 1 if max_site_in_pocket >= label_threshold else 0

        # Assemble row
        row_data = {
            "Pockets_pdb": entry_id,
            "Pockets_pocket": pname,
            "Pockets_nres": nres,
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
    pdb_file: Optional[str] = None,
    label_threshold: float = 0.65,
    workers: int = 6,
    output_filename: str = "pockets_dataset.parquet",
    use_minimal_chains: bool = True,
    outlier_filter: bool = False,
) -> pd.DataFrame:
    """
    Main entrypoint to generate training data directly from database.db.
    Supports multi-threaded parallel extraction across PDB complexes.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    resolved_db = Path(db_path)
    if not resolved_db.exists() and Path("database.db").exists():
        resolved_db = Path("database.db")

    out_path = Path(output_dir)
    cache_dir = out_path / ("cache_minimal" if use_minimal_chains else "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_path / ("work_minimal" if use_minimal_chains else "work")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load PDB list from file if provided
    if pdb_file:
        pf = Path(pdb_file)
        if pf.suffix.lower() == ".csv":
            df_p = pd.read_csv(pf)
            col = "pdb" if "pdb" in df_p.columns else df_p.columns[0]
            pdb_list = df_p[col].dropna().astype(str).tolist()
        else:
            with open(pf, "r") as f:
                pdb_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Connect to database.db and pre-fetch sites to avoid worker lock contention
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

    # Pre-fetch site annotations into memory
    cursor.execute("SELECT pdb_id, site, modulator, info FROM site")
    sites_by_pdb: Dict[str, List[Dict]] = defaultdict(list)
    for p_id, s_raw, m_raw, i_raw in cursor.fetchall():
        try:
            sites_by_pdb[str(p_id)].append(
                {
                    "site": json.loads(s_raw) if isinstance(s_raw, str) else s_raw,
                    "modulator": json.loads(m_raw) if isinstance(m_raw, str) else m_raw,
                    "info": json.loads(i_raw) if isinstance(i_raw, str) else i_raw,
                }
            )
        except Exception:
            continue
    conn.close()

    logger.info(
        f"Preparing dataset for {len(all_entries)} structures from {db_path} using {workers} workers (minimal_chains={use_minimal_chains})..."
    )

    def _process_entry(entry_id: str) -> Optional[pd.DataFrame]:
        cache_file = cache_dir / f"{entry_id}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        site_records = sites_by_pdb.get(entry_id, [])
        try:
            df_entry = process_pdb_structure(
                entry_id=entry_id,
                site_records=site_records,
                work_dir=work_dir,
                label_threshold=label_threshold,
                use_minimal_chains=use_minimal_chains,
            )
            if not df_entry.empty:
                df_entry.to_parquet(cache_file, index=False)
                return df_entry
        except Exception as e:
            logger.warning(f"Failed to process {entry_id}: {e}")
        return None

    results = []
    if workers <= 1:
        for entry_id in tqdm(all_entries, desc="Extracting PDB complexes"):
            res = _process_entry(entry_id)
            if res is not None and not res.empty:
                results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_pdb = {executor.submit(_process_entry, eid): eid for eid in all_entries}
            for future in tqdm(
                as_completed(future_to_pdb), total=len(all_entries), desc="Extracting PDB complexes"
            ):
                res = future.result()
                if res is not None and not res.empty:
                    results.append(res)

    if not results:
        logger.error("No pockets extracted.")
        return pd.DataFrame()

    full_dataset = pd.concat(results, ignore_index=True)
    # Sort deterministically by PDB and pocket
    full_dataset = full_dataset.sort_values(["Pockets_pdb", "Pockets_pocket"]).reset_index(
        drop=True
    )

    if outlier_filter and len(full_dataset) > 1:
        std_val = full_dataset["Pockets_nres"].std()
        if std_val > 0:
            z = (full_dataset["Pockets_nres"] - full_dataset["Pockets_nres"].mean()).abs() / std_val
            before_len = len(full_dataset)
            full_dataset = full_dataset[z < 3].reset_index(drop=True)
            logger.info(
                f"Applied nres outlier filter (|z| < 3): {before_len} -> {len(full_dataset)} pockets remaining."
            )

    full_dataset.to_parquet(out_path / output_filename, index=False)

    pos_count = int((full_dataset["Label_label"] == 1).sum())
    neg_count = int((full_dataset["Label_label"] == 0).sum())
    logger.info(
        f"Dataset ready: {len(full_dataset)} total pockets across {len(results)} PDBs. "
        f"Positives: {pos_count} ({pos_count/len(full_dataset)*100:.1f}%), Negatives: {neg_count}."
    )
    return full_dataset
