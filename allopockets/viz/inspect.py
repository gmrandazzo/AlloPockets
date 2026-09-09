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

Interactive 3D Pocket Inspector and Diagnostics tool.
"""

import json
import logging
import sqlite3
import webbrowser
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from allopockets.pockets.pocket import Pocket
from allopockets.viz.html_viewer import generate_3d_pocket_html

logger = logging.getLogger(__name__)


def compute_centroid(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.array([0.0, 0.0, 0.0])
    return coords.mean(axis=0)


def inspect_pocket_cli(
    pdb_id: str,
    pocket_id: str,
    path: str = "predict",
    db_path: str = "data/database.db",
    html_output: Optional[str] = None,
    open_browser: bool = False,
) -> Dict:
    """
    Inspect 3D coordinates, alpha spheres, and ground-truth alignment for a candidate pocket.
    """
    pdb_id = pdb_id.lower()
    base_dir = Path(path)
    pocket_cif = base_dir / pdb_id / f"{pdb_id}_out" / "pockets" / f"{pocket_id}_atm.cif"

    if not pocket_cif.exists():
        # Check if parent directory exists
        if not (base_dir / pdb_id).exists():
            from allopockets.predict import get_cif, get_clean_pdb
            from allopockets.pockets.fpocket import run_fpocket

            print(
                f"Structure {pdb_id} not yet processed in {path}. Fetching and running fpocket..."
            )
            pdb = get_cif(pdb_id=pdb_id, path=path)
            clean_pdb = get_clean_pdb(pdb, protein_chains=None, path=path)
            run_fpocket(clean_pdb, path=path)

    if not pocket_cif.exists():
        raise FileNotFoundError(f"Pocket file not found: {pocket_cif}")

    pkt = Pocket(str(pocket_cif))
    feats = pkt.feats

    # 1. Pocket coordinates & centroid
    pkt_atoms = pkt.atoms
    if not pkt_atoms.empty and {"Cartn_x", "Cartn_y", "Cartn_z"}.issubset(pkt_atoms.columns):
        pkt_coords = pkt_atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].astype(float).values
        pkt_centroid = compute_centroid(pkt_coords)
    else:
        pkt_coords = np.empty((0, 3))
        pkt_centroid = np.array([0.0, 0.0, 0.0])

    # 2. Query ground-truth modulator & site from database.db
    mod_centroid = None
    mod_distance = None
    site_in_pocket = 0.0
    pocket_in_site = 0.0
    site_residues_list = []

    resolved_db = Path(db_path) if Path(db_path).exists() else Path("database.db")
    if resolved_db.exists():
        try:
            conn = sqlite3.connect(str(resolved_db))
            c = conn.cursor()
            c.execute("SELECT site, modulator FROM site WHERE pdb_id = ?", (pdb_id,))
            rows = c.fetchall()
            if rows:
                site_json = json.loads(rows[0][0]) if isinstance(rows[0][0], str) else rows[0][0]
                chains = site_json.get("label_asym_id") or site_json.get("auth_asym_id") or []
                seqs = site_json.get("label_seq_id") or site_json.get("auth_seq_id") or []

                site_set = set(zip([str(x) for x in chains], [str(x) for x in seqs]))
                for ch, sq in site_set:
                    try:
                        site_residues_list.append({"chain": ch, "resi": int(sq)})
                    except ValueError:
                        pass

                # Calculate overlap
                pocket_res_set = set()
                if hasattr(pkt, "residues") and not pkt.residues.empty:
                    for _, r in pkt.residues.iterrows():
                        cid = r.get("label_asym_id") or r.get("auth_asym_id", "")
                        seqid = r.get("label_seq_id") or r.get("auth_seq_id", "")
                        pocket_res_set.add((str(cid), str(seqid)))

                common = pocket_res_set.intersection(site_set)
                site_in_pocket = len(common) / len(site_set) if len(site_set) > 0 else 0.0
                pocket_in_site = (
                    len(common) / len(pocket_res_set) if len(pocket_res_set) > 0 else 0.0
                )
        except Exception as e:
            logger.debug(f"Database query skipped: {e}")

    # Format diagnostic summary
    label = 1 if site_in_pocket >= 0.65 else 0
    vol = (
        feats.get("Pocket volume (convex hull)") or feats.get("Pocket volume (Monte Carlo)") or 0.0
    )
    nspheres = int(feats.get("Number of alpha spheres") or 0)

    print("\n" + "=" * 65)
    print(f"       ALLOPOCKETS 3D POCKET DIAGNOSTICS: {pdb_id.upper()} / {pocket_id}")
    print("=" * 65)
    print(f"  Classification Label:    {'1 (ALLOSTERIC)' if label == 1 else '0 (NON-ALLOSTERIC)'}")
    print(f"  Site-in-Pocket Overlap:  {site_in_pocket * 100:.1f}%")
    print(f"  Pocket-in-Site Overlap:  {pocket_in_site * 100:.1f}%")
    print(f"  Alpha Spheres Count:     {nspheres}")
    print(f"  Convex Hull Volume:      {vol:.1f} Å³")
    print(
        f"  Pocket Centroid (X,Y,Z): ({pkt_centroid[0]:.2f}, {pkt_centroid[1]:.2f}, {pkt_centroid[2]:.2f})"
    )
    if mod_distance is not None:
        print(f"  Distance to Modulator:   {mod_distance:.2f} Å")
    print(
        f"  Contacting Residues:     {len(pocket_res_set) if 'pocket_res_set' in locals() else 'N/A'}"
    )
    print("-" * 65)
    print("  Key FPocket Descriptors:")
    for k in ["Pocket Score", "Drug Score", "Hydrophobicity Score", "Polarity Score", "Total SASA"]:
        if k in feats:
            print(f"    - {k:24s}: {feats[k]:.3f}")
    print("=" * 65 + "\n")

    # Generate HTML viewer
    out_html = html_output or f"{pdb_id}_{pocket_id}_debug.html"
    parent_cif = base_dir / f"{pdb_id}_updated.cif.gz"
    if not parent_cif.exists():
        parent_cif = base_dir / f"{pdb_id}.cif"

    pdb_cif_text = ""
    if parent_cif.exists():
        from allopockets.features.utils import Cif

        pdb_cif_text = Cif(pdb_id, str(parent_cif)).cif.text

    pocket_cif_text = pkt.cif.text if hasattr(pkt, "cif") else ""

    metrics = {
        "label": label,
        "site_in_pocket": site_in_pocket,
        "pocket_in_site": pocket_in_site,
        "num_spheres": nspheres,
        "volume": vol,
        "modulator_distance": mod_distance,
    }

    generate_3d_pocket_html(
        pdb_id=pdb_id,
        pocket_id=pocket_id,
        pdb_cif_content=pdb_cif_text,
        pocket_cif_content=pocket_cif_text,
        metrics=metrics,
        site_residues=site_residues_list,
        output_path=out_html,
    )
    print(f"Interactive 3D HTML Viewer written to: {out_html}")

    if open_browser:
        webbrowser.open(f"file://{Path(out_html).resolve()}")

    return metrics
