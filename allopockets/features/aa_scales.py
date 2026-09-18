"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Authors: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

This module provides 100% local, dependency-free lookup tables for the 67
physicochemical features previously computed via Graphein:
- 7 Meiler amino acid embedding dimensions (Meiler et al. 2001)
- 60 Expasy ProtScale scales (Kyte-Doolittle, Hopp-Woods, Eisenberg, Janin, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

try:
    from importlib.resources import files

    _data_file = files("allopockets.features").joinpath("aa_scales_data.json")
    with _data_file.open("r", encoding="utf-8") as _f:
        _SCALES_DATA = json.load(_f)
except Exception:
    _DATA_PATH = Path(__file__).resolve().parent / "aa_scales_data.json"
    with open(_DATA_PATH, "r", encoding="utf-8") as _f:
        _SCALES_DATA = json.load(_f)

MEILER_DATA: Dict[str, Dict[str, float]] = _SCALES_DATA["meiler"]
EXPASY_DATA: Dict[str, Dict[str, float]] = _SCALES_DATA["expasy"]

# Expected Expasy feature names used in AlloPockets (excluding hphob_fauchere)
EXPASY_FEATURE_NAMES: List[str] = [
    "pka_cooh_alpha",
    "pka_nh3",
    "pka_rgroup",
    "isoelectric_points",
    "molecularweight",
    "numbercodons",
    "bulkiness",
    "polarityzimmerman",
    "polaritygrantham",
    "refractivity",
    "recognitionfactors",
    "hphob_eisenberg",
    "hphob_sweet",
    "hphob_woods",
    "hphob_doolittle",
    "hphob_manavalan",
    "hphob_leo",
    "hphob_black",
    "hphob_breese",
    "hphob_guy",
    "hphob_janin",
    "hphob_miyazawa",
    "hphob_argos",
    "hphob_roseman",
    "hphob_tanford",
    "hphob_wolfenden",
    "hphob_welling",
    "hphob_wilson",
    "hphob_parker",
    "hphob_ph3_4",
    "hphob_ph7_5",
    "hphob_mobility",
    "hplchfba",
    "hplctfa",
    "transmembranetendency",
    "hplc2_1",
    "hplc7_4",
    "buriedresidues",
    "accessibleresidues",
    "hphob_chothia",
    "hphob_rose",
    "ratioside",
    "averageburied",
    "averageflexibility",
    "alpha_helixfasman",
    "beta_sheetfasman",
    "beta_turnfasman",
    "alpha_helixroux",
    "beta_sheetroux",
    "beta_turnroux",
    "coilroux",
    "alpha_helixlevitt",
    "beta_sheetlevitt",
    "beta_turnlevitt",
    "totalbeta_strand",
    "antiparallelbeta_strand",
    "parallelbeta_strand",
    "a_a_composition",
    "a_a_swiss_prot",
    "relativemutability",
]

MEILER_FEATURE_NAMES: List[str] = [f"dim_{i}" for i in range(1, 8)]

ALL_GRAPH_SCALE_NAMES: List[str] = MEILER_FEATURE_NAMES + EXPASY_FEATURE_NAMES


def get_residue_scales(res_name_3: str) -> Dict[str, float]:
    """Return all 67 Meiler + Expasy scales for a 3-letter amino acid name."""
    res = res_name_3.upper().strip()
    # Default to 0.0 for unknown residues
    meiler = MEILER_DATA.get(res, {k: 0.0 for k in MEILER_FEATURE_NAMES})
    expasy = EXPASY_DATA.get(res, {k: 0.0 for k in EXPASY_FEATURE_NAMES})

    result: Dict[str, float] = {}
    for k in MEILER_FEATURE_NAMES:
        result[k] = float(meiler.get(k, 0.0))
    for k in EXPASY_FEATURE_NAMES:
        result[k] = float(expasy.get(k, 0.0))
    return result


def get_residues_scales_df(residues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a residues DataFrame containing 'auth_asym_id', 'auth_seq_id', 'label_comp_id',
    return a DataFrame with the residue keys and all 67 scale columns.
    """
    records = []
    comp_col = "label_comp_id" if "label_comp_id" in residues_df.columns else "auth_comp_id"
    for _, row in residues_df.iterrows():
        res_name = str(row[comp_col])
        scales = get_residue_scales(res_name)
        d: Dict[str, Any] = {
            "auth_asym_id": str(row["auth_asym_id"]),
            "auth_seq_id": str(row["auth_seq_id"]),
            "pdbx_PDB_ins_code": str(row.get("pdbx_PDB_ins_code", "?")),
        }
        d.update(scales)
        records.append(d)
    return pd.DataFrame(records)
