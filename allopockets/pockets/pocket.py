"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Original Author: Francho Nerín Fonz <fnerin@bioacademy.gr>
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

Pocket representation and feature extraction from fpocket outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from allopockets.features.utils import Cif

res_cols = ["label_asym_id", "auth_asym_id", "auth_comp_id", "auth_seq_id", "pdbx_PDB_ins_code"]


class Pocket(Cif):
    """
    Class to manage the .cif file of an fpocket-identified pocket.
    Extracts geometric descriptors, residue contacts, and 3D spatial properties.
    """

    def __init__(self, filename):
        name = Path(filename).stem.replace(".cif", "")
        super().__init__(name, filename=str(filename), name=name)
        self.filename = str(filename)
        self._name = name

    @property
    def cif_feats(self):
        """
        Extract pocket features stored in the cif in '_struct.pdbx_descriptor'.
        """
        try:
            descriptor = self.cif.data["_struct"]["pdbx_descriptor"]
            raw_text = " ".join(descriptor + ["15"]).replace(
                " 10 -Pocket volume (convex hull)", " 10 - Pocket volume (convex hull)"
            )
            splits = raw_text.split(" - ")[1:]
            feats = {}
            for feat in splits:
                parts = feat.strip().split(" ")
                key = " ".join(parts[:-3])
                val = float(parts[-2])
                feats[key] = val
            return feats
        except Exception:
            return {}

    @property
    def info_feats(self):
        """
        Extract pocket features stored in the whole-PDB FPocket output _info.txt file.
        """
        try:
            p = Path(self.filename)
            # path is typically .../<pdb>_out/pockets/pocketN_atm.cif
            out_dir = p.parent.parent
            pdb_name = out_dir.name.replace("_out", "")
            info_file = out_dir / f"{pdb_name}_info.txt"
            if not info_file.exists():
                return {}

            pocket_num = self._name.replace("pocket", "").replace("_atm", "").replace("_vert", "")
            with open(info_file, "r") as f:
                content = f.read()

            for block in content.split("Pocket"):
                lines = [line_item.strip() for line_item in block.splitlines() if line_item.strip()]
                if not lines:
                    continue
                if lines[0].startswith(f"{pocket_num} :"):
                    res = {}
                    for line in lines[1:]:
                        if ":" in line:
                            k, v = line.split(":", 1)
                            try:
                                res[k.strip()] = float(v.strip())
                            except ValueError:
                                continue
                    return res
            return {}
        except Exception:
            return {}

    @property
    def feats(self):
        """
        Combine features from both the pocket CIF and the _info.txt file.
        """
        common = {
            "Pocket Score": "Score",
            "Drug Score": "Druggability Score",
            "Number of alpha spheres": "Number of Alpha Spheres",
            "Mean alpha-sphere radius": "Mean alpha-sphere radius",
            "Mean alpha-sphere Solvent Acc.": "Mean alp. sph. solvent access",
            "Hydrophobicity Score": "Hydrophobicity score",
            "Polarity Score": "Polarity score",
            "Amino Acid based volume Score": "Volume score",
            "Pocket volume (Monte Carlo)": "Volume",
            "Charge Score": "Charge score",
            "Local hydrophobic density Score": "Mean local hydrophobic density",
            "Proportion of apolar alpha sphere": "Apolar alpha sphere proportion",
        }
        cif_exclusive = [
            "Mean B-factor of pocket residues",
            "Pocket volume (convex hull)",
            "Number of apolar alpha sphere",
        ]
        info_exclusive = [
            "Total SASA",
            "Polar SASA",
            "Apolar SASA",
            "Proportion of polar atoms",
            "Alpha sphere density",
            "Cent. of mass - Alpha Sphere max dist",
            "Flexibility",
        ]

        c_feats = self.cif_feats
        i_feats = self.info_feats

        combined = {}
        for k in list(common.keys()) + cif_exclusive:
            if k in c_feats:
                combined[k] = c_feats[k]
            elif k in common and common[k] in i_feats:
                combined[k] = i_feats[common[k]]

        for k in info_exclusive:
            if k in i_feats:
                combined[k] = i_feats[k]

        return combined

    def get_centroid(self):
        """Calculate the geometric center (X, Y, Z) of the pocket atoms."""
        atoms = self.atoms
        if atoms.empty:
            return np.array([0.0, 0.0, 0.0])
        coords = atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].astype(float).values
        return coords.mean(axis=0)


def get_pockets_info(pdb_id, pockets_df, pockets_path):
    """
    Extract geometric metrics for all candidate pockets of a PDB.
    """
    records = []
    for _, row in pockets_df.iterrows():
        pname = row["pocket"]
        cif_file = Path(pockets_path) / pdb_id / f"{pdb_id}_out" / "pockets" / f"{pname}_atm.cif"
        if cif_file.exists():
            pkt = Pocket(str(cif_file))
            feats = pkt.feats
            feats["pdb"] = pdb_id
            feats["pocket"] = pname
            feats["nres"] = len(pkt.residues)
            records.append(feats)
    return pd.DataFrame(records)


def get_mean_pocket_features(pdb, pocket, pdb_features, pockets_path):
    """
    Given a PDB ID, a pocket and a pd.DataFrame of features of that PDB, return the average of the features of the residues of the pocket.
    """
    pocket_cif = Path(pockets_path) / pdb / f"{pdb}_out" / "pockets" / f"{pocket}_atm.cif"
    if not pocket_cif.exists():
        return pd.Series(dtype=float)

    pkt_res = Pocket(str(pocket_cif)).residues
    cols = [c for c in res_cols if c in pkt_res.columns]
    merged = pdb_features.merge(
        pkt_res[cols].set_axis(pd.MultiIndex.from_tuples([("Residues", c) for c in cols]), axis=1)
    )
    drop_levels = [lvl for lvl in ["Residues", "Label"] if lvl in merged.columns.levels[0]]
    return merged.drop(columns=drop_levels, level=0).mean()
