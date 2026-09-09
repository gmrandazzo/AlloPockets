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

Wrapper for executing fpocket and retrieving pocket outputs.
"""

import shutil
import subprocess
from pathlib import Path
import pandas as pd
from allopockets.features.utils import Cif


def is_fpocket_available() -> bool:
    """Check if fpocket executable is found in system PATH."""
    return shutil.which("fpocket") is not None


def run_fpocket(
    clean_pdb, path="predict", min_spheres=3, max_spheres=6, min_radius=35
) -> pd.DataFrame:
    """
    Run fpocket on a cleaned protein structure and return a DataFrame of detected pocket IDs.
    """
    entry_id = clean_pdb.entry_id
    target_dir = Path(path) / entry_id
    out_dir = target_dir / f"{entry_id}_out"
    pockets_dir = out_dir / "pockets"

    if not out_dir.is_dir() or not pockets_dir.is_dir():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(clean_pdb.filename, target_dir / f"{entry_id}.cif")

        if not is_fpocket_available():
            raise FileNotFoundError(
                "fpocket executable was not found in system PATH. "
                "Please activate the environment with fpocket (e.g., conda activate allopockets)."
            )

        cmd = [
            "fpocket",
            "-m",
            str(min_spheres),
            "-M",
            str(max_spheres),
            "-i",
            str(min_radius),
            "--file",
            f"{entry_id}.cif",
        ]
        subprocess.run(cmd, cwd=str(target_dir), check=True, capture_output=True)

    if not pockets_dir.is_dir():
        return pd.DataFrame(columns=["pocket"])

    pockets = [f.stem.split("_")[0] for f in pockets_dir.glob("*.cif") if f.name.endswith(".cif")]
    # Remove duplicates and sort naturally
    pockets = sorted(
        list(set(pockets)),
        key=lambda x: int(x.replace("pocket", "")) if x.replace("pocket", "").isdigit() else 9999,
    )
    return pd.DataFrame({"pocket": pockets})


def get_pocket_atoms(pdb_id: str, pocket_id: str, path="predict") -> pd.DataFrame:
    """
    Retrieve alpha sphere STP pseudo-atoms for a given pocket from the global fpocket CIF.
    """
    pocket_num = pocket_id.replace("pocket", "")
    out_cif = Path(path) / pdb_id / f"{pdb_id}_out" / f"{pdb_id}_out.cif"
    if not out_cif.is_file():
        return pd.DataFrame()

    atoms = Cif(pdb_id, str(out_cif), name=f"{pdb_id}_out").atoms
    if atoms.empty:
        return pd.DataFrame()

    pocket_atoms = atoms.query(f"label_comp_id == 'STP' and label_seq_id == '{pocket_num}'").copy()
    pocket_atoms["label_asym_id"] = "ZZZ"
    pocket_atoms["label_entity_id"] = "99"
    return pocket_atoms
