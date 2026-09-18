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
"""

from allopockets.database.utils import Cif as BaseCifFile
from allopockets.database.cifutils import MMCIF2Dict, CifFileWriter

import tempfile, pickle
from functools import cached_property, cache
from contextlib import contextmanager
import pandas as pd
from tqdm.notebook import tqdm


class CifFile(BaseCifFile):
    """
    Base class to manage .cif files. Inherits from allodb.utils Cif but overrides intialization to read from a file.
    """

    def __init__(self, parent, filename=None, name=None):
        self.filename = filename or parent.filename
        self._name = name or parent._name

    @cached_property
    def data(self) -> dict:
        parsed = MMCIF2Dict().parse(self.filename)
        if self._name in parsed:
            return parsed[self._name]
        # Case-insensitive fallback
        for k, v in parsed.items():
            if k.lower() == self._name.lower():
                return v
        raise KeyError(f"Block '{self._name}' not found in {self.filename}")


class Cif:
    """
    Base class to work with a structure from a file: a "minimal" structure which also has a parent "original" file. Mimicks allodb.utils Structure class.
    """

    path = "."
    original_cifs_path = "."

    def __init__(self, pdb, filename=None, name=None):
        self.entry_id = pdb.lower()
        self._name = name or self.entry_id.upper()
        self.filename = filename or f"{self.path}/{self.entry_id}.cif"
        self.orig_filename = f"{self.original_cifs_path}/{self.entry_id}_updated.cif.gz"

    @cached_property
    def cif(self):
        return CifFile(self)

    @cached_property
    def origcif(self):
        return CifFile(self, filename=self.orig_filename)

    @cached_property
    def atoms(self) -> pd.DataFrame:
        return pd.DataFrame(self.cif.data["_atom_site"], dtype=str)

    @cached_property
    def residues(self) -> pd.DataFrame:
        return self.atoms.drop(
            columns=[
                c
                for c in self.atoms.columns
                if c
                not in [
                    "label_comp_id",
                    "label_asym_id",
                    "label_entity_id",
                    "label_seq_id",
                    "pdbx_PDB_ins_code",
                    "auth_seq_id",
                    "auth_comp_id",
                    "auth_asym_id",
                    "pdbx_PDB_model_num",
                    "pdbx_label_index",
                    "pdbx_sifts_xref_db_name",
                    "pdbx_sifts_xref_db_acc",
                    "pdbx_sifts_xref_db_num",
                    "pdbx_sifts_xref_db_res",
                ]
            ]
        ).drop_duplicates()

    @contextmanager
    def _extended_temp_ciff(self, extra_data: dict):
        """
        To be used as: 'with <obj>._extended_temp_ciff(extra_data) as f:' to work with a cif file with the data of the current object plus the extra_data passed
        """
        data = self.cif.data
        data.update(extra_data)
        try:
            file = tempfile.NamedTemporaryFile("w+", suffix=".cif")
            writer = CifFileWriter(file.name)
            writer.write({self.entry_id.upper(): data})
            yield file
        finally:
            if file:
                file.close()
