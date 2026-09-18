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

from allopockets.database.utils import Assembly


from .utils import *
import os
import requests, gzip, tempfile


class PDBCif:
    """
    Base class to manage .cif files of new PDBs not in allodb. Mimicks CifFile but retrieves the .cif content from online, and has other utils to manipulate the downloaded information.
    """

    def __init__(self, parent, filename=None, name=None):
        self.filename = filename or parent.filename
        self._name = name or parent._name

        # Check local cache first before downloading
        self._cif_content = None
        candidates = [
            f"{getattr(parent, 'path', '.')}/{self._name.lower()}_updated.cif.gz",
            f"{getattr(parent, 'path', '.')}/{self._name.lower()}.cif.gz",
            f"data/benchmark/work/{self._name.lower()}_updated.cif.gz",
            f"data/work/{self._name.lower()}_updated.cif.gz",
        ]
        for c in candidates:
            if c and os.path.isfile(c) and c.endswith(".cif.gz") and os.path.getsize(c) > 0:
                try:
                    with open(c, "rb") as f:
                        content = f.read()
                        if content.startswith(b"\x1f\x8b"):
                            self._cif_content = content
                            break
                except Exception:
                    pass

        if self._cif_content is None:
            # Download the SIFTS-standardized .cif.gz file from PDBe
            headers = {"User-Agent": "AlloPockets/1.0 (https://github.com/fnerin/AlloPockets)"}
            response = requests.get(
                f"https://www.ebi.ac.uk/pdbe/entry-files/{self._name.lower()}_updated.cif.gz",
                timeout=30,
                headers=headers,
            )

            if response.ok and response.content.startswith(b"\x1f\x8b"):
                self._cif_content = response.content
            else:
                # Fallback to RCSB
                response = requests.get(
                    f"https://files.rcsb.org/download/{self._name.lower()}.cif.gz",
                    timeout=30,
                    headers=headers,
                )
                if response.ok and response.content.startswith(b"\x1f\x8b"):
                    self._cif_content = response.content
                else:
                    raise RuntimeError(
                        f"Failed to download valid PDB {self._name} from both PDBe and RCSB."
                    )

    @cached_property
    def data(self):
        with self.ciff() as f:
            parsed = MMCIF2Dict().parse(f.name)
            if self._name in parsed:
                data = parsed[self._name]
            else:
                # Case-insensitive fallback
                for k, v in parsed.items():
                    if k.lower() == self._name.lower():
                        data = v
                        break
                else:
                    raise KeyError(f"Block '{self._name}' not found in downloaded mmCIF.")
        return data

    @contextmanager
    def ciff(self):
        """
        To be used as: 'with <obj>.ciff() as f:' to work with the contents of the cif as a .cif.gz file
        """
        try:
            file = tempfile.NamedTemporaryFile("wb+", suffix=".cif.gz")
            content = self._cif_content or b""
            file.write(content)
            file.flush()
            yield file
        finally:
            if file:
                file.close()

    @property
    def text(self):
        """
        Return the contents of the cif as plain text
        """
        content = self._cif_content or b""
        return gzip.decompress(content).decode()


class Pdb(Cif):
    """
    Inherits from Cif, to manage structures of new PDBs not in allodb. Uses PdbCif and has extra methods of the original allodb.PDB class to retrieve information from the new PDB.
    """

    path = "."

    def __init__(self, pdb, filename=None, name=None):
        self.entry_id = pdb.lower()
        self._name = name or self.entry_id.upper()
        self.filename = filename or f"{self.path}/{self.entry_id}.cif"

    @cached_property
    def cif(self):
        return PDBCif(self)

    @cached_property
    def _entities(self):
        return pd.DF(self.cif.data["_entity_poly"], dtype=str)

    @cached_property
    def _protein_entities(self):
        return self._entities.query("type == 'polypeptide(L)'").entity_id.to_list()

    @cached_property
    def _assembly(self):
        try:
            ass = Assembly(self)

            @contextmanager
            def ciff(self):
                try:
                    file = tempfile.NamedTemporaryFile("w+", suffix=".cif")
                    file.write(self.text)
                    yield file
                finally:
                    if file:
                        file.close()

            ass.cif.ciff = ciff.__get__(ass.cif)
            return ass
        except Assembly._AssemblyIsModel:
            return None

    @cached_property
    def _assembly_asyms(self):
        if self.assembly is not None:
            return pd.DF(self.cif.data["_pdbx_struct_assembly_gen"], dtype=str).sort_values(
                "assembly_id"
            )

    @property
    def assembly(self):
        if self._assembly is not None:
            return self._assembly
