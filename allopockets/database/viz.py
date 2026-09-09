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

try:
    from ipymolstar import PDBeMolstar
except ImportError:

    class _FallbackPDBeMolstar:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ipymolstar is required for Jupyter interactive Mol* widget. "
                "Install it with: pip install ipymolstar"
            )

    PDBeMolstar = _FallbackPDBeMolstar  # type: ignore[misc,assignment]


class Viz(PDBeMolstar):
    """
    entity_id: Optional[str]
    ??? start: Optional[Any]
    ??? end: Optional[Any]

    struct_asym_id: Optional[str]
    label_comp_id: Optional[str]
    residue_number: Optional[int]
    start_residue_number: Optional[int]
    end_residue_number: Optional[int]
    atoms: Optional[List[str]]
    atom_id: Optional[List[int]]

    auth_asym_id: Optional[str]
    ??? auth_seq_id: Optional[int]
    auth_residue_number: Optional[int],        auth_ins_code_id: Optional[str]
    start_auth_residue_number: Optional[int],  start_auth_ins_code_id: Optional[str]
    end_auth_residue_number: Optional[int],    end_auth_ins_code_id: Optional[str]

    uniprot_accession: Optional[str]
    uniprot_residue_number: Optional[int]
    start_uniprot_residue_number: Optional[int]
    end_uniprot_residue_number: Optional[int]

    color: Optional[Color]

    sideChain: Optional[bool]
    representation: Optional[str]
    representationColor: Optional[Color]
    focus: Optional[bool]
    tooltip: Optional[str]
    """

    def __init__(self, pdb, assembly_id=""):
        self._pdb = pdb

        super().__init__(
            custom_data={
                "data": self._pdb.cif.text,
                "format": "cif",
                "binary": False,
            },
            sequence_panel=True,
            assembly_id=str(assembly_id),
            # For non-white structure, don't pass this arg
            color_data={
                "data": [{"color": "white"}],
                "nonSelectedColor": None,
                "keepColors": False,
                "keepRepresentations": False,
            },
        )

    def colorful(self):
        self.color(
            data=[{"color": None}],
        )

    def color_residues(self, residues):
        """
        residues: [
            {
                "res": pd.DataFrame of residues,
                "color": color
            },
            { ... }
        ]
        """
        self.color(
            [
                {
                    k: v
                    for k, v in {
                        "struct_asym_id": r["label_asym_id"],
                        "residue_number": (
                            int(r["label_seq_id"]) if r["label_seq_id"] != "." else ""
                        ),
                        "color": color,
                        "focus": True,
                    }.items()
                    if v != ""
                }
                for s in residues
                for res, color in (s.values(),)
                for i, r in res.iterrows()
            ],
            keep_colors=True,
            keep_representations=True,
        )

    def color_site(
        self, site, include_modulator=True, site_color="black", modulator_color="purple"
    ):
        self.color_residues(
            [
                {"res": site.residues.query("label_comp_id != 'HOH'"), "color": site_color},
            ]
        )
        if hasattr(site, "modulator_residues"):
            self.color_residues(
                [
                    {"res": site.modulator_residues, "color": modulator_color},
                ]
            )

    def color_sites(self, sites, **kwargs):
        "Same kwargs as color_site. 'sites' can be a joint peewee query: pdb.orthosites | pdb.sites ((¿ also pdb.orthosites or pdb.sites ?))"
        for site in sites:
            self.color_site(site, **kwargs)
