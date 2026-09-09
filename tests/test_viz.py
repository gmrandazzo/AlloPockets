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

Test 3D visualization and HTML generation.
"""

import tempfile
from pathlib import Path
from allopockets.viz.html_viewer import generate_3d_pocket_html


def test_html_viewer_generation():
    pdb_cif = "data_test\n#\nloop_\n_atom_site.id\n1\n"
    pocket_cif = "data_pocket\n#\nloop_\n_atom_site.id\n1\n"
    metrics = {
        "label": 1,
        "site_in_pocket": 0.85,
        "volume": 650.0,
        "num_spheres": 42,
        "modulator_distance": 2.1,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_viewer.html"
        html = generate_3d_pocket_html(
            pdb_id="6t4k",
            pocket_id="pocket1",
            pdb_cif_content=pdb_cif,
            pocket_cif_content=pocket_cif,
            metrics=metrics,
            site_residues=[{"chain": "A", "resi": 123}],
            output_path=out_path,
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 500
        assert "6T4K - pocket1" in html
        assert "85.0%" in html
        assert "Positive (Allosteric)" in html
        assert "3Dmol" in html
