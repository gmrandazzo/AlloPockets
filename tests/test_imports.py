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

Test package importability and namespace structure.
"""


def test_import_allopockets():
    import allopockets

    assert hasattr(allopockets, "__version__")
    assert allopockets.__version__ == "1.0.0"
    assert hasattr(allopockets, "predict")
    assert hasattr(allopockets, "Pocket")
    assert hasattr(allopockets, "PocketClassifier")


def test_import_submodules():
    import allopockets.pockets
    import allopockets.ml
    import allopockets.viz
    import allopockets.cli

    assert hasattr(allopockets.pockets, "Pocket")
    assert hasattr(allopockets.ml, "PocketClassifier")
    assert hasattr(allopockets.viz, "generate_3d_pocket_html")
    assert hasattr(allopockets.cli, "main")


def test_backward_compatible_predict():
    import predict

    assert hasattr(predict, "get_cif")
    assert hasattr(predict, "predict")
    assert hasattr(predict, "view_pockets")
