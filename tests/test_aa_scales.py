"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Author: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

Unit tests for Meiler and Expasy ProtScale physicochemical lookup module.
"""

import unittest
import pandas as pd
from allopockets.features.aa_scales import get_residue_scales, get_residues_scales_df


class TestAAScales(unittest.TestCase):
    def test_single_residue_lookup(self):
        scales_ala = get_residue_scales("ALA")
        self.assertIn("dim_1", scales_ala)
        self.assertIn("hphob_doolittle", scales_ala)
        self.assertAlmostEqual(scales_ala["dim_1"], 1.28, places=2)
        self.assertAlmostEqual(scales_ala["hphob_doolittle"], 1.8, places=1)

    def test_unknown_residue_fallback(self):
        scales_unk = get_residue_scales("XYZ")
        self.assertEqual(scales_unk["dim_1"], 0.0)
        self.assertEqual(scales_unk["hphob_doolittle"], 0.0)

    def test_residues_df_scaling(self):
        df_res = pd.DataFrame(
            {
                "auth_asym_id": ["A", "A", "B"],
                "auth_seq_id": ["1", "2", "1"],
                "label_comp_id": ["ALA", "GLY", "LEU"],
                "pdbx_PDB_ins_code": ["?", "?", "?"],
            }
        )
        df_out = get_residues_scales_df(df_res)
        self.assertEqual(len(df_out), 3)
        self.assertIn("dim_1", df_out.columns)
        self.assertIn("hphob_doolittle", df_out.columns)
        self.assertEqual(len(df_out.columns), 70)  # 3 identifiers + 67 scales


if __name__ == "__main__":
    unittest.main()
