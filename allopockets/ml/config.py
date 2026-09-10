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

Configuration and default parameters for AlloPockets ML pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_FEATURE_NAMES = [
    # FPocket (22)
    "FPocket_Pocket Score",
    "FPocket_Drug Score",
    "FPocket_Number of alpha spheres",
    "FPocket_Mean alpha-sphere radius",
    "FPocket_Mean alpha-sphere Solvent Acc.",
    "FPocket_Mean B-factor of pocket residues",
    "FPocket_Hydrophobicity Score",
    "FPocket_Polarity Score",
    "FPocket_Amino Acid based volume Score",
    "FPocket_Pocket volume (Monte Carlo)",
    "FPocket_Pocket volume (convex hull)",
    "FPocket_Charge Score",
    "FPocket_Local hydrophobic density Score",
    "FPocket_Number of apolar alpha sphere",
    "FPocket_Proportion of apolar alpha sphere",
    "FPocket_Total SASA",
    "FPocket_Polar SASA",
    "FPocket_Apolar SASA",
    "FPocket_Proportion of polar atoms",
    "FPocket_Alpha sphere density",
    "FPocket_Cent. of mass - Alpha Sphere max dist",
    "FPocket_Flexibility",
    # Amino acid compositions (20)
    "Amino acids_label_comp_id_A",
    "Amino acids_label_comp_id_C",
    "Amino acids_label_comp_id_D",
    "Amino acids_label_comp_id_E",
    "Amino acids_label_comp_id_F",
    "Amino acids_label_comp_id_G",
    "Amino acids_label_comp_id_H",
    "Amino acids_label_comp_id_I",
    "Amino acids_label_comp_id_K",
    "Amino acids_label_comp_id_L",
    "Amino acids_label_comp_id_M",
    "Amino acids_label_comp_id_N",
    "Amino acids_label_comp_id_P",
    "Amino acids_label_comp_id_Q",
    "Amino acids_label_comp_id_R",
    "Amino acids_label_comp_id_T",
    "Amino acids_label_comp_id_V",
    "Amino acids_label_comp_id_W",
    "Amino acids_label_comp_id_Y",
    "Amino acids_label_comp_id_S",
    # Graphein physicochemical descriptors (67)
    "Graphein_dim_1",
    "Graphein_dim_2",
    "Graphein_dim_3",
    "Graphein_dim_4",
    "Graphein_dim_5",
    "Graphein_dim_6",
    "Graphein_dim_7",
    "Graphein_pka_cooh_alpha",
    "Graphein_pka_nh3",
    "Graphein_pka_rgroup",
    "Graphein_isoelectric_points",
    "Graphein_molecularweight",
    "Graphein_numbercodons",
    "Graphein_bulkiness",
    "Graphein_polarityzimmerman",
    "Graphein_polaritygrantham",
    "Graphein_refractivity",
    "Graphein_recognitionfactors",
    "Graphein_hphob_eisenberg",
    "Graphein_hphob_sweet",
    "Graphein_hphob_woods",
    "Graphein_hphob_doolittle",
    "Graphein_hphob_manavalan",
    "Graphein_hphob_leo",
    "Graphein_hphob_black",
    "Graphein_hphob_breese",
    "Graphein_hphob_guy",
    "Graphein_hphob_janin",
    "Graphein_hphob_miyazawa",
    "Graphein_hphob_argos",
    "Graphein_hphob_roseman",
    "Graphein_hphob_tanford",
    "Graphein_hphob_wolfenden",
    "Graphein_hphob_welling",
    "Graphein_hphob_wilson",
    "Graphein_hphob_parker",
    "Graphein_hphob_ph3_4",
    "Graphein_hphob_ph7_5",
    "Graphein_hphob_mobility",
    "Graphein_hplchfba",
    "Graphein_hplctfa",
    "Graphein_transmembranetendency",
    "Graphein_hplc2_1",
    "Graphein_hplc7_4",
    "Graphein_buriedresidues",
    "Graphein_accessibleresidues",
    "Graphein_hphob_chothia",
    "Graphein_hphob_rose",
    "Graphein_ratioside",
    "Graphein_averageburied",
    "Graphein_averageflexibility",
    "Graphein_alpha_helixfasman",
    "Graphein_beta_sheetfasman",
    "Graphein_beta_turnfasman",
    "Graphein_alpha_helixroux",
    "Graphein_beta_sheetroux",
    "Graphein_beta_turnroux",
    "Graphein_coilroux",
    "Graphein_alpha_helixlevitt",
    "Graphein_beta_sheetlevitt",
    "Graphein_beta_turnlevitt",
    "Graphein_totalbeta_strand",
    "Graphein_antiparallelbeta_strand",
    "Graphein_parallelbeta_strand",
    "Graphein_a_a_composition",
    "Graphein_a_a_swiss_prot",
    "Graphein_relativemutability",
    # FreeSASA (9)
    "FreeSASA_area_total",
    "FreeSASA_area_polar",
    "FreeSASA_area_apolar",
    "FreeSASA_area_main-chain",
    "FreeSASA_area_side-chain",
    "FreeSASA_relative-area_total",
    "FreeSASA_relative-area_polar",
    "FreeSASA_relative-area_apolar",
    "FreeSASA_relative-area_main-chain",
    # DSSP (18)
    "DSSP_phi",
    "DSSP_psi",
    "DSSP_NH_O_1_relidx",
    "DSSP_NH_O_1_energy",
    "DSSP_O_NH_1_relidx",
    "DSSP_O_NH_1_energy",
    "DSSP_NH_O_2_relidx",
    "DSSP_NH_O_2_energy",
    "DSSP_O_NH_2_relidx",
    "DSSP_O_NH_2_energy",
    "DSSP_relative ASA",
    "DSSP_secondary structure_H",
    "DSSP_secondary structure_B",
    "DSSP_secondary structure_E",
    "DSSP_secondary structure_G",
    "DSSP_secondary structure_I",
    "DSSP_secondary structure_T",
    "DSSP_secondary structure_S",
    "DSSP_secondary structure_-",
    # Melodia (7)
    "Melodia_curvature",
    "Melodia_torsion",
    "Melodia_arc_len",
    "Melodia_writhing",
    "Melodia_phi",
    "Melodia_psi",
    "Melodia_propensity",
    # Biopython (5)
    "Biopython_EXP_HSE_B_U",
    "Biopython_EXP_HSE_B_D",
    "Biopython_EXP_CN",
    "Biopython_EXP_RD",
    "Biopython_EXP_RD_CA",
    # PyRosetta (1)
    "PyRosetta_ddG",
    # ProDy (5)
    "ProDy_prs_effectiveness",
    "ProDy_prs_sensitivity",
    "ProDy_mechstiff",
    "ProDy_rmsf",
    "ProDy_essa",
    # Transfer Entropy (1)
    "TransferEntropy_TE",
    # HHBlits / Evolutionary Profile (30)
    "HHBlits_A",
    "HHBlits_C",
    "HHBlits_D",
    "HHBlits_E",
    "HHBlits_F",
    "HHBlits_G",
    "HHBlits_H",
    "HHBlits_I",
    "HHBlits_K",
    "HHBlits_L",
    "HHBlits_M",
    "HHBlits_N",
    "HHBlits_P",
    "HHBlits_Q",
    "HHBlits_R",
    "HHBlits_S",
    "HHBlits_T",
    "HHBlits_V",
    "HHBlits_W",
    "HHBlits_Y",
    "HHBlits_M->M",
    "HHBlits_M->I",
    "HHBlits_M->D",
    "HHBlits_I->M",
    "HHBlits_I->I",
    "HHBlits_D->M",
    "HHBlits_D->D",
    "HHBlits_Neff",
    "HHBlits_Neff_I",
    "HHBlits_Neff_D",
]


@dataclass
class ModelConfig:
    model_type: str = (
        "hist_gradient_boost"  # "hist_gradient_boost", "lightgbm", "xgboost", "autogluon"
    )
    n_estimators: int = 300
    learning_rate: float = 0.03
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    scale_pos_weight: Optional[float] = None  # None = compute automatically
    seed: int = 42
    n_jobs: int = -1
    time_limit: Optional[int] = None


@dataclass
class TrainConfig:
    data_path: str = "data/pockets_dataset.parquet"
    output_dir: str = "models/lgbm_pocket_classifier"
    n_splits: int = 5
    seed: int = 42
    target_col: str = "Label_label"
    group_col: str = "Pockets_pdb"
    model: ModelConfig = field(default_factory=ModelConfig)
