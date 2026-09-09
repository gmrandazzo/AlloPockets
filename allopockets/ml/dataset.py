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

Dataset loading, feature preparation, and Grouped Stratified K-Fold splitting.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from allopockets.ml.config import DEFAULT_FEATURE_NAMES


def flatten_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level column names if loaded from legacy AlloPockets pickles."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if "Pockets" in df.columns.levels[0]:
            if "pdb" in df["Pockets"].columns and "pocket" in df["Pockets"].columns:
                df.index = df["Pockets"][["pdb", "pocket"]].apply(
                    lambda x: "_".join(x.astype(str)), axis=1
                )
            df = df.drop(columns=["Pockets"], level=0)
        df.columns = [
            "_".join([str(lvl) for lvl in col if str(lvl)]).strip("_") for col in df.columns.values
        ]
    return df


def load_pocket_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Load pocket dataset from Parquet, Pickle, or CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")

    return flatten_dataframe_columns(df)


def get_group_splits(
    df: pd.DataFrame,
    group_col: str = "Pockets_pdb",
    target_col: str = "Label_label",
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Stratified Group K-Fold cross-validation splits.
    Ensures all candidate pockets of the same PDB structure stay together in either train or val.
    """
    # Auto-detect target column
    if target_col not in df.columns:
        possible_targets = [c for c in df.columns if "label" in c.lower()]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            raise KeyError(f"Target column '{target_col}' not found in dataset.")

    # Auto-detect group column
    if group_col not in df.columns:
        possible_groups = [c for c in df.columns if "pdb" in c.lower()]
        if possible_groups:
            group_col = possible_groups[0]
        else:
            raise KeyError(f"Group column '{group_col}' not found in dataset.")

    y = df[target_col].astype(int).values
    groups = df[group_col].astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(X=df, y=y, groups=groups))


def impute_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Impute NaNs with medians computed strictly on train_df (prevents data leakage).
    """
    cols = feature_cols or [c for c in DEFAULT_FEATURE_NAMES if c in train_df.columns]
    train_out = train_df.copy()
    val_out = val_df.copy() if val_df is not None else None

    medians = train_out[cols].median().fillna(0.0)
    train_out[cols] = train_out[cols].fillna(medians)

    if val_out is not None:
        val_out[cols] = val_out[cols].fillna(medians)

    return train_out, val_out
