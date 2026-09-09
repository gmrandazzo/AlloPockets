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

Evaluation metrics for allosteric pocket classification and ranking.
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    # Area under curves (only valid if both classes are present)
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(np.mean(y_true))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)
    return metrics


def compute_topk_retrieval(
    df: pd.DataFrame,
    pdb_col: str = "pdb",
    label_col: str = "label",
    score_col: str = "score",
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Union[int, float]]:
    """
    Calculate Top-K pocket retrieval rate across protein structures.
    For each PDB structure with at least one annotated positive pocket:
      Ranks all candidate pockets by predicted score descending.
      Checks if at least one true pocket is present within the Top-K candidates.
    """
    pdb_groups = df.groupby(pdb_col)
    eval_pdbs = 0
    hits = {k: 0 for k in ks}

    for pdb_id, group in pdb_groups:
        # Only evaluate structures that have at least one ground-truth positive pocket
        pos_count = (group[label_col] == 1).sum()
        if pos_count == 0:
            continue

        eval_pdbs += 1
        sorted_group = group.sort_values(by=score_col, ascending=False)
        labels = sorted_group[label_col].values

        for k in ks:
            if np.any(labels[:k] == 1):
                hits[k] += 1

    results: Dict[str, Union[int, float]] = {
        "evaluated_pdbs": eval_pdbs,
    }
    for k in ks:
        results[f"top_{k}_accuracy"] = float(hits[k] / eval_pdbs) if eval_pdbs > 0 else 0.0

    return results
