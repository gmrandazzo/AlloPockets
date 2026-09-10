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

Test machine learning models, metrics, and dataset preparation.
"""

import tempfile
import numpy as np
import pandas as pd
import pytest

from allopockets.ml.config import ModelConfig
from allopockets.ml.dataset import get_group_splits
from allopockets.ml.metrics import compute_classification_metrics, compute_topk_retrieval
from allopockets.ml.models import PocketClassifier


@pytest.fixture
def synthetic_pocket_data():
    """Create synthetic multi-structure pocket dataset for testing."""
    np.random.seed(42)
    n_pdbs = 10
    rows = []
    feature_cols = [f"feat_{i}" for i in range(10)]

    for p in range(n_pdbs):
        pdb_id = f"pdb_{p:03d}"
        n_pockets = np.random.randint(5, 12)
        # Exactly one pocket is ground-truth allosteric
        pos_pocket = np.random.randint(0, n_pockets)

        for pkt in range(n_pockets):
            is_pos = int(pkt == pos_pocket)
            # Make positive pocket have slightly higher feature values
            feats = np.random.randn(10) + (1.5 if is_pos else -0.5)
            row = {
                "Pockets_pdb": pdb_id,
                "Pockets_pocket": f"pocket{pkt}",
                "Label_label": is_pos,
            }
            for i, col in enumerate(feature_cols):
                row[col] = feats[i]
            rows.append(row)

    df = pd.DataFrame(rows)
    return df, feature_cols


def test_metrics_computation():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    m = compute_classification_metrics(y_true, y_prob)
    assert "mcc" in m
    assert "roc_auc" in m
    assert "pr_auc" in m
    assert m["mcc"] == 1.0  # Perfect separation
    assert m["roc_auc"] == 1.0


def test_topk_retrieval():
    df = pd.DataFrame(
        {
            "pdb": ["1a", "1a", "1a", "2b", "2b", "2b"],
            "pocket": ["p1", "p2", "p3", "p1", "p2", "p3"],
            "label": [0, 1, 0, 1, 0, 0],
            "score": [0.3, 0.9, 0.1, 0.8, 0.2, 0.4],
        }
    )

    ret = compute_topk_retrieval(df, pdb_col="pdb", label_col="label", score_col="score")
    assert ret["evaluated_pdbs"] == 2
    assert ret["top_1_accuracy"] == 1.0
    assert ret["top_3_accuracy"] == 1.0


def test_stratified_group_splits(synthetic_pocket_data):
    df, _ = synthetic_pocket_data
    splits = get_group_splits(df, group_col="Pockets_pdb", target_col="Label_label", n_splits=3)
    assert len(splits) == 3

    for train_idx, val_idx in splits:
        train_pdbs = set(df.iloc[train_idx]["Pockets_pdb"])
        val_pdbs = set(df.iloc[val_idx]["Pockets_pdb"])
        # Ensure zero leakage between train and val
        assert len(train_pdbs.intersection(val_pdbs)) == 0


def test_pocket_classifier_fit_save_load(synthetic_pocket_data):
    df, feature_cols = synthetic_pocket_data
    cfg = ModelConfig(model_type="hist_gradient_boost", n_estimators=20, seed=42)
    clf = PocketClassifier(config=cfg, feature_names=feature_cols)

    X = df[feature_cols].values
    y = df["Label_label"].values

    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(df), 2)
    scores = clf.predict_score(X)
    assert len(scores) == len(df)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save(tmpdir)
        loaded_clf = PocketClassifier.load(tmpdir)
        loaded_scores = loaded_clf.predict_score(X)
        np.testing.assert_allclose(scores, loaded_scores)


def test_pocket_classifier_regularized_backends(synthetic_pocket_data):
    df, feature_cols = synthetic_pocket_data
    X = df[feature_cols].values
    y = df["Label_label"].values

    for backend in ("lightgbm", "xgboost"):
        cfg = ModelConfig(
            model_type=backend,
            n_estimators=10,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=0.1,
            seed=42,
        )
        clf = PocketClassifier(config=cfg, feature_names=feature_cols)
        clf.fit(X, y)
        scores = clf.predict_score(X)
        assert len(scores) == len(df)
        assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_pocket_classifier_autogluon_handling():
    cfg = ModelConfig(model_type="autogluon")
    try:
        clf = PocketClassifier(config=cfg)
        assert clf.model_backend == "autogluon"
    except ImportError as e:
        assert "AutoGluon" in str(e)


def test_train_pipeline_mean_fold_metrics(synthetic_pocket_data, tmp_path):
    from allopockets.ml.train import train_pipeline

    df, _ = synthetic_pocket_data
    data_file = tmp_path / "cv_data.parquet"
    df.to_parquet(data_file)
    out_dir = tmp_path / "cv_model_out"

    res = train_pipeline(
        data_path=data_file,
        model_type="hist_gradient_boost",
        n_splits=3,
        n_estimators=10,
        subsample=0.8,
        colsample=0.8,
        reg_lambda=1.5,
        output_dir=str(out_dir),
    )

    assert "mean_fold_classification_metrics" in res
    assert "std_fold_classification_metrics" in res
    assert "pr_auc" in res["mean_fold_classification_metrics"]
    assert "roc_auc" in res["mean_fold_classification_metrics"]
    assert "mcc" in res["mean_fold_classification_metrics"]
    assert not np.isnan(res["mean_fold_classification_metrics"]["pr_auc"])
