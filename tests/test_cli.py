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

Test CLI binary commands.
"""

from click.testing import CliRunner
from allopockets.cli import main, predict_cli, prepare_data_cli, train_cli, inspect_3d_cli


def test_main_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "AlloPockets: Allosteric Pocket Prediction" in result.output
    assert "predict" in result.output
    assert "prepare-data" in result.output
    assert "train" in result.output
    assert "inspect-3d" in result.output


def test_subcommand_helps():
    runner = CliRunner()
    for cmd in [predict_cli, prepare_data_cli, train_cli, inspect_3d_cli]:
        result = runner.invoke(cmd, ["--help"])
        assert result.exit_code == 0


def test_train_cli_execution(tmp_path):
    import numpy as np
    import pandas as pd
    from allopockets.ml.config import DEFAULT_FEATURE_NAMES

    # Create small synthetic dataset
    rows = []
    for p in range(6):
        for k in range(4):
            is_pos = int(k == 0)
            r = {"Pockets_pdb": f"pdb_{p}", "Pockets_pocket": f"pkt_{k}", "Label_label": is_pos}
            for feat in DEFAULT_FEATURE_NAMES:
                r[feat] = np.random.randn() + (1.0 if is_pos else -0.5)
            rows.append(r)
    df = pd.DataFrame(rows)
    data_file = tmp_path / "train_data.parquet"
    df.to_parquet(data_file)

    out_dir = tmp_path / "model_out"
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        [
            "--data",
            str(data_file),
            "--splits",
            "2",
            "--n-estimators",
            "10",
            "--subsample",
            "0.7",
            "--colsample",
            "0.7",
            "--reg-lambda",
            "2.0",
            "--outdir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "metrics.json").exists()

    import json

    with open(out_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
    assert "mean_fold_classification_metrics" in metrics
    assert "std_fold_classification_metrics" in metrics
    assert "oof_classification_metrics" in metrics

    # Test with test dataset and --n-splits alias
    test_file = tmp_path / "test_data.parquet"
    df.iloc[:8].to_parquet(test_file)
    test_out_dir = tmp_path / "test_model_out"
    result_test = runner.invoke(
        train_cli,
        [
            "--data",
            str(data_file),
            "--test-data",
            str(test_file),
            "--n-splits",
            "2",
            "--n-estimators",
            "10",
            "--subsample",
            "0.8",
            "--colsample",
            "0.8",
            "--reg-lambda",
            "1.0",
            "--outdir",
            str(test_out_dir),
        ],
    )
    assert result_test.exit_code == 0
    assert (test_out_dir / "metrics.json").exists()


def test_train_cli_autogluon_uninstalled(tmp_path):
    import numpy as np
    import pandas as pd
    from allopockets.ml.config import DEFAULT_FEATURE_NAMES

    # Create small synthetic dataset
    rows = []
    for p in range(4):
        for k in range(2):
            is_pos = int(k == 0)
            r = {"Pockets_pdb": f"pdb_{p}", "Pockets_pocket": f"pkt_{k}", "Label_label": is_pos}
            for feat in DEFAULT_FEATURE_NAMES:
                r[feat] = np.random.randn()
            rows.append(r)
    df = pd.DataFrame(rows)
    data_file = tmp_path / "ag_train_data.parquet"
    df.to_parquet(data_file)

    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        [
            "--data",
            str(data_file),
            "--model",
            "autogluon",
            "--splits",
            "2",
            "--outdir",
            str(tmp_path / "ag_out"),
        ],
    )
    # AutoGluon requires Python <= 3.11, so on 3.13 it gracefully exits with code 1 and error message
    assert result.exit_code != 0
    assert "AutoGluon" in result.output


def test_db_default_paths():
    runner = CliRunner()
    res1 = runner.invoke(prepare_data_cli, ["--help"])
    assert "data/database.db" in res1.output

    res2 = runner.invoke(inspect_3d_cli, ["--help"])
    assert "data/database.db" in res2.output
