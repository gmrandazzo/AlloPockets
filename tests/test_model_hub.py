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

Unit tests for model hub, cache resolution, and weight downloading.
"""

from pathlib import Path
import tempfile
from click.testing import CliRunner
import pytest

from allopockets.cli import main
from allopockets.ml.hub import (
    AVAILABLE_MODELS,
    clear_cache,
    get_cache_dir,
    get_model_dir,
    list_available_models,
    load_model,
)
from allopockets.ml.models import PocketClassifier


def test_get_cache_dir_default():
    cache_dir = get_cache_dir()
    assert isinstance(cache_dir, Path)
    assert cache_dir.name == "allopockets"
    assert cache_dir.exists()


def test_get_cache_dir_xdg_env(monkeypatch):
    with tempfile.TemporaryDirectory() as custom_dir:
        monkeypatch.setenv("XDG_CACHE_HOME", custom_dir)
        cache_dir = get_cache_dir()
        assert cache_dir == Path(custom_dir) / "allopockets"
        assert cache_dir.exists()


def test_list_available_models():
    models = list_available_models()
    assert "minimal_lgbm" in models
    assert "minimal_xgboost" in models

    lgbm_meta = models["minimal_lgbm"]
    assert "description" in lgbm_meta
    assert "size_kb" in lgbm_meta
    assert lgbm_meta["default"] is True
    assert isinstance(lgbm_meta["cached"], bool)


def test_get_model_dir_local_repo():
    # When running from repo checkout, get_model_dir finds models/minimal_lgbm
    model_dir = get_model_dir("minimal_lgbm", auto_download=False)
    assert model_dir.exists()
    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "features.json").exists()


def test_get_model_dir_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model 'non_existent_model'"):
        get_model_dir("non_existent_model", auto_download=True)


def test_load_model():
    clf = load_model("minimal_lgbm")
    assert isinstance(clf, PocketClassifier)
    assert len(clf.feature_names) == 186


def test_clear_cache():
    cache_dir = get_cache_dir() / "models" / "dummy_test_model"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = cache_dir / "model.joblib"
    dummy_file.write_text("test")
    assert dummy_file.exists()

    clear_cache("dummy_test_model")
    assert not cache_dir.exists()


def test_cli_models_commands():
    runner = CliRunner()

    # Test models list
    res = runner.invoke(main, ["models", "list"])
    assert res.exit_code == 0
    assert "minimal_lgbm" in res.output
    assert "minimal_xgboost" in res.output

    # Test models path
    res = runner.invoke(main, ["models", "path", "--model", "minimal_lgbm"])
    assert res.exit_code == 0
    assert "minimal_lgbm" in res.output

    # Test download-model help
    res = runner.invoke(main, ["download-model", "--help"])
    assert res.exit_code == 0
    assert "--model" in res.output
