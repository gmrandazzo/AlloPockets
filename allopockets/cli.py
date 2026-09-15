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

Command Line Interface (CLI) binaries and entrypoints for AlloPockets.
"""

from typing import Optional
import click


@click.group()
@click.version_option(version="1.0.0", prog_name="allopockets")
def main():
    """AlloPockets: Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging."""
    pass


@click.command("predict")
@click.option(
    "--pdb",
    "pdb_input",
    required=True,
    help="PDB ID (e.g. 6t4k) or path to structure file (.cif/.pdb)",
)
@click.option("--chains", default=None, help="Comma-separated protein chains (e.g. 'A,B')")
@click.option("--email", default=None, help="Email for ColabFold MSA server")
@click.option(
    "--uniref-path", default=None, help="Path to local UniRef30 database for offline HHBlits"
)
@click.option("--outdir", default="predict", help="Output directory (default: ./predict)")
@click.option("--model-path", default=None, help="Path to custom model directory")
@click.option(
    "--model",
    "model_name",
    default="minimal_lgbm",
    type=click.Choice(["minimal_lgbm", "minimal_xgboost"]),
    help="Pretrained model name to use if --model-path is not set (default: minimal_lgbm)",
)
def predict_cli(pdb_input, chains, email, uniref_path, outdir, model_path, model_name):
    """Run allosteric pocket prediction on a protein structure."""
    from allopockets.predict import run_prediction_cli

    run_prediction_cli(
        pdb_input=pdb_input,
        chains=chains,
        email=email,
        uniref_path=uniref_path,
        outdir=outdir,
        model_path=model_path,
        model_name=model_name,
    )


@click.command("prepare-data")
@click.option(
    "--db",
    "db_path",
    default="data/database.db",
    help="Path to database.db (default: data/database.db)",
)
@click.option(
    "--outdir", default="data", help="Output directory for generated dataset (default: ./data)"
)
@click.option("--limit", default=None, type=int, help="Limit number of PDBs for quick testing")
@click.option(
    "--pdb-file",
    default=None,
    help="Path to CSV or text file containing PDB IDs to extract",
)
@click.option(
    "--threshold",
    default=0.65,
    type=float,
    help="Overlap threshold for positive allosteric pockets (default: 0.65)",
)
@click.option(
    "--workers",
    default=6,
    type=int,
    help="Number of parallel extraction workers (default: 6)",
)
@click.option(
    "--filename",
    "output_filename",
    default="pockets_dataset.parquet",
    help="Output dataset filename (default: pockets_dataset.parquet)",
)
@click.option(
    "--minimal-chains/--all-chains",
    default=True,
    help="Restrict cavity detection to minimal interacting protein chains (default: True)",
)
@click.option(
    "--outlier-filter/--no-outlier-filter",
    default=False,
    help="Apply author nres outlier filter (|z| < 3) (default: False)",
)
def prepare_data_cli(
    db_path,
    outdir,
    limit,
    pdb_file,
    threshold,
    workers,
    output_filename,
    minimal_chains,
    outlier_filter,
):
    """Extract and featurize pocket dataset directly from database.db."""
    from allopockets.ml.prepare import prepare_dataset

    prepare_dataset(
        db_path=db_path,
        output_dir=outdir,
        limit=limit,
        pdb_file=pdb_file,
        label_threshold=threshold,
        workers=workers,
        output_filename=output_filename,
        use_minimal_chains=minimal_chains,
        outlier_filter=outlier_filter,
    )


@click.command("train")
@click.option(
    "--data", "data_path", required=True, help="Path to prepared dataset (.parquet, .pkl, or .csv)"
)
@click.option(
    "--model",
    "model_type",
    default="hist_gradient_boost",
    type=click.Choice(["hist_gradient_boost", "lightgbm", "xgboost", "autogluon"]),
    help="Model architecture: hist_gradient_boost, lightgbm, xgboost, or autogluon",
)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility (default: 42)")
@click.option(
    "--splits",
    "--n-splits",
    "splits",
    default=5,
    type=int,
    help="Number of CV splits (default: 5)",
)
@click.option("--lr", default=0.03, type=float, help="Learning rate (default: 0.03)")
@click.option(
    "--n-estimators", default=300, type=int, help="Number of boosting trees (default: 300)"
)
@click.option("--max-depth", default=6, type=int, help="Maximum tree depth (default: 6)")
@click.option(
    "--subsample",
    default=0.8,
    type=float,
    help="Row subsampling ratio per tree (default: 0.8)",
)
@click.option(
    "--colsample",
    "--colsample-bytree",
    "colsample",
    default=0.8,
    type=float,
    help="Feature subsampling ratio per tree (default: 0.8)",
)
@click.option(
    "--reg-lambda",
    default=1.0,
    type=float,
    help="L2 regularization strength (default: 1.0)",
)
@click.option(
    "--time-limit",
    default=None,
    type=int,
    help="Time limit in seconds per fit for AutoML backends (e.g. autogluon)",
)
@click.option(
    "--outdir", default="models/lgbm_pocket_classifier", help="Directory to save trained model"
)
@click.option(
    "--test-data",
    "test_data_path",
    default=None,
    help="Path to independent held-out test dataset (.parquet, .pkl, or .csv)",
)
def train_cli(
    data_path,
    model_type,
    seed,
    splits,
    lr,
    n_estimators,
    max_depth,
    subsample,
    colsample,
    reg_lambda,
    time_limit,
    outdir,
    test_data_path,
):
    """Train reproducible Gradient Boost pocket classifier with Grouped Stratified CV."""
    from allopockets.ml.train import train_pipeline

    try:
        train_pipeline(
            data_path=data_path,
            model_type=model_type,
            seed=seed,
            n_splits=splits,
            output_dir=outdir,
            learning_rate=lr,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample=colsample,
            reg_lambda=reg_lambda,
            time_limit=time_limit,
            test_data_path=test_data_path,
        )
    except ImportError as e:
        raise click.ClickException(str(e))


@click.command("inspect-3d")
@click.option("--pdb", "pdb_id", required=True, help="PDB ID (e.g. 6t4k)")
@click.option("--pocket", "pocket_id", required=True, help="Pocket ID (e.g. pocket1)")
@click.option(
    "--path", default="predict", help="Path to prediction/fpocket workspace (default: ./predict)"
)
@click.option(
    "--db",
    "db_path",
    default="data/database.db",
    help="Path to database.db (default: data/database.db)",
)
@click.option("--html", default=None, help="Output path for standalone interactive 3D HTML viewer")
@click.option("--open-browser", is_flag=True, help="Open viewer in default browser automatically")
def inspect_3d_cli(pdb_id, pocket_id, path, db_path, html, open_browser):
    """Inspect and debug 3D pocket coordinates, alpha spheres, and ground truth alignment."""
    from allopockets.viz.inspect import inspect_pocket_cli

    inspect_pocket_cli(
        pdb_id=pdb_id,
        pocket_id=pocket_id,
        path=path,
        db_path=db_path,
        html_output=html,
        open_browser=open_browser,
    )


@click.command("download-model")
@click.option(
    "--model",
    "model_name",
    default="minimal_lgbm",
    type=click.Choice(["minimal_lgbm", "minimal_xgboost"]),
    help="Model name to download (default: minimal_lgbm)",
)
@click.option("--force", is_flag=True, help="Force re-download even if already cached")
def download_model_cli(model_name: str, force: bool):
    """Download pretrained model weights into local user cache."""
    from allopockets.ml.hub import download_model

    click.echo(f"Downloading model '{model_name}'...")
    path = download_model(model_name=model_name, force=force)
    click.echo(f"Model '{model_name}' ready at: {path}")


@click.group("models")
def models_cli():
    """Manage pretrained AlloPockets model weights and cache."""
    pass


@models_cli.command("list")
def models_list_cli():
    """List available pretrained models and local cache status."""
    from allopockets.ml.hub import list_available_models

    models = list_available_models()
    click.echo("\nAvailable Pretrained Models:")
    click.echo("=" * 60)
    for name, meta in models.items():
        status = "CACHED" if meta["cached"] else "NOT CACHED"
        default_flag = " (DEFAULT)" if meta["default"] else ""
        click.echo(f"• {name}{default_flag} [{status}] ({meta['size_kb']} KB)")
        click.echo(f"  Description: {meta['description']}")
        if meta["local_path"]:
            click.echo(f"  Path: {meta['local_path']}")
        click.echo("-" * 60)


@models_cli.command("download")
@click.option(
    "--model",
    "model_name",
    default="minimal_lgbm",
    type=click.Choice(["minimal_lgbm", "minimal_xgboost"]),
    help="Model name to download (default: minimal_lgbm)",
)
@click.option("--force", is_flag=True, help="Force re-download even if already cached")
def models_download_cli(model_name: str, force: bool):
    """Download pretrained model weights into local user cache."""
    from allopockets.ml.hub import download_model

    click.echo(f"Downloading model '{model_name}'...")
    path = download_model(model_name=model_name, force=force)
    click.echo(f"Model '{model_name}' ready at: {path}")


@models_cli.command("path")
@click.option(
    "--model",
    "model_name",
    default="minimal_lgbm",
    type=click.Choice(["minimal_lgbm", "minimal_xgboost"]),
    help="Model name to locate (default: minimal_lgbm)",
)
def models_path_cli(model_name: str):
    """Print the local filesystem directory path for a model."""
    from allopockets.ml.hub import get_model_dir

    try:
        path = get_model_dir(model_name=model_name, auto_download=False)
        click.echo(str(path))
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


@models_cli.command("clean")
@click.option(
    "--model",
    "model_name",
    default=None,
    type=click.Choice(["minimal_lgbm", "minimal_xgboost"]),
    help="Specific model to remove from cache (default: all)",
)
def models_clean_cli(model_name: Optional[str]):
    """Remove cached model weights from local user cache."""
    from allopockets.ml.hub import clear_cache

    clear_cache(model_name=model_name)
    target = f"model '{model_name}'" if model_name else "all models"
    click.echo(f"Cleared cache for {target}.")


# Attach subcommands to main group
main.add_command(predict_cli, name="predict")
main.add_command(prepare_data_cli, name="prepare-data")
main.add_command(train_cli, name="train")
main.add_command(inspect_3d_cli, name="inspect-3d")
main.add_command(download_model_cli, name="download-model")
main.add_command(models_cli, name="models")

if __name__ == "__main__":
    main()
