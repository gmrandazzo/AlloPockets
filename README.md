# AlloPockets

**AlloPockets** is a modular Python package and CLI suite for machine learning-based allosteric pocket identification, allosteric communication pathway tracing, and 3D cavity debugging.

Given a protein structure (PDB or mmCIF), AlloPockets detects candidate cavities with `fpocket`, extracts comprehensive geometric and physico-chemical descriptors, and ranks pockets using a reproducible Gradient Boost classifier trained on a curated database of >3,000 allosteric protein complexes.

---

## What AlloPockets Does Now

- **Standard Python Package**: Formally structured under [`allopockets/`](allopockets/) with PEP 561 typing compliance (`py.typed`).
- **5 Standalone CLI Binaries**:
  - `allopockets`: Main unified CLI dispatcher.
  - `allopockets-predict`: End-to-end structure prediction and communication pathway tracing.
  - `allopockets-prepare-data`: Direct training dataset extraction from `data/database.db` with ground-truth labeling (`site_in_pocket >= 0.65`).
  - `allopockets-train`: Reproducible Gradient Boost model training with 5-fold Stratified Group Cross-Validation.
  - `allopockets-inspect-3d`: Interactive 3D visual pocket debugger with terminal diagnostics and standalone 3Dmol.js HTML viewer generation.
- **Reproducible ML Pipeline**: Replaced opaque AutoGluon dependencies with a transparent Gradient Boost architecture (`HistGradientBoostingClassifier`, `LightGBM`, `XGBoost`), tracking MCC, ROC-AUC, PR-AUC, F1, and structural Top-1 / Top-3 / Top-5 retrieval rates.
- **Dedicated Data Directory**: Database files moved out of the package root into [`data/database.db`](data/database.db).
- **Consolidated Notebooks**: All analysis and benchmark notebooks are organized in [`notebooks/`](notebooks/).

---

## Installation

### 1. Create and Activate an Isolated Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install AlloPockets in Editable Mode
```bash
# Core package + test dependencies
pip install -e ".[dev]"

# Optional: with LightGBM and XGBoost support
pip install -e ".[dev,gbm]"
```

---

## Quickstart Examples

### A. Command Line Interface (CLI)

#### 1. Allosteric Pocket Prediction
Run pocket detection, feature extraction, and allosteric probability ranking on a query structure:
```bash
allopockets-predict --pdb 6t4k --chains A --outdir predict_results
```

#### 2. Extract Training Data from `database.db`
Extract candidate cavities and 186 biophysical features directly from `data/database.db`:
```bash
allopockets-prepare-data --db data/database.db --outdir data/training --threshold 0.65
```

#### 3. Train a Reproducible Pocket Classifier
Train a 5-fold grouped cross-validated model (`hist_gradient_boost`, `lightgbm`, `xgboost`, or `autogluon`) with regularization controls, mean-fold metrics, and export serialized artifacts (`model.joblib`, `features.json`, `metrics.json`):
```bash
allopockets-train --data data/training/dataset.parquet --model lightgbm --subsample 0.8 --colsample 0.8 --reg-lambda 1.0 --splits 5 --outdir models/my_pocket_model
```

#### 4. 3D Pocket Visual Debugger & Inspector
Inspect 3D coordinates, alpha spheres, volume, and alignment with the ground-truth modulator site:
```bash
allopockets-inspect-3d --pdb 6t4k --pocket pocket1 --db data/database.db --html 6t4k_pocket1_debug.html
```

#### 5. Unified Command Dispatcher
All tools can also be invoked via the root `allopockets` command:
```bash
allopockets --help
allopockets predict --help
allopockets prepare-data --help
allopockets train --help
allopockets inspect-3d --help
```

---

### B. Python API

#### Running Prediction in Python:
```python
from allopockets import predict, get_cif

# Fetch structure
pdb = get_cif(pdb_id="6t4k")

# Predict allosteric pockets and rank by score
clean_pdb, predictions = predict(
    pdb,
    protein_chains=["A"],
    email="you@institution.edu"  # for ColabFold MSA retrieval (optional)
)

print(predictions.head())
```

#### Training and Model Inspection:
```python
from allopockets.ml.models import PocketClassifier
from allopockets.ml.config import ModelConfig
from allopockets.ml.dataset import load_pocket_dataset

# Load dataset
df = load_pocket_dataset("data/training/dataset.parquet")

# Initialize and train classifier
clf = PocketClassifier(config=ModelConfig(model_type="hist_gradient_boost", learning_rate=0.03))
# Fit and score
clf.save("models/my_model")
```

#### 3D HTML Viewer Generation:
```python
from allopockets.viz.html_viewer import generate_3d_pocket_html

html = generate_3d_pocket_html(
    pdb_id="6t4k",
    pocket_id="pocket1",
    pdb_cif_content=pdb_cif_text,
    pocket_cif_content=pocket_cif_text,
    metrics={"site_in_pocket": 0.85, "volume": 720.0, "label": 1},
    output_path="viewer.html"
)
```

---

## Benchmark Training Experiment & Evaluation

AlloPockets includes a reproducible full-benchmark training and evaluation pipeline comparing 229 train protein complexes against 60 independent held-out test complexes.

### Run the Experiment
Run all stages automatically via:
```bash
./scripts/run_full_training_experiment.sh
```

For complete step-by-step CLI commands, dataset featurization details, and feature importances, see [docs/BENCHMARK_EXPERIMENT.md](docs/BENCHMARK_EXPERIMENT.md).

### Results Summary (LightGBM vs. XGBoost)

| Evaluation Stage | Metric | LightGBM | XGBoost | Observation |
|---|---|:---:|:---:|---|
| **5-Fold Cross-Validation (OOF)** | **ROC-AUC** | 0.7879 | **0.8487** | +0.0608 global ranking gain |
| (229 PDBs, 8,970 pockets) | **PR-AUC** | **0.0251** | 0.0235 | ~5x above random baseline (0.0051) |
| | **MCC (th=0.5)** | **0.0234** | -0.0038 | Near zero due to 194:1 class imbalance |
| | **Top-1 Pocket Retrieval** | 16.3% | **18.6%** | Higher chance of true pocket at rank #1 |
| | **Top-3 Pocket Retrieval** | 34.9% | **44.2%** | **+9.3%** boost in candidate shortlisting |
| **Independent Held-Out Test Set** | **Test ROC-AUC** | 0.7443 | **0.8908** | **+0.1465** separation on unseen proteins |
| (60 PDBs, 2,521 pockets) | **Test PR-AUC** | **0.0311** | 0.0201 | -0.0110 |
| | **Test MCC (th=0.5)** | -0.0022 | -0.0019 | Low at default th; reaches +0.11 at th=0.02 |
| | **Test Top-1 Retrieval** | 12.5% | 12.5% | Equal |
| | **Test Top-3 Retrieval** | 25.0% | 25.0% | Equal |

> [!TIP]
> **Reproduced Author Curation Protocol**: Filtering out non-informative zero-positive structures (author's `6.Training_sets.ipynb` protocol) increases training positive prevalence to ~1.7–5.3%, boosting **PR-AUC to 0.12–0.18**, **MCC to 0.19–0.21**, **Top-1 retrieval to 25.0–32.5%**, and **Top-3 retrieval to 50.0%**. See [docs/BENCHMARK_EXPERIMENT.md](docs/BENCHMARK_EXPERIMENT.md) for full benchmarks.

---

## Repository Layout

```
AlloPockets/
├── allopockets/             # Main installable Python package
│   ├── py.typed             # PEP 561 typing marker
│   ├── cli.py               # Click CLI entrypoints
│   ├── predict.py           # Prediction & pathway calculation
│   ├── database/            # Database ORM models (PDB, Site) & CIF utilities
│   ├── features/            # Biophysical descriptor extraction (DSSP, FreeSASA, etc.)
│   ├── ml/                  # ML models, CV splits, data preparation & training
│   ├── pockets/             # fpocket execution wrapper & Pocket geometry
│   └── viz/                 # Interactive 3Dmol.js HTML visualizer & 3D debugger
├── data/                    # Data directory (external to python package)
│   ├── database.db          # SQLite database of curated allosteric structures
│   ├── benchmark/           # Featurized train & test parquet datasets
│   └── README.md            # Data sources and download links
├── docs/                    # Technical documentation
│   └── BENCHMARK_EXPERIMENT.md  # Detailed benchmark training & evaluation guide
├── models/                  # Pre-trained model weights & benchmark outputs
│   ├── benchmark_experiment/ # Benchmark model, metrics & feature importance
│   └── pockets_physchem_deploy/
├── notebooks/               # Centralized Jupyter notebooks directory
│   ├── predict.ipynb        # User interactive prediction notebook
│   ├── predict_advanced.ipynb
│   ├── database.ipynb       # Database exploration notebook
│   ├── database/            # Database curation & statistics notebooks
│   ├── training_data/       # Clustering, minimal structures & feature notebooks
│   └── models/              # Model ablation, benchmarking & comparison notebooks
├── scripts/                 # Automation & utility scripts
│   └── run_full_training_experiment.sh # End-to-end benchmark script
├── tests/                   # Pytest test suite
├── predict.py               # Root backward-compatible wrapper
├── pyproject.toml           # Build system, dependencies, and configuration
└── README.md
```

---

## Development & Verification

Run the automated test suite:
```bash
pytest tests/ -v
```

Run static type checking with `mypy`:
```bash
mypy allopockets tests
```

---

## HHBlits for Offline Computations

By default, ColabFold's MSA server is used for sequence profile extraction. For batch offline workflows:
1. Download and extract the [UniRef30 database](https://wwwuser.gwdguser.de/~compbiol/uniclust/2023_02).
2. Pass the database directory via `--uniref-path` in the CLI or `uniref_path=` in `get_features()`.

---

## Cite

If you use AlloPockets, please cite:

```bibtex
@software{AlloPockets,
  title  = {AlloPockets},
  author = {AlloPockets authors},
  url    = {https://github.com/frannerin/AlloPockets},
  year   = {2026}
}
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

**Note:** PyRosetta dependency requires separate licensing for commercial use ([www.pyrosetta.org](https://www.pyrosetta.org)).

![6T4K](https://github.com/user-attachments/assets/f392b49f-500a-4e38-acd3-fd431f0aaa9d)
