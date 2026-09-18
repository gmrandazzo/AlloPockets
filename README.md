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

### Standard Installation from PyPI
```bash
pip install allopockets
```

### External C/C++ Bioinformatics Dependencies
AlloPockets relies on three standard bioinformatics binaries to extract features from structures:
- `fpocket` (for pocket detection)
- `dssp` / `mkdssp` (for secondary structure and relative ASA)
- `hhsuite` / `hhmake` (for generating HMM profiles)

If you already have these in your system `$PATH` (e.g. via Conda), AlloPockets will automatically detect and use them!

If you do **not** use Conda (e.g. on macOS or a fresh Linux install) and don't want to compile them manually, AlloPockets includes an automated installer that will fetch and compile them into a local hidden directory (`~/.allopockets/bin`) without requiring `sudo`:

```bash
allopockets-install-deps
```
*(Note: `fpocket` and `hhsuite` will be compiled automatically. For `dssp`, the script will guide you to install it via your OS package manager like Homebrew or apt-get).*

### Install in an Isolated Virtual Environment (Development)
```bash
python3 -m venv .venv
source .venv/bin/activate

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
Run all stages of the baseline experiment automatically via:
```bash
./scripts/run_full_training_experiment.sh
```

### Reproduce All 3 Feature Generations (Gen 1, Gen 2, Gen 3)
To reproduce and evaluate the multi-generation benchmark:
```bash
# Fast mode (< 1 min): train and evaluate LightGBM & XGBoost across Gen 1, Gen 2, and Gen 3
./scripts/reproduce_multigen_benchmark.sh

# Report-only mode: instantaneously print the Grand Master comparison matrix and takeaways
./scripts/reproduce_multigen_benchmark.sh --report-only

# Full extraction mode: re-extract Route 1 multi-modal features from raw CIF structures
./scripts/reproduce_multigen_benchmark.sh --reextract
```

For complete step-by-step CLI commands, dataset featurization details, feature importances, and the Grand Master comparison matrix, see [docs/BENCHMARK_EXPERIMENT.md](docs/BENCHMARK_EXPERIMENT.md).

### Results Summary: Baselines, Regularized Models & AutoGluon

| Model & Protocol | Mean Fold PR-AUC | Test ROC-AUC | Test PR-AUC | Test MCC | Test Top-1 | Test Top-3 | Key Highlights |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Baseline LightGBM (Raw)** | N/A | 0.7443 | 0.0311 | -0.0022 | 12.5% | 25.0% | Baseline benchmark configuration |
| **Baseline XGBoost (Raw)** | N/A | 0.8908 | 0.0201 | -0.0019 | 12.5% | 25.0% | Strong raw discrimination (+0.15 ROC-AUC) |
| **Regularized LightGBM (Raw)** | 0.0297 ± 0.009 | **0.9037** | **0.0365** | **+0.0993** | 12.5% | 25.0% | Bagging & L2 eliminate memorization; positive MCC |
| **Regularized XGBoost (Raw)** | 0.0330 ± 0.008 | **0.9218** | **0.0331** | **+0.1163** | 12.5% | **50.0%** | **Doubled Top-3 candidate retrieval** to 50% |
| **Curated Regularized LightGBM** | 0.1325 ± 0.054 | **0.8992** | **0.1854** | **+0.2242** | 12.5% | 37.5% | **Highest PR-AUC** (0.1854) and robust positive MCC |
| **Curated Regularized XGBoost** | **0.1525 ± 0.024** | **0.9057** | 0.1491 | **+0.1729** | **25.0%** | 37.5% | **Highest cross-validation PR-AUC** (0.1525 ± 0.02) |
| **AutoGluon (Curated)** | 0.1399 ± 0.064 | **0.9495** | 0.1683 | -0.0039 | 0.0% | **75.0%** | **75% Top-3 retrieval** on held-out test proteins |
| **Minimal Regularized LightGBM (186 feats)** | 0.5392 ± 0.041 | **0.9703** | **0.6540** | **+0.5864** | **69.5%** | **83.1%** | **Beats author model5** in Test ROC-AUC (0.9703) & PR-AUC (0.6540) |
| **Minimal Regularized XGBoost (186 feats)** | 0.5344 ± 0.022 | **0.9684** | **0.6758** | **+0.6093** | **67.8%** | **88.1%** | **Highest PR-AUC** (0.6758, +0.095 over model5) & **88.1% Top-3 retrieval** |
| **Author's AutoGluon (`model5`)** | N/A | **0.9631** | **0.5811** | **+0.6554** | **83.3%** | **87.5%** | Pre-computed author deployment reference (186 feats) |

> [!TIP]
> **Minimal Chains Extraction (`--minimal-chains`)**: Slicing structures to interacting protein chains and aligning coordinate numbering via `auth_seq_id` recovers the exact 186-feature dataset used in `model5`. Minimal models attain **0.654–0.676 Test PR-AUC** and **88.1% Top-3 retrieval**, directly surpassing the author's reference model. See [docs/BENCHMARK_EXPERIMENT.md](docs/BENCHMARK_EXPERIMENT.md) for full benchmarks.

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
│   ├── reproduce_multigen_benchmark.sh # Master multi-generation reproduction runner
│   ├── run_full_training_experiment.sh # End-to-end baseline benchmark script (23 feats)
│   ├── extract_minimal_structures_benchmark.py # Recreate author minimal dataset & evaluate vs model5
│   ├── extract_route1_features.py      # Route 1 parallel multi-modal feature extractor (155 feats)
│   ├── generate_route2_features.py     # Route 2 186-feature dataset generator and trainer
│   └── train_and_evaluate_route1.py    # Route 1 training and evaluation runner
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

## Authors & Contributors

- **Francho Nerín Fonz** (<fnerin@bioacademy.gr>) – Original author and methodology design ([frannerin/AlloPockets](https://github.com/frannerin/AlloPockets))
- **Giuseppe Marco Randazzo** (<gmrandazzo@gmail.com>) – Modernization, packaging, model distribution hub, CLI suite, and maintainer ([gmrandazzo/AlloPockets](https://github.com/gmrandazzo/AlloPockets))

---

## Cite

If you use AlloPockets, please cite:

```bibtex
@software{AlloPockets,
  title  = {AlloPockets: Machine Learning Allosteric Pocket Prediction, Pathway Tracing & 3D Visualization},
  author = {Ner{\'\i}n Fonz, Francho and Randazzo, Giuseppe Marco},
  url    = {https://github.com/gmrandazzo/AlloPockets},
  year   = {2026}
}
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

**Note:** PyRosetta dependency requires separate licensing for commercial use ([www.pyrosetta.org](https://www.pyrosetta.org)).

![6T4K](https://github.com/user-attachments/assets/f392b49f-500a-4e38-acd3-fd431f0aaa9d)
