#!/usr/bin/env bash
#
# AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.
#
# Author: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
# License: GNU General Public License v3.0 (GPL-3.0)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "======================================================================"
echo "  AlloPockets: Full Benchmark Training & Evaluation Experiment"
echo "======================================================================"

# 1. Environment & Prerequisite Checks
export PATH="${REPO_ROOT}/.venv/bin:${HOME}/.local/bin:${PATH}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

if ! command -v fpocket &>/dev/null; then
    echo "ERROR: fpocket binary not found in PATH." >&2
    exit 1
fi

DB_FILE="${REPO_ROOT}/data/database.db"
TRAIN_PDB_FILE="${REPO_ROOT}/models/trainset_protnames.csv"
TEST_PDB_FILE="${REPO_ROOT}/models/testset_protnames.csv"

if [ ! -f "${DB_FILE}" ]; then
    echo "ERROR: Database file not found: ${DB_FILE}" >&2
    exit 1
fi

if [ ! -f "${TRAIN_PDB_FILE}" ]; then
    echo "ERROR: Train set PDB file not found: ${TRAIN_PDB_FILE}" >&2
    exit 1
fi

if [ ! -f "${TEST_PDB_FILE}" ]; then
    echo "ERROR: Test set PDB file not found: ${TEST_PDB_FILE}" >&2
    exit 1
fi

WORKERS="${WORKERS:-6}"
BENCHMARK_DATA_DIR="${REPO_ROOT}/data/benchmark"
EXPERIMENT_MODEL_DIR="${REPO_ROOT}/models/benchmark_experiment"
mkdir -p "${BENCHMARK_DATA_DIR}" "${EXPERIMENT_MODEL_DIR}"

TRAIN_PARQUET="${BENCHMARK_DATA_DIR}/train_pockets.parquet"
TEST_PARQUET="${BENCHMARK_DATA_DIR}/test_pockets.parquet"

echo "Configuration:"
echo "  Repository root:   ${REPO_ROOT}"
echo "  Parallel workers:  ${WORKERS}"
echo "  Database:          ${DB_FILE}"
echo "  Train PDBs:        ${TRAIN_PDB_FILE}"
echo "  Test PDBs:         ${TEST_PDB_FILE}"
echo "  Data output dir:   ${BENCHMARK_DATA_DIR}"
echo "  Model output dir:  ${EXPERIMENT_MODEL_DIR}"
echo "----------------------------------------------------------------------"

# 2. Stage 1: Extract and Featurize Benchmark Train Set
if [ -f "${TRAIN_PARQUET}" ] && [ "${FORCE_REEXTRACT:-0}" != "1" ]; then
    echo "[Stage 1/4] Train dataset already exists at ${TRAIN_PARQUET}, skipping extraction (set FORCE_REEXTRACT=1 to redo)."
else
    echo "[Stage 1/4] Extracting pockets & features for 229 Train PDBs..."
    python -m allopockets.cli prepare-data \
        --db "${DB_FILE}" \
        --pdb-file "${TRAIN_PDB_FILE}" \
        --outdir "${BENCHMARK_DATA_DIR}" \
        --filename "train_pockets.parquet" \
        --workers "${WORKERS}"
    echo "Train dataset created at: ${TRAIN_PARQUET}"
fi

# 3. Stage 2: Extract and Featurize Benchmark Test Set
if [ -f "${TEST_PARQUET}" ] && [ "${FORCE_REEXTRACT:-0}" != "1" ]; then
    echo "[Stage 2/4] Test dataset already exists at ${TEST_PARQUET}, skipping extraction (set FORCE_REEXTRACT=1 to redo)."
else
    echo "[Stage 2/4] Extracting pockets & features for 60 Test PDBs..."
    python -m allopockets.cli prepare-data \
        --db "${DB_FILE}" \
        --pdb-file "${TEST_PDB_FILE}" \
        --outdir "${BENCHMARK_DATA_DIR}" \
        --filename "test_pockets.parquet" \
        --workers "${WORKERS}"
    echo "Test dataset created at: ${TEST_PARQUET}"
fi

# 4. Stage 3: Train 5-Fold Stratified Group K-Fold LightGBM & Evaluate Test Set
echo "[Stage 3/4] Training LightGBM with 5-Fold CV & evaluating test set..."
python -m allopockets.cli train \
    --data "${TRAIN_PARQUET}" \
    --test-data "${TEST_PARQUET}" \
    --model lightgbm \
    --outdir "${EXPERIMENT_MODEL_DIR}" \
    --n-estimators 300 \
    --lr 0.03 \
    --max-depth 6 \
    --splits 5

# 5. Stage 4: Experiment Summary Report
echo "[Stage 4/4] Experiment complete. Summary of metrics:"
echo "----------------------------------------------------------------------"
if [ -f "${EXPERIMENT_MODEL_DIR}/metrics.json" ]; then
    python - "${EXPERIMENT_MODEL_DIR}/metrics.json" << 'EOF'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
with open(metrics_path) as f:
    d = json.load(f)

model_type = str(d.get("model_type", "unknown")).upper()
n_samples = d.get("n_samples", 0)
n_features = d.get("n_features", 0)
print(f"Model Type:     {model_type}")
print(f"Train Samples:  {n_samples} pockets ({n_features} features)")

oof = d.get("oof_classification_metrics", {})
print("Cross-Validation (Out-of-Fold):")
print(f"  MCC:          {oof.get('mcc', 0.0):.4f}")
print(f"  ROC-AUC:      {oof.get('roc_auc', 0.0):.4f}")
print(f"  PR-AUC:       {oof.get('pr_auc', 0.0):.4f}")

oof_topk = d.get("oof_topk_retrieval", {})
top1 = oof_topk.get("top_1_accuracy", 0.0) * 100
top3 = oof_topk.get("top_3_accuracy", 0.0) * 100
print(f"  Top-1 Acc:    {top1:.1f}%")
print(f"  Top-3 Acc:    {top3:.1f}%")

if "test_metrics" in d:
    tm = d["test_metrics"]
    t_cls = tm.get("classification_metrics", {})
    t_topk = tm.get("topk_retrieval", {})
    t_samples = tm.get("n_samples", 0)
    print(f"\nHeld-Out Test Set ({t_samples} pockets):")
    print(f"  Test MCC:     {t_cls.get('mcc', 0.0):.4f}")
    print(f"  Test ROC-AUC: {t_cls.get('roc_auc', 0.0):.4f}")
    print(f"  Test PR-AUC:  {t_cls.get('pr_auc', 0.0):.4f}")
    test_top1 = t_topk.get("top_1_accuracy", 0.0) * 100
    test_top3 = t_topk.get("top_3_accuracy", 0.0) * 100
    print(f"  Test Top-1:   {test_top1:.1f}%")
    print(f"  Test Top-3:   {test_top3:.1f}%")
EOF
fi

echo "----------------------------------------------------------------------"
echo "Artifacts generated:"
echo "  - Train data:          ${TRAIN_PARQUET}"
echo "  - Test data:           ${TEST_PARQUET}"
echo "  - Model checkpoint:    ${EXPERIMENT_MODEL_DIR}/model.joblib"
echo "  - Metrics summary:     ${EXPERIMENT_MODEL_DIR}/metrics.json"
echo "  - Feature importance:  ${EXPERIMENT_MODEL_DIR}/feature_importance.csv"
echo "======================================================================"
