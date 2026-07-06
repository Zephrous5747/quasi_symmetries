#!/bin/bash
#SBATCH --job-name=n2_k_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=n2_sen_diag_%j.out
#SBATCH --error=n2_sen_diag_%j.err

set -euo pipefail

# --- paths (edit if repo lives elsewhere on the cluster) ---
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# =============================================================================
# Python environment (Alliance / Trillium)
# =============================================================================
# Modules provide: numpy, scipy, matplotlib (via scipy-stack).
# Venv must already contain h5py, tqdm, openfermion (install once outside this job).
# See: https://docs.alliancecan.ca/wiki/Python
# =============================================================================
module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack/2024a

VENV="${VENV:-${SCRATCH:-$HOME}/venvs/quasi_symmetries}"
if [[ ! -d "$VENV" ]]; then
  echo "[setup] creating venv at $VENV"
  virtualenv --no-download "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import importlib
required = ("numpy", "scipy", "matplotlib", "h5py", "tqdm", "openfermion")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(f"Missing Python packages after module+venv setup: {missing}")
print("[setup] ok:", ", ".join(required))
PY

# --- thread control: one BLAS thread per worker process ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- sparse-energy parallelism ---
SPARSE_WORKERS="${SPARSE_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-192} - 8 ))}"
export SPARSE_ENERGY_WORKERS="$SPARSE_WORKERS"
export SPARSE_ENERGY_PROFILE=1

# Geometries in parallel (3 fits one node if RAM allows; use 1 if OOM).
export GEOM_WORKERS="${GEOM_WORKERS:-3}"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "REPO=$REPO"
echo "VENV=$VENV"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK:-?} sparse_workers=$SPARSE_ENERGY_WORKERS geom_workers=$GEOM_WORKERS"
date

python -u scripts/n2/benchmark_parity_seniority_energy.py \
  --geom-workers "$GEOM_WORKERS" \
  --sparse-workers "$SPARSE_ENERGY_WORKERS"

python -u scripts/n2/merge_action_diagnostics.py

python -u scripts/plot/n2_operator_diagnostics.py \
  --seniority-csv tables/n2/sto3g/parity_seniority_summary.csv \
  --mixed-pool-csv tables/n2/sto3g/mixed_pool_summary.csv \
  --output images/diagnostics/n2/sto3g/mixed_pool_diagnostics.png

echo "=== done ==="
date
