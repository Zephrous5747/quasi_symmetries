#!/bin/bash
#SBATCH --job-name=h2o_sym_rerun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=h2o_sym_rerun_%j.out
#SBATCH --error=h2o_sym_rerun_%j.err

set -euo pipefail

# --- paths (edit if repo lives elsewhere on the cluster) ---
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# =============================================================================
# Python environment (Alliance / Trillium)
# =============================================================================
# Modules provide: numpy, scipy, matplotlib (via scipy-stack).
# Venv must already contain h5py, tqdm, openfermion, pyscf (install once outside this job).
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
required = (
    "numpy", "scipy", "matplotlib", "h5py", "tqdm",
    "openfermion", "pyscf",
)
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing Python packages after module+venv setup: " + ", ".join(missing)
    )
print("[setup] ok:", ", ".join(required))
PY

# --- thread control: one BLAS thread per worker process ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Geometries in parallel (3 fits one node if RAM allows; use 1 if OOM).
export GEOM_WORKERS="${GEOM_WORKERS:-3}"
export OPTIMIZER_MAXFEV="${OPTIMIZER_MAXFEV:-1000}"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "REPO=$REPO"
echo "VENV=$VENV"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK:-?} geom_workers=$GEOM_WORKERS optimizer_maxfev=$OPTIMIZER_MAXFEV"
date

# 1. Regenerate symmetry-adapted H2O STO-3G caches (10 geometries).
python -u scripts/generate_hamiltonians.py \
  --molecule h2o \
  --overwrite \
  --hoh-angle-deg 104.5

# 2. Seniority + mixed pool + variance heatmaps + energy CSVs.
python -u scripts/h2o/parity_optimization_heatmaps.py \
  --max-workers "$GEOM_WORKERS" \
  --n-restarts "${N_RESTARTS:-1}" \
  --optimizer-maxfev "$OPTIMIZER_MAXFEV"

# 3. Operator diagnostic figure (canonical output paths).
python -u scripts/plot/h2o_operator_diagnostics.py \
  --seniority-csv tables/h2o/sto3g/parity_seniority_diagnostics.csv \
  --mixed-pool-csv tables/h2o/sto3g/mixed_pool_energy_diagnostics.csv \
  --output images/diagnostics/h2o/sto3g/mixed_pool_diagnostics.png

echo "=== done ==="
date
