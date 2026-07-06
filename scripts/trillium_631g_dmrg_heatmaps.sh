#!/bin/bash
#SBATCH --job-name=631g_dmrg_heat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=out/631g_dmrg_heat_%j.out
#SBATCH --error=out/631g_dmrg_heat_%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# H2O + N2 canonical parity heatmaps on the full default geometry grids (6-31G, DMRG).
# Parallelism: one Block2 DMRG per Python process (ProcessPoolExecutor).
# Keep GEOM_WORKERS <= --cpus-per-task; do NOT use all 192 cores in one Python job.
export BASIS="${BASIS:-6-31g}"
export GEOM_WORKERS="${GEOM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
export DMRG_CACHE_ROOT="${DMRG_CACHE_ROOT:-$REPO/dmrg_cache}"
export DMRG_BOND_DIMENSION="${DMRG_BOND_DIMENSION:-512}"
export DMRG_MAX_SWEEPS="${DMRG_MAX_SWEEPS:-30}"
export DMRG_THREADS="${DMRG_THREADS:-1}"

export H2O_HEATMAP_DIR="${H2O_HEATMAP_DIR:-$REPO/images/orbital_heatmaps/h2o_631g_dmrg}"
export N2_HEATMAP_DIR="${N2_HEATMAP_DIR:-$REPO/images/orbital_heatmaps/n2_631g_dmrg}"
export H2O_SUMMARY_CSV="${H2O_SUMMARY_CSV:-$REPO/tables/631g/h2o_dmrg_canonical_variance_summary.csv}"
export N2_SUMMARY_CSV="${N2_SUMMARY_CSV:-$REPO/tables/631g/n2_dmrg_canonical_variance_summary.csv}"

mkdir -p out "$DMRG_CACHE_ROOT" \
  "$H2O_HEATMAP_DIR" "$N2_HEATMAP_DIR" \
  "$(dirname "$H2O_SUMMARY_CSV")" "$(dirname "$N2_SUMMARY_CSV")"

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack/2024a

VENV="${VENV:-${SCRATCH:-$HOME}/venvs/quasi_symmetries}"
if [[ ! -d "$VENV" ]]; then
  echo "[setup] missing venv: $VENV" >&2
  echo "[setup] create it outside the job and install pyscf openfermionpyscf block2" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python - <<'PY'
import importlib
required = (
    "numpy", "scipy", "matplotlib", "h5py", "tqdm",
    "openfermion", "openfermionpyscf", "pyscf", "pyblock2.driver.core",
)
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages in venv: " + ", ".join(missing))
print("[setup] ok:", ", ".join(required))
PY

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "REPO=$REPO"
echo "BASIS=$BASIS"
echo "GEOM_WORKERS=$GEOM_WORKERS (one Block2 DMRG per process)"
echo "DMRG_BOND_DIMENSION=$DMRG_BOND_DIMENSION"
echo "DMRG_MAX_SWEEPS=$DMRG_MAX_SWEEPS"
echo "DMRG_THREADS=$DMRG_THREADS"
date

COMMON_ARGS=(
  --basis "$BASIS"
  --wavefunction dmrg
  --dmrg-cache-root "$DMRG_CACHE_ROOT"
  --dmrg-bond-dimension "$DMRG_BOND_DIMENSION"
  --dmrg-max-sweeps "$DMRG_MAX_SWEEPS"
  --dmrg-threads "$DMRG_THREADS"
  --max-workers "$GEOM_WORKERS"
  --overwrite
)

python -u - <<'PY'
from quasi_symmetries.hamiltonian.geometry import default_grid_for_molecule

for molecule in ("h2o", "n2"):
    grid = [float(x) for x in default_grid_for_molecule(molecule)]
    print(f"[grid] {molecule}: {len(grid)} geometries -> {grid}")
PY

echo "=== [1/2] H2O 6-31G DMRG heatmaps (parallel over geometries) ==="
python -u scripts/h2o/generate_basis_variance_heatmaps.py \
  "${COMMON_ARGS[@]}" \
  --molecule h2o \
  --heatmap-dir "$H2O_HEATMAP_DIR" \
  --summary-csv "$H2O_SUMMARY_CSV"

echo "=== [2/2] N2 6-31G DMRG heatmaps (parallel over geometries) ==="
python -u scripts/h2o/generate_basis_variance_heatmaps.py \
  "${COMMON_ARGS[@]}" \
  --molecule n2 \
  --heatmap-dir "$N2_HEATMAP_DIR" \
  --summary-csv "$N2_SUMMARY_CSV"

echo "=== summaries ==="
column -t -s, "$H2O_SUMMARY_CSV" || cat "$H2O_SUMMARY_CSV"
echo "---"
column -t -s, "$N2_SUMMARY_CSV" || cat "$N2_SUMMARY_CSV"

echo "=== done ==="
date
