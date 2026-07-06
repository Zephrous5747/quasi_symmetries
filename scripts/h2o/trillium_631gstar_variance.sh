#!/bin/bash
#SBATCH --job-name=h2o_631gs_var
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=out/h2o_631gs_var_%j.out
#SBATCH --error=out/h2o_631gs_var_%j.err

set -euo pipefail

# --- paths (edit if repo lives elsewhere on the cluster) ---
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# =============================================================================
# H2O 6-31G* (C2v symmetry-adapted MOs): generate HDF5 caches + variance heatmaps.
# Outputs are isolated from STO-3G (hamiltonian_cache/*.h5 without _631gs suffix).
# =============================================================================
export BASIS="${BASIS:-6-31g*}"
export CACHE_631GS_DIR="${CACHE_631GS_DIR:-$REPO/hamiltonian_cache/631gs}"
export HEATMAP_631GS_DIR="${HEATMAP_631GS_DIR:-$REPO/images/orbital_heatmaps/h2o/631gs/canonical/exact}"
export SUMMARY_631GS_CSV="${SUMMARY_631GS_CSV:-$REPO/tables/h2o/631gs/exact_canonical_variance_summary.csv}"
# H2O 6-31G* fixed-N FCI is sparse and memory-heavy; use 1 worker unless you have headroom.
export GEOM_WORKERS="${GEOM_WORKERS:-1}"

mkdir -p "$CACHE_631GS_DIR" "$HEATMAP_631GS_DIR" "$(dirname "$SUMMARY_631GS_CSV")"

# =============================================================================
# Python environment (Alliance / Trillium)
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
import sys

required = (
    "numpy", "scipy", "matplotlib", "h5py", "tqdm",
    "openfermion", "openfermionpyscf", "pyscf",
)
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing Python packages (install in venv before submitting): "
        + ", ".join(missing)
    )
print("[setup] ok:", ", ".join(required))
PY

# --- thread control: one BLAS thread per worker process ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "REPO=$REPO"
echo "VENV=$VENV"
echo "BASIS=$BASIS"
echo "CACHE_631GS_DIR=$CACHE_631GS_DIR"
echo "HEATMAP_631GS_DIR=$HEATMAP_631GS_DIR"
echo "SUMMARY_631GS_CSV=$SUMMARY_631GS_CSV"
echo "geom_workers=$GEOM_WORKERS"
date

# --- PySCF sizing estimate (fail early if basis / symmetry setup is wrong) ---
python -u - <<PY
import math
import os
import sys

from quasi_symmetries.hamiltonian.geometry import get_geometry_and_description
from quasi_symmetries.optimization import fixed_n_subspace_dim
from quasi_symmetries.symmetry.labels import molecule_point_group

basis = os.environ["BASIS"]
x = 0.958
geometry, _ = get_geometry_and_description("h2o", x, hoh_angle_deg=104.5)
point_group = molecule_point_group("h2o")

from pyscf import gto, scf

mol = gto.M(
    atom=[(atom, coords) for atom, coords in geometry],
    basis=basis,
    symmetry=point_group,
    charge=0,
    spin=0,
)
mol.build()
mf = scf.RHF(mol).run(verbose=0)
n_spatial = mf.mo_coeff.shape[1]
n_electrons = mol.nelectron
dim_sub = fixed_n_subspace_dim(n_spatial, n_electrons)
log10_dim = math.log10(dim_sub) if dim_sub > 0 else 0.0

print("=== H2O 6-31G* preflight (equilibrium geometry) ===")
print(f"basis:              {basis}")
print(f"point_group:        {point_group}")
print(f"n_spatial:          {n_spatial}")
print(f"n_electrons:        {n_electrons}")
print(f"dim_sub:            {dim_sub:,}  (~10^{log10_dim:.1f})")
print(f"E_HF:               {mf.e_tot:.8f}")

if n_spatial < 13:
    print(
        f"ERROR: n_spatial={n_spatial} looks like STO-3G (7), not 6-31G* (~18).",
        file=sys.stderr,
    )
    sys.exit(1)
print("[preflight] ok")
PY

# --- drop mislabeled STO-3G caches that may have been copied into 631gs/ ---
python -u - <<PY
import json
import os
from pathlib import Path

import h5py

cache_dir = Path(os.environ["CACHE_631GS_DIR"])
expected_basis = os.environ["BASIS"].strip().lower()
removed = []
for path in sorted(cache_dir.glob("*.h5")):
    with h5py.File(path, "r") as handle:
        meta = json.loads(handle.attrs["meta_json"])
        meta_basis = str(meta.get("basis", "")).strip().lower()
        n_spatial = int(handle.attrs["n_spatial"])
    if meta_basis != expected_basis or n_spatial < 13:
        print(f"[purge] removing stale cache {path.name} (basis={meta_basis!r}, n_spatial={n_spatial})")
        path.unlink()
        removed.append(path.name)
if removed:
    print(f"[purge] removed {len(removed)} stale file(s); will regenerate with --overwrite")
else:
    print("[purge] no stale STO-3G caches in 631gs/")
PY

# --- generate symmetry-adapted Hamiltonian + FCI wavefunction, then heatmaps ---
python -u scripts/h2o/generate_basis_variance_heatmaps.py \
  --molecule h2o \
  --basis "6-31g*" \
  --wavefunction exact \
  --cache-dir "$CACHE_631GS_DIR" \
  --heatmap-dir "$HEATMAP_631GS_DIR" \
  --summary-csv "$SUMMARY_631GS_CSV" \
  --max-workers "$GEOM_WORKERS" \
  --overwrite

# --- post-run sanity check on summary CSV ---
python -u - <<PY
import csv
import os
import sys
from pathlib import Path

summary = Path(os.environ["SUMMARY_631GS_CSV"])
expected_basis = os.environ["BASIS"].strip().lower()
rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
if not rows:
    raise SystemExit(f"No rows in {summary}")

bad = []
for row in rows:
    n_spatial = int(row["n_spatial"])
    basis = str(row["Basis"]).strip().lower()
    if basis != expected_basis or n_spatial < 13:
        bad.append((row["Geometry_Param"], basis, n_spatial))

if bad:
    print("ERROR: summary contains non-6-31G* rows:", bad, file=sys.stderr)
    sys.exit(1)

n_spatial_vals = sorted({int(r["n_spatial"]) for r in rows})
print(f"[verify] {len(rows)} geometries, n_spatial={n_spatial_vals}, summary={summary}")
print(f"[verify] heatmaps in {os.environ['HEATMAP_631GS_DIR']}")
PY

echo "=== done ==="
date
