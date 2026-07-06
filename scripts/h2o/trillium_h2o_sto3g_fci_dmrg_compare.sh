#!/bin/bash
#SBATCH --job-name=h2o_sto3g_cmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=out/h2o_sto3g_cmp_%j.out
#SBATCH --error=out/h2o_sto3g_cmp_%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# --- STO-3G H2O: FCI vs DMRG canonical parity heatmap comparison ---
export BASIS="${BASIS:-sto-3g}"
export GEOM_WORKERS="${GEOM_WORKERS:-1}"

# FCI uses hamiltonian_cache/ (no basis subdir for STO-3G)
export EXACT_CACHE_DIR="${EXACT_CACHE_DIR:-$REPO/hamiltonian_cache}"
export DMRG_CACHE_ROOT="${DMRG_CACHE_ROOT:-$REPO/dmrg_cache}"

export FCI_HEATMAP_DIR="${FCI_HEATMAP_DIR:-$REPO/images/orbital_heatmaps/h2o/sto3g/canonical/exact}"
export DMRG_HEATMAP_DIR="${DMRG_HEATMAP_DIR:-$REPO/images/orbital_heatmaps/h2o/sto3g/canonical/dmrg}"
export FCI_SUMMARY_CSV="${FCI_SUMMARY_CSV:-$REPO/tables/h2o/sto3g/exact_canonical_variance_summary.csv}"
export DMRG_SUMMARY_CSV="${DMRG_SUMMARY_CSV:-$REPO/tables/h2o/sto3g/dmrg_canonical_variance_summary.csv}"

# STO-3G is small; 256 x 30 is ample for exact agreement with FCI
export DMRG_BOND_DIMENSION="${DMRG_BOND_DIMENSION:-256}"
export DMRG_MAX_SWEEPS="${DMRG_MAX_SWEEPS:-30}"
export DMRG_THREADS="${DMRG_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

mkdir -p out "$EXACT_CACHE_DIR" "$DMRG_CACHE_ROOT" \
  "$FCI_HEATMAP_DIR" "$DMRG_HEATMAP_DIR" \
  "$(dirname "$FCI_SUMMARY_CSV")" "$(dirname "$DMRG_SUMMARY_CSV")"

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack/2024a

VENV="${VENV:-${SCRATCH:-$HOME}/venvs/quasi_symmetries}"
if [[ ! -d "$VENV" ]]; then
  echo "[setup] missing venv: $VENV" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - <<'PY'
import importlib
required = (
    "numpy", "scipy", "matplotlib", "h5py", "tqdm",
    "openfermion", "openfermionpyscf", "pyscf", "pyblock2.driver.core",
)
missing = [name for name in required if not importlib.util.find_spec(name)]
if missing:
    raise SystemExit("Missing: " + ", ".join(missing))
print("[setup] ok")
PY

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "BASIS=$BASIS"
echo "DMRG_BOND_DIMENSION=$DMRG_BOND_DIMENSION  DMRG_MAX_SWEEPS=$DMRG_MAX_SWEEPS"
date

# Preflight: STO-3G H2O should have 7 spatial orbitals at equilibrium geometry
python -u - <<'PY'
from pyscf import gto, scf
from quasi_symmetries.hamiltonian.geometry import get_geometry_and_description
from quasi_symmetries.symmetry.labels import molecule_point_group

basis = "sto-3g"
x = 0.958
geometry, _ = get_geometry_and_description("h2o", x, hoh_angle_deg=104.5)
mol = gto.M(
    atom=[(atom, coords) for atom, coords in geometry],
    basis=basis,
    symmetry=molecule_point_group("h2o"),
    charge=0,
    spin=0,
)
mol.build()
mf = scf.RHF(mol).run(verbose=0)
n_spatial = mf.mo_coeff.shape[1]
print(f"[preflight] basis={basis} n_spatial={n_spatial} n_electrons={mol.nelectron} E_HF={mf.e_tot:.10f}")
if basis.strip().lower() in {"sto-3g", "sto3g"} and n_spatial != 7:
    raise SystemExit(f"Expected STO-3G H2O to have 7 spatial orbitals, got {n_spatial}")
PY

COMMON_ARGS=(
  --basis "$BASIS"
  --cache-dir "$EXACT_CACHE_DIR"
  --dmrg-cache-root "$DMRG_CACHE_ROOT"
  --dmrg-bond-dimension "$DMRG_BOND_DIMENSION"
  --dmrg-max-sweeps "$DMRG_MAX_SWEEPS"
  --dmrg-threads "$DMRG_THREADS"
  --max-workers "$GEOM_WORKERS"
  --overwrite
)

echo "=== [1/2] FCI exact heatmap ==="
python -u scripts/h2o/generate_basis_variance_heatmaps.py \
  "${COMMON_ARGS[@]}" \
  --wavefunction exact \
  --heatmap-dir "$FCI_HEATMAP_DIR" \
  --summary-csv "$FCI_SUMMARY_CSV"

echo "=== [2/2] DMRG heatmap + FCI validation ==="
python -u scripts/h2o/generate_basis_variance_heatmaps.py \
  "${COMMON_ARGS[@]}" \
  --wavefunction dmrg \
  --heatmap-dir "$DMRG_HEATMAP_DIR" \
  --summary-csv "$DMRG_SUMMARY_CSV" \
  --validate-ci-vector

echo "=== summaries ==="
column -t -s, "$FCI_SUMMARY_CSV" || cat "$FCI_SUMMARY_CSV"
echo "---"
column -t -s, "$DMRG_SUMMARY_CSV" || cat "$DMRG_SUMMARY_CSV"

echo "=== full-grid NPZ comparison ==="
python -u - <<'PY'
import csv
import numpy as np
from pathlib import Path

import os

fci_dir = Path(os.environ["FCI_HEATMAP_DIR"])
dmrg_dir = Path(os.environ["DMRG_HEATMAP_DIR"])
rows = []

for fci_npz in sorted(fci_dir.glob("h2o_*_canonical_variance_data.npz")):
    dmrg_npz = dmrg_dir / fci_npz.name
    if not dmrg_npz.is_file():
        print(f"[missing] {dmrg_npz}")
        continue
    fci = np.load(fci_npz)
    dmrg = np.load(dmrg_npz)
    vf = np.asarray(fci["variance_canonical"], dtype=float)
    vd = np.asarray(dmrg["variance_canonical"], dtype=float)
    diff = vd - vf
    mask = np.isfinite(vf) & np.isfinite(vd)
    x = float(fci["geometry_param"])
    rows.append(
        {
            "x": x,
            "delta_e": abs(float(dmrg["reference_energy"]) - float(fci["reference_energy"])),
            "max_abs_diff": float(np.max(np.abs(diff[mask]))),
            "rms_diff": float(np.sqrt(np.mean(diff[mask] ** 2))),
            "sum_diag_fci": float(np.nansum(np.diag(vf))),
            "sum_diag_dmrg": float(np.nansum(np.diag(vd))),
        }
    )

rows.sort(key=lambda row: row["x"])
for row in rows:
    print(
        f"x={row['x']:.6g} |E_DMRG-E_FCI|={row['delta_e']:.3e} "
        f"max|dV|={row['max_abs_diff']:.3e} rms={row['rms_diff']:.3e} "
        f"sumdiag(FCI)={row['sum_diag_fci']:.6g} sumdiag(DMRG)={row['sum_diag_dmrg']:.6g}"
    )
PY

echo "=== done ==="
date