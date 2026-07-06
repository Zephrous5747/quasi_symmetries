#!/bin/bash
#SBATCH --job-name=h2o_dmrg_heat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=out/h2o_dmrg_heat_%j.out
#SBATCH --error=out/h2o_dmrg_heat_%j.err

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# H2O 6-31G DMRG canonical heatmap smoke run.
export BASIS="${BASIS:-6-31g}"
export WAVEFUNCTION="${WAVEFUNCTION:-dmrg}"
export OH_BOND="${OH_BOND:-0.958}"
export DMRG_BOND_DIMENSION="${DMRG_BOND_DIMENSION:-512}"
export DMRG_MAX_SWEEPS="${DMRG_MAX_SWEEPS:-20}"
export DMRG_THREADS="${DMRG_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export GEOM_WORKERS="${GEOM_WORKERS:-1}"
export DMRG_CACHE_ROOT="${DMRG_CACHE_ROOT:-$REPO/dmrg_cache}"
export EXACT_CACHE_DIR="${EXACT_CACHE_DIR:-$REPO/hamiltonian_cache/631g}"
export HEATMAP_DIR="${HEATMAP_DIR:-$REPO/images/orbital_heatmaps/h2o/631g/canonical/dmrg}"
export SUMMARY_CSV="${SUMMARY_CSV:-$REPO/tables/h2o/631g/dmrg_canonical_variance_summary.csv}"

mkdir -p out "$DMRG_CACHE_ROOT" "$EXACT_CACHE_DIR" "$HEATMAP_DIR" "$(dirname "$SUMMARY_CSV")"

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack/2024a

VENV="${VENV:-${SCRATCH:-$HOME}/venvs/quasi_symmetries}"
if [[ ! -d "$VENV" ]]; then
  echo "[setup] missing venv: $VENV" >&2
  echo "[setup] create it outside the job and install h5py tqdm openfermion openfermionpyscf pyscf pyblock2/block2" >&2
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
echo "VENV=$VENV"
echo "BASIS=$BASIS"
echo "WAVEFUNCTION=$WAVEFUNCTION"
echo "OH_BOND=$OH_BOND"
echo "DMRG_BOND_DIMENSION=$DMRG_BOND_DIMENSION"
echo "DMRG_MAX_SWEEPS=$DMRG_MAX_SWEEPS"
echo "DMRG_THREADS=$DMRG_THREADS"
echo "DMRG_CACHE_ROOT=$DMRG_CACHE_ROOT"
echo "HEATMAP_DIR=$HEATMAP_DIR"
echo "SUMMARY_CSV=$SUMMARY_CSV"
date

python -u - <<'PY'
import math
import os
from pyscf import gto, scf

from quasi_symmetries.hamiltonian.geometry import get_geometry_and_description
from quasi_symmetries.optimization import fixed_n_subspace_dim
from quasi_symmetries.symmetry.labels import extract_labels_from_pyscf, molecule_point_group

basis = os.environ["BASIS"]
x = float(os.environ["OH_BOND"])
geometry, _ = get_geometry_and_description("h2o", x, hoh_angle_deg=104.5)
point_group = molecule_point_group("h2o")
mol = gto.M(atom=[(atom, coords) for atom, coords in geometry], basis=basis, symmetry=point_group, charge=0, spin=0)
mol.build()
mf = scf.RHF(mol).run(verbose=0)
labels = extract_labels_from_pyscf(geometry, "h2o", basis=basis, charge=0, spin=0)
n_spatial = mf.mo_coeff.shape[1]
dim_sub = fixed_n_subspace_dim(n_spatial, mol.nelectron)
print("=== H2O DMRG preflight ===")
print(f"basis:       {basis}")
print(f"point_group: {point_group}")
print(f"n_spatial:   {n_spatial}")
print(f"n_electrons: {mol.nelectron}")
print(f"dim_sub:     {dim_sub:,} (~10^{math.log10(dim_sub):.1f})")
print(f"E_HF:        {mf.e_tot:.10f}")
print("irreps:      " + " ".join(labels.irrep_labels))
if basis.strip().lower() == "6-31g" and n_spatial != 13:
    raise SystemExit(f"Expected H2O 6-31G to have 13 spatial orbitals, got {n_spatial}")
print("[preflight] ok")
PY

python -u scripts/h2o/generate_basis_variance_heatmaps.py \
  --basis "$BASIS" \
  --wavefunction "$WAVEFUNCTION" \
  --x "$OH_BOND" \
  --cache-dir "$EXACT_CACHE_DIR" \
  --dmrg-cache-root "$DMRG_CACHE_ROOT" \
  --heatmap-dir "$HEATMAP_DIR" \
  --summary-csv "$SUMMARY_CSV" \
  --dmrg-bond-dimension "$DMRG_BOND_DIMENSION" \
  --dmrg-max-sweeps "$DMRG_MAX_SWEEPS" \
  --dmrg-threads "$DMRG_THREADS" \
  --max-workers "$GEOM_WORKERS" \
  --overwrite

echo "=== done ==="
date
