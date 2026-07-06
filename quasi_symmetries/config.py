"""Central configuration: repo paths and optimizer constants."""

from __future__ import annotations

import os
from pathlib import Path

from quasi_symmetries.hamiltonian.geometry import basis_cache_slug

REPO_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = REPO_ROOT / "hamiltonian_cache"
TABLES_DIR = REPO_ROOT / "tables"
OPT_RESULTS_DIR = REPO_ROOT / "opt_results"
IMAGES_DIR = REPO_ROOT / "images"

LEGACY_ABC_ROOT = REPO_ROOT / "legacy" / "abc"
LEGACY_ABC_TABLES_DIR = LEGACY_ABC_ROOT / "tables"
LEGACY_ABC_OPT_RESULTS_DIR = LEGACY_ABC_ROOT / "opt_results"

# Quantum chemistry defaults
LIH_BOND_ANGSTROM = 1.60
CHARGE = 0
MULTIPLICITY = 1
BASIS = "sto-3g"

# Optimizer settings
OPT_METHOD = "Powell"
MAXITER = 200
N_RESTARTS = 5
ANGLE_INIT_SCALE = 0.2
RANDOM_SEED = 0
TOPK_ANGLES_TO_PRINT = 10
EVAL_STATE_SPECIFIC_COMMUTATIVITY = True
OP_COEF_TOL = 1e-12

# Fixed-N determinant counts above this use sparse / subspace-only post-processing.
DENSE_SUBSPACE_MAX = 8_000

# Coupled-energy greedy selection (perturbation pre-filter near-degeneracy).
COUPLED_ENERGY_DEGENERACY_FLOOR = 1e-8

# Sparse energy-sector diagnostics (N2-scale).
STATES_PER_SECTOR = 20
if "SPARSE_ENERGY_WORKERS" in os.environ:
    SPARSE_ENERGY_MAX_WORKERS = max(1, int(os.environ["SPARSE_ENERGY_WORKERS"]))
else:
    SPARSE_ENERGY_MAX_WORKERS = max(1, min(8, os.cpu_count() or 1))

# Backward-compatible alias used by cache module
DEFAULT_CACHE_DIR = str(CACHE_DIR)


def normalized_basis_slug(basis: str | None = None) -> str:
    """Short basis tag for artifact paths (``sto3g`` when basis is default STO-3G)."""
    slug = basis_cache_slug(basis if basis is not None else BASIS)
    return slug or "sto3g"


def molecule_basis_dir(root: Path, molecule: str, basis: str | None = None) -> Path:
    return root / molecule.lower() / normalized_basis_slug(basis)


def heatmap_optimization_dir(molecule: str, basis: str | None = None) -> Path:
    return molecule_basis_dir(IMAGES_DIR / "orbital_heatmaps", molecule, basis) / "optimization"


def heatmap_canonical_dir(molecule: str, basis: str, backend: str) -> Path:
    return (
        molecule_basis_dir(IMAGES_DIR / "orbital_heatmaps", molecule, basis)
        / "canonical"
        / backend.lower()
    )


def diagnostics_dir(molecule: str, basis: str | None = None) -> Path:
    return molecule_basis_dir(IMAGES_DIR / "diagnostics", molecule, basis)


def scans_dir(molecule: str, basis: str | None = None) -> Path:
    return molecule_basis_dir(IMAGES_DIR / "scans", molecule, basis)


def table_dir(molecule: str, basis: str | None = None) -> Path:
    return molecule_basis_dir(TABLES_DIR, molecule, basis)


def table_path(molecule: str, name: str, basis: str | None = None) -> Path:
    return table_dir(molecule, basis) / name
