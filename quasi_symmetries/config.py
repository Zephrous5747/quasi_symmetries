"""Central configuration: repo paths and optimizer constants."""

from __future__ import annotations

import os
from pathlib import Path

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

# Sparse energy-sector diagnostics (N2-scale).
STATES_PER_SECTOR = 20
if "SPARSE_ENERGY_WORKERS" in os.environ:
    SPARSE_ENERGY_MAX_WORKERS = max(1, int(os.environ["SPARSE_ENERGY_WORKERS"]))
else:
    SPARSE_ENERGY_MAX_WORKERS = max(1, min(8, os.cpu_count() or 1))

# Backward-compatible alias used by cache module
DEFAULT_CACHE_DIR = str(CACHE_DIR)
