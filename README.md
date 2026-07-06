# Quasi-Symmetries

Computational research code for studying quasi-symmetries via parity-parent Hamiltonians on small molecules (LiH, H₂O, H₄, N₂) in STO-3G.

## Layout

```
quasi_symmetries/     Library: theory, hamiltonians, optimization, workflows
scripts/              CLI entry points (runners, plots, analysis)
legacy/abc/           Archived fixed/shared/local ABC workflows and results
tests/                unittest suite
hamiltonian_cache/    Precomputed HDF5 Hamiltonians
tables/               CSV summaries: tables/{molecule}/{basis}/
opt_results/          Scratch runs (archived under opt_results/scratch/)
images/               Plots and NPZ caches:
                      - orbital_heatmaps/{molecule}/{basis}/optimization/
                      - orbital_heatmaps/{molecule}/{basis}/canonical/{exact|dmrg}/
                      - diagnostics/{molecule}/{basis}/
                      - scans/{molecule}/{basis}/
```

Path helpers live in `quasi_symmetries.config` (`table_path`, `heatmap_optimization_dir`, etc.).

## Setup

```bash
pip install -r requirements.txt
```

Optional (Hamiltonian generation): `pip install pyscf openfermionpyscf`

## Running

From the repo root, set `PYTHONPATH` to the repo root:

```powershell
# PowerShell
$env:PYTHONPATH = "."
python scripts/run_tests.py -v
python scripts/generate_hamiltonians.py --molecule lih
python scripts/run_quartets.py
python scripts/h2o/mixed_pool_scan.py
```

```bash
# Bash
export PYTHONPATH=.
python scripts/run_tests.py -v
```

## Workflows

- **Quartet baselines** (`quasi_symmetries.workflows.quartet`, `scripts/run_quartets.py`)
- **Parity seniority** (`scripts/h2o/continue_h2o_operator_optimization.py`, `scripts/n2/parity_optimization_heatmaps.py`)
- **Mixed pools** (H₂O/N₂): `scripts/h2o/mixed_pool_scan.py`, `scripts/n2/parity_optimization_heatmaps.py`

Archived **fixed/shared/local ABC** workflows live under [`legacy/abc/`](legacy/abc/README.md).
