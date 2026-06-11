# Quasi-Symmetries

Computational research code for studying quasi-symmetries via parity-parent Hamiltonians on small molecules (LiH, H₂O, H₄, N₂) in STO-3G.

## Layout

```
quasi_symmetries/     Library: theory, hamiltonians, optimization, workflows
scripts/              CLI entry points (runners, plots, analysis)
tests/                unittest suite
hamiltonian_cache/    Precomputed HDF5 Hamiltonians
tables/               CSV scan summaries
opt_results/          Detailed optimization outputs
images/               Generated plots
```

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
python scripts/h2o/mixed_pool_scan.py
```

```bash
# Bash
export PYTHONPATH=.
python scripts/run_tests.py -v
```

## Workflows

- **ABC optimization** (`quasi_symmetries.workflows.abc`): `fixed_abc`, `shared_abc`, `local_abc`
- **Quartet baselines** (`quasi_symmetries.workflows.quartet`)
- **Mixed pools** (H₂O): `scripts/h2o/mixed_pool_scan.py`
