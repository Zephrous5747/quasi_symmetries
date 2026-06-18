# Archived ABC workflows

Fixed, shared, and local ABC quasi-symmetry scans live here, outside the main quartet / parity-seniority / mixed-pool workflow.

## Layout

- `quasi_symmetries_abc/` — library (`workflows/abc.py`, `optimization/variance.py`, `optimization/local_abc.py`)
- `scripts/` — CLI runners and plots
- `tables/` — published ABC CSV summaries
- `opt_results/` — detailed ABC outputs and rotation NPZ artifacts
- `tests/` — ABC-specific unit tests

## Usage

From the repo root, include both the repo and this folder on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = ".;legacy/abc"
python legacy/abc/scripts/h2o/seniority_fixed_abc.py
python -m unittest discover -s legacy/abc/tests -v
```

```bash
export PYTHONPATH=.:legacy/abc
python legacy/abc/scripts/h2o/seniority_fixed_abc.py
python -m unittest discover -s legacy/abc/tests -v
```
