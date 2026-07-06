"""Run quartet summary scans for the default h4_linear and LiH grids."""

from __future__ import annotations

from pathlib import Path

from quasi_symmetries.config import TABLES_DIR, table_path
from quasi_symmetries.hamiltonian.geometry import default_grid_for_molecule
from quasi_symmetries.workflows import quartet as quartet_optimization_workflow as quartet_workflow


def run_one(molecule: str, tables_dir: Path) -> None:
    output_csv = table_path(molecule, "quartet_baseline_summary.csv")
    print(f"[runner] Starting {molecule}; output={output_csv}", flush=True)
    quartet_workflow.main(
        molecule=molecule,
        grid=default_grid_for_molecule(molecule),
        csv_filename=str(output_csv),
        verbose=True,
    )
    print(f"[runner] Finished {molecule}; output={output_csv}", flush=True)


def main() -> None:
    tables_dir = TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    for molecule in ("h4_linear", "lih"):
        run_one(molecule, tables_dir)


if __name__ == "__main__":
    main()
