"""Run quartet summary scans for the default h4_linear and LiH grids."""

from __future__ import annotations

from pathlib import Path

from hamiltonian_geometry import default_grid_for_molecule
import quartet_optimization_workflow as quartet_workflow


def run_one(molecule: str, opt_results_dir: Path) -> None:
    output_csv = opt_results_dir / f"{molecule}_quartet_baseline_summary.csv"
    print(f"[runner] Starting {molecule}; output={output_csv}", flush=True)
    quartet_workflow.main(
        molecule=molecule,
        grid=default_grid_for_molecule(molecule),
        csv_filename=str(output_csv),
        verbose=True,
    )
    print(f"[runner] Finished {molecule}; output={output_csv}", flush=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    opt_results_dir = repo_root / "opt_results"
    opt_results_dir.mkdir(parents=True, exist_ok=True)

    for molecule in ("h4_linear", "lih"):
        run_one(molecule, opt_results_dir)


if __name__ == "__main__":
    main()
