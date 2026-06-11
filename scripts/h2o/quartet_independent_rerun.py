from quasi_symmetries.config import CACHE_DIR, IMAGES_DIR, OPT_RESULTS_DIR, TABLES_DIR
"""Rerun H2O quartet baselines with independent edge sets and warm-started rotations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from quasi_symmetries.hamiltonian.geometry import default_grid_for_molecule
from quasi_symmetries.workflows import quartet as quartet_optimization_workflow as workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="h2o")
    parser.add_argument(
        "--warm-start-csv",
        type=Path,
        default=TABLES_DIR / 'h2o_quartet_baseline_summary.csv"),
        help="Previous quartet summary CSV with Thetas_JSON warm starts.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=TABLES_DIR / 'h2o_quartet_baseline_summary.csv"),
    )
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--parallel-geometries",
        action="store_true",
        default=True,
        help="Run multiple geometries concurrently (default: on).",
    )
    args = parser.parse_args()

    molecule = args.molecule.lower()
    grid = [float(x) for x in default_grid_for_molecule(molecule)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def run_geometry(x: float) -> list[dict]:
        geom_kwargs = {"hoh_angle_deg": 104.5} if molecule == "h2o" else {}
        return workflow.evaluate_single_geometry(
            molecule,
            x,
            n_restarts=args.n_restarts,
            post_diagnostics="none",
            parallel_baselines=True,
            parallel_restarts=False,
            max_workers=args.max_workers,
            quartet_warm_start_csv=str(args.warm_start_csv),
            verbose=True,
            **geom_kwargs,
        )

    rows: list[dict] = []
    if args.parallel_geometries and len(grid) > 1:
        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(grid))) as executor:
            futures = {executor.submit(run_geometry, x): x for x in grid}
            for future in as_completed(futures):
                rows.extend(future.result())
    else:
        for x in grid:
            rows.extend(run_geometry(x))

    rows.sort(key=lambda row: (float(row["Geometry_Param"]), row["Baseline"]))
    import csv

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=workflow.quartet_summary_csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
