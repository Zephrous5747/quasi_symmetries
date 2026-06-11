"""Compute dense energy indicators for H2O mixed-pool optimizations (parallel)."""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hamiltonian_cache import load_reference_state
from hamiltonian_geometry import default_grid_for_molecule
from mixed_pool_diagnostics import mixed_pool_energy_indicators
from optimization_abc_utils import (
    closed_shell_hf_bitstring,
    popcount,
    solve_cisd_state,
)
from quartet_optimization_utils import H2O_MIXED_POOL, MixedOperatorPool

ENERGY_FIELDS = [
    "Sum_CommSq_Identity",
    "Sum_CommSq_Optimized",
    "Sum_Sexp_Identity",
    "Sum_Sexp_Optimized",
    "Coarse_Entropy_Identity",
    "Coarse_Entropy_Optimized",
    "Fine_Entropy_Identity",
    "Fine_Entropy_Optimized",
    "Edec_Identity",
    "Edec_Optimized",
    "Ecoupled_Identity",
    "Ecoupled_Optimized",
    "Kcoupled_Identity",
    "Kcoupled_Optimized",
    "EBO_Identity",
    "EBO_Optimized",
    "NumSectors_Identity",
    "NumSectors_Optimized",
    "DenseDiagnosticsSkipped",
    "Operator_Count",
    "Energy_Diagnostics_Seconds",
]


def _geometry_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")


def run_geometry(
    x: float,
    *,
    pool: MixedOperatorPool,
    cache_dir: str,
    npz_dir: Path,
    summary_row: dict[str, str] | None,
) -> dict:
    start = time.perf_counter()
    ref = load_reference_state(
        "h2o",
        x,
        cache_dir=cache_dir,
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        hoh_angle_deg=104.5,
    )
    npz_path = npz_dir / f"h2o_{_geometry_tag(x)}_mixed_pool_data.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing mixed-pool cache: {npz_path}")
    data = np.load(npz_path)
    u_optimized = np.asarray(data["u_mixed"], dtype=np.complex128)

    energy = mixed_pool_energy_indicators(ref, pool, u_optimized)
    elapsed = time.perf_counter() - start

    row = dict(summary_row) if summary_row is not None else {
        "Workflow": "mixed_pool",
        "Molecule": "h2o",
        "Geometry_Param": x,
        "Pool_Singles": " ".join(str(i) for i in pool.singles),
        "Pool_Quartets": " ".join(f"{p}-{q}" for p, q in pool.quartets),
    }
    row.update(energy)
    row["Energy_Diagnostics_Seconds"] = elapsed
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="hamiltonian_cache")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("tables/h2o_mixed_pool_summary.csv"),
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path("images/orbital_heatmaps/h2o"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("tables/h2o_mixed_pool_energy_diagnostics.csv"),
    )
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()

    pool = H2O_MIXED_POOL
    grid = [float(x) for x in default_grid_for_molecule("h2o")]

    summary_by_x: dict[float, dict[str, str]] = {}
    if args.summary_csv.is_file():
        with args.summary_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                summary_by_x[float(row["Geometry_Param"])] = row

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(grid))) as executor:
        futures = {
            executor.submit(
                run_geometry,
                x,
                pool=pool,
                cache_dir=args.cache_dir,
                npz_dir=args.npz_dir,
                summary_row=summary_by_x.get(x),
            ): x
            for x in grid
        }
        for future in as_completed(futures):
            x = futures[future]
            row = future.result()
            print(
                f"[ok] h2o x={x:.6g}: "
                f"Edec {row['Edec_Identity']:.8f} -> {row['Edec_Optimized']:.8f}, "
                f"Ecoupled {row['Ecoupled_Identity']:.8f} -> {row['Ecoupled_Optimized']:.8f} "
                f"({row['Energy_Diagnostics_Seconds']:.1f}s)",
                flush=True,
            )
            rows.append(row)

    rows.sort(key=lambda row: float(row["Geometry_Param"]))
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for field in ENERGY_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
