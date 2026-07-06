"""Benchmark fast parity-seniority energy diagnostics on representative N2 geometries."""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from quasi_symmetries.config import table_path
from quasi_symmetries.diagnostics.n2_action import _diagnose_one_geometry
from quasi_symmetries.hamiltonian.geometry import N2_REPRESENTATIVE_GRID


def _load_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grid = {float(x) for x in N2_REPRESENTATIVE_GRID}
    selected = [row for row in rows if float(row["Geometry_Param"]) in grid]
    if not selected:
        raise ValueError(
            f"No rows in {input_csv} match N2_REPRESENTATIVE_GRID {tuple(N2_REPRESENTATIVE_GRID)}."
        )
    return sorted(selected, key=lambda row: float(row["Geometry_Param"]))


def _worker(row: dict[str, str]) -> dict[str, object]:
    os.environ["SPARSE_ENERGY_PROFILE"] = "1"
    x = float(row["Geometry_Param"])
    start = time.perf_counter()
    result = _diagnose_one_geometry(row, profile=True)
    elapsed = time.perf_counter() - start
    print(
        f"[benchmark] x={x:.1f} Energy_Seconds={result.get('Energy_Seconds')} "
        f"K_opt={result.get('Kcoupled_Optimized')} wall={elapsed:.1f}s",
        flush=True,
    )
    return result


def _default_geom_workers() -> int:
    if "GEOM_WORKERS" in os.environ:
        return max(1, int(os.environ["GEOM_WORKERS"]))
    return 1


def _default_sparse_workers() -> int | None:
    if "SPARSE_ENERGY_WORKERS" in os.environ:
        return max(1, int(os.environ["SPARSE_ENERGY_WORKERS"]))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return None


def _run_sequential(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in tqdm(rows, desc="N2 geometries"):
        results.append(_worker(row))
    return results


def _run_parallel(rows: list[dict[str, str]], geom_workers: int) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=geom_workers) as executor:
        futures = {executor.submit(_worker, row): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="N2 geometries"):
            row = futures[future]
            x = float(row["Geometry_Param"])
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"geometry x={x} failed") from exc
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=table_path("n2", "parity_seniority_summary.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=table_path("n2", "parity_seniority_action_diagnostics.csv"),
    )
    parser.add_argument(
        "--geom-workers",
        type=int,
        default=_default_geom_workers(),
        help="Geometries in parallel (default 1 on cluster; set GEOM_WORKERS env).",
    )
    parser.add_argument(
        "--sparse-workers",
        type=int,
        default=_default_sparse_workers(),
        help="ProcessPool workers per geometry (default SLURM_CPUS_PER_TASK or SPARSE_ENERGY_WORKERS).",
    )
    args = parser.parse_args()

    if args.sparse_workers is not None:
        os.environ["SPARSE_ENERGY_WORKERS"] = str(args.sparse_workers)

    rows = _load_rows(args.input_csv)
    geom_workers = min(max(1, args.geom_workers), len(rows))
    sparse_workers = os.environ.get("SPARSE_ENERGY_WORKERS", "auto")
    geom_labels = ", ".join(f"{float(row['Geometry_Param']):.1f}" for row in rows)
    print(
        f"[benchmark] {len(rows)} geometries ({geom_labels} A) "
        f"geom_workers={geom_workers} sparse_workers={sparse_workers}",
        flush=True,
    )

    total_start = time.perf_counter()
    if geom_workers == 1:
        results = _run_sequential(rows)
    else:
        results = _run_parallel(rows, geom_workers)

    total_elapsed = time.perf_counter() - total_start
    per_geom = total_elapsed / len(rows)
    print(
        f"[benchmark] total wall={total_elapsed:.1f}s "
        f"mean={per_geom:.1f}s/geometry",
        flush=True,
    )

    results.sort(key=lambda row: float(row["Geometry_Param"]))  # type: ignore[arg-type]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[ok] wrote {len(results)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
