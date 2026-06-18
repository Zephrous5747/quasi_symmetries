"""Benchmark fast parity-seniority energy diagnostics on representative N2 geometries."""

from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from quasi_symmetries.config import TABLES_DIR
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


def main() -> None:
    input_csv = TABLES_DIR / "n2_parity_seniority_summary.csv"
    output_csv = TABLES_DIR / "n2_parity_seniority_action_diagnostics.csv"
    rows = _load_rows(input_csv)
    geom_labels = ", ".join(f"{float(row['Geometry_Param']):.1f}" for row in rows)
    print(
        f"[benchmark] running {len(rows)} geometries in parallel ({geom_labels} A)",
        flush=True,
    )

    total_start = time.perf_counter()
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=len(rows)) as executor:
        futures = {executor.submit(_worker, row): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="N2 geometries"):
            row = futures[future]
            x = float(row["Geometry_Param"])
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"geometry x={x} failed") from exc

    total_elapsed = time.perf_counter() - total_start
    per_geom = total_elapsed / len(rows)
    print(
        f"[benchmark] total wall={total_elapsed:.1f}s "
        f"mean={per_geom:.1f}s/geometry "
        f"projected_full_run={per_geom * len(rows):.1f}s",
        flush=True,
    )

    results.sort(key=lambda row: float(row["Geometry_Param"]))  # type: ignore[arg-type]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[ok] wrote {len(results)} rows to {output_csv}", flush=True)


if __name__ == "__main__":
    main()
