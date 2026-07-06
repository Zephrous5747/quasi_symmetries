"""Continue H2O mixed-pool and parity-seniority optimization from saved thetas."""

from __future__ import annotations

from quasi_symmetries.config import CACHE_DIR, diagnostics_dir, table_path

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from quasi_symmetries.diagnostics.mixed_pool import mixed_pool_energy_indicators
from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.hamiltonian.geometry import default_grid_for_molecule
from quasi_symmetries.optimization import (
    closed_shell_hf_bitstring,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import (
    MixedOperatorPool,
    mixed_pool_cost_for_u,
    normalize_edge,
    optimize_mixed_operator_pool,
)

H2O_USER_MIXED_POOL = MixedOperatorPool(
    singles=(0, 1, 2),
    quartets=(normalize_edge((3, 6)), normalize_edge((4, 5))),
)

SUMMARY_FIELDS = [
    "Workflow",
    "Molecule",
    "Geometry_Param",
    "E_HF",
    "E_FCI",
    "E_CISD",
    "n_spatial",
    "Pool_Singles",
    "Pool_Quartets",
    "V_Identity",
    "V_Optimized",
    "Single_Expectations",
    "Single_Variances",
    "Quartet_Expectations",
    "Quartet_Variances",
    "Thetas_JSON",
    "Rotation_Pairs_JSON",
    "Optimizer_Success",
    "Optimizer_Status",
    "Optimizer_Message",
    "Optimizer_Nit",
    "Optimizer_Nfev",
    "N_Restarts",
    "Elapsed_Seconds",
]


def _json_list(values) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def _edge_json(edges) -> str:
    return json.dumps([[int(p), int(q)] for p, q in edges], separators=(",", ":"))


def _pool_labels(pool: MixedOperatorPool) -> tuple[str, str]:
    singles = " ".join(str(i) for i in pool.singles)
    quartets = " ".join(f"{p}-{q}" for p, q in pool.quartets)
    return singles, quartets


def _load_thetas_by_geometry(csv_path: Path) -> dict[float, np.ndarray]:
    by_x: dict[float, np.ndarray] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x = float(row["Geometry_Param"])
            thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
            by_x[x] = thetas
    return by_x


def _match_geometry(x: float, table: dict[float, np.ndarray], tol: float = 1e-9) -> np.ndarray | None:
    for key, thetas in table.items():
        if abs(key - x) <= tol:
            return thetas
    return None


def _load_reference(x: float, *, cache_dir: str) -> dict:
    return load_reference_state(
        "h2o",
        x,
        cache_dir=cache_dir,
        load_hamiltonian=True,
        load_full_hamiltonian=False,
        compute_rdms=False,
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        hoh_angle_deg=104.5,
    )


def _summary_row_from_best(
    *,
    workflow: str,
    ref: dict,
    pool: MixedOperatorPool,
    best: dict,
    v_identity: float,
    elapsed: float,
) -> dict:
    res = best["res"]
    singles_label, quartets_label = _pool_labels(pool)
    return {
        "Workflow": workflow,
        "Molecule": "h2o",
        "Geometry_Param": ref["geometry_param"],
        "E_HF": ref["energy_hf"],
        "E_FCI": ref["energy_fci"],
        "E_CISD": ref["energy_cisd"],
        "n_spatial": ref["n_spatial"],
        "Pool_Singles": singles_label,
        "Pool_Quartets": quartets_label,
        "V_Identity": v_identity,
        "V_Optimized": float(best["cost"]),
        "Single_Expectations": _json_list(stat.expectation for stat in best["single_stats"]),
        "Single_Variances": _json_list(stat.variance for stat in best["single_stats"]),
        "Quartet_Expectations": _json_list(stat.expectation for stat in best["quartet_stats"]),
        "Quartet_Variances": _json_list(stat.variance for stat in best["quartet_stats"]),
        "Thetas_JSON": _json_list(res.x),
        "Rotation_Pairs_JSON": _edge_json(best["pairs"]),
        "Optimizer_Success": bool(getattr(res, "success", False)),
        "Optimizer_Status": getattr(res, "status", ""),
        "Optimizer_Message": str(getattr(res, "message", "")),
        "Optimizer_Nit": getattr(res, "nit", ""),
        "Optimizer_Nfev": getattr(res, "nfev", ""),
        "N_Restarts": int(best["n_restarts"]),
        "Elapsed_Seconds": elapsed,
    }


def run_mixed_pool_continue(
    x: float,
    *,
    pool: MixedOperatorPool,
    cache_dir: str,
    warm_thetas: np.ndarray,
    maxfev: int,
) -> dict:
    start = time.perf_counter()
    ref = _load_reference(x, cache_dir=cache_dir)
    ref["geometry_param"] = x
    v_sub = ref["v_sub"]
    basis = ref["basis_bitstrings"]
    n_spatial = ref["n_spatial"]
    u_identity = np.eye(n_spatial, dtype=np.complex128)
    v_identity = mixed_pool_cost_for_u(v_sub, basis, u_identity, n_spatial, pool)

    best = optimize_mixed_operator_pool(
        v_sub,
        basis,
        n_spatial,
        pool,
        n_restarts=1,
        initial_thetas=warm_thetas,
        include_zero_start=True,
        parallel=False,
        maxfev=maxfev,
    )
    row = _summary_row_from_best(
        workflow="mixed_pool",
        ref=ref,
        pool=pool,
        best=best,
        v_identity=v_identity,
        elapsed=time.perf_counter() - start,
    )
    row.update(mixed_pool_energy_indicators(ref, pool, best["u_spatial"]))
    row["Energy_Diagnostics_Seconds"] = row["Elapsed_Seconds"]
    return row


def run_parity_seniority(
    x: float,
    *,
    cache_dir: str,
    warm_thetas: np.ndarray,
    maxfev: int,
) -> dict:
    start = time.perf_counter()
    ref = _load_reference(x, cache_dir=cache_dir)
    ref["geometry_param"] = x
    n_spatial = ref["n_spatial"]
    pool = MixedOperatorPool(singles=tuple(range(n_spatial)), quartets=())
    v_sub = ref["v_sub"]
    basis = ref["basis_bitstrings"]
    u_identity = np.eye(n_spatial, dtype=np.complex128)
    v_identity = mixed_pool_cost_for_u(v_sub, basis, u_identity, n_spatial, pool)

    best = optimize_mixed_operator_pool(
        v_sub,
        basis,
        n_spatial,
        pool,
        n_restarts=1,
        initial_thetas=warm_thetas,
        include_zero_start=True,
        parallel=False,
        maxfev=maxfev,
    )
    row = _summary_row_from_best(
        workflow="parity_seniority",
        ref=ref,
        pool=pool,
        best=best,
        v_identity=v_identity,
        elapsed=time.perf_counter() - start,
    )
    row.update(mixed_pool_energy_indicators(ref, pool, best["u_spatial"]))
    row["Energy_Diagnostics_Seconds"] = row["Elapsed_Seconds"]
    return row


def _write_csv(path: Path, rows: list[dict], summary_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = list(summary_fields)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument(
        "--mixed-warm-start-csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_energy_diagnostics.csv"),
    )
    parser.add_argument(
        "--seniority-warm-start-csv",
        type=Path,
        default=table_path("h2o", "parity_seniority_diagnostics.csv"),
    )
    parser.add_argument(
        "--mixed-summary-csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_summary.csv"),
    )
    parser.add_argument(
        "--mixed-diagnostics-csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_energy_diagnostics.csv"),
    )
    parser.add_argument(
        "--seniority-diagnostics-csv",
        type=Path,
        default=table_path("h2o", "parity_seniority_diagnostics.csv"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=diagnostics_dir("h2o") / "mixed_pool_diagnostics.png",
    )
    parser.add_argument("--optimizer-maxfev", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument("--skip-seniority", action="store_true")
    args = parser.parse_args()

    grid = [float(x) for x in default_grid_for_molecule("h2o")]
    mixed_warm_table = _load_thetas_by_geometry(args.mixed_warm_start_csv)
    seniority_warm_table = _load_thetas_by_geometry(args.seniority_warm_start_csv)
    pool = H2O_USER_MIXED_POOL

    mixed_rows: list[dict] = []
    if not args.skip_mixed:
        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(grid))) as executor:
            futures = {}
            for x in grid:
                warm = _match_geometry(x, mixed_warm_table)
                if warm is None:
                    raise KeyError(
                        f"No warm-start thetas for geometry x={x} in {args.mixed_warm_start_csv}"
                    )
                futures[
                    executor.submit(
                        run_mixed_pool_continue,
                        x,
                        pool=pool,
                        cache_dir=args.cache_dir,
                        warm_thetas=warm,
                        maxfev=args.optimizer_maxfev,
                    )
                ] = x
            for future in as_completed(futures):
                x = futures[future]
                row = future.result()
                print(
                    f"[mixed] h2o x={x:.6g}: V {row['V_Identity']:.6g} -> {row['V_Optimized']:.6g}, "
                    f"Edec err {abs(float(row['Edec_Optimized']) - float(row['E_FCI'])):.2e}, "
                    f"K={row['Kcoupled_Optimized']} ({row['Elapsed_Seconds']:.1f}s)",
                    flush=True,
                )
                mixed_rows.append(row)
        mixed_rows.sort(key=lambda row: float(row["Geometry_Param"]))
        _write_csv(args.mixed_summary_csv, mixed_rows, SUMMARY_FIELDS)
        _write_csv(args.mixed_diagnostics_csv, mixed_rows, SUMMARY_FIELDS)
        print(f"[ok] wrote mixed pool results to {args.mixed_diagnostics_csv}", flush=True)

    seniority_rows: list[dict] = []
    if not args.skip_seniority:
        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(grid))) as executor:
            futures = {}
            for x in grid:
                warm = _match_geometry(x, seniority_warm_table)
                if warm is None:
                    raise KeyError(
                        f"No warm-start thetas for geometry x={x} in {args.seniority_warm_start_csv}"
                    )
                futures[
                    executor.submit(
                        run_parity_seniority,
                        x,
                        cache_dir=args.cache_dir,
                        warm_thetas=warm,
                        maxfev=args.optimizer_maxfev,
                    )
                ] = x
            for future in as_completed(futures):
                x = futures[future]
                row = future.result()
                print(
                    f"[parity] h2o x={x:.6g}: V {row['V_Identity']:.6g} -> {row['V_Optimized']:.6g}, "
                    f"Edec err {abs(float(row['Edec_Optimized']) - float(row['E_FCI'])):.2e}, "
                    f"K={row['Kcoupled_Optimized']} ({row['Elapsed_Seconds']:.1f}s)",
                    flush=True,
                )
                seniority_rows.append(row)
        seniority_rows.sort(key=lambda row: float(row["Geometry_Param"]))
        _write_csv(args.seniority_diagnostics_csv, seniority_rows, SUMMARY_FIELDS)
        print(f"[ok] wrote parity seniority results to {args.seniority_diagnostics_csv}", flush=True)

    from scripts.plot.h2o_operator_diagnostics import plot_h2o_operator_diagnostics

    mixed_csv = args.mixed_diagnostics_csv if mixed_rows else args.mixed_diagnostics_csv
    seniority_csv = args.seniority_diagnostics_csv
    if not seniority_rows and not seniority_csv.is_file():
        raise FileNotFoundError(
            f"Parity seniority diagnostics not found: {seniority_csv}. "
            "Run with --skip-seniority or provide an existing CSV."
        )

    plot_h2o_operator_diagnostics(
        seniority_csv=seniority_csv,
        mixed_pool_csv=mixed_csv,
        output_path=args.plot_output,
    )


if __name__ == "__main__":
    main()
