"""Rerun H2O mixed pool (s0,s1,s2 + s36,s45) and produce operator diagnostics plot."""

from __future__ import annotations

from quasi_symmetries.config import CACHE_DIR, diagnostics_dir, table_path

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.hamiltonian.geometry import default_grid_for_molecule
from quasi_symmetries.diagnostics.mixed_pool import mixed_pool_energy_indicators
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

MIXED_POOL_CSV_FIELDS = [
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


def _pool_labels(pool: MixedOperatorPool) -> tuple[str, str]:
    singles = " ".join(str(i) for i in pool.singles)
    quartets = " ".join(f"{p}-{q}" for p, q in pool.quartets)
    return singles, quartets


def _json_list(values) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def _edge_json(edges) -> str:
    return json.dumps([[int(p), int(q)] for p, q in edges], separators=(",", ":"))


def run_geometry(
    x: float,
    *,
    pool: MixedOperatorPool,
    cache_dir: str,
    n_restarts: int,
    maxfev: int | None,
) -> dict:
    start = time.perf_counter()
    ref = load_reference_state(
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
    n_spatial = ref["n_spatial"]
    v_sub = ref["v_sub"]
    basis = ref["basis_bitstrings"]
    u_identity = np.eye(n_spatial, dtype=np.complex128)
    v_identity = mixed_pool_cost_for_u(v_sub, basis, u_identity, n_spatial, pool)

    best = optimize_mixed_operator_pool(
        v_sub,
        basis,
        n_spatial,
        pool,
        n_restarts=n_restarts,
        include_zero_start=True,
        parallel=False,
        maxfev=maxfev,
    )
    res = best["res"]
    singles_label, quartets_label = _pool_labels(pool)
    energy = mixed_pool_energy_indicators(ref, pool, best["u_spatial"])
    elapsed = time.perf_counter() - start

    row = {
        "Workflow": "mixed_pool",
        "Molecule": "h2o",
        "Geometry_Param": x,
        "E_HF": ref["energy_hf"],
        "E_FCI": ref["energy_fci"],
        "E_CISD": ref["energy_cisd"],
        "n_spatial": n_spatial,
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
        **energy,
        "Energy_Diagnostics_Seconds": elapsed,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_summary.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_energy_diagnostics.csv"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=diagnostics_dir("h2o") / "mixed_pool_diagnostics.png",
    )
    parser.add_argument(
        "--seniority-csv",
        type=Path,
        default=table_path("h2o", "parity_seniority_diagnostics.csv"),
    )
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--optimizer-maxfev", type=int, default=500)
    args = parser.parse_args()

    pool = H2O_USER_MIXED_POOL
    grid = [float(x) for x in default_grid_for_molecule("h2o")]
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(grid))) as executor:
        futures = {
            executor.submit(
                run_geometry,
                x,
                pool=pool,
                cache_dir=args.cache_dir,
                n_restarts=args.n_restarts,
                maxfev=args.optimizer_maxfev,
            ): x
            for x in grid
        }
        for future in as_completed(futures):
            x = futures[future]
            row = future.result()
            print(
                f"[ok] h2o x={x:.6g}: V {row['V_Identity']:.6g} -> {row['V_Optimized']:.6g}, "
                f"Edec err {abs(float(row['Edec_Optimized']) - float(row['E_FCI'])):.2e}, "
                f"K={row['Kcoupled_Optimized']} ({row['Elapsed_Seconds']:.1f}s)",
                flush=True,
            )
            rows.append(row)

    rows.sort(key=lambda row: float(row["Geometry_Param"]))

    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MIXED_POOL_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in MIXED_POOL_CSV_FIELDS if key in row})
    print(f"[ok] wrote {len(rows)} rows to {args.summary_csv}", flush=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote diagnostics to {args.output_csv}", flush=True)

    from scripts.plot.h2o_operator_diagnostics import plot_h2o_operator_diagnostics

    plot_h2o_operator_diagnostics(
        seniority_csv=args.seniority_csv,
        mixed_pool_csv=args.output_csv,
        output_path=args.plot_output,
    )


if __name__ == "__main__":
    main()
