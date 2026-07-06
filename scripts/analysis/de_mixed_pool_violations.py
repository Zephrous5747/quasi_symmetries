from quasi_symmetries.config import CACHE_DIR, heatmap_optimization_dir, table_path
"""Differential-evolution mixed-pool rerun for remaining violation geometries."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path


import numpy as np
from scipy.optimize import differential_evolution

from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.optimization import (
    build_U_from_thetas,
    closed_shell_hf_bitstring,
    pair_list_for_n,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import (
    H2O_MIXED_POOL,
    mixed_pool_cost_for_u,
    mixed_pool_cost_from_thetas,
    mixed_pool_stats_for_u,
)
from scripts.h2o.mixed_pool_rerun_violations import (
    MIXED_POOL_CSV_FIELDS,
    _edge_json,
    _json_list,
    _merge_rows,
    _pool_labels,
    _save_npz,
)

GEOMETRIES = (0.958, 1.1293333333333333, 1.3006666666666666)


def de_optimize_geometry(x: float, *, cache_dir: str = "hamiltonian_cache") -> dict:
    ref = load_reference_state(
        "h2o",
        x,
        cache_dir=cache_dir,
        load_hamiltonian=False,
        load_full_hamiltonian=False,
        compute_rdms=False,
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        hoh_angle_deg=104.5,
    )
    pairs = pair_list_for_n(ref["n_spatial"])
    pool = H2O_MIXED_POOL

    def objective(thetas: np.ndarray) -> float:
        return mixed_pool_cost_from_thetas(
            thetas, pool, ref["v_sub"], ref["basis_bitstrings"], ref["n_spatial"], pairs
        )

    summary_csv = table_path("h2o", "mixed_pool_summary.csv")
    mix_row = next(
        row
        for row in csv.DictReader(summary_csv.open(newline="", encoding="utf-8"))
        if abs(float(row["Geometry_Param"]) - x) < 1e-9
    )
    x0 = np.asarray(json.loads(mix_row["Thetas_JSON"]), dtype=float)
    start = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=[(-3.2, 3.2)] * len(pairs),
        x0=x0,
        maxiter=80,
        popsize=10,
        seed=1,
        polish=True,
        tol=1e-9,
    )
    elapsed = time.perf_counter() - start
    print(
        f"[de] x={x:.6g}: {float(mix_row['V_Optimized']):.6g} -> {result.fun:.6g} "
        f"({elapsed:.1f}s, success={result.success})",
        flush=True,
    )

    u_spatial = build_U_from_thetas(ref["n_spatial"], result.x, pairs)
    single_stats, quartet_stats = mixed_pool_stats_for_u(
        ref["v_sub"], ref["basis_bitstrings"], u_spatial, ref["n_spatial"], pool
    )
    singles_label, quartets_label = _pool_labels(pool)
    u_identity = np.eye(ref["n_spatial"], dtype=np.complex128)
    return {
        "Workflow": "mixed_pool",
        "Molecule": "h2o",
        "Geometry_Param": x,
        "E_HF": ref["energy_hf"],
        "E_FCI": ref["energy_fci"],
        "E_CISD": ref["energy_cisd"],
        "n_spatial": ref["n_spatial"],
        "Pool_Singles": singles_label,
        "Pool_Quartets": quartets_label,
        "V_Identity": mixed_pool_cost_for_u(
            ref["v_sub"], ref["basis_bitstrings"], u_identity, ref["n_spatial"], pool
        ),
        "V_Optimized": float(result.fun),
        "Single_Expectations": _json_list(stat.expectation for _, stat in single_stats),
        "Single_Variances": _json_list(stat.variance for _, stat in single_stats),
        "Quartet_Expectations": _json_list(stat.expectation for _, stat in quartet_stats),
        "Quartet_Variances": _json_list(stat.variance for _, stat in quartet_stats),
        "Thetas_JSON": _json_list(result.x),
        "Rotation_Pairs_JSON": _edge_json(pairs),
        "Optimizer_Success": bool(result.success),
        "Optimizer_Status": getattr(result, "status", ""),
        "Optimizer_Message": str(getattr(result, "message", "")),
        "Optimizer_Nit": getattr(result, "nit", ""),
        "Optimizer_Nfev": getattr(result, "nfev", ""),
        "N_Restarts": 1,
        "Elapsed_Seconds": elapsed,
        "_u_spatial": u_spatial,
        "_pool": pool,
        "_ref": ref,
    }


def main() -> None:
    rows = [de_optimize_geometry(x) for x in GEOMETRIES]
    summary_csv = table_path("h2o", "mixed_pool_summary.csv")
    _merge_rows(summary_csv, rows, MIXED_POOL_CSV_FIELDS)
    npz_dir = heatmap_optimization_dir("h2o")
    for row in rows:
        _save_npz(row, npz_dir)
    print(f"[ok] updated {len(rows)} geometries in {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
