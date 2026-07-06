"""Rerun H2O optimizations at geometries where parity seniority beats mixed variance."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from quasi_symmetries.config import heatmap_optimization_dir, table_path
from quasi_symmetries.diagnostics.mixed_pool import mixed_pool_energy_indicators
from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.optimization import (
    closed_shell_hf_bitstring,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import (
    H2O_MIXED_POOL,
    MixedOperatorPool,
    mixed_pool_cost_for_u,
    optimize_mixed_operator_pool,
)
from scripts.plot.orbital_heatmaps import parity_variance_matrix

VIOLATION_GEOMETRIES = (
    0.958,
    1.1293333333333333,
    1.3006666666666666,
    2.5,
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


def _geometry_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")


def _json_list(values) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def _edge_json(edges) -> str:
    return json.dumps([[int(p), int(q)] for p, q in edges], separators=(",", ":"))


def _pool_labels(pool: MixedOperatorPool) -> tuple[str, str]:
    singles = " ".join(str(i) for i in pool.singles)
    quartets = " ".join(f"{p}-{q}" for p, q in pool.quartets)
    return singles, quartets


def _load_thetas_by_geometry(csv_path: Path) -> dict[float, np.ndarray]:
    if not csv_path.is_file():
        return {}
    by_x: dict[float, np.ndarray] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x = float(row["Geometry_Param"])
            if "Thetas_JSON" in row:
                thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
            else:
                thetas = np.asarray(json.loads(row["Thetas"]), dtype=float)
            by_x[x] = thetas
    return by_x


def _match_geometry(x: float, table: dict[float, np.ndarray], tol: float = 1e-9) -> np.ndarray | None:
    for key, thetas in table.items():
        if abs(key - x) <= tol:
            return thetas
    return None


def optimize_geometry_aggressive(
    x: float,
    *,
    pool: MixedOperatorPool,
    cache_dir: str,
    n_restarts: int,
    maxfev: int,
    warm_starts: list[np.ndarray | None],
) -> dict:
    start = time.perf_counter()
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
    n_spatial = ref["n_spatial"]
    v_sub = ref["v_sub"]
    basis = ref["basis_bitstrings"]
    u_identity = np.eye(n_spatial, dtype=np.complex128)
    v_identity = mixed_pool_cost_for_u(v_sub, basis, u_identity, n_spatial, pool)

    best: dict | None = None
    total_restarts = 0
    for warm_idx, warm in enumerate(warm_starts):
        trial = optimize_mixed_operator_pool(
            v_sub,
            basis,
            n_spatial,
            pool,
            n_restarts=n_restarts,
            random_seed=17 + warm_idx,
            initial_thetas=warm,
            include_zero_start=warm is None,
            parallel=False,
            maxfev=maxfev,
            maxiter=2000,
        )
        total_restarts += int(trial["n_restarts"])
        if best is None or trial["cost"] < best["cost"]:
            best = trial

    assert best is not None
    res = best["res"]
    singles_label, quartets_label = _pool_labels(pool)
    return {
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
        "N_Restarts": total_restarts,
        "Elapsed_Seconds": time.perf_counter() - start,
        "_u_spatial": best["u_spatial"],
        "_pool": pool,
        "_ref": ref,
    }


def _merge_rows(csv_path: Path, new_rows: list[dict], fieldnames: list[str]) -> None:
    by_x = {float(row["Geometry_Param"]): row for row in new_rows}
    merged: list[dict] = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                x = float(row["Geometry_Param"])
                merged.append(by_x.pop(x, row))
    for x in sorted(by_x):
        merged.append({key: by_x[x].get(key, "") for key in fieldnames})
    merged.sort(key=lambda row: float(row["Geometry_Param"]))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _save_npz(row: dict, out_dir: Path) -> None:
    x = float(row["Geometry_Param"])
    tag = _geometry_tag(x)
    ref = row["_ref"]
    pool = row["_pool"]
    n_spatial = ref["n_spatial"]
    u_canonical = np.eye(n_spatial, dtype=np.complex128)
    u_mixed = row["_u_spatial"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"h2o_{tag}_mixed_pool_data.npz",
        geometry_param=x,
        pool_singles=np.asarray(pool.singles, dtype=int),
        pool_quartets=np.asarray(pool.quartets, dtype=int),
        u_canonical=u_canonical,
        u_mixed=u_mixed,
        variance_canonical=parity_variance_matrix(
            ref["v_sub"], ref["basis_bitstrings"], u_canonical, n_spatial
        ),
        variance_mixed=parity_variance_matrix(
            ref["v_sub"], ref["basis_bitstrings"], u_mixed, n_spatial
        ),
    )


PARITY_SENIORITY_FIELDS = [
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


def _parity_summary_row(full_row: dict[str, Any]) -> dict[str, str]:
    return {key: full_row.get(key, "") for key in PARITY_SENIORITY_FIELDS}


def rerun_parity_seniority_geometry(
    x: float,
    *,
    cache_dir: str,
    warm_thetas: list[np.ndarray],
    n_restarts: int,
    maxfev: int,
) -> dict[str, Any]:
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
    ref["geometry_param"] = x
    n_spatial = ref["n_spatial"]
    pool = MixedOperatorPool(singles=tuple(range(n_spatial)), quartets=())
    v_sub = ref["v_sub"]
    basis = ref["basis_bitstrings"]
    u_identity = np.eye(n_spatial, dtype=np.complex128)
    v_identity = mixed_pool_cost_for_u(v_sub, basis, u_identity, n_spatial, pool)

    best: dict | None = None
    total_restarts = 0
    warm_starts: list[np.ndarray | None] = list(warm_thetas) + [None]
    for warm_idx, warm in enumerate(warm_starts):
        trial = optimize_mixed_operator_pool(
            v_sub,
            basis,
            n_spatial,
            pool,
            n_restarts=n_restarts,
            random_seed=31 + warm_idx,
            initial_thetas=warm,
            include_zero_start=warm is None,
            parallel=False,
            maxfev=maxfev,
            maxiter=2000,
        )
        total_restarts += int(trial["n_restarts"])
        if best is None or trial["cost"] < best["cost"]:
            best = trial

    assert best is not None
    res = best["res"]
    singles_label, quartets_label = _pool_labels(pool)
    row = {
        "Workflow": "parity_seniority",
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
        "N_Restarts": total_restarts,
        "Elapsed_Seconds": time.perf_counter() - start,
    }
    row.update(mixed_pool_energy_indicators(ref, pool, best["u_spatial"]))
    row["Energy_Diagnostics_Seconds"] = row["Elapsed_Seconds"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="hamiltonian_cache")
    parser.add_argument(
        "--mixed-summary-csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_summary.csv"),
    )
    parser.add_argument(
        "--parity-seniority-csv",
        type=Path,
        default=table_path("h2o", "parity_seniority_diagnostics.csv"),
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=heatmap_optimization_dir("h2o"),
    )
    parser.add_argument("--n-restarts", type=int, default=20)
    parser.add_argument("--optimizer-maxfev", type=int, default=3000)
    parser.add_argument("--skip-parity-seniority", action="store_true")
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument(
        "--geometries",
        type=float,
        nargs="*",
        default=None,
        help="Geometry list (default: violation set).",
    )
    args = parser.parse_args()
    geometries = tuple(args.geometries) if args.geometries else VIOLATION_GEOMETRIES

    pool = H2O_MIXED_POOL
    parity_thetas = _load_thetas_by_geometry(args.parity_seniority_csv)
    mixed_thetas = _load_thetas_by_geometry(args.mixed_summary_csv)

    if not args.skip_mixed:
        new_rows: list[dict] = []
        for x in geometries:
            warm_starts: list[np.ndarray | None] = [
                _match_geometry(x, mixed_thetas),
                _match_geometry(x, parity_thetas),
                None,
            ]
            row = optimize_geometry_aggressive(
                x,
                pool=pool,
                cache_dir=args.cache_dir,
                n_restarts=args.n_restarts,
                maxfev=args.optimizer_maxfev,
                warm_starts=warm_starts,
            )
            print(
                f"[mixed] h2o x={x:.6g}: V {row['V_Identity']:.6g} -> {row['V_Optimized']:.6g} "
                f"(success={row['Optimizer_Success']}, nfev={row['Optimizer_Nfev']}, "
                f"{row['Elapsed_Seconds']:.1f}s)",
                flush=True,
            )
            _save_npz(row, args.npz_dir)
            new_rows.append(row)

        _merge_rows(args.mixed_summary_csv, new_rows, MIXED_POOL_CSV_FIELDS)
        print(f"[ok] merged {len(new_rows)} mixed rows into {args.mixed_summary_csv}", flush=True)

    if not args.skip_parity_seniority:
        parity_rows: list[dict] = []
        for x in geometries:
            warm = [
                theta
                for theta in (
                    _match_geometry(x, parity_thetas),
                    _match_geometry(x, mixed_thetas),
                )
                if theta is not None
            ]
            row = rerun_parity_seniority_geometry(
                x,
                cache_dir=args.cache_dir,
                warm_thetas=warm,
                n_restarts=args.n_restarts,
                maxfev=args.optimizer_maxfev,
            )
            print(
                f"[parity] h2o x={x:.6g}: V {row['V_Identity']:.6g} -> {row['V_Optimized']:.6g} "
                f"(success={row['Optimizer_Success']}, nfev={row['Optimizer_Nfev']})",
                flush=True,
            )
            parity_rows.append(_parity_summary_row(row))

        _merge_rows(args.parity_seniority_csv, parity_rows, PARITY_SENIORITY_FIELDS)
        print(
            f"[ok] merged {len(parity_rows)} parity-seniority rows into {args.parity_seniority_csv}",
            flush=True,
        )


if __name__ == "__main__":
    main()
