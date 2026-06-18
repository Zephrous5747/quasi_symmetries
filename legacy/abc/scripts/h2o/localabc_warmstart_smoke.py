"""Debug smoke: H2O quartet optimization from local-ABC angles only."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from quasi_symmetries.config import LEGACY_ABC_OPT_RESULTS_DIR, LEGACY_ABC_TABLES_DIR
from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.optimization import (
    build_U_from_thetas,
    closed_shell_hf_bitstring,
    pair_list_for_n,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import (
    balanced_tree_plus_edges,
    matching_edges,
    quartet_cost_for_u,
    ring_edges,
    run_fixed_topology_baseline,
    run_matching_greedy_baseline,
)


def _load_ref(molecule: str, x: float, cache_dir: str) -> dict:
    return load_reference_state(
        molecule,
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


def main() -> None:
    source = LEGACY_ABC_TABLES_DIR / "h2o_quasi_symmetry_local_abc.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        local_rows = list(csv.DictReader(handle))
    row = next(item for item in local_rows if abs(float(item["Geometry_Param"]) - 0.958) < 1e-12)
    thetas = np.asarray(json.loads(row["Thetas"]), dtype=float)

    ref = _load_ref("h2o", 0.958, "hamiltonian_cache")
    n_spatial = ref["n_spatial"]
    v_sub = ref["v_sub"]
    basis_bitstrings = ref["basis_bitstrings"]
    pairs = pair_list_for_n(n_spatial)
    u_initial = build_U_from_thetas(n_spatial, thetas, pairs)

    jobs = [
        (
            "greedy",
            matching_edges(n_spatial),
            lambda: run_matching_greedy_baseline(
                v_sub,
                basis_bitstrings,
                n_spatial,
                final_reoptimize=True,
                n_restarts=1,
                include_zero_start=True,
                parallel_restarts=False,
                maxfev=500,
                initial_thetas=thetas,
            ),
        ),
        (
            "ring",
            ring_edges(n_spatial),
            lambda: run_fixed_topology_baseline(
                v_sub,
                basis_bitstrings,
                n_spatial,
                "ring",
                n_restarts=1,
                include_zero_start=True,
                parallel_restarts=False,
                maxfev=500,
                initial_thetas=thetas,
            ),
        ),
        (
            "balanced_tree",
            balanced_tree_plus_edges(n_spatial),
            lambda: run_fixed_topology_baseline(
                v_sub,
                basis_bitstrings,
                n_spatial,
                "balanced_tree",
                n_restarts=1,
                include_zero_start=True,
                parallel_restarts=False,
                maxfev=500,
                initial_thetas=thetas,
            ),
        ),
    ]

    out_rows = []
    for baseline, initial_edges, run in jobs:
        initial_cost = quartet_cost_for_u(v_sub, basis_bitstrings, u_initial, n_spatial, initial_edges)
        start = time.perf_counter()
        result = run()
        elapsed = time.perf_counter() - start
        final = result["final"]
        opt = final["res"]
        out = {
            "Baseline": baseline,
            "Initial_LocalABC_Cost": initial_cost,
            "Optimized_Cost": float(final["cost"]),
            "Elapsed_Seconds": elapsed,
            "Optimizer_Nfev": int(getattr(opt, "nfev", -1)),
            "Optimizer_Nit": int(getattr(opt, "nit", -1)),
            "Optimizer_Success": bool(getattr(opt, "success", False)),
            "Optimizer_Message": str(getattr(opt, "message", "")),
        }
        out_rows.append(out)
        print("[h2o-localabc-smoke] " + json.dumps(out), flush=True)

    out_path = LEGACY_ABC_OPT_RESULTS_DIR / "h2o_quartet_localabc_warmstart_smoke.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[h2o-localabc-smoke] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
