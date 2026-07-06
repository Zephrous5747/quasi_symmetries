"""Quick benchmark for energy-sector diagnostics (N2 sparse / H2O dense mixed-pool)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from quasi_symmetries.config import CACHE_DIR, diagnostics_dir, heatmap_optimization_dir, table_path
from quasi_symmetries.diagnostics.mixed_pool import (
    mixed_pool_energy_indicators,
    mixed_pool_sectors,
)
from quasi_symmetries.hamiltonian.cache import list_cached_hamiltonians, load_reference_state
from quasi_symmetries.diagnostics.n2_action import (
    OrbitalRotationAction,
    RotatedHamiltonian,
    _parse_edges,
    _quartet_sectors,
)
from quasi_symmetries.optimization import (
    SparseSubspaceHamiltonian,
    build_U_from_thetas,
    closed_shell_hf_bitstring,
    coupled_energy_lazy,
    coupled_energy_test,
    decoupled_energy_lazy,
    decoupled_energy_test,
    diagonalize_sector_blocks,
    energy_sector_diagnostics_sparse,
    pair_list_for_n,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import H2O_MIXED_POOL


def _geometry_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")


def _load_h2o_row(x: float | None) -> dict[str, str]:
    csv_path = table_path("h2o", "mixed_pool_energy_diagnostics.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")
    if x is None:
        return rows[0]
    for row in rows:
        if abs(float(row["Geometry_Param"]) - x) < 1e-9:
            return row
    raise ValueError(f"geometry x={x} not found in {csv_path}")


def benchmark_n2(*, x: float | None = None, profile: bool = False) -> None:
    with table_path("n2", "quartet_variance_summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    if x is None:
        x = float(row["Geometry_Param"])
    print(f"[n2] geometry x={x}", flush=True)

    start = time.perf_counter()
    ref = load_reference_state("n2", x, cache_dir=str(CACHE_DIR))
    print(f"[n2] load_ref {time.perf_counter() - start:.1f}s dim={ref['dim_sub']}", flush=True)

    edges = _parse_edges(row["Edges"])
    sectors = _quartet_sectors(ref["basis_bitstrings"], edges, ref["n_spatial"])
    print(f"[n2] sectors={len(sectors)}", flush=True)

    h_identity = SparseSubspaceHamiltonian(ref["h_sub"])
    start = time.perf_counter()
    identity = energy_sector_diagnostics_sparse(
        h_identity,
        sectors,
        ref["energy_fci"],
        tol=1e-3,
        profile=profile,
    )
    elapsed = time.perf_counter() - start
    print(f"[n2] identity_total {elapsed:.1f}s {identity}", flush=True)
    if profile and "_profile" in identity:
        print(f"[n2] identity_profile {identity['_profile']}", flush=True)

    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    u_spatial = build_U_from_thetas(ref["n_spatial"], thetas, pair_list_for_n(ref["n_spatial"]))
    action = OrbitalRotationAction(u_spatial, ref["basis_bitstrings"], ref["n_spatial"])
    h_rot = RotatedHamiltonian(ref["h_sub"], action)

    start = time.perf_counter()
    e_dec, _, _ = decoupled_energy_lazy(h_rot, sectors)
    print(f"[n2] optimized_edec {time.perf_counter() - start:.1f}s {e_dec}", flush=True)

    start = time.perf_counter()
    e_coupled, k_coupled, converged, _ = coupled_energy_lazy(
        h_rot,
        sectors,
        E_exact=ref["energy_fci"],
        tol=1e-3,
    )
    print(
        f"[n2] optimized_coupled {time.perf_counter() - start:.1f}s "
        f"E={e_coupled} K={k_coupled} converged={converged}",
        flush=True,
    )


def benchmark_h2o(*, x: float | None = None) -> None:
    row = _load_h2o_row(x)
    x = float(row["Geometry_Param"])
    baseline_seconds = float(row["Energy_Diagnostics_Seconds"])
    baseline_k_id = int(row["Kcoupled_Identity"])
    baseline_k_opt = int(row["Kcoupled_Optimized"])
    print(f"[h2o] geometry x={x}", flush=True)
    print(
        f"[h2o] csv baseline Energy_Diagnostics_Seconds={baseline_seconds:.1f}s "
        f"K_id={baseline_k_id} K_opt={baseline_k_opt}",
        flush=True,
    )

    start = time.perf_counter()
    ref = load_reference_state(
        "h2o",
        x,
        cache_dir=str(CACHE_DIR),
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        hoh_angle_deg=104.5,
    )
    print(
        f"[h2o] load_ref {time.perf_counter() - start:.1f}s "
        f"dim={ref['dim_sub']} use_dense={ref.get('use_dense')}",
        flush=True,
    )

    npz_path = heatmap_optimization_dir("h2o") / f"h2o_{_geometry_tag(x)}_mixed_pool_data.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing mixed-pool cache: {npz_path}")
    data = np.load(npz_path)
    key = "u_optimized" if "u_optimized" in data.files else "u_mixed"
    u_optimized = np.asarray(data[key], dtype=np.complex128)

    pool = H2O_MIXED_POOL
    start = time.perf_counter()
    results = mixed_pool_energy_indicators(ref, pool, u_optimized)
    total_elapsed = time.perf_counter() - start
    print(f"[h2o] mixed_pool_energy_indicators {total_elapsed:.1f}s", flush=True)
    print(
        f"[h2o] K_id={results['Kcoupled_Identity']} K_opt={results['Kcoupled_Optimized']} "
        f"Ecoupled_id={results['Ecoupled_Identity']:.8f} "
        f"Ecoupled_opt={results['Ecoupled_Optimized']:.8f}",
        flush=True,
    )
    speedup = baseline_seconds / total_elapsed if total_elapsed > 0 else float("inf")
    print(
        f"[h2o] vs csv baseline: {total_elapsed:.1f}s now / {baseline_seconds:.1f}s before "
        f"({speedup:.2f}x)",
        flush=True,
    )

    if not ref.get("use_dense", False):
        print("[h2o] dense diagnostics skipped in cache; coupled-only timing omitted", flush=True)
        return

    h_dense = ref["h_sub"].toarray().astype(np.complex128)
    h_dense = 0.5 * (h_dense + h_dense.conj().T)
    sectors = mixed_pool_sectors(ref["basis_bitstrings"], pool, ref["n_spatial"])
    sector_data = diagonalize_sector_blocks(h_dense, sectors)

    start = time.perf_counter()
    e_dec, _, _ = decoupled_energy_test(h_dense, sectors)
    print(f"[h2o] identity_edec {time.perf_counter() - start:.3f}s {e_dec}", flush=True)

    start = time.perf_counter()
    e_coupled, k_coupled, converged, _ = coupled_energy_test(
        h_dense,
        sector_data,
        E_exact=ref["energy_fci"],
        tol=1e-3,
    )
    print(
        f"[h2o] identity_coupled {time.perf_counter() - start:.3f}s "
        f"E={e_coupled} K={k_coupled} converged={converged}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        choices=("n2", "h2o", "both"),
        default="both",
        help="Which benchmark to run (default: both).",
    )
    parser.add_argument(
        "--geometry",
        type=float,
        default=None,
        help="Geometry parameter x (default: first row in summary CSV).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable sparse energy profiling (N2 only).",
    )
    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="List hamiltonian_cache entries and exit.",
    )
    args = parser.parse_args()

    if args.list_cache:
        cached = list_cached_hamiltonians(CACHE_DIR)
        print(f"hamiltonian_cache ({CACHE_DIR}): {len(cached)} files")
        for name in cached:
            print(f"  {name}")
        n2 = [name for name in cached if name.startswith("n2_")]
        h2o = [name for name in cached if name.startswith("h2o_")]
        print(f"  n2: {len(n2)}  h2o: {len(h2o)}")
        return

    if args.molecule in ("n2", "both"):
        benchmark_n2(x=args.geometry, profile=args.profile)
    if args.molecule in ("h2o", "both"):
        benchmark_h2o(x=args.geometry)


if __name__ == "__main__":
    main()
