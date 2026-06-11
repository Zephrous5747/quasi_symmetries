"""Quick benchmark for sparse N2 energy-sector diagnostics."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from hamiltonian_cache import load_reference_state
from n2_action_diagnostics import (
    OrbitalRotationAction,
    RotatedHamiltonian,
    _parse_edges,
    _quartet_sectors,
)
from optimization_abc_utils import (
    SparseSubspaceHamiltonian,
    build_U_from_thetas,
    coupled_energy_lazy,
    decoupled_energy_lazy,
    energy_sector_diagnostics_sparse,
    pair_list_for_n,
)


def main() -> None:
    with Path("tables/n2_quartet_variance_summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    x = float(row["Geometry_Param"])
    print(f"geometry x={x}", flush=True)

    start = time.perf_counter()
    ref = load_reference_state("n2", x, cache_dir="hamiltonian_cache")
    print(f"load_ref {time.perf_counter() - start:.1f}s", flush=True)

    edges = _parse_edges(row["Edges"])
    sectors = _quartet_sectors(ref["basis_bitstrings"], edges, ref["n_spatial"])
    print(f"sectors={len(sectors)}", flush=True)

    h_identity = SparseSubspaceHamiltonian(ref["h_sub"])
    start = time.perf_counter()
    identity = energy_sector_diagnostics_sparse(
        h_identity,
        sectors,
        ref["energy_fci"],
        tol=1e-3,
    )
    print(f"identity_total {time.perf_counter() - start:.1f}s {identity}", flush=True)

    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    u_spatial = build_U_from_thetas(ref["n_spatial"], thetas, pair_list_for_n(ref["n_spatial"]))
    action = OrbitalRotationAction(u_spatial, ref["basis_bitstrings"], ref["n_spatial"])
    h_rot = RotatedHamiltonian(ref["h_sub"], action)

    start = time.perf_counter()
    e_dec, _, _ = decoupled_energy_lazy(h_rot, sectors)
    print(f"optimized_edec {time.perf_counter() - start:.1f}s {e_dec}", flush=True)

    start = time.perf_counter()
    e_coupled, k_coupled, converged, _ = coupled_energy_lazy(
        h_rot,
        sectors,
        E_exact=ref["energy_fci"],
        tol=1e-3,
    )
    print(
        f"optimized_coupled {time.perf_counter() - start:.1f}s "
        f"E={e_coupled} K={k_coupled} converged={converged}",
        flush=True,
    )


if __name__ == "__main__":
    main()
