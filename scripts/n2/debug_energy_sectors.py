"""Profile sparse energy-sector diagnostics for one N2 geometry."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from quasi_symmetries.config import CACHE_DIR, TABLES_DIR
from quasi_symmetries.diagnostics.mixed_pool import mixed_pool_diagonals
from quasi_symmetries.diagnostics.n2_action import (
    OrbitalRotationAction,
    RotatedHamiltonian,
    _exact_symmetry_allowed_indices,
    _mixed_pool_sectors,
    _parse_pairs,
    _parse_pool,
)
from quasi_symmetries.diagnostics.sparse_energy import (
    SparseSubspaceHamiltonian,
    energy_sector_diagnostics_sparse,
)
from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.optimization import build_U_from_thetas
from quasi_symmetries.optimization.quartet import MixedOperatorPool


def _load_row(csv_path: Path, x: float) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["Geometry_Param"]) - x) <= 1e-9:
                return row
    raise KeyError(f"No row for x={x} in {csv_path}")


def _profile_energy(
    label: str,
    h_op,
    sectors: dict,
    energy_fci: float,
    *,
    tol: float,
    max_workers: int,
) -> dict:
    print(f"\n=== {label} ===", flush=True)
    print(f"  sectors={len(sectors)}  dim={h_op.shape[0]}", flush=True)
    max_dim = max(len(idxs) for idxs in sectors.values())
    print(f"  max_sector_dim={max_dim}", flush=True)
    start = time.perf_counter()
    result = energy_sector_diagnostics_sparse(
        h_op,
        sectors,
        energy_fci,
        tol=tol,
        max_workers=max_workers,
        profile=True,
    )
    wall = time.perf_counter() - start
    profile = result.pop("_profile", {})
    print(f"  Edec={result['Edec']:.8f}  K={result['Kcoupled']}  converged={result['Coupled_Converged']}", flush=True)
    print(f"  |Edec-FCI|={abs(result['Edec'] - energy_fci):.6e}", flush=True)
    for key, value in profile.items():
        print(f"  {key}={value:.3f}s", flush=True)
    print(f"  wall_seconds={wall:.3f}", flush=True)
    return {**result, **profile, "wall_seconds": wall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=2.2)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--workflow",
        choices=("parity_seniority", "mixed_pool"),
        default="parity_seniority",
    )
    parser.add_argument("--skip-optimized", action="store_true")
    args = parser.parse_args()

    if args.workflow == "parity_seniority":
        csv_path = TABLES_DIR / "n2_parity_seniority_summary.csv"
        row = _load_row(csv_path, args.x)
        ref = load_reference_state("n2", args.x, cache_dir=str(CACHE_DIR), compute_rdms=False)
        pool = MixedOperatorPool(singles=tuple(range(ref["n_spatial"])), quartets=())
        pairs = _parse_pairs(row["Rotation_Pairs_JSON"])
        thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    else:
        csv_path = TABLES_DIR / "n2_mixed_pool_summary.csv"
        row = _load_row(csv_path, args.x)
        ref = load_reference_state("n2", args.x, cache_dir=str(CACHE_DIR), compute_rdms=False)
        pool = _parse_pool(row, ref["n_spatial"])
        pairs = _parse_pairs(row["Rotation_Pairs_JSON"])
        thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)

    diagonals = mixed_pool_diagonals(ref["basis_bitstrings"], pool, ref["n_spatial"])
    allowed = _exact_symmetry_allowed_indices(ref)
    sectors = _mixed_pool_sectors(ref, pool, allowed_indices=allowed)
    energy_fci = float(ref["energy_fci"])

    print(
        f"workflow={args.workflow} x={args.x} operators={len(diagonals)} "
        f"E_FCI={energy_fci:.8f}",
        flush=True,
    )

    h_identity = SparseSubspaceHamiltonian(ref["h_sub"])
    _profile_energy(
        "identity",
        h_identity,
        sectors,
        energy_fci,
        tol=args.tol,
        max_workers=args.workers,
    )

    if not args.skip_optimized:
        u_spatial = build_U_from_thetas(ref["n_spatial"], thetas, pairs)
        action = OrbitalRotationAction(u_spatial, ref["basis_bitstrings"], ref["n_spatial"])
        h_rot = RotatedHamiltonian(ref["h_sub"], action)
        _profile_energy(
            "optimized",
            h_rot,
            sectors,
            energy_fci,
            tol=args.tol,
            max_workers=args.workers,
        )


if __name__ == "__main__":
    main()
