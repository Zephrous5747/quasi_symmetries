"""Recompute K_coupled / E_coupled from stored orbital rotations (no re-optimization)."""

from __future__ import annotations

from quasi_symmetries.config import CACHE_DIR, diagnostics_dir, table_path

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from quasi_symmetries.diagnostics.mixed_pool import mixed_pool_energy_indicators
from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.optimization import (
    closed_shell_hf_bitstring,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import MixedOperatorPool, normalize_edge
from quasi_symmetries.optimization.rotations import build_U_from_thetas
from quasi_symmetries.workflows import quartet as quartet_workflow


ENERGY_FIELDS = (
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
)


def _parse_edges(raw: str) -> list[tuple[int, int]]:
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        pairs = json.loads(text)
        return [(int(p), int(q)) for p, q in pairs]
    return [normalize_edge(tuple(int(part) for part in item.split("-"))) for item in text.split()]


def _parse_pairs(raw: str) -> list[tuple[int, int]]:
    return [(int(p), int(q)) for p, q in json.loads(raw)]


def _parse_pool(row: dict[str, str], n_spatial: int) -> MixedOperatorPool:
    singles = tuple(int(item) for item in row.get("Pool_Singles", "").split() if item)
    quartets_raw = row.get("Pool_Quartets", "").strip()
    if quartets_raw:
        quartets = tuple(
            normalize_edge(tuple(int(part) for part in item.split("-")))
            for item in quartets_raw.split()
        )
    else:
        quartets = ()
    if not singles and n_spatial:
        singles = tuple(range(n_spatial))
    return MixedOperatorPool(singles=singles, quartets=quartets)


def _load_reference(molecule: str, x: float, *, cache_dir: str, **geometry_kwargs) -> dict:
    return load_reference_state(
        molecule,
        x,
        cache_dir=cache_dir,
        load_hamiltonian=True,
        load_full_hamiltonian=False,
        compute_rdms=False,
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        **geometry_kwargs,
    )


def _recompute_quartet_baseline_row(
    row: dict[str, str],
    *,
    cache_dir: str,
    geometry_kwargs: dict,
) -> dict[str, str]:
    molecule = row["Molecule"].lower()
    x = float(row["Geometry_Param"])
    ref = _load_reference(molecule, x, cache_dir=cache_dir, **geometry_kwargs)
    if not ref["use_dense"]:
        print(f"[skip] {molecule} x={x}: dense diagnostics unavailable", flush=True)
        return row

    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    pairs = _parse_pairs(row["Rotation_Pairs_JSON"])
    edges = _parse_edges(row["Edges"])
    u_optimized = build_U_from_thetas(int(ref["n_spatial"]), thetas, pairs)
    diagnostics = quartet_workflow._quartet_diagnostics(ref, edges, u_optimized)

    updated = dict(row)
    for key in ENERGY_FIELDS:
        if key in diagnostics:
            updated[key] = diagnostics[key]
    print(
        f"[ok] quartet {row.get('Baseline', '')} x={x:.6g}: "
        f"K_id={updated.get('Kcoupled_Identity')} K_opt={updated.get('Kcoupled_Optimized')}",
        flush=True,
    )
    return updated


def _recompute_mixed_pool_row(
    row: dict[str, str],
    *,
    cache_dir: str,
    geometry_kwargs: dict,
) -> dict[str, str]:
    molecule = row["Molecule"].lower()
    x = float(row["Geometry_Param"])
    ref = _load_reference(molecule, x, cache_dir=cache_dir, **geometry_kwargs)
    if not ref["use_dense"]:
        print(f"[skip] {molecule} x={x}: dense diagnostics unavailable", flush=True)
        return row

    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    pairs = _parse_pairs(row["Rotation_Pairs_JSON"])
    pool = _parse_pool(row, int(ref["n_spatial"]))
    u_optimized = build_U_from_thetas(int(ref["n_spatial"]), thetas, pairs)
    energy = mixed_pool_energy_indicators(ref, pool, u_optimized)

    updated = dict(row)
    mapping = {
        "Coarse_Entropy_Identity": "Coarse_Entropy_Identity",
        "Coarse_Entropy_Optimized": "Coarse_Entropy_Optimized",
        "Fine_Entropy_Identity": "Fine_Entropy_Identity",
        "Fine_Entropy_Optimized": "Fine_Entropy_Optimized",
        "Edec_Identity": "Edec_Identity",
        "Edec_Optimized": "Edec_Optimized",
        "Ecoupled_Identity": "Ecoupled_Identity",
        "Ecoupled_Optimized": "Ecoupled_Optimized",
        "Kcoupled_Identity": "Kcoupled_Identity",
        "Kcoupled_Optimized": "Kcoupled_Optimized",
        "EBO_Identity": "EBO_Identity",
        "EBO_Optimized": "EBO_Optimized",
        "NumSectors_Identity": "NumSectors_Identity",
        "NumSectors_Optimized": "NumSectors_Optimized",
        "DenseDiagnosticsSkipped": "DenseDiagnosticsSkipped",
    }
    for src, dst in mapping.items():
        if src in energy:
            updated[dst] = energy[src]
    print(
        f"[ok] {row.get('Workflow', 'mixed_pool')} x={x:.6g}: "
        f"K_id={updated.get('Kcoupled_Identity')} K_opt={updated.get('Kcoupled_Optimized')}",
        flush=True,
    )
    return updated


def recompute_row(
    row: dict[str, str],
    *,
    cache_dir: str,
    geometry_kwargs: dict,
) -> dict[str, str]:
    workflow = row.get("Workflow", "")
    if workflow == quartet_workflow.WORKFLOW_QUARTET_BASELINE:
        return _recompute_quartet_baseline_row(row, cache_dir=cache_dir, geometry_kwargs=geometry_kwargs)
    if workflow in {"mixed_pool", "parity_seniority"}:
        return _recompute_mixed_pool_row(row, cache_dir=cache_dir, geometry_kwargs=geometry_kwargs)
    raise ValueError(f"Unsupported workflow '{workflow}' in row for x={row.get('Geometry_Param')}.")


def recompute_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    cache_dir: str,
    geometry_kwargs: dict | None = None,
) -> None:
    geometry_kwargs = geometry_kwargs or {}
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {input_csv}")

    updated_rows = [
        recompute_row(row, cache_dir=cache_dir, geometry_kwargs=geometry_kwargs)
        for row in rows
    ]

    fieldnames = list(rows[0].keys())
    for row in updated_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f"[ok] wrote {len(updated_rows)} rows to {output_csv}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=table_path("h2o", "quartet_baseline_summary.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to overwriting --input-csv.",
    )
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--hoh-angle-deg", type=float, default=104.5)
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=diagnostics_dir("h2o") / "quartet_baseline_diagnostics.png",
        help="If set with --plot, write H2O operator diagnostics figure here.",
    )
    parser.add_argument(
        "--seniority-csv",
        type=Path,
        default=table_path("h2o", "parity_seniority_diagnostics.csv"),
    )
    parser.add_argument(
        "--mixed-pool-csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_energy_diagnostics.csv"),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After recompute, regenerate the H2O operator diagnostics plot.",
    )
    parser.add_argument(
        "--recompute-h2o-operator-csvs",
        action="store_true",
        help="Also recompute seniority and mixed-pool CSVs used by the H2O plot.",
    )
    args = parser.parse_args()

    geometry_kwargs = {"hoh_angle_deg": args.hoh_angle_deg}
    output_csv = args.output_csv or args.input_csv

    recompute_csv(
        args.input_csv,
        output_csv,
        cache_dir=args.cache_dir,
        geometry_kwargs=geometry_kwargs,
    )

    if args.recompute_h2o_operator_csvs:
        for path in (args.seniority_csv, args.mixed_pool_csv):
            recompute_csv(
                path,
                path,
                cache_dir=args.cache_dir,
                geometry_kwargs=geometry_kwargs,
            )

    if args.plot or args.recompute_h2o_operator_csvs:
        from scripts.plot.h2o_operator_diagnostics import plot_h2o_operator_diagnostics

        plot_h2o_operator_diagnostics(
            seniority_csv=args.seniority_csv,
            mixed_pool_csv=args.mixed_pool_csv,
            output_path=args.plot_output,
        )


if __name__ == "__main__":
    main()
