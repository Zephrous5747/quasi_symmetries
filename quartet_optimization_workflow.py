"""Workflow entry points for quartet parity baseline scans."""

from __future__ import annotations

import csv
from typing import Any, Iterable

from hamiltonian_cache import DEFAULT_CACHE_DIR, load_reference_state
from hamiltonian_geometry import default_grid_for_molecule
from optimization_abc_utils import closed_shell_hf_bitstring, popcount, solve_cisd_state
from quartet_optimization_utils import (
    baseline_rows_from_result,
    edge_set_jaccard,
    iter_topologies,
    plot_quartet_comparison,
    quartet_csv_fieldnames,
    run_fixed_topology_baseline,
    run_matching_greedy_baseline,
)

WORKFLOW_QUARTET_BASELINE = "quartet_baseline"


def _split_workflow_kwargs(kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cache_dir = str(kwargs.pop("hamiltonian_cache_dir", DEFAULT_CACHE_DIR))
    geom_kw = {k: kwargs[k] for k in ("hoh_angle_deg", "aspect_ratio") if k in kwargs}
    return cache_dir, geom_kw


def _default_csv_name(molecule: str) -> str:
    return f"{molecule}_quasi_symmetry_quartet_baseline.csv"


def _load_ref(molecule: str, x: float, cache_dir: str, **geometry_kwargs: Any) -> dict[str, Any]:
    return load_reference_state(
        molecule,
        x,
        cache_dir=cache_dir,
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        **geometry_kwargs,
    )


def evaluate_single_geometry(
    molecule: str,
    x: float,
    *,
    include_hub: bool = False,
    final_reoptimize_matching: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Evaluate matching-greedy and prescribed quartet topologies for one geometry."""
    workflow_kwargs = dict(kwargs)
    cache_dir, geom_kw = _split_workflow_kwargs(workflow_kwargs)
    ref = _load_ref(molecule, x, cache_dir, **geom_kw)
    n_spatial = ref["n_spatial"]

    common = {
        "molecule": molecule,
        "geometry_param": x,
        "energy_hf": ref["energy_hf"],
        "energy_fci": ref["energy_fci"],
        "energy_cisd": ref["energy_cisd"],
        "n_spatial": n_spatial,
    }
    v_sub = ref["v_sub"]
    basis_bitstrings = ref["basis_bitstrings"]

    results = [
        run_matching_greedy_baseline(
            v_sub,
            basis_bitstrings,
            n_spatial,
            final_reoptimize=final_reoptimize_matching,
        )
    ]
    for topology in iter_topologies(include_hub=include_hub):
        results.append(run_fixed_topology_baseline(v_sub, basis_bitstrings, n_spatial, topology))

    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(baseline_rows_from_result(result, **common))
    return rows


def add_edge_continuity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate rows with edge-set Jaccard continuity versus previous geometry."""
    previous: dict[tuple[str, str], set[tuple[int, int]]] = {}
    current_key: tuple[str, str, float] | None = None
    current_edges: set[tuple[int, int]] = set()
    current_rows: list[dict[str, Any]] = []
    annotated: list[dict[str, Any]] = []

    def flush() -> None:
        if current_key is None:
            return
        molecule, baseline, _ = current_key
        prev_edges = previous.get((molecule, baseline))
        continuity = float("nan") if prev_edges is None else edge_set_jaccard(prev_edges, current_edges)
        for row in current_rows:
            row["Edge_Jaccard_Prev"] = continuity
            annotated.append(row)
        previous[(molecule, baseline)] = set(current_edges)

    for row in sorted(rows, key=lambda r: (r["Molecule"], r["Baseline"], float(r["Geometry_Param"]), r["Edge_Index"])):
        key = (row["Molecule"], row["Baseline"], float(row["Geometry_Param"]))
        if current_key is not None and key != current_key:
            flush()
            current_edges = set()
            current_rows = []
        current_key = key
        current_edges.add((int(row["Edge_P"]), int(row["Edge_Q"])))
        current_rows.append(row)
    flush()
    return annotated


def run_scan(
    molecule: str,
    grid: Iterable[float],
    csv_filename: str | None = None,
    *,
    include_hub: bool = False,
    final_reoptimize_matching: bool = True,
    plot_prefix: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for x in grid:
        rows.extend(
            evaluate_single_geometry(
                molecule,
                float(x),
                include_hub=include_hub,
                final_reoptimize_matching=final_reoptimize_matching,
                **kwargs,
            )
        )
    rows = add_edge_continuity(rows)

    if csv_filename is not None:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=quartet_csv_fieldnames())
            writer.writeheader()
            writer.writerows(rows)

    if plot_prefix is not None:
        plot_quartet_comparison(rows, output_prefix=plot_prefix)

    return rows


def main(
    molecule: str = "lih",
    grid: Iterable[float] | None = None,
    csv_filename: str | None = None,
    *,
    include_hub: bool = False,
    final_reoptimize_matching: bool = True,
    plot_prefix: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    molecule_name = molecule.lower()
    scan_grid = default_grid_for_molecule(molecule_name) if grid is None else grid
    output_csv = _default_csv_name(molecule_name) if csv_filename is None else csv_filename
    return run_scan(
        molecule_name,
        scan_grid,
        csv_filename=output_csv,
        include_hub=include_hub,
        final_reoptimize_matching=final_reoptimize_matching,
        plot_prefix=plot_prefix,
        **kwargs,
    )


if __name__ == "__main__":
    main()
