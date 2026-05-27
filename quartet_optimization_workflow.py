"""Workflow entry points for quartet parity baseline scans."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp

from hamiltonian_cache import DEFAULT_CACHE_DIR, load_reference_state
from hamiltonian_geometry import default_grid_for_molecule
from optimization_abc_utils import (
    bo_like_coupled_energy_test,
    closed_shell_hf_bitstring,
    comm_state_norm_sq,
    coupled_energy_test,
    decoupled_energy_test,
    diagonalize_sector_blocks,
    orbital_rotation_representation_R,
    popcount,
    shannon_block_decomposition,
    solve_cisd_state,
)
from quartet_optimization_utils import (
    graph_diagnostics,
    iter_topologies,
    quartet_cost_for_u,
    quartet_parity_diagonal,
    run_fixed_topology_baseline,
    run_matching_greedy_baseline,
)

WORKFLOW_QUARTET_BASELINE = "quartet_baseline"


def _trace(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(f"[quartet] {message}", flush=True)


def _split_workflow_kwargs(kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cache_dir = str(kwargs.pop("hamiltonian_cache_dir", DEFAULT_CACHE_DIR))
    geom_kw = {k: kwargs[k] for k in ("hoh_angle_deg", "aspect_ratio") if k in kwargs}
    return cache_dir, geom_kw


def _default_csv_name(molecule: str) -> str:
    return f"{molecule}_quartet_baseline_summary.csv"


def quartet_summary_csv_fieldnames() -> list[str]:
    return [
        "Workflow",
        "Baseline",
        "Molecule",
        "Geometry_Param",
        "E_HF",
        "E_FCI",
        "E_CISD",
        "n_spatial",
        "V_Identity",
        "V_Optimized",
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
        "Edge_Count",
        "Mean_Abs_Expectation",
        "Min_Abs_Expectation",
        "Max_Variance",
        "Degree_Sequence",
        "Components",
        "Component_Count",
        "Cycle_Count",
        "Algebraic_Rank",
        "Vertices_Involved",
        "Edges",
        "Sources",
        "Expectations",
        "Variances",
        "Thetas_JSON",
        "Rotation_Pairs_JSON",
        "Optimizer_Success",
        "Optimizer_Status",
        "Optimizer_Message",
        "Optimizer_Nit",
        "Optimizer_Nfev",
        "N_Restarts",
        "Initial_Frame_Edges_JSON",
        "Initial_Frame_Sources_JSON",
        "Initial_Frame_Expectations_JSON",
        "Initial_Frame_Variances_JSON",
        "Elapsed_Seconds",
    ]


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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _edge_json(edges: Iterable[tuple[int, int]]) -> str:
    return _json_dumps([[int(p), int(q)] for p, q in edges])


def _float_list_json(values: Iterable[Any]) -> str:
    return _json_dumps([float(value) for value in values])


def _optimizer_value(res: Any, name: str) -> Any:
    return getattr(res, name, "")


def _quartet_sectors(
    basis_bitstrings: list[int],
    edges: Iterable[tuple[int, int]],
    n_spatial: int,
) -> dict[tuple[int, ...], list[int]]:
    edge_list = list(edges)
    sectors: dict[tuple[int, ...], list[int]] = {}
    for index, bitstring in enumerate(basis_bitstrings):
        key = tuple(
            int(quartet_parity_diagonal([int(bitstring)], edge, n_spatial)[0])
            for edge in edge_list
        )
        sectors.setdefault(key, []).append(index)
    return sectors


def _quartet_commutativity(
    h_mat: Any,
    psi: np.ndarray,
    basis_bitstrings: list[int],
    edges: Iterable[tuple[int, int]],
    n_spatial: int,
) -> tuple[float, float]:
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    sum_comm_sq = 0.0
    sum_exp = 0.0
    dim = len(basis_bitstrings)
    for edge in edges:
        diagonal = quartet_parity_diagonal(basis_bitstrings, edge, n_spatial)
        op = sp.diags(diagonal, offsets=0, shape=(dim, dim), format="csc")
        sum_exp += float(np.real(np.vdot(psi, op.dot(psi))))
        comm_sq, _ = comm_state_norm_sq(h_mat, op, psi, check_eigenstate=False)
        sum_comm_sq += comm_sq
    return float(sum_comm_sq), float(sum_exp)


def _quartet_entropy_and_energy(
    h_dense: np.ndarray,
    psi: np.ndarray,
    basis_bitstrings: list[int],
    edges: Iterable[tuple[int, int]],
    n_spatial: int,
    energy_fci: float,
) -> dict[str, Any]:
    sectors = _quartet_sectors(basis_bitstrings, edges, n_spatial)
    entropy_fine, entropy_coarse, _ = shannon_block_decomposition(h_dense, psi, sectors)
    sector_data = diagonalize_sector_blocks(h_dense, sectors)
    e_dec_min, _, _ = decoupled_energy_test(h_dense, sectors)
    e_coupled, k_coupled, _, _ = coupled_energy_test(h_dense, sector_data, E_exact=energy_fci, tol=1e-3)
    e_bo, _, _ = bo_like_coupled_energy_test(h_dense, sector_data)
    return {
        "Coarse_Entropy": entropy_coarse,
        "Fine_Entropy": entropy_fine,
        "Edec": e_dec_min,
        "Ecoupled": e_coupled,
        "Kcoupled": k_coupled,
        "EBO": e_bo,
        "NumSectors": len(sectors),
    }


def _skipped_quartet_diagnostics() -> dict[str, Any]:
    nan = float("nan")
    return {
        "Coarse_Entropy_Identity": nan,
        "Coarse_Entropy_Optimized": nan,
        "Fine_Entropy_Identity": nan,
        "Fine_Entropy_Optimized": nan,
        "Edec_Identity": nan,
        "Edec_Optimized": nan,
        "Ecoupled_Identity": nan,
        "Ecoupled_Optimized": nan,
        "Kcoupled_Identity": 0,
        "Kcoupled_Optimized": 0,
        "EBO_Identity": nan,
        "EBO_Optimized": nan,
        "NumSectors_Identity": 0,
        "NumSectors_Optimized": 0,
        "DenseDiagnosticsSkipped": True,
    }


def _quartet_diagnostics(
    ref: dict[str, Any],
    edges: list[tuple[int, int]],
    u_optimized: np.ndarray,
) -> dict[str, Any]:
    n_spatial = ref["n_spatial"]
    basis_bitstrings = ref["basis_bitstrings"]
    psi_identity = ref["v_sub"] / np.linalg.norm(ref["v_sub"])
    h_identity_sparse = ref["h_sub"]
    comm_id, sexp_id = _quartet_commutativity(
        h_identity_sparse, psi_identity, basis_bitstrings, edges, n_spatial
    )

    r_opt = orbital_rotation_representation_R(u_optimized, basis_bitstrings, n_spatial)
    psi_optimized = r_opt.conj().T @ psi_identity
    h_optimized_sparse = r_opt.conj().T @ (h_identity_sparse @ r_opt)
    h_optimized_sparse = sp.csc_matrix(0.5 * (h_optimized_sparse + h_optimized_sparse.conj().T))
    comm_opt, sexp_opt = _quartet_commutativity(
        h_optimized_sparse, psi_optimized, basis_bitstrings, edges, n_spatial
    )

    if not ref["use_dense"]:
        return {
            "Sum_CommSq_Identity": comm_id,
            "Sum_CommSq_Optimized": comm_opt,
            "Sum_Sexp_Identity": sexp_id,
            "Sum_Sexp_Optimized": sexp_opt,
            **_skipped_quartet_diagnostics(),
        }

    h_identity = ref["h_sub"].toarray().astype(np.complex128)
    h_identity = 0.5 * (h_identity + h_identity.conj().T)
    h_optimized = np.asarray(h_optimized_sparse.toarray(), dtype=np.complex128)
    identity_post = _quartet_entropy_and_energy(
        h_identity, psi_identity, basis_bitstrings, edges, n_spatial, ref["energy_fci"]
    )
    optimized_post = _quartet_entropy_and_energy(
        h_optimized, psi_optimized, basis_bitstrings, edges, n_spatial, ref["energy_fci"]
    )
    return {
        "Sum_CommSq_Identity": comm_id,
        "Sum_CommSq_Optimized": comm_opt,
        "Sum_Sexp_Identity": sexp_id,
        "Sum_Sexp_Optimized": sexp_opt,
        "Coarse_Entropy_Identity": identity_post["Coarse_Entropy"],
        "Coarse_Entropy_Optimized": optimized_post["Coarse_Entropy"],
        "Fine_Entropy_Identity": identity_post["Fine_Entropy"],
        "Fine_Entropy_Optimized": optimized_post["Fine_Entropy"],
        "Edec_Identity": identity_post["Edec"],
        "Edec_Optimized": optimized_post["Edec"],
        "Ecoupled_Identity": identity_post["Ecoupled"],
        "Ecoupled_Optimized": optimized_post["Ecoupled"],
        "Kcoupled_Identity": identity_post["Kcoupled"],
        "Kcoupled_Optimized": optimized_post["Kcoupled"],
        "EBO_Identity": identity_post["EBO"],
        "EBO_Optimized": optimized_post["EBO"],
        "NumSectors_Identity": identity_post["NumSectors"],
        "NumSectors_Optimized": optimized_post["NumSectors"],
        "DenseDiagnosticsSkipped": False,
    }


def _summary_row_from_result(
    best: dict[str, Any],
    *,
    common: dict[str, Any],
    baseline: str,
    ref: dict[str, Any],
    sources: dict[tuple[int, int], str] | None = None,
    elapsed_seconds: float,
    initial_frame_edge_stats: list[tuple[tuple[int, int], str, Any]] | None = None,
) -> dict[str, Any]:
    res = best["res"]
    edges = best["edges"]
    pairs = best["pairs"]
    stats = best["stats"]
    source_map = sources or {}
    initial_frame_edge_stats = initial_frame_edge_stats or []
    u_identity = np.eye(common["n_spatial"])
    v_identity = quartet_cost_for_u(
        ref["v_sub"], ref["basis_bitstrings"], u_identity, common["n_spatial"], edges
    )
    diagnostics = graph_diagnostics(edges, common["n_spatial"])
    expectations = [stat.expectation for stat in stats]
    variances = [stat.variance for stat in stats]
    return {
        "Workflow": WORKFLOW_QUARTET_BASELINE,
        "Baseline": baseline,
        "Molecule": common["molecule"],
        "Geometry_Param": common["geometry_param"],
        "E_HF": common["energy_hf"],
        "E_FCI": common["energy_fci"],
        "E_CISD": common["energy_cisd"],
        "n_spatial": common["n_spatial"],
        "V_Identity": v_identity,
        "V_Optimized": float(best["cost"]),
        **_quartet_diagnostics(ref, edges, best["u_spatial"]),
        "Edge_Count": len(edges),
        "Mean_Abs_Expectation": float(np.mean(np.abs(expectations))) if expectations else float("nan"),
        "Min_Abs_Expectation": float(np.min(np.abs(expectations))) if expectations else float("nan"),
        "Max_Variance": float(np.max(variances)) if variances else float("nan"),
        "Degree_Sequence": " ".join(str(v) for v in diagnostics["degree_sequence"]),
        "Components": ";".join("-".join(str(v) for v in comp) for comp in diagnostics["components"]),
        "Component_Count": diagnostics["component_count"],
        "Cycle_Count": diagnostics["cycle_count"],
        "Algebraic_Rank": diagnostics["algebraic_rank"],
        "Vertices_Involved": diagnostics["vertices_involved"],
        "Edges": " ".join(f"{p}-{q}" for p, q in edges),
        "Sources": " ".join(source_map.get(edge, baseline) for edge in edges),
        "Expectations": " ".join(f"{value:.12g}" for value in expectations),
        "Variances": " ".join(f"{value:.12g}" for value in variances),
        "Thetas_JSON": _float_list_json(res.x),
        "Rotation_Pairs_JSON": _edge_json(pairs),
        "Optimizer_Success": bool(_optimizer_value(res, "success")),
        "Optimizer_Status": _optimizer_value(res, "status"),
        "Optimizer_Message": str(_optimizer_value(res, "message")),
        "Optimizer_Nit": _optimizer_value(res, "nit"),
        "Optimizer_Nfev": _optimizer_value(res, "nfev"),
        "N_Restarts": int(best["n_restarts"]),
        "Initial_Frame_Edges_JSON": _edge_json(edge for edge, _, _ in initial_frame_edge_stats),
        "Initial_Frame_Sources_JSON": _json_dumps([source for _, source, _ in initial_frame_edge_stats]),
        "Initial_Frame_Expectations_JSON": _float_list_json(
            stat.expectation for _, _, stat in initial_frame_edge_stats
        ),
        "Initial_Frame_Variances_JSON": _float_list_json(
            stat.variance for _, _, stat in initial_frame_edge_stats
        ),
        "Elapsed_Seconds": elapsed_seconds,
    }


def evaluate_single_geometry(
    molecule: str,
    x: float,
    *,
    include_hub: bool = False,
    final_reoptimize_matching: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run optimization stages and return raw optimization-result rows."""
    workflow_kwargs = dict(kwargs)
    cache_dir, geom_kw = _split_workflow_kwargs(workflow_kwargs)
    start = time.perf_counter()
    _trace(f"{molecule} x={x:.6g}: loading reference state from {cache_dir}", verbose=verbose)
    ref = _load_ref(molecule, x, cache_dir, **geom_kw)
    n_spatial = ref["n_spatial"]
    _trace(
        f"{molecule} x={x:.6g}: loaded n_spatial={n_spatial}, "
        f"E_FCI={ref['energy_fci']:.12g}",
        verbose=verbose,
    )

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

    rows: list[dict[str, Any]] = []
    _trace(f"{molecule} x={x:.6g}: optimizing greedy baseline", verbose=verbose)
    baseline_start = time.perf_counter()
    matching_result = run_matching_greedy_baseline(
        v_sub,
        basis_bitstrings,
        n_spatial,
        final_reoptimize=final_reoptimize_matching,
    )
    baseline_elapsed = time.perf_counter() - baseline_start
    rows.append(
        _summary_row_from_result(
            matching_result["final"],
            common=common,
            ref=ref,
            baseline="greedy",
            sources=matching_result["sources"],
            elapsed_seconds=baseline_elapsed,
            initial_frame_edge_stats=matching_result["initial_frame_edge_stats"],
        )
    )
    _trace(
        f"{molecule} x={x:.6g}: finished greedy in "
        f"{baseline_elapsed:.1f}s",
        verbose=verbose,
    )
    for topology in iter_topologies(include_hub=include_hub):
        _trace(f"{molecule} x={x:.6g}: optimizing {topology} baseline", verbose=verbose)
        baseline_start = time.perf_counter()
        topology_result = run_fixed_topology_baseline(v_sub, basis_bitstrings, n_spatial, topology)
        baseline_elapsed = time.perf_counter() - baseline_start
        rows.append(
            _summary_row_from_result(
                topology_result["final"],
                common=common,
                ref=ref,
                baseline=topology,
                sources=topology_result["sources"],
                elapsed_seconds=baseline_elapsed,
            )
        )
        _trace(
            f"{molecule} x={x:.6g}: finished {topology} in "
            f"{baseline_elapsed:.1f}s",
            verbose=verbose,
        )

    _trace(
        f"{molecule} x={x:.6g}: completed geometry with {len(rows)} summary rows in "
        f"{time.perf_counter() - start:.1f}s",
        verbose=verbose,
    )
    return rows


def run_scan(
    molecule: str,
    grid: Iterable[float],
    csv_filename: str | None = None,
    *,
    include_hub: bool = False,
    final_reoptimize_matching: bool = True,
    plot_prefix: str | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid_values = [float(x) for x in grid]
    writer: csv.DictWriter[str] | None = None
    handle = None

    if csv_filename is not None:
        csv_path = Path(csv_filename)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(csv_path, mode="w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=quartet_summary_csv_fieldnames())
        writer.writeheader()
        handle.flush()
        _trace(f"{molecule}: writing incremental quartet summary CSV to {csv_path}", verbose=verbose)

    try:
        _trace(f"{molecule}: starting scan over {len(grid_values)} geometries", verbose=verbose)
        for index, x in enumerate(grid_values, start=1):
            _trace(f"{molecule}: geometry {index}/{len(grid_values)} x={x:.6g} started", verbose=verbose)
            geometry_rows = evaluate_single_geometry(
                molecule,
                x,
                include_hub=include_hub,
                final_reoptimize_matching=final_reoptimize_matching,
                verbose=verbose,
                **kwargs,
            )
            rows.extend(geometry_rows)

            if writer is not None and handle is not None:
                writer.writerows(geometry_rows)
                handle.flush()
                _trace(
                    f"{molecule}: wrote {len(geometry_rows)} summary rows for x={x:.6g}",
                    verbose=verbose,
                )
            _trace(f"{molecule}: geometry {index}/{len(grid_values)} x={x:.6g} done", verbose=verbose)
    finally:
        if handle is not None:
            handle.close()

    if plot_prefix is not None:
        _trace("plot_prefix ignored; quartet summary data is written for separate analysis", verbose=verbose)

    return rows


def main(
    molecule: str = "lih",
    grid: Iterable[float] | None = None,
    csv_filename: str | None = None,
    *,
    include_hub: bool = False,
    final_reoptimize_matching: bool = True,
    plot_prefix: str | None = None,
    verbose: bool = True,
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
        verbose=verbose,
        **kwargs,
    )


if __name__ == "__main__":
    main()
