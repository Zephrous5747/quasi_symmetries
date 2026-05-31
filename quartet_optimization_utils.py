"""Quartet parity baselines for orbital-frame quasi-symmetry experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from optimization_abc_utils import (
    ANGLE_INIT_SCALE,
    MAXITER,
    N_RESTARTS,
    OPT_METHOD,
    RANDOM_SEED,
    build_U_from_thetas,
    mode_is_occupied,
    pair_list_for_n,
)

Edge = tuple[int, int]


@dataclass(frozen=True)
class ParityStats:
    """Expectation and variance for an involutory parity operator."""

    expectation: float
    variance: float


def normalize_edge(edge: Edge) -> Edge:
    p, q = edge
    if p == q:
        raise ValueError("Quartet parity edges must connect two distinct orbitals.")
    return (p, q) if p < q else (q, p)


def validate_edges(edges: Iterable[Edge], n_spatial: int) -> list[Edge]:
    normalized: list[Edge] = []
    seen: set[Edge] = set()
    for edge in edges:
        p, q = normalize_edge(edge)
        if not (0 <= p < n_spatial and 0 <= q < n_spatial):
            raise ValueError(f"Edge {(p, q)} is out of range for n_spatial={n_spatial}.")
        if (p, q) in seen:
            raise ValueError(f"Duplicate quartet edge {(p, q)}.")
        normalized.append((p, q))
        seen.add((p, q))
    return normalized


def single_orbital_parity_value(bitstring: int, orbital: int, n_spatial: int) -> int:
    """Return (-1)^(n_alpha + n_beta) for one spatial orbital."""
    n_qubits = 2 * n_spatial
    occ = mode_is_occupied(bitstring, 2 * orbital, n_qubits)
    occ += mode_is_occupied(bitstring, 2 * orbital + 1, n_qubits)
    return -1 if occ % 2 else 1


def quartet_parity_value(bitstring: int, edge: Edge, n_spatial: int) -> int:
    """Return the product of single-orbital parities on an edge."""
    p, q = normalize_edge(edge)
    return (
        single_orbital_parity_value(bitstring, p, n_spatial)
        * single_orbital_parity_value(bitstring, q, n_spatial)
    )


def single_parity_diagonal(basis_bitstrings: Iterable[int], orbital: int, n_spatial: int) -> np.ndarray:
    return np.array(
        [single_orbital_parity_value(int(bitstring), orbital, n_spatial) for bitstring in basis_bitstrings],
        dtype=np.float64,
    )


def quartet_parity_diagonal(basis_bitstrings: Iterable[int], edge: Edge, n_spatial: int) -> np.ndarray:
    if isinstance(basis_bitstrings, list):
        return _quartet_parity_diagonal_cached(tuple(int(b) for b in basis_bitstrings), normalize_edge(edge), n_spatial)
    return np.array(
        [quartet_parity_value(int(bitstring), edge, n_spatial) for bitstring in basis_bitstrings],
        dtype=np.float64,
    )


@lru_cache(maxsize=512)
def _quartet_parity_diagonal_cached(
    basis_bitstrings: tuple[int, ...],
    edge: Edge,
    n_spatial: int,
) -> np.ndarray:
    return np.array(
        [quartet_parity_value(int(bitstring), edge, n_spatial) for bitstring in basis_bitstrings],
        dtype=np.float64,
    )


@lru_cache(maxsize=64)
def _spin_occupation_data(
    basis_bitstrings: tuple[int, ...],
    n_spatial: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    n_qubits = 2 * n_spatial
    alpha_occs: list[tuple[int, ...]] = []
    beta_occs: list[tuple[int, ...]] = []
    n_alpha_values: list[int] = []
    spin_order_signs: list[int] = []
    alpha_keys: set[tuple[int, ...]] = set()
    beta_keys: set[tuple[int, ...]] = set()

    for bitstring in basis_bitstrings:
        alpha = tuple(i for i in range(n_spatial) if mode_is_occupied(int(bitstring), 2 * i, n_qubits))
        beta = tuple(i for i in range(n_spatial) if mode_is_occupied(int(bitstring), 2 * i + 1, n_qubits))
        grouped_modes = [2 * i for i in alpha] + [2 * i + 1 for i in beta]
        inversions = sum(
            1
            for i, left in enumerate(grouped_modes)
            for right in grouped_modes[i + 1 :]
            if left > right
        )
        alpha_occs.append(alpha)
        beta_occs.append(beta)
        n_alpha_values.append(len(alpha))
        spin_order_signs.append(-1 if inversions % 2 else 1)
        alpha_keys.add(alpha)
        beta_keys.add(beta)

    sorted_alpha = tuple(sorted(alpha_keys, key=lambda item: (len(item), item)))
    sorted_beta = tuple(sorted(beta_keys, key=lambda item: (len(item), item)))
    alpha_index = {occ: index for index, occ in enumerate(sorted_alpha)}
    beta_index = {occ: index for index, occ in enumerate(sorted_beta)}
    alpha_positions = tuple(alpha_index[occ] for occ in alpha_occs)
    beta_positions = tuple(beta_index[occ] for occ in beta_occs)
    return sorted_alpha, sorted_beta, tuple(n_alpha_values), alpha_positions, beta_positions, tuple(spin_order_signs)


def _determinant_transform_matrix(
    u_spatial: np.ndarray,
    source_occs: list[tuple[int, ...]],
    target_occs: list[tuple[int, ...]],
) -> np.ndarray:
    if not source_occs or not target_occs:
        return np.zeros((len(source_occs), len(target_occs)), dtype=np.complex128)
    if len(source_occs[0]) == 0:
        return np.ones((len(source_occs), len(target_occs)), dtype=np.complex128)
    out = np.empty((len(source_occs), len(target_occs)), dtype=np.complex128)
    for i, source in enumerate(source_occs):
        source_idx = np.asarray(source, dtype=int)
        for j, target in enumerate(target_occs):
            out[i, j] = np.linalg.det(u_spatial[np.ix_(source_idx, np.asarray(target, dtype=int))])
    return out


def rotate_state_to_orbital_frame_fast(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    u_spatial: np.ndarray,
    n_spatial: int,
) -> np.ndarray:
    """Express the fixed-N state in a rotated orbital frame via determinant overlaps."""
    basis_tuple = tuple(int(bitstring) for bitstring in basis_bitstrings)
    alpha_occs, beta_occs, n_alpha_values, alpha_positions, beta_positions, spin_order_signs = _spin_occupation_data(
        basis_tuple,
        n_spatial,
    )
    coeffs = np.asarray(v_sub, dtype=np.complex128)
    rotated = np.zeros_like(coeffs)
    u_spatial = np.asarray(u_spatial, dtype=np.complex128)

    alpha_by_count = {
        count: [occ for occ in alpha_occs if len(occ) == count]
        for count in sorted(set(n_alpha_values))
    }
    beta_by_count = {
        count: [occ for occ in beta_occs if len(occ) == count]
        for count in sorted({len(occ) for occ in beta_occs})
    }
    alpha_lookup = {occ: idx for idx, occ in enumerate(alpha_occs)}
    beta_lookup = {occ: idx for idx, occ in enumerate(beta_occs)}

    for n_alpha in sorted(set(n_alpha_values)):
        source_indices = [
            index
            for index, count in enumerate(n_alpha_values)
            if count == n_alpha
        ]
        beta_count = len(beta_occs[beta_positions[source_indices[0]]])
        alpha_block = alpha_by_count[n_alpha]
        beta_block = beta_by_count[beta_count]
        alpha_global = [alpha_lookup[occ] for occ in alpha_block]
        beta_global = [beta_lookup[occ] for occ in beta_block]
        alpha_local = {global_idx: local_idx for local_idx, global_idx in enumerate(alpha_global)}
        beta_local = {global_idx: local_idx for local_idx, global_idx in enumerate(beta_global)}

        c_block = np.zeros((len(alpha_block), len(beta_block)), dtype=np.complex128)
        for index in source_indices:
            c_block[alpha_local[alpha_positions[index]], beta_local[beta_positions[index]]] = (
                spin_order_signs[index] * coeffs[index]
            )

        t_alpha = _determinant_transform_matrix(u_spatial, alpha_block, alpha_block)
        t_beta = _determinant_transform_matrix(u_spatial, beta_block, beta_block)
        rotated_block = t_alpha.conj().T @ c_block @ t_beta.conj()

        for index in source_indices:
            rotated[index] = rotated_block[
                alpha_local[alpha_positions[index]],
                beta_local[beta_positions[index]],
            ] * spin_order_signs[index]

    norm = np.linalg.norm(rotated)
    if norm == 0:
        raise ValueError("Cannot evaluate parity expectations for a zero-norm state.")
    return rotated / norm


def orbital_rotation_representation_R_fast(
    u_spatial: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
) -> np.ndarray:
    """Many-body orbital-rotation matrix using determinant overlaps."""
    basis_tuple = tuple(int(bitstring) for bitstring in basis_bitstrings)
    alpha_occs, beta_occs, n_alpha_values, alpha_positions, beta_positions, spin_order_signs = _spin_occupation_data(
        basis_tuple,
        n_spatial,
    )
    dim = len(basis_bitstrings)
    r_sub = np.zeros((dim, dim), dtype=np.complex128)
    u_spatial = np.asarray(u_spatial, dtype=np.complex128)

    alpha_by_count = {
        count: [occ for occ in alpha_occs if len(occ) == count]
        for count in sorted(set(n_alpha_values))
    }
    beta_by_count = {
        count: [occ for occ in beta_occs if len(occ) == count]
        for count in sorted({len(occ) for occ in beta_occs})
    }
    alpha_lookup = {occ: idx for idx, occ in enumerate(alpha_occs)}
    beta_lookup = {occ: idx for idx, occ in enumerate(beta_occs)}

    for n_alpha in sorted(set(n_alpha_values)):
        block_indices = [
            index
            for index, count in enumerate(n_alpha_values)
            if count == n_alpha
        ]
        beta_count = len(beta_occs[beta_positions[block_indices[0]]])
        alpha_block = alpha_by_count[n_alpha]
        beta_block = beta_by_count[beta_count]
        alpha_global = [alpha_lookup[occ] for occ in alpha_block]
        beta_global = [beta_lookup[occ] for occ in beta_block]
        alpha_local = {global_idx: local_idx for local_idx, global_idx in enumerate(alpha_global)}
        beta_local = {global_idx: local_idx for local_idx, global_idx in enumerate(beta_global)}
        t_alpha = _determinant_transform_matrix(u_spatial, alpha_block, alpha_block)
        t_beta = _determinant_transform_matrix(u_spatial, beta_block, beta_block)

        for source in block_indices:
            source_alpha = alpha_local[alpha_positions[source]]
            source_beta = beta_local[beta_positions[source]]
            source_sign = spin_order_signs[source]
            for target in block_indices:
                target_alpha = alpha_local[alpha_positions[target]]
                target_beta = beta_local[beta_positions[target]]
                target_sign = spin_order_signs[target]
                r_sub[source, target] = (
                    source_sign
                    * target_sign
                    * t_alpha[source_alpha, target_alpha]
                    * t_beta[source_beta, target_beta]
                )

    return r_sub


def rotate_state_to_orbital_frame(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    u_spatial: np.ndarray,
    n_spatial: int,
) -> np.ndarray:
    """Express the fixed-N state in the determinant basis of the rotated orbital frame."""
    return rotate_state_to_orbital_frame_fast(v_sub, basis_bitstrings, u_spatial, n_spatial)


def parity_stats_from_diagonal(state: np.ndarray, diagonal: np.ndarray) -> ParityStats:
    weights = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    expectation = float(np.real(np.dot(weights, np.asarray(diagonal, dtype=np.float64))))
    expectation = float(np.clip(expectation, -1.0, 1.0))
    variance = max(0.0, 1.0 - expectation**2)
    return ParityStats(expectation=expectation, variance=variance)


def single_parity_expectations(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    u_spatial: np.ndarray,
    n_spatial: int,
) -> list[ParityStats]:
    rotated = rotate_state_to_orbital_frame(v_sub, basis_bitstrings, u_spatial, n_spatial)
    return [
        parity_stats_from_diagonal(rotated, single_parity_diagonal(basis_bitstrings, i, n_spatial))
        for i in range(n_spatial)
    ]


def quartet_parity_expectations(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    u_spatial: np.ndarray,
    n_spatial: int,
    edges: Iterable[Edge],
) -> list[ParityStats]:
    edge_list = validate_edges(edges, n_spatial)
    rotated = rotate_state_to_orbital_frame(v_sub, basis_bitstrings, u_spatial, n_spatial)
    return [
        parity_stats_from_diagonal(rotated, quartet_parity_diagonal(basis_bitstrings, edge, n_spatial))
        for edge in edge_list
    ]


def quartet_cost_for_u(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    u_spatial: np.ndarray,
    n_spatial: int,
    edges: Iterable[Edge],
) -> float:
    stats = quartet_parity_expectations(v_sub, basis_bitstrings, u_spatial, n_spatial, edges)
    return float(sum(stat.variance for stat in stats))


def quartet_cost_from_thetas(
    thetas: np.ndarray,
    edges: Iterable[Edge],
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    rotation_pairs: list[Edge] | None = None,
) -> float:
    pairs = rotation_pairs or pair_list_for_n(n_spatial)
    u_spatial = build_U_from_thetas(n_spatial, thetas, pairs)
    return quartet_cost_for_u(v_sub, basis_bitstrings, u_spatial, n_spatial, edges)


def optimize_fixed_edge_quartets(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    edges: Iterable[Edge],
    *,
    n_restarts: int = N_RESTARTS,
    random_seed: int = RANDOM_SEED,
    initial_thetas: np.ndarray | None = None,
    include_zero_start: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    maxfev: int | None = 500,
    maxiter: int | None = None,
) -> dict[str, Any]:
    """Optimize a fixed quartet edge family over the existing Givens rotation angles."""
    edge_list = validate_edges(edges, n_spatial)
    pairs = pair_list_for_n(n_spatial)
    rng = np.random.default_rng(random_seed)

    def obj(thetas: np.ndarray) -> float:
        return quartet_cost_from_thetas(thetas, edge_list, v_sub, basis_bitstrings, n_spatial, pairs)

    starts = max(1, int(n_restarts))
    x0s: list[np.ndarray] = []
    for restart in range(starts):
        if restart == 0 and include_zero_start:
            x0 = np.zeros(len(pairs), dtype=float)
            if initial_thetas is not None:
                x0[: len(initial_thetas)] = np.asarray(initial_thetas, dtype=float)
        else:
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(pairs))
        x0s.append(x0)

    def run_start(x0: np.ndarray) -> tuple[Any, float]:
        method = "Powell" if OPT_METHOD.upper() == "POWELL" else OPT_METHOD
        options: dict[str, Any] = {"maxiter": MAXITER if maxiter is None else int(maxiter), "disp": False}
        if maxfev is not None:
            options["maxfev"] = int(maxfev)
        result = minimize(obj, x0=x0, method=method, options=options)
        return result, float(obj(result.x))

    if parallel and len(x0s) > 1:
        workers = max_workers if max_workers is not None else len(x0s)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_start, x0s))
    else:
        results = [run_start(x0) for x0 in x0s]

    best: dict[str, Any] | None = None
    for result, score in results:
        if best is None or score < best["cost"]:
            u_spatial = build_U_from_thetas(n_spatial, result.x, pairs)
            stats = quartet_parity_expectations(v_sub, basis_bitstrings, u_spatial, n_spatial, edge_list)
            best = {
                "res": result,
                "cost": score,
                "edges": edge_list,
                "pairs": pairs,
                "u_spatial": u_spatial,
                "stats": stats,
                "n_restarts": starts,
            }

    assert best is not None
    return best


def matching_edges(n_spatial: int) -> list[Edge]:
    """Non-overlapping seed matching: (0,1), (2,3), ..."""
    return [(i, i + 1) for i in range(0, n_spatial - 1, 2)]


def ring_edges(n_spatial: int) -> list[Edge]:
    if n_spatial < 3:
        raise ValueError("A simple M-edge ring requires at least 3 orbitals.")
    return [normalize_edge((i, (i + 1) % n_spatial)) for i in range(n_spatial)]


def balanced_tree_plus_edges(n_spatial: int) -> list[Edge]:
    if n_spatial < 3:
        raise ValueError("A balanced-tree-plus-one graph requires at least 3 orbitals.")
    edges = [normalize_edge(((child - 1) // 2, child)) for child in range(1, n_spatial)]
    extra = normalize_edge((0, n_spatial - 1))
    if extra in edges:
        extra = normalize_edge((1, n_spatial - 1))
    return [*edges, extra]


def hub_edges(n_spatial: int) -> list[Edge]:
    if n_spatial < 3:
        raise ValueError("A hub-plus-one graph requires at least 3 orbitals.")
    edges = [normalize_edge((0, i)) for i in range(1, n_spatial)]
    extra = normalize_edge((1, 2))
    return [*edges, extra]


def all_candidate_edges(n_spatial: int) -> list[Edge]:
    return [(p, q) for p in range(n_spatial) for q in range(p + 1, n_spatial)]


def add_greedy_edges_by_expectation(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    u_spatial: np.ndarray,
    n_spatial: int,
    seed_edges: Iterable[Edge],
    target_count: int | None = None,
) -> list[tuple[Edge, str, ParityStats]]:
    target = n_spatial if target_count is None else int(target_count)
    selected = validate_edges(seed_edges, n_spatial)
    if len(selected) > target:
        raise ValueError("Seed edge family already exceeds the requested target count.")

    selected_set = set(selected)
    rows: list[tuple[Edge, str, ParityStats]] = []
    seed_stats = quartet_parity_expectations(v_sub, basis_bitstrings, u_spatial, n_spatial, selected)
    rows.extend((edge, "matching_seed", stat) for edge, stat in zip(selected, seed_stats))

    candidates = [edge for edge in all_candidate_edges(n_spatial) if edge not in selected_set]
    candidate_stats = quartet_parity_expectations(v_sub, basis_bitstrings, u_spatial, n_spatial, candidates)
    ranked = sorted(
        zip(candidates, candidate_stats),
        key=lambda item: (-abs(item[1].expectation), item[0]),
    )
    for edge, stat in ranked:
        if len(rows) >= target:
            break
        rows.append((edge, "greedy_added", stat))

    if len(rows) != target:
        raise ValueError(f"Could only build {len(rows)} quartet edges; requested {target}.")
    return rows


def run_matching_greedy_baseline(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    *,
    final_reoptimize: bool = True,
    n_restarts: int | None = None,
    include_zero_start: bool = False,
    parallel_restarts: bool = True,
    max_workers: int | None = None,
    maxfev: int | None = 500,
    maxiter: int | None = None,
    initial_thetas: np.ndarray | None = None,
) -> dict[str, Any]:
    seed_edges = matching_edges(n_spatial)
    restart_count = N_RESTARTS if n_restarts is None else int(n_restarts)
    seed_best = optimize_fixed_edge_quartets(
        v_sub,
        basis_bitstrings,
        n_spatial,
        seed_edges,
        n_restarts=restart_count,
        include_zero_start=include_zero_start,
        parallel=parallel_restarts,
        max_workers=max_workers,
        maxfev=maxfev,
        maxiter=maxiter,
        initial_thetas=initial_thetas,
    )
    greedy_rows = add_greedy_edges_by_expectation(
        v_sub,
        basis_bitstrings,
        seed_best["u_spatial"],
        n_spatial,
        seed_edges,
    )
    final_edges = [edge for edge, _, _ in greedy_rows]
    initial_frame_stats = [
        (edge, source, stat) for edge, source, stat in greedy_rows
    ]

    if final_reoptimize:
        final_best = optimize_fixed_edge_quartets(
            v_sub,
            basis_bitstrings,
            n_spatial,
            final_edges,
            n_restarts=restart_count,
            include_zero_start=include_zero_start,
            parallel=parallel_restarts,
            max_workers=max_workers,
            maxfev=maxfev,
            maxiter=maxiter,
            initial_thetas=initial_thetas,
        )
    else:
        final_stats = quartet_parity_expectations(
            v_sub, basis_bitstrings, seed_best["u_spatial"], n_spatial, final_edges
        )
        final_best = {
            "res": seed_best["res"],
            "cost": float(sum(stat.variance for stat in final_stats)),
            "edges": final_edges,
            "pairs": seed_best["pairs"],
            "u_spatial": seed_best["u_spatial"],
            "stats": final_stats,
            "n_restarts": seed_best["n_restarts"],
        }

    return {
        "baseline": "matching_greedy",
        "seed": seed_best,
        "initial_frame_edge_stats": initial_frame_stats,
        "final": final_best,
        "sources": {edge: source for edge, source, _ in greedy_rows},
    }


def run_fixed_topology_baseline(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    topology: str,
    *,
    n_restarts: int | None = None,
    include_zero_start: bool = False,
    parallel_restarts: bool = True,
    max_workers: int | None = None,
    maxfev: int | None = 500,
    maxiter: int | None = None,
    initial_thetas: np.ndarray | None = None,
) -> dict[str, Any]:
    constructors = {
        "ring": ring_edges,
        "balanced_tree": balanced_tree_plus_edges,
        "hub": hub_edges,
    }
    if topology not in constructors:
        raise ValueError(f"Unknown quartet topology '{topology}'. Choose from {sorted(constructors)}.")
    edges = constructors[topology](n_spatial)
    restart_count = N_RESTARTS if n_restarts is None else int(n_restarts)
    best = optimize_fixed_edge_quartets(
        v_sub,
        basis_bitstrings,
        n_spatial,
        edges,
        n_restarts=restart_count,
        include_zero_start=include_zero_start,
        parallel=parallel_restarts,
        max_workers=max_workers,
        maxfev=maxfev,
        maxiter=maxiter,
        initial_thetas=initial_thetas,
    )
    return {
        "baseline": topology,
        "final": best,
        "sources": {edge: topology for edge in edges},
    }


def graph_diagnostics(edges: Iterable[Edge], n_spatial: int) -> dict[str, Any]:
    edge_list = validate_edges(edges, n_spatial)
    adjacency = {i: set() for i in range(n_spatial)}
    for p, q in edge_list:
        adjacency[p].add(q)
        adjacency[q].add(p)

    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(n_spatial):
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        comp: list[int] = []
        visited.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in adjacency[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))

    vertices_involved = sum(1 for i in range(n_spatial) if adjacency[i])
    nonempty_components = len(components)
    cycle_count = len(edge_list) - vertices_involved + nonempty_components
    return {
        "edge_count": len(edge_list),
        "degree_sequence": sorted((len(adjacency[i]) for i in range(n_spatial)), reverse=True),
        "components": components,
        "component_count": nonempty_components,
        "cycle_count": cycle_count,
        "algebraic_rank": vertices_involved - nonempty_components,
        "vertices_involved": vertices_involved,
    }


def edge_set_jaccard(edges_a: Iterable[Edge], edges_b: Iterable[Edge]) -> float:
    a = {normalize_edge(edge) for edge in edges_a}
    b = {normalize_edge(edge) for edge in edges_b}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def summarize_baseline_row(
    result: dict[str, Any],
    *,
    molecule: str,
    geometry_param: float,
    energy_hf: float,
    energy_fci: float,
    energy_cisd: float,
    n_spatial: int,
) -> dict[str, Any]:
    final = result["final"]
    edges = final["edges"]
    stats = final["stats"]
    diagnostics = graph_diagnostics(edges, n_spatial)
    expectations = [stat.expectation for stat in stats]
    variances = [stat.variance for stat in stats]
    return {
        "Workflow": "quartet_baseline",
        "Baseline": result["baseline"],
        "Molecule": molecule,
        "Geometry_Param": geometry_param,
        "E_HF": energy_hf,
        "E_FCI": energy_fci,
        "E_CISD": energy_cisd,
        "Edge_Count": len(edges),
        "Cost_Final": float(sum(variances)),
        "Mean_Abs_Expectation": float(np.mean(np.abs(expectations))) if expectations else float("nan"),
        "Min_Abs_Expectation": float(np.min(np.abs(expectations))) if expectations else float("nan"),
        "Max_Variance": float(np.max(variances)) if variances else float("nan"),
        "Degree_Sequence": " ".join(str(v) for v in diagnostics["degree_sequence"]),
        "Component_Count": diagnostics["component_count"],
        "Cycle_Count": diagnostics["cycle_count"],
        "Algebraic_Rank": diagnostics["algebraic_rank"],
        "Edges": " ".join(f"{p}-{q}" for p, q in edges),
        "Sources": " ".join(result["sources"].get(edge, result["baseline"]) for edge in edges),
        "Expectations": " ".join(f"{value:.12g}" for value in expectations),
        "Variances": " ".join(f"{value:.12g}" for value in variances),
    }


def baseline_rows_from_result(
    result: dict[str, Any],
    *,
    molecule: str,
    geometry_param: float,
    energy_hf: float,
    energy_fci: float,
    energy_cisd: float,
    n_spatial: int,
) -> list[dict[str, Any]]:
    final = result["final"]
    edges = final["edges"]
    stats = final["stats"]
    diagnostics = graph_diagnostics(edges, n_spatial)
    expectations = [stat.expectation for stat in stats]
    variances = [stat.variance for stat in stats]

    rows: list[dict[str, Any]] = []
    for edge_index, (edge, stat) in enumerate(zip(edges, stats)):
        rows.append(
            {
                "Workflow": "quartet_baseline",
                "Baseline": result["baseline"],
                "Molecule": molecule,
                "Geometry_Param": geometry_param,
                "E_HF": energy_hf,
                "E_FCI": energy_fci,
                "E_CISD": energy_cisd,
                "n_spatial": n_spatial,
                "Edge_Index": edge_index,
                "Edge_P": edge[0],
                "Edge_Q": edge[1],
                "Selection_Source": result["sources"].get(edge, result["baseline"]),
                "Expectation": stat.expectation,
                "Variance": stat.variance,
                "Cost_Final": float(sum(variances)),
                "Mean_Abs_Expectation": float(np.mean(np.abs(expectations))),
                "Min_Abs_Expectation": float(np.min(np.abs(expectations))),
                "Max_Variance": float(np.max(variances)),
                "Degree_Sequence": " ".join(str(v) for v in diagnostics["degree_sequence"]),
                "Components": ";".join("-".join(str(v) for v in comp) for comp in diagnostics["components"]),
                "Component_Count": diagnostics["component_count"],
                "Cycle_Count": diagnostics["cycle_count"],
                "Algebraic_Rank": diagnostics["algebraic_rank"],
                "Vertices_Involved": diagnostics["vertices_involved"],
                "Edge_Jaccard_Prev": float("nan"),
            }
        )
    return rows


def quartet_csv_fieldnames() -> list[str]:
    return [
        "Workflow",
        "Baseline",
        "Molecule",
        "Geometry_Param",
        "E_HF",
        "E_FCI",
        "E_CISD",
        "n_spatial",
        "Edge_Index",
        "Edge_P",
        "Edge_Q",
        "Selection_Source",
        "Expectation",
        "Variance",
        "Cost_Final",
        "Mean_Abs_Expectation",
        "Min_Abs_Expectation",
        "Max_Variance",
        "Degree_Sequence",
        "Components",
        "Component_Count",
        "Cycle_Count",
        "Algebraic_Rank",
        "Vertices_Involved",
        "Edge_Jaccard_Prev",
    ]


def plot_quartet_comparison(rows: list[dict[str, Any]], output_prefix: str | None = None) -> None:
    """Plot fair baseline comparisons: total cost and weakest parity per geometry."""
    if not rows:
        return

    summaries: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        key = (row["Baseline"], float(row["Geometry_Param"]))
        summaries.setdefault(key, row)

    baselines = sorted({baseline for baseline, _ in summaries})
    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    for baseline in baselines:
        points = sorted(
            ((geom, data) for (name, geom), data in summaries.items() if name == baseline),
            key=lambda item: item[0],
        )
        x_vals = [geom for geom, _ in points]
        axes[0].plot(x_vals, [data["Cost_Final"] for _, data in points], marker="o", label=baseline)
        axes[1].plot(
            x_vals,
            [data["Min_Abs_Expectation"] for _, data in points],
            marker="o",
            label=baseline,
        )

    axes[0].set_xlabel("Geometry parameter")
    axes[0].set_ylabel("Sum quartet variance")
    axes[0].legend()
    axes[1].set_xlabel("Geometry parameter")
    axes[1].set_ylabel("Minimum |<s_pq>|")
    axes[1].legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_quartet_comparison.png", dpi=200)
    else:
        plt.show()


def iter_topologies(include_hub: bool = False) -> Iterable[str]:
    yield from ("ring", "balanced_tree")
    if include_hub:
        yield "hub"
