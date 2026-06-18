"""Sparse action-based diagnostics for large N2 quasi-symmetry tables.

This avoids dense many-body rotation and dense Hamiltonian construction.  It
computes expectation, variance, coarse entropy, and state-specific commutator
norms from precomputed optimized angles.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from quasi_symmetries.config import CACHE_DIR, LEGACY_ABC_TABLES_DIR, TABLES_DIR
from quasi_symmetries.diagnostics.mixed_pool import (
    mixed_pool_diagonals,
    mixed_pool_sectors,
)
from quasi_symmetries.diagnostics.sparse_energy import (
    SparseSubspaceHamiltonian,
    build_rotated_h_sub_csc,
    energy_sector_diagnostics_sparse,
)
from quasi_symmetries.fermion.bitstring import mode_is_occupied
from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.optimization import (
    build_U_from_thetas,
    build_generalized_sectors,
    pair_list_for_n,
)
from quasi_symmetries.optimization.quartet import (
    MixedOperatorPool,
    _determinant_transform_matrix,
    _spin_occupation_data,
    normalize_edge,
    quartet_parity_diagonal,
)
from quasi_symmetries.symmetry.exact import (
    exact_symmetry_data_for_reference,
    ground_state_sector_indices,
)
from quasi_symmetries.symmetry.labels import load_symmetry_labels
from quasi_symmetries.optimization.rotations import symmetry_blocked_pair_list


@dataclass
class RotationBlock:
    indices: list[int]
    alpha_local: dict[int, int]
    beta_local: dict[int, int]
    alpha_positions: tuple[int, ...]
    beta_positions: tuple[int, ...]
    signs: tuple[int, ...]
    t_alpha: np.ndarray
    t_beta: np.ndarray


class OrbitalRotationAction:
    """Apply R and R^dagger in the fixed-N determinant basis using cached blocks."""

    def __init__(self, u_spatial: np.ndarray, basis_bitstrings: list[int], n_spatial: int):
        self.u_spatial = np.asarray(u_spatial, dtype=np.complex128)
        self.basis_bitstrings = list(basis_bitstrings)
        self.n_spatial = int(n_spatial)
        self.dim = len(basis_bitstrings)
        basis_tuple = tuple(int(bitstring) for bitstring in basis_bitstrings)
        (
            alpha_occs,
            beta_occs,
            n_alpha_values,
            alpha_positions,
            beta_positions,
            spin_order_signs,
        ) = _spin_occupation_data(basis_tuple, n_spatial)
        alpha_lookup = {occ: idx for idx, occ in enumerate(alpha_occs)}
        beta_lookup = {occ: idx for idx, occ in enumerate(beta_occs)}
        alpha_by_count = {
            count: [occ for occ in alpha_occs if len(occ) == count]
            for count in sorted(set(n_alpha_values))
        }
        beta_by_count = {
            count: [occ for occ in beta_occs if len(occ) == count]
            for count in sorted({len(occ) for occ in beta_occs})
        }

        self.blocks: list[RotationBlock] = []
        for n_alpha in sorted(set(n_alpha_values)):
            indices = [index for index, count in enumerate(n_alpha_values) if count == n_alpha]
            beta_count = len(beta_occs[beta_positions[indices[0]]])
            alpha_block = alpha_by_count[n_alpha]
            beta_block = beta_by_count[beta_count]
            alpha_global = [alpha_lookup[occ] for occ in alpha_block]
            beta_global = [beta_lookup[occ] for occ in beta_block]
            alpha_local = {global_idx: local_idx for local_idx, global_idx in enumerate(alpha_global)}
            beta_local = {global_idx: local_idx for local_idx, global_idx in enumerate(beta_global)}
            self.blocks.append(
                RotationBlock(
                    indices=indices,
                    alpha_local=alpha_local,
                    beta_local=beta_local,
                    alpha_positions=alpha_positions,
                    beta_positions=beta_positions,
                    signs=spin_order_signs,
                    t_alpha=_determinant_transform_matrix(u_spatial, alpha_block, alpha_block),
                    t_beta=_determinant_transform_matrix(u_spatial, beta_block, beta_block),
                )
            )

    def _to_blocks(self, vector: np.ndarray, block: RotationBlock) -> np.ndarray:
        matrix = np.zeros((block.t_alpha.shape[0], block.t_beta.shape[0]), dtype=np.complex128)
        for index in block.indices:
            matrix[
                block.alpha_local[block.alpha_positions[index]],
                block.beta_local[block.beta_positions[index]],
            ] = block.signs[index] * vector[index]
        return matrix

    def _from_blocks(self, matrix: np.ndarray, block: RotationBlock, out: np.ndarray) -> None:
        for index in block.indices:
            out[index] = (
                matrix[
                    block.alpha_local[block.alpha_positions[index]],
                    block.beta_local[block.beta_positions[index]],
                ]
                * block.signs[index]
            )

    def apply_dagger(self, vector: np.ndarray) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.complex128)
        for block in self.blocks:
            matrix = self._to_blocks(vector, block)
            rotated = block.t_alpha.conj().T @ matrix @ block.t_beta.conj()
            self._from_blocks(rotated, block, out)
        return out

    def apply(self, vector: np.ndarray) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.complex128)
        for block in self.blocks:
            matrix = self._to_blocks(vector, block)
            rotated = block.t_alpha @ matrix @ block.t_beta.T
            self._from_blocks(rotated, block, out)
        return out


class RotatedHamiltonian:
    """Apply H_rot = R^dagger H R without forming dense many-body R."""

    def __init__(self, h_sub, action: OrbitalRotationAction):
        self.h_sub = h_sub
        self.action = action
        self.shape = h_sub.shape

    def dot(self, vector: np.ndarray) -> np.ndarray:
        return self.action.apply_dagger(self.h_sub.dot(self.action.apply(vector)))

    def sector_block_dense(self, idxs: Iterable[int]) -> np.ndarray:
        idxs_arr = np.asarray(list(idxs), dtype=np.int64)
        dim = self.shape[0]
        block = np.zeros((idxs_arr.size, idxs_arr.size), dtype=np.complex128)
        for col, index in enumerate(idxs_arr):
            vector = np.zeros(dim, dtype=np.complex128)
            vector[index] = 1.0
            block[:, col] = self.dot(vector)[idxs_arr]
        return 0.5 * (block + block.conj().T)


def _json_list(values: Iterable[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _parse_edges(value: str) -> list[tuple[int, int]]:
    return [tuple(int(part) for part in edge.split("-")) for edge in value.split()]


def _parse_pairs(value: str) -> list[tuple[int, int]]:
    return [(int(p), int(q)) for p, q in json.loads(value)]


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


def _mixed_pool_sectors(
    ref: dict,
    pool: MixedOperatorPool,
    *,
    allowed_indices: list[int] | None = None,
) -> dict[tuple[int, ...], list[int]]:
    sectors = mixed_pool_sectors(ref["basis_bitstrings"], pool, ref["n_spatial"])
    return _filter_sectors_to_indices(sectors, allowed_indices)


def _exact_symmetry_allowed_indices(ref: dict, *, use_exact_symmetries: bool = True) -> list[int] | None:
    if not use_exact_symmetries:
        return None
    payload = exact_symmetry_data_for_reference(ref)
    if payload is None:
        return None
    _labels, parities = payload
    return ground_state_sector_indices(parities, target=1)


def _filter_sectors_to_indices(
    sectors: dict[tuple[int, ...], list[int]],
    allowed: list[int] | None,
) -> dict[tuple[int, ...], list[int]]:
    if allowed is None:
        return sectors
    allowed_set = set(allowed)
    filtered: dict[tuple[int, ...], list[int]] = {}
    for key, indices in sectors.items():
        kept = [index for index in indices if index in allowed_set]
        if kept:
            filtered[key] = kept
    return filtered


def _rotation_pairs_for_ref(ref: dict, use_exact_symmetries: bool = True) -> list[tuple[int, int]]:
    n_spatial = int(ref["n_spatial"])
    if use_exact_symmetries:
        labels = load_symmetry_labels(ref)
        if labels is not None:
            return symmetry_blocked_pair_list(n_spatial, labels.irrep_labels)
    return pair_list_for_n(n_spatial)


def _quartet_sectors(
    basis_bitstrings: list[int],
    edges: Iterable[tuple[int, int]],
    n_spatial: int,
    *,
    allowed_indices: list[int] | None = None,
) -> dict[tuple[int, ...], list[int]]:
    edge_list = list(edges)
    sectors: dict[tuple[int, ...], list[int]] = {}
    allowed_set = set(allowed_indices) if allowed_indices is not None else None
    for index, bitstring in enumerate(basis_bitstrings):
        if allowed_set is not None and index not in allowed_set:
            continue
        key = tuple(
            int(quartet_parity_diagonal([int(bitstring)], edge, n_spatial)[0])
            for edge in edge_list
        )
        sectors.setdefault(key, []).append(index)
    return sectors


def _generalized_sectors_symmetry_restricted(
    ref: dict,
    a: float,
    b: float,
    c: float,
    *,
    use_exact_symmetries: bool = True,
) -> dict[tuple, list[int]]:
    from quasi_symmetries.diagnostics.energy_sectors import build_sectors_with_exact_symmetries

    payload = exact_symmetry_data_for_reference(ref) if use_exact_symmetries else None
    if payload is None:
        return build_generalized_sectors(
            ref["basis_bitstrings"],
            ref["n_spatial"],
            ref["n_qubits"],
            a,
            b,
            c,
        )
    _labels, parities = payload
    sectors, _ = build_sectors_with_exact_symmetries(
        ref["basis_bitstrings"],
        ref["n_spatial"],
        ref["n_qubits"],
        a,
        b,
        c,
        parities,
    )
    return sectors


def _energy_diagnostics(
    ref: dict,
    sectors: dict[tuple[int, ...], list[int]],
    action: OrbitalRotationAction | None,
    *,
    energy_tol: float = 1e-3,
    profile: bool = False,
    max_workers: int | None = None,
) -> dict[str, float | int | bool]:
    h_sub = ref["h_sub"]
    energy_fci = float(ref["energy_fci"])
    if action is None:
        h_op = SparseSubspaceHamiltonian(h_sub)
    else:
        rotate_start = time.perf_counter()
        h_rot = build_rotated_h_sub_csc(
            h_sub,
            action,
            max_workers=max_workers,
            profile=profile,
        )
        rotate_seconds = time.perf_counter() - rotate_start
        h_op = SparseSubspaceHamiltonian(h_rot)
    result = energy_sector_diagnostics_sparse(
        h_op,
        sectors,
        energy_fci,
        tol=energy_tol,
        max_workers=max_workers,
        profile=profile,
    )
    if action is not None:
        profile_data = result.get("_profile")
        if isinstance(profile_data, dict):
            profile_data["rotate_seconds"] = rotate_seconds
    return result


def _seniority_diagonal(
    basis_bitstrings: list[int],
    orbital: int,
    n_spatial: int,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    n_qubits = 2 * n_spatial
    values = []
    for bitstring in basis_bitstrings:
        occ_a = mode_is_occupied(int(bitstring), 2 * orbital, n_qubits)
        occ_b = mode_is_occupied(int(bitstring), 2 * orbital + 1, n_qubits)
        values.append(a * occ_a + b * occ_b + c * occ_a * occ_b)
    return np.asarray(values, dtype=np.float64)


def _coarse_entropy(weights_by_key: dict[tuple[float, ...], float]) -> float:
    weights = np.asarray([weight for weight in weights_by_key.values() if weight > 1e-15], dtype=float)
    return float(-np.sum(weights * np.log(weights))) if weights.size else 0.0


def _diagnose_diagonals(
    ref: dict,
    thetas: np.ndarray,
    diagonals: list[np.ndarray],
    sectors: dict[tuple[int, ...], list[int]],
    *,
    pairs: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    n_spatial = ref["n_spatial"]
    if pairs is None:
        pairs = _rotation_pairs_for_ref(ref)
    u_spatial = build_U_from_thetas(n_spatial, thetas, pairs)
    start = time.perf_counter()
    action = OrbitalRotationAction(u_spatial, ref["basis_bitstrings"], n_spatial)
    build_seconds = time.perf_counter() - start

    psi = np.asarray(ref["v_sub"], dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    psi_rot = action.apply_dagger(psi)
    weights = np.abs(psi_rot) ** 2
    h_sub = ref["h_sub"]
    energy = float(ref["energy_fci"])

    expectations: list[float] = []
    variances: list[float] = []
    comm_sq_values: list[float] = []
    sector_weights: dict[tuple[float, ...], float] = {}

    rounded_diagonals = [np.round(diagonal.astype(float), 12) for diagonal in diagonals]
    for idx in range(len(weights)):
        key = tuple(float(diagonal[idx]) for diagonal in rounded_diagonals)
        sector_weights[key] = sector_weights.get(key, 0.0) + float(weights[idx])

    for diagonal in diagonals:
        expectation = float(np.real(np.dot(weights, diagonal)))
        second = float(np.real(np.dot(weights, diagonal * diagonal)))
        expectations.append(expectation)
        variances.append(max(0.0, second - expectation * expectation))
        s_psi = action.apply(diagonal * psi_rot)
        residual = h_sub.dot(s_psi) - energy * s_psi
        comm_sq_values.append(float(np.real(np.vdot(residual, residual))))

    energy_start = time.perf_counter()
    profile_energy = os.environ.get("SPARSE_ENERGY_PROFILE", "").lower() in {"1", "true", "yes"}
    x_geom = float(ref.get("geometry_param", ref.get("x", float("nan"))))
    identity_energy = _energy_diagnostics(ref, sectors, action=None, profile=profile_energy)
    if profile_energy:
        id_profile = identity_energy.get("_profile", {})
        print(
            f"  [energy] diagonalize_id={id_profile.get('diagonalize_seconds', float('nan')):.1f}s "
            f"K_id={identity_energy.get('Kcoupled')} "
            f"Edec_id={identity_energy.get('Edec'):+.8f}",
            flush=True,
        )
    optimized_energy = _energy_diagnostics(ref, sectors, action=action, profile=profile_energy)
    energy_seconds = time.perf_counter() - energy_start
    if profile_energy:
        opt_profile = optimized_energy.get("_profile", {})
        print(
            f"  [energy] rotate_h={opt_profile.get('rotate_seconds', float('nan')):.1f}s "
            f"diagonalize_opt={opt_profile.get('diagonalize_seconds', float('nan')):.1f}s "
            f"K_opt={optimized_energy.get('Kcoupled')} "
            f"Edec_opt={optimized_energy.get('Edec'):+.8f} "
            f"total_energy={energy_seconds:.1f}s",
            flush=True,
        )

    return {
        "Build_Seconds": build_seconds,
        "Operator_Count": len(diagonals),
        "Sum_CommSq_Action": float(np.sum(comm_sq_values)),
        "Sum_Expectation_Action": float(np.sum(expectations)),
        "Sum_Variance_Action": float(np.sum(variances)),
        "Coarse_Entropy_Action": _coarse_entropy(sector_weights),
        "NumSectors_Action": len(sector_weights),
        "Expectations_Action_JSON": _json_list(expectations),
        "Variances_Action_JSON": _json_list(variances),
        "CommSq_Action_JSON": _json_list(comm_sq_values),
        "Energy_Seconds": energy_seconds,
        "Edec_Identity": identity_energy["Edec"],
        "Edec_Optimized": optimized_energy["Edec"],
        "Ecoupled_Identity": identity_energy["Ecoupled"],
        "Ecoupled_Optimized": optimized_energy["Ecoupled"],
        "Kcoupled_Identity": identity_energy["Kcoupled"],
        "Kcoupled_Optimized": optimized_energy["Kcoupled"],
        "Coupled_Converged_Identity": identity_energy["Coupled_Converged"],
        "Coupled_Converged_Optimized": optimized_energy["Coupled_Converged"],
    }


def run_fixed_abc(
    input_csv: Path = LEGACY_ABC_TABLES_DIR / "n2_quasi_symmetry_fixed_abc.csv",
    output_csv: Path = LEGACY_ABC_TABLES_DIR / "n2_fixed_abc_action_diagnostics.csv",
) -> None:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out_rows = []
    for row in rows:
        x = float(row["Geometry_Param"])
        print(f"[action] fixed_abc x={x}", flush=True)
        ref = load_reference_state("n2", x, cache_dir=str(CACHE_DIR))
        thetas = np.asarray(json.loads(row["Thetas"]), dtype=float)
        a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
        diagonals = [
            _seniority_diagonal(ref["basis_bitstrings"], i, ref["n_spatial"], a, b, c)
            for i in range(ref["n_spatial"])
        ]
        sectors = _generalized_sectors_symmetry_restricted(ref, a, b, c)
        allowed = _exact_symmetry_allowed_indices(ref)
        if allowed is not None:
            sectors = _filter_sectors_to_indices(sectors, allowed)
        out_rows.append({**row, **_diagnose_diagonals(ref, thetas, diagonals, sectors)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def run_quartets(
    input_csv: Path = TABLES_DIR / "n2_quartet_variance_summary.csv",
    output_csv: Path = TABLES_DIR / "n2_quartet_action_diagnostics.csv",
) -> None:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out_rows = []
    for row in rows:
        x = float(row["Geometry_Param"])
        baseline = row["Baseline"]
        print(f"[action] quartet {baseline} x={x}", flush=True)
        ref = load_reference_state("n2", x, cache_dir=str(CACHE_DIR))
        thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
        edges = _parse_edges(row["Edges"])
        diagonals = [
            quartet_parity_diagonal(ref["basis_bitstrings"], edge, ref["n_spatial"])
            for edge in edges
        ]
        allowed = _exact_symmetry_allowed_indices(ref)
        sectors = _quartet_sectors(
            ref["basis_bitstrings"],
            edges,
            ref["n_spatial"],
            allowed_indices=allowed,
        )
        out_rows.append({**row, **_diagnose_diagonals(ref, thetas, diagonals, sectors)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def run_mixed_pool(
    input_csv: Path = TABLES_DIR / "n2_mixed_pool_summary.csv",
    output_csv: Path = TABLES_DIR / "n2_mixed_pool_action_diagnostics.csv",
) -> None:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out_rows = []
    for row in rows:
        x = float(row["Geometry_Param"])
        print(f"[action] mixed_pool x={x}", flush=True)
        ref = load_reference_state("n2", x, cache_dir=str(CACHE_DIR), compute_rdms=False)
        pool = _parse_pool(row, ref["n_spatial"])
        thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
        pairs = _parse_pairs(row["Rotation_Pairs_JSON"])
        diagonals = mixed_pool_diagonals(ref["basis_bitstrings"], pool, ref["n_spatial"])
        allowed = _exact_symmetry_allowed_indices(ref)
        sectors = _mixed_pool_sectors(ref, pool, allowed_indices=allowed)
        out_rows.append(
            {**row, **_diagnose_diagonals(ref, thetas, diagonals, sectors, pairs=pairs)}
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[ok] wrote {len(out_rows)} rows to {output_csv}", flush=True)


def _diagnose_one_geometry(
    row: dict,
    *,
    profile: bool = False,
) -> dict[str, object]:
    """Run parity-seniority action diagnostics for one N2 geometry row."""
    x = float(row["Geometry_Param"])
    if profile:
        os.environ["SPARSE_ENERGY_PROFILE"] = "1"
    ref = load_reference_state("n2", x, cache_dir=str(CACHE_DIR), compute_rdms=False)
    n_spatial = ref["n_spatial"]
    pool = MixedOperatorPool(singles=tuple(range(n_spatial)), quartets=())
    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    pairs = _parse_pairs(row["Rotation_Pairs_JSON"])
    diagonals = mixed_pool_diagonals(ref["basis_bitstrings"], pool, n_spatial)
    allowed = _exact_symmetry_allowed_indices(ref)
    sectors = _mixed_pool_sectors(ref, pool, allowed_indices=allowed)
    return {**row, **_diagnose_diagonals(ref, thetas, diagonals, sectors, pairs=pairs)}


def run_parity_seniority(
    input_csv: Path = TABLES_DIR / "n2_parity_seniority_summary.csv",
    output_csv: Path = TABLES_DIR / "n2_parity_seniority_action_diagnostics.csv",
) -> None:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out_rows = []
    for row in rows:
        x = float(row["Geometry_Param"])
        print(f"[action] parity_seniority x={x}", flush=True)
        out_rows.append(_diagnose_one_geometry(row))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[ok] wrote {len(out_rows)} rows to {output_csv}", flush=True)


def benchmark_first_quartet() -> None:
    with (TABLES_DIR / "n2_quartet_variance_summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    x = float(row["Geometry_Param"])
    ref = load_reference_state("n2", x, cache_dir=str(CACHE_DIR))
    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    edges = _parse_edges(row["Edges"])
    diagonals = [
        quartet_parity_diagonal(ref["basis_bitstrings"], edge, ref["n_spatial"])
        for edge in edges
    ]
    sectors = _quartet_sectors(ref["basis_bitstrings"], edges, ref["n_spatial"])
    start = time.perf_counter()
    result = _diagnose_diagonals(ref, thetas, diagonals, sectors)
    elapsed = time.perf_counter() - start
    print(json.dumps({"Elapsed_Seconds": elapsed, **result}, indent=2), flush=True)


def benchmark_first_mixed_pool() -> None:
    with (TABLES_DIR / "n2_mixed_pool_summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    x = float(row["Geometry_Param"])
    ref = load_reference_state("n2", x, cache_dir=str(CACHE_DIR), compute_rdms=False)
    pool = _parse_pool(row, ref["n_spatial"])
    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    pairs = _parse_pairs(row["Rotation_Pairs_JSON"])
    diagonals = mixed_pool_diagonals(ref["basis_bitstrings"], pool, ref["n_spatial"])
    allowed = _exact_symmetry_allowed_indices(ref)
    sectors = _mixed_pool_sectors(ref, pool, allowed_indices=allowed)
    max_sector = max(len(idxs) for idxs in sectors.values())
    start = time.perf_counter()
    result = _diagnose_diagonals(ref, thetas, diagonals, sectors, pairs=pairs)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "Geometry_Param": x,
                "NumSectors": len(sectors),
                "MaxSectorDim": max_sector,
                "Elapsed_Seconds": elapsed,
                **result,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run_fixed_abc()
    run_quartets()
    run_mixed_pool()
