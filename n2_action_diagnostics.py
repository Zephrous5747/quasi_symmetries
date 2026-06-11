"""Sparse action-based diagnostics for large N2 quasi-symmetry tables.

This avoids dense many-body rotation and dense Hamiltonian construction.  It
computes expectation, variance, coarse entropy, and state-specific commutator
norms from precomputed optimized angles.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from hamiltonian_cache import load_reference_state
from optimization_abc_utils import (
    SparseSubspaceHamiltonian,
    build_U_from_thetas,
    build_generalized_sectors,
    energy_sector_diagnostics_sparse,
    mode_is_occupied,
    pair_list_for_n,
)
from quartet_optimization_utils import (
    _determinant_transform_matrix,
    _spin_occupation_data,
    quartet_parity_diagonal,
)


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


def _energy_diagnostics(
    ref: dict,
    sectors: dict[tuple[int, ...], list[int]],
    action: OrbitalRotationAction | None,
    *,
    energy_tol: float = 1e-3,
) -> dict[str, float | int | bool]:
    h_sub = ref["h_sub"]
    energy_fci = float(ref["energy_fci"])
    lazy = action is not None
    if action is None:
        h_op = SparseSubspaceHamiltonian(h_sub)
    else:
        h_op = RotatedHamiltonian(h_sub, action)
    return energy_sector_diagnostics_sparse(
        h_op,
        sectors,
        energy_fci,
        tol=energy_tol,
        lazy=lazy,
    )


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
) -> dict[str, object]:
    n_spatial = ref["n_spatial"]
    pairs = pair_list_for_n(n_spatial)
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
    identity_energy = _energy_diagnostics(ref, sectors, action=None)
    optimized_energy = _energy_diagnostics(ref, sectors, action=action)
    energy_seconds = time.perf_counter() - energy_start

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
    input_csv: Path = Path("tables/n2_quasi_symmetry_fixed_abc.csv"),
    output_csv: Path = Path("tables/n2_fixed_abc_action_diagnostics.csv"),
) -> None:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out_rows = []
    for row in rows:
        x = float(row["Geometry_Param"])
        print(f"[action] fixed_abc x={x}", flush=True)
        ref = load_reference_state("n2", x, cache_dir="hamiltonian_cache")
        thetas = np.asarray(json.loads(row["Thetas"]), dtype=float)
        a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
        diagonals = [
            _seniority_diagonal(ref["basis_bitstrings"], i, ref["n_spatial"], a, b, c)
            for i in range(ref["n_spatial"])
        ]
        sectors = build_generalized_sectors(
            ref["basis_bitstrings"],
            ref["n_spatial"],
            ref["n_qubits"],
            a,
            b,
            c,
        )
        out_rows.append({**row, **_diagnose_diagonals(ref, thetas, diagonals, sectors)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def run_quartets(
    input_csv: Path = Path("tables/n2_quartet_variance_summary.csv"),
    output_csv: Path = Path("tables/n2_quartet_action_diagnostics.csv"),
) -> None:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out_rows = []
    for row in rows:
        x = float(row["Geometry_Param"])
        baseline = row["Baseline"]
        print(f"[action] quartet {baseline} x={x}", flush=True)
        ref = load_reference_state("n2", x, cache_dir="hamiltonian_cache")
        thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
        edges = _parse_edges(row["Edges"])
        diagonals = [
            quartet_parity_diagonal(ref["basis_bitstrings"], edge, ref["n_spatial"])
            for edge in edges
        ]
        sectors = _quartet_sectors(ref["basis_bitstrings"], edges, ref["n_spatial"])
        out_rows.append({**row, **_diagnose_diagonals(ref, thetas, diagonals, sectors)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def benchmark_first_quartet() -> None:
    with Path("tables/n2_quartet_variance_summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    x = float(row["Geometry_Param"])
    ref = load_reference_state("n2", x, cache_dir="hamiltonian_cache")
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


if __name__ == "__main__":
    run_fixed_abc()
    run_quartets()
