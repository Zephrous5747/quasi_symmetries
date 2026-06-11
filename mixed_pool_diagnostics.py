"""Energy-sector diagnostics for fixed mixed seniority + quartet operator pools."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp

from optimization_abc_utils import (
    bo_like_coupled_energy_test,
    comm_state_norm_sq,
    coupled_energy_test,
    decoupled_energy_test,
    diagonalize_sector_blocks,
    shannon_block_decomposition,
)
from quartet_optimization_utils import (
    MixedOperatorPool,
    orbital_rotation_representation_R_fast,
    quartet_parity_diagonal,
    single_parity_diagonal,
)


def mixed_pool_diagonals(
    basis_bitstrings: list[int],
    pool: MixedOperatorPool,
    n_spatial: int,
) -> list[np.ndarray]:
    pool.validate(n_spatial)
    diagonals = [
        single_parity_diagonal(basis_bitstrings, orbital, n_spatial)
        for orbital in pool.singles
    ]
    diagonals.extend(
        quartet_parity_diagonal(basis_bitstrings, edge, n_spatial)
        for edge in pool.quartets
    )
    return diagonals


def mixed_pool_sectors(
    basis_bitstrings: list[int],
    pool: MixedOperatorPool,
    n_spatial: int,
) -> dict[tuple[int, ...], list[int]]:
    diagonals = mixed_pool_diagonals(basis_bitstrings, pool, n_spatial)
    sectors: dict[tuple[int, ...], list[int]] = {}
    for index, bitstring in enumerate(basis_bitstrings):
        key = tuple(int(diagonal[index]) for diagonal in diagonals)
        sectors.setdefault(key, []).append(index)
    return sectors


def _mixed_pool_commutativity(
    h_mat: Any,
    psi: np.ndarray,
    basis_bitstrings: list[int],
    pool: MixedOperatorPool,
    n_spatial: int,
) -> tuple[float, float]:
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    diagonals = mixed_pool_diagonals(basis_bitstrings, pool, n_spatial)
    dim = len(basis_bitstrings)
    sum_comm_sq = 0.0
    sum_exp = 0.0
    for diagonal in diagonals:
        op = sp.diags(diagonal, offsets=0, shape=(dim, dim), format="csc")
        sum_exp += float(np.real(np.vdot(psi, op.dot(psi))))
        comm_sq, _ = comm_state_norm_sq(h_mat, op, psi, check_eigenstate=False)
        sum_comm_sq += comm_sq
    return float(sum_comm_sq), float(sum_exp)


def _mixed_pool_entropy_and_energy(
    h_dense: np.ndarray,
    psi: np.ndarray,
    basis_bitstrings: list[int],
    pool: MixedOperatorPool,
    n_spatial: int,
    energy_fci: float,
) -> dict[str, Any]:
    sectors = mixed_pool_sectors(basis_bitstrings, pool, n_spatial)
    entropy_fine, entropy_coarse, _ = shannon_block_decomposition(h_dense, psi, sectors)
    sector_data = diagonalize_sector_blocks(h_dense, sectors)
    e_dec_min, _, _ = decoupled_energy_test(h_dense, sectors)
    e_coupled, k_coupled, _, _ = coupled_energy_test(
        h_dense, sector_data, E_exact=energy_fci, tol=1e-3
    )
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


def mixed_pool_energy_indicators(
    ref: dict[str, Any],
    pool: MixedOperatorPool,
    u_optimized: np.ndarray,
) -> dict[str, Any]:
    """Identity vs optimized energy indicators for a mixed operator pool."""
    n_spatial = ref["n_spatial"]
    basis_bitstrings = ref["basis_bitstrings"]
    psi_identity = ref["v_sub"] / np.linalg.norm(ref["v_sub"])
    h_identity_sparse = ref["h_sub"]

    comm_id, sexp_id = _mixed_pool_commutativity(
        h_identity_sparse, psi_identity, basis_bitstrings, pool, n_spatial
    )

    r_opt = orbital_rotation_representation_R_fast(
        u_optimized, basis_bitstrings, n_spatial
    )
    psi_optimized = r_opt.conj().T @ psi_identity
    h_optimized_sparse = r_opt.conj().T @ (h_identity_sparse @ r_opt)
    h_optimized_sparse = sp.csc_matrix(0.5 * (h_optimized_sparse + h_optimized_sparse.conj().T))
    comm_opt, sexp_opt = _mixed_pool_commutativity(
        h_optimized_sparse, psi_optimized, basis_bitstrings, pool, n_spatial
    )

    if not ref.get("use_dense", False):
        nan = float("nan")
        return {
            "Sum_CommSq_Identity": comm_id,
            "Sum_CommSq_Optimized": comm_opt,
            "Sum_Sexp_Identity": sexp_id,
            "Sum_Sexp_Optimized": sexp_opt,
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
            "Operator_Count": len(pool.singles) + len(pool.quartets),
        }

    h_identity = ref["h_sub"].toarray().astype(np.complex128)
    h_identity = 0.5 * (h_identity + h_identity.conj().T)
    h_optimized = np.asarray(h_optimized_sparse.toarray(), dtype=np.complex128)
    identity_post = _mixed_pool_entropy_and_energy(
        h_identity, psi_identity, basis_bitstrings, pool, n_spatial, ref["energy_fci"]
    )
    optimized_post = _mixed_pool_entropy_and_energy(
        h_optimized, psi_optimized, basis_bitstrings, pool, n_spatial, ref["energy_fci"]
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
        "Operator_Count": len(pool.singles) + len(pool.quartets),
    }
