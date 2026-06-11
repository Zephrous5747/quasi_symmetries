"""Spatial orbital rotations and many-body representation R."""

from __future__ import annotations

import numpy as np
from openfermion import FermionOperator
from scipy.linalg import expm, logm

from quasi_symmetries.config import OP_COEF_TOL
from quasi_symmetries.fermion.bitstring import mode_is_occupied, omega_mask_from_bitstring
from quasi_symmetries.fermion.operators import (
    fermion_to_sparse_qubit,
    rotated_seniority_orbital_fermion,
)

def givens(n, p, q, theta):
    G = np.eye(n)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    G[p, p] = c
    G[q, q] = c
    G[p, q] = s
    G[q, p] = -s
    return G
def pair_list_for_n(n): #order of givens
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def symmetry_blocked_pair_list(n_spatial: int, irrep_labels) -> list[tuple[int, int]]:
    """Rotation pairs that preserve symmetry-adapted orbital blocks."""
    from quasi_symmetries.symmetry.labels import symmetry_adapted_pair_list

    return symmetry_adapted_pair_list(n_spatial, irrep_labels)


def build_U_from_thetas_symmetry_blocked(
    n_spatial: int,
    thetas,
    irrep_labels,
    pairs: list[tuple[int, int]] | None = None,
):
    """Build a spatial unitary using only intra-irrep Givens rotations."""
    blocked_pairs = pairs or symmetry_blocked_pair_list(n_spatial, irrep_labels)
    return build_U_from_thetas(n_spatial, thetas, blocked_pairs)


def build_U_from_thetas(n, thetas, pairs): #U size = n_spatial x n_spatial
    U = np.eye(n)
    for th, (p, q) in zip(thetas, pairs):
        U = U @ givens(n, p, q, th)
    return U
def orbital_rotation_representation_R(U_spatial, basis_bitstrings, n_spatial, tol=1e-12):
    n_qubits = 2 * n_spatial
    idx = np.asarray(basis_bitstrings, dtype=int)

    U_spatial = np.asarray(U_spatial, dtype=np.complex128)
    K = logm(U_spatial)
    K = 0.5 * (K - K.conj().T)  # enforce anti-Hermitian numerically

    # 2) Lift K to the fermionic generator κ = sum_{pqσ} K_pq a†_{pσ} a_{qσ}
    kappa = FermionOperator()
    for p in range(n_spatial):
        for q in range(n_spatial):
            coef = K[p, q]
            if abs(coef) <= tol:
                continue
            # alpha
            p_a = 2 * p
            q_a = 2 * q
            kappa += FermionOperator(((p_a, 1), (q_a, 0)), coef)
            # beta
            p_b = 2 * p + 1
            q_b = 2 * q + 1
            kappa += FermionOperator(((p_b, 1), (q_b, 0)), coef)

    # 3) Matrix of κ on the full Fock space, then restrict to the fixed-N subspace
    kappa_mat_full = fermion_to_sparse_qubit(kappa, n_qubits)
    kappa_sub = kappa_mat_full[idx, :][:, idx].toarray().astype(np.complex128)

    # 4) Exponentiate on the fixed-N subspace
    R_sub = expm(kappa_sub)

    return R_sub
def build_seniority_sectors(basis_bitstrings, n_spatial: int): #partition in2 sectors
    sectors = {}
    for k, b in enumerate(basis_bitstrings):
        m = omega_mask_from_bitstring(int(b), n_spatial)
        sectors.setdefault(m, []).append(k)
    return sectors
def sector_weights_from_vec(vec, sectors): #get weights for print out
    # vec is in the SAME basis ordering as basis_bitstrings / H_sub
    w = {}
    for m, idxs in sectors.items():
        w[m] = float(np.sum(np.abs(vec[idxs])**2))
    return w
def direct_seniority_variance_check(U_spatial, psi, n_spatial, n_qubits, a, b, c, label=""):
    print(f"\n=== Direct Ω_i(U) variance check: {label} ===")

    total_var = 0.0
    total_exp = 0.0

    for i in range(n_spatial):
        Si_ferm = rotated_seniority_orbital_fermion(U_spatial, i, n_spatial, a, b, c, tol=OP_COEF_TOL)
        Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)

        Spsi = Si_mat.dot(psi)
        S2psi = Si_mat.dot(Spsi)

        exp1 = np.vdot(psi, Spsi)
        exp2 = np.vdot(psi, S2psi)

        exp1_r = float(np.real_if_close(exp1))
        exp2_r = float(np.real_if_close(exp2))
        var_i = exp2_r - exp1_r**2

        # for an exact projector, exp2 should equal exp1
        proj_defect = exp2_r - exp1_r

        total_exp += exp1_r
        total_var += var_i

        print(
            f"i={i:2d}   <Ω_i>={exp1_r:+.12f}   "
            f"<Ω_i^2>={exp2_r:+.12f}   "
            f"Var={var_i:+.12e}   "
            f"(<Ω_i^2>-<Ω_i>)={proj_defect:+.3e}"
        )

    print(f"sum_i <Ω_i>   = {total_exp:+.12f}")
    print(f"sum_i Var(Ω_i)= {total_var:+.12e}")
def check_R_vs_direct_seniority(U_spatial, basis_bitstrings, n_spatial, n_qubits, a, b, c, psi_old=None):
    """
    Compare the many-body rotation matrix R against the direct rotated seniority operators.

    In the determinant basis used by basis_bitstrings / H_sub:
      - D_total is the diagonal total-seniority operator in the rotated determinant basis
      - R maps rotated determinant basis -> old determinant basis
      - so the old-basis operator predicted by R is: S_via_R = R D_total R^dagger

    This should match the direct old-basis matrix built from the fermion operators:
      S_direct = sum_i Ω_i(U)
    restricted to the same subspace ordering as basis_bitstrings.
    """

    idx = np.asarray(basis_bitstrings, dtype=int)
    dim_sub = len(idx)

    # --- build R in the same subspace basis ordering ---
    R = orbital_rotation_representation_R(U_spatial, basis_bitstrings, n_spatial)

    # --- diagonal total seniority operator D in the rotated determinant basis ---
    sen_diag = np.zeros(dim_sub, dtype=np.float64)
    D_orb_list = []
    for i in range(n_spatial):
        d_i = np.zeros(dim_sub, dtype=np.float64)
        for k, bit_str in enumerate(basis_bitstrings):
            oa = mode_is_occupied(int(bit_str), 2 * i, n_qubits)
            ob = mode_is_occupied(int(bit_str), 2 * i + 1, n_qubits)

            # The exact eigenvalue of the parameterized operator for this determinant
            d_i[k] = float(a * oa + b * ob + c * (oa * ob))
            sen_diag[k] += d_i[k]
        D_orb_list.append(np.diag(d_i))

    D_total = np.diag(sen_diag)

    # --- operator predicted by R, expressed back in the old basis ---
    S_via_R_total = R @ D_total @ R.conj().T
    S_via_R_orb = [R @ D_i @ R.conj().T for D_i in D_orb_list]

    # --- direct rotated operators, restricted to the same subspace/order ---
    S_direct_total = np.zeros((dim_sub, dim_sub), dtype=np.complex128)
    S_direct_orb = []

    for i in range(n_spatial):
        Si_ferm = rotated_seniority_orbital_fermion(U_spatial, i, n_spatial, a, b, c, tol=OP_COEF_TOL)
        Si_full = fermion_to_sparse_qubit(Si_ferm, n_qubits)
        Si_sub = Si_full[idx, :][:, idx].toarray().astype(np.complex128)
        S_direct_orb.append(Si_sub)
        S_direct_total += Si_sub

    # --- main matrix checks ---
    diff_total = S_via_R_total - S_direct_total
    fro_total = np.linalg.norm(diff_total)
    rel_total = fro_total / max(np.linalg.norm(S_direct_total), 1e-15)

    print("\n=== Check: R D R^dag vs direct rotated seniority operator ===")
    print(f"||R^dag R - I||_F                      = {np.linalg.norm(R.conj().T @ R - np.eye(dim_sub)):.6e}")
    print(f"||R D_total R^dag - S_direct_total||_F = {fro_total:.6e}")
    print(f"relative total mismatch            = {rel_total:.6e}")

    # Equivalent check in the rotated basis: R† S_direct R should be diagonal and equal to D_total
    backrot_total = R.conj().T @ S_direct_total @ R
    diag_mismatch = np.linalg.norm(backrot_total - D_total)
    offdiag_only = backrot_total - np.diag(np.diag(backrot_total))
    print(f"||R^dag S_direct_total R - D_total||_F = {diag_mismatch:.6e}")
    print(f"offdiag norm of R^dag S_direct R         = {np.linalg.norm(offdiag_only):.6e}")

    # --- per-orbital checks (very useful for spotting exactly where it fails) ---
    print("\nPer-orbital Ω_i checks:")
    for i in range(n_spatial):
        dmat = S_via_R_orb[i] - S_direct_orb[i]
        fro_i = np.linalg.norm(dmat)
        rel_i = fro_i / max(np.linalg.norm(S_direct_orb[i]), 1e-15)

        backrot_i = R.conj().T @ S_direct_orb[i] @ R
        diag_target_i = D_orb_list[i]
        offdiag_i = backrot_i - np.diag(np.diag(backrot_i))
        diag_err_i = np.linalg.norm(backrot_i - diag_target_i)

        print(
            f"  i={i:2d}  "
            f"||R D_i R^dag - Omega_i(U)||_F = {fro_i:.6e}   "
            f"rel = {rel_i:.6e}   "
            f"||R^dag Omega_i R - D_i||_F = {diag_err_i:.6e}   "
            f"offdiag = {np.linalg.norm(offdiag_i):.6e}"
        )

    # --- optional state expectation check in the old basis ---
    if psi_old is not None:
        psi = np.asarray(psi_old, dtype=np.complex128)
        psi = psi / np.linalg.norm(psi)

        exp_via_R = np.vdot(psi, S_via_R_total @ psi)
        exp_direct = np.vdot(psi, S_direct_total @ psi)

        print("\nExpectation on supplied state (old basis):")
        print(f"<psi|R D_total R^dag|psi>             = {float(np.real_if_close(exp_via_R)):.12f}")
        print(f"<psi|S_direct_total|psi>           = {float(np.real_if_close(exp_direct)):.12f}")
        print(f"difference                         = {float(np.real_if_close(exp_via_R - exp_direct)):.6e}")

    return {
        "R": R,
        "D_total": D_total,
        "S_via_R_total": S_via_R_total,
        "S_direct_total": S_direct_total,
        "fro_total": fro_total,
        "rel_total": rel_total,
    }
