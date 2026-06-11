"""Rotated fermion operators and qubit matrices."""

from __future__ import annotations

import numpy as np
from openfermion import FermionOperator, get_sparse_operator, jordan_wigner, normal_ordered

from quasi_symmetries.config import OP_COEF_TOL

def rotated_number_operator_fermion(U_spatial, i_spatial, spin_offset, n_spatial, tol=1e-12):
    op = FermionOperator()
    for p in range(n_spatial):
        for q in range(n_spatial):
            coef = np.conjugate(U_spatial[p, i_spatial]) * U_spatial[q, i_spatial]
            if abs(coef) <= tol:
                continue
            p_mode = 2 * p + spin_offset
            q_mode = 2 * q + spin_offset
            op += FermionOperator(((p_mode, 1), (q_mode, 0)), coef)
    return op
def rotated_seniority_orbital_fermion(U_spatial, i_spatial, n_spatial, a, b, c, tol=1e-12):
    n_a = rotated_number_operator_fermion(U_spatial, i_spatial, spin_offset=0, n_spatial=n_spatial, tol=tol)
    n_b = rotated_number_operator_fermion(U_spatial, i_spatial, spin_offset=1, n_spatial=n_spatial, tol=tol)

    # Generalized operator
    omega = normal_ordered(a * n_a + b * n_b + c * (n_a * n_b))
    return omega
def fermion_to_sparse_qubit(op_fermion, n_qubits): # in qubit matrix
    op_qubit = jordan_wigner(op_fermion)
    return get_sparse_operator(op_qubit, n_qubits).tocsc()
def comm_expect_comm_sq_abs(H_mat, S_mat, psi):
  #evaluate non commutativity
    Apsi = H_mat.dot(S_mat.dot(psi)) - S_mat.dot(H_mat.dot(psi))
    A2psi = H_mat.dot(S_mat.dot(Apsi)) - S_mat.dot(H_mat.dot(Apsi)) #eas
    exp = np.vdot(psi, A2psi)  # should be real if H,S Hermitian (numerical imag ~0)
    exp = np.vdot(psi, A2psi)

    norm2 = np.real(np.vdot(Apsi, Apsi))
    assert np.allclose(exp, -norm2, atol=1e-10), "Not equal to <Apsi|Apsi>"

    Spsi = S_mat.dot(psi)
    HSpsi = H_mat.dot(Spsi)
    E0 = np.vdot(psi, H_mat.dot(psi))   # energy expectation; exact eigenvalue if psi is FCI eigenstate

    norm3 = (
        np.vdot(HSpsi, HSpsi)
        - 2 * E0 * np.vdot(Spsi, HSpsi)
        + E0**2 * np.vdot(Spsi, Spsi)
    )
    assert np.allclose(np.real(norm2), np.real(norm3), atol=1e-10), "Not equal to Expanded"
    return float(abs(exp)), exp
def build_total_operator(U_spatial, n_spatial, a, b, c, tol=1e-12):
    S_ferm = FermionOperator()
    for i in range(n_spatial):
        S_ferm += rotated_seniority_orbital_fermion(
            U_spatial, i, n_spatial, a, b, c, tol=tol
        )
    return normal_ordered(S_ferm)
