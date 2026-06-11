"""Spin reduced density matrices from statevectors."""

from __future__ import annotations

import numpy as np

from quasi_symmetries.fermion.bitstring import apply_annihilate, apply_create

def compute_spin_rdms_from_statevector(statevec, n_spatial):
    n_qubits = 2 * n_spatial
    dim = 1 << n_qubits
    if statevec.shape[0] != dim:
        raise ValueError("state dim doesn't match")

    psi = statevec
    gamma_a = np.zeros((n_spatial, n_spatial), dtype=np.complex128)
    gamma_b = np.zeros((n_spatial, n_spatial), dtype=np.complex128)
    Gamma_ab = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=np.complex128)

    nz = np.nonzero(np.abs(psi) > 0)[0]

    def fill_gamma(gamma, spin_offset):
        for q in range(n_spatial):
            q_mode = 2 * q + spin_offset
            for x in nz:
                amp_x = psi[x]
                x1, s1 = apply_annihilate(int(x), q_mode, n_qubits)
                if x1 is None:
                    continue
                for p in range(n_spatial):
                    p_mode = 2 * p + spin_offset
                    x2, s2 = apply_create(x1, p_mode, n_qubits)
                    if x2 is None:
                        continue
                    gamma[p, q] += np.conjugate(psi[x2]) * amp_x * (s1 * s2)

    fill_gamma(gamma_a, 0)
    fill_gamma(gamma_b, 1)

    for p in range(n_spatial):
        p_mode = 2 * p
        for q in range(n_spatial):
            q_mode = 2 * q + 1
            for r in range(n_spatial):
                r_mode = 2 * r
                for s in range(n_spatial):
                    s_mode = 2 * s + 1
                    val = 0.0 + 0.0j
                    for x in nz:
                        amp_x = psi[x]
                        x1, sr = apply_annihilate(int(x), r_mode, n_qubits)
                        if x1 is None:
                            continue
                        x2, ss = apply_annihilate(x1, s_mode, n_qubits)
                        if x2 is None:
                            continue
                        x3, sq = apply_create(x2, q_mode, n_qubits)
                        if x3 is None:
                            continue
                        x4, sp_ = apply_create(x3, p_mode, n_qubits)
                        if x4 is None:
                            continue
                        val += np.conjugate(psi[x4]) * amp_x * (sr * ss * sq * sp_)
                    Gamma_ab[p, q, r, s] = val

    return gamma_a, gamma_b, Gamma_ab
def compute_spin_rdms_from_subspace_state(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spin RDMs from a fixed-N state without embedding in the full Fock space."""
    n_qubits = 2 * n_spatial
    coeffs = np.asarray(v_sub, dtype=np.complex128)
    det_to_amp = {int(b): coeffs[i] for i, b in enumerate(basis_bitstrings)}

    gamma_a = np.zeros((n_spatial, n_spatial), dtype=np.complex128)
    gamma_b = np.zeros((n_spatial, n_spatial), dtype=np.complex128)
    Gamma_ab = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=np.complex128)

    nz = [(b, det_to_amp[b]) for b in basis_bitstrings if abs(det_to_amp[b]) > 0]

    def fill_gamma(gamma, spin_offset):
        for q in range(n_spatial):
            q_mode = 2 * q + spin_offset
            for x, amp_x in nz:
                x1, s1 = apply_annihilate(int(x), q_mode, n_qubits)
                if x1 is None:
                    continue
                for p in range(n_spatial):
                    p_mode = 2 * p + spin_offset
                    x2, s2 = apply_create(x1, p_mode, n_qubits)
                    if x2 is None:
                        continue
                    amp_x2 = det_to_amp.get(x2)
                    if amp_x2 is None:
                        continue
                    gamma[p, q] += np.conjugate(amp_x2) * amp_x * (s1 * s2)

    fill_gamma(gamma_a, 0)
    fill_gamma(gamma_b, 1)

    for p in range(n_spatial):
        p_mode = 2 * p
        for q in range(n_spatial):
            q_mode = 2 * q + 1
            for r in range(n_spatial):
                r_mode = 2 * r
                for s in range(n_spatial):
                    s_mode = 2 * s + 1
                    val = 0.0 + 0.0j
                    for x, amp_x in nz:
                        x1, sr = apply_annihilate(int(x), r_mode, n_qubits)
                        if x1 is None:
                            continue
                        x2, ss = apply_annihilate(x1, s_mode, n_qubits)
                        if x2 is None:
                            continue
                        x3, sq = apply_create(x2, q_mode, n_qubits)
                        if x3 is None:
                            continue
                        x4, sp_ = apply_create(x3, p_mode, n_qubits)
                        if x4 is None:
                            continue
                        amp_x4 = det_to_amp.get(x4)
                        if amp_x4 is None:
                            continue
                        val += np.conjugate(amp_x4) * amp_x * (sr * ss * sq * sp_)
                    Gamma_ab[p, q, r, s] = val

    return gamma_a, gamma_b, Gamma_ab
