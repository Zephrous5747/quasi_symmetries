"""Local-ABC parameterization and operators."""

from __future__ import annotations

import numpy as np
from openfermion import FermionOperator, normal_ordered
from scipy.optimize import minimize

from quasi_symmetries.config import (
    ANGLE_INIT_SCALE,
    MAXITER,
    N_RESTARTS,
    OP_COEF_TOL,
    OPT_METHOD,
    RANDOM_SEED,
)
from quasi_symmetries.optimization.rotations import build_U_from_thetas, pair_list_for_n
from quasi_symmetries.optimization.variance import OptLog
from quasi_symmetries.config import EVAL_STATE_SPECIFIC_COMMUTATIVITY
from quasi_symmetries.diagnostics.energy_sectors import (
    analyze_single_operator_leakage,
    comm_state_norm_sq,
    solve_cisd_state,
)
from quasi_symmetries.fermion.bitstring import closed_shell_hf_bitstring, mode_is_occupied, popcount
from quasi_symmetries.fermion.subspace import restrict_operator_to_subspace
from quasi_symmetries.fermion.operators import (
    fermion_to_sparse_qubit,
    rotated_seniority_orbital_fermion,
)

def unpack_local_abc_params(x_params, n, m):
    """
    x_params layout:
      [theta_orbital_rotations (m entries),
       phi1_0, phi2_0,
       phi1_1, phi2_1,
       ...
       phi1_{n-1}, phi2_{n-1}]
    """
    thetas = x_params[:m]
    local_abcs = []

    offset = m
    for i in range(n):
        phi1 = x_params[offset + 2*i]
        phi2 = x_params[offset + 2*i + 1]

        a_i = np.sin(phi1) * np.cos(phi2)
        b_i = np.sin(phi1) * np.sin(phi2)
        c_i = np.cos(phi1)

        local_abcs.append((a_i, b_i, c_i))

    return thetas, local_abcs
def variance_restricted_local_abc(gamma_a, gamma_b, Gamma_ab, x_params, pairs):
    """
    Optimize
        sum_i Var(S_i)
    where each S_i has its own normalized (a_i, b_i, c_i),
    while U is shared.
    """
    n = gamma_a.shape[0]
    m = len(pairs)

    thetas, local_abcs = unpack_local_abc_params(x_params, n, m)

    U = build_U_from_thetas(n, thetas, pairs)
    Ua = U.T @ gamma_a @ U
    Ub = U.T @ gamma_b @ U

    exp_vals = np.zeros(n, dtype=float)
    V_total = 0.0

    for i in range(n):
        a_i, b_i, c_i = local_abcs[i]

        u = U[:, i]
        G_i = np.einsum("p,q,r,s,pqrs->", u, u, u, u, Gamma_ab, optimize=True).real
        N_a = Ua[i, i].real
        N_b = Ub[i, i].real

        exp_omega = a_i * N_a + b_i * N_b + c_i * G_i
        exp_omega_sq = (
            a_i**2 * N_a
            + b_i**2 * N_b
            + (2*a_i*b_i + 2*a_i*c_i + 2*b_i*c_i + c_i**2) * G_i
        )

        exp_vals[i] = exp_omega
        V_total += float(exp_omega_sq - exp_omega**2)

    return V_total, exp_vals, U, local_abcs
def optimize_variance_restricted_local_abc(gamma_a, gamma_b, Gamma_ab, pairs=None):
    np.random.seed(RANDOM_SEED)
    n = gamma_a.shape[0]
    pairs = pair_list_for_n(n) if pairs is None else list(pairs)
    m = len(pairs)

    num_params = m + 2 * n

    def obj(x):
        V, _, _, _ = variance_restricted_local_abc(
            gamma_a, gamma_b, Gamma_ab, x, pairs
        )
        return V

    best = None
    for r in range(N_RESTARTS):
        x0 = np.zeros(num_params)

        if r == 0:
            # initialize every orbital near normalized 1:1:-2
            for i in range(n):
                x0[m + 2*i] = np.arccos(-2.0 / np.sqrt(6.0))  # phi1_i
                x0[m + 2*i + 1] = np.pi / 4.0                 # phi2_i
        else:
            x0[:m] = ANGLE_INIT_SCALE * np.random.randn(m)
            for i in range(n):
                x0[m + 2*i] = np.random.uniform(0, np.pi)
                x0[m + 2*i + 1] = np.random.uniform(0, 2*np.pi)

        log = OptLog(V=[], nOmega=[], x=[])

        def callback(xk):
            V, nO, _, _ = variance_restricted_local_abc(
                gamma_a, gamma_b, Gamma_ab, xk, pairs
            )
            log.V.append(V)
            log.nOmega.append(nO)
            log.x.append(np.array(xk, copy=True))

        if OPT_METHOD.upper() == "POWELL":
            res = minimize(
                obj,
                x0=x0,
                method="Powell",
                options={"maxiter": MAXITER, "disp": False}
            )
            callback(res.x)
        else:
            res = minimize(
                obj,
                x0=x0,
                method=OPT_METHOD,
                options={"maxiter": MAXITER, "disp": False},
                callback=callback
            )

        V_fin = obj(res.x)
        if best is None or V_fin < best["V"]:
            best = {
                "res": res,
                "log": log,
                "V": V_fin,
                "pairs": pairs,
            }

    return best
def omega_eigenvalue_on_orbital_local(oa: int, ob: int, a_i: float, b_i: float, c_i: float) -> float:
    return float(a_i * oa + b_i * ob + c_i * (oa * ob))
def omega_eigenvalues_from_bitstring_local_abc(
    bitstring: int,
    n_spatial: int,
    n_qubits: int,
    local_abcs,
    tol_decimals: int = 8,
) -> tuple[float, ...]:
    """Joint eigenvalue vector (omega_0, ..., omega_{n-1}) with orbital-dependent (a_i, b_i, c_i)."""
    values = []
    for i in range(n_spatial):
        a_i, b_i, c_i = local_abcs[i]
        oa = mode_is_occupied(bitstring, 2 * i, n_qubits)
        ob = mode_is_occupied(bitstring, 2 * i + 1, n_qubits)
        values.append(round(omega_eigenvalue_on_orbital_local(oa, ob, a_i, b_i, c_i), tol_decimals))
    return tuple(values)
def generalized_eigenvalue_from_bitstring_local_abc(bitstring, n_spatial, n_qubits, local_abcs):
    """Sum eigenvalue of sum_i Omega_i^(i) (diagnostic only)."""
    eigval = 0.0
    for i in range(n_spatial):
        a_i, b_i, c_i = local_abcs[i]
        oa = mode_is_occupied(bitstring, 2 * i, n_qubits)
        ob = mode_is_occupied(bitstring, 2 * i + 1, n_qubits)
        eigval += omega_eigenvalue_on_orbital_local(oa, ob, a_i, b_i, c_i)
    return eigval
def build_generalized_sectors_local_abc(basis_bitstrings, n_spatial, n_qubits, local_abcs, tol_decimals=8):
    """
    Partition determinants by joint eigenvalues of local Omega_i with per-orbital (a_i, b_i, c_i).
    """
    sectors: dict[tuple[float, ...], list[int]] = {}
    for k, bit_str in enumerate(basis_bitstrings):
        sector_key = omega_eigenvalues_from_bitstring_local_abc(
            int(bit_str), n_spatial, n_qubits, local_abcs, tol_decimals=tol_decimals
        )
        sectors.setdefault(sector_key, []).append(k)
    return sectors
def decoupled_min_energy(H, sectors): # min eigval in each sector
    Emin = None
    for idxs in sectors.values():
        blk = H[np.ix_(idxs, idxs)]
        e0 = float(np.linalg.eigvalsh(blk)[0])
        Emin = e0 if Emin is None else min(Emin, e0)
    return float(Emin)
def build_single_local_operator(U_spatial, n_spatial, i, local_abcs, tol=1e-12):
    a_i, b_i, c_i = local_abcs[i]
    return normal_ordered(
        rotated_seniority_orbital_fermion(
            U_spatial, i, n_spatial, a_i, b_i, c_i, tol=tol
        )
    )
def build_total_operator_local_abc(U_spatial, n_spatial, local_abcs, tol=1e-12):
    S_ferm = FermionOperator()
    for i in range(n_spatial):
        S_ferm += build_single_local_operator(U_spatial, n_spatial, i, local_abcs, tol=tol)
    return normal_ordered(S_ferm)
def analyze_individual_symmetry_operators_with_leakage_local_abc(
    H_mat,
    psi,
    U_spatial,
    n_spatial,
    n_qubits,
    local_abcs,
    label="",
    tol=1e-12,
    check_eigenstate=True,
):
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)

    exp_vals = []
    delta_norm_vals = []
    delta_energy_vals = []
    delta_Eavg_vals = []
    comm_sq_vals = []
    comm_vals = []

    print(f"\n=== Individual operator + leakage analysis: {label} ===")
    print(" i        <S_i>                  ||delta_i||           <delta_i|H|delta_i>       <H>_delta_i-E0         ||[H,S_i]psi||^2          (a_i,b_i,c_i)")

    for i in range(n_spatial):
        a_i, b_i, c_i = local_abcs[i]

        Si_ferm = build_single_local_operator(U_spatial, n_spatial, i, local_abcs, tol=tol)
        Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)

        leak = analyze_single_operator_leakage(
            H_mat, Si_mat, psi, label=f"{label} / S_{i}", atol=tol
        )

        comm_sq_i, comm_i = comm_state_norm_sq(
            H_mat, Si_mat, psi, check_eigenstate=check_eigenstate
        )

        E_shift = np.nan
        if leak["delta_norm2"] > tol and not np.isnan(np.real(leak["E_delta"])):
            E_shift = np.real(leak["E_delta"]) - leak["E0"]

        exp_vals.append(leak["s"])
        delta_norm_vals.append(leak["delta_norm"])
        delta_energy_vals.append(leak["delta_H_delta"])
        delta_Eavg_vals.append(leak["E_delta"])
        comm_sq_vals.append(comm_sq_i)
        comm_vals.append(comm_i)

        print(
            f"{i:2d}   "
            f"{leak['s'].real:+.12f}{leak['s'].imag:+.2e}j   "
            f"{leak['delta_norm']:+.12e}   "
            f"{leak['delta_H_delta'].real:+.12e}   "
            f"{E_shift:+.12e}   "
            f"{comm_sq_i:+.12e}   "
            f"({a_i:+.4f},{b_i:+.4f},{c_i:+.4f})"
        )

    total_comm_sq = float(np.sum(comm_sq_vals))
    total_exp = np.sum(exp_vals)

    print(f"\nsum_i <S_i>                 = {total_exp.real:+.12f}{total_exp.imag:+.2e}j")
    print(f"sum_i ||[H,S_i]psi||^2      = {total_comm_sq:+.12e}")

    return {
        "exp_vals": exp_vals,
        "delta_norm_vals": delta_norm_vals,
        "delta_energy_vals": delta_energy_vals,
        "delta_Eavg_vals": delta_Eavg_vals,
        "comm_sq_vals": comm_sq_vals,
        "comm_vals": comm_vals,
        "sum_exp": total_exp,
        "sum_comm_sq": total_comm_sq,
    }
def analyze_individual_symmetry_operators_with_leakage_local_abc_subspace(
    h_sub,
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    U_spatial,
    n_spatial: int,
    n_qubits: int,
    local_abcs,
    label: str = "",
    tol: float = 1e-12,
    check_eigenstate: bool = True,
):
    """Local-ABC leakage / commutator analysis on the fixed-N subspace."""
    psi = np.asarray(v_sub, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)

    exp_vals = []
    comm_sq_vals = []

    print(f"\n=== Individual operator + leakage analysis (subspace): {label} ===")
    print(
        " i        <S_i>                  ||delta_i||           "
        "<delta_i|H|delta_i>       <H>_delta_i-E0         ||[H,S_i]psi||^2          (a_i,b_i,c_i)"
    )

    for i in range(n_spatial):
        a_i, b_i, c_i = local_abcs[i]
        Si_ferm = build_single_local_operator(U_spatial, n_spatial, i, local_abcs, tol=tol)
        Si_full = fermion_to_sparse_qubit(Si_ferm, n_qubits)
        Si_sub = restrict_operator_to_subspace(Si_full, basis_bitstrings)

        leak = analyze_single_operator_leakage(
            h_sub, Si_sub, psi, label=f"{label} / S_{i}", atol=tol
        )
        comm_sq_i, _ = comm_state_norm_sq(
            h_sub, Si_sub, psi, check_eigenstate=check_eigenstate
        )

        E_shift = np.nan
        if leak["delta_norm2"] > tol and not np.isnan(np.real(leak["E_delta"])):
            E_shift = np.real(leak["E_delta"]) - leak["E0"]

        exp_vals.append(leak["s"])
        comm_sq_vals.append(comm_sq_i)

        print(
            f"{i:2d}   "
            f"{leak['s'].real:+.12f}{leak['s'].imag:+.2e}j   "
            f"{leak['delta_norm']:+.12e}   "
            f"{leak['delta_H_delta'].real:+.12e}   "
            f"{E_shift:+.12e}   "
            f"{comm_sq_i:+.12e}   "
            f"({a_i:+.4f},{b_i:+.4f},{c_i:+.4f})"
        )

    total_comm_sq = float(np.sum(comm_sq_vals))
    total_exp = np.sum(exp_vals)

    print(f"\nsum_i <S_i>                 = {total_exp.real:+.12f}{total_exp.imag:+.2e}j")
    print(f"sum_i ||[H,S_i]psi||^2      = {total_comm_sq:+.12e}")

    return {
        "sum_exp": total_exp,
        "sum_comm_sq": total_comm_sq,
    }
