import csv
import math
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
from scipy.linalg import expm, logm
from scipy.optimize import minimize

from openfermion import (
    FermionOperator,
    get_fermion_operator,
    get_sparse_operator,
    jordan_wigner,
    normal_ordered,
)

LIH_BOND_ANGSTROM = 1.60  # ~equilibrium Li–H bond length rough starting point (can vary by basis/method)

CHARGE = 0

MULTIPLICITY = 1

BASIS = "sto-3g"

OPT_METHOD = "Powell" #web says powell good

MAXITER = 200

N_RESTARTS = 5

ANGLE_INIT_SCALE = 0.2

RANDOM_SEED = 0

TOPK_ANGLES_TO_PRINT = 10

EVAL_STATE_SPECIFIC_COMMUTATIVITY = True

OP_COEF_TOL = 1e-12

# Fixed-N determinant counts above this use sparse / subspace-only post-processing.
DENSE_SUBSPACE_MAX = 8_000


def fixed_n_subspace_dim(n_spatial: int, n_electrons: int) -> int:
    return math.comb(2 * n_spatial, n_electrons)


def use_dense_subspace_ops(n_spatial: int, n_electrons: int) -> bool:
    return fixed_n_subspace_dim(n_spatial, n_electrons) <= DENSE_SUBSPACE_MAX


def popcount(x: int) -> int:
    return int(x.bit_count())

def mode_to_bitpos(mode: int, n_qubits: int) -> int:
    """
    OpenFermion-consistent mapping inferred from your identity check:
    fermionic mode 0 is the LEFTMOST bit in the printed binary string.
    """
    if not (0 <= mode < n_qubits):
        raise ValueError(f"mode {mode} out of range for n_qubits={n_qubits}")
    return n_qubits - 1 - mode

def mode_is_occupied(bitstring: int, mode: int, n_qubits: int) -> int:
    pos = mode_to_bitpos(mode, n_qubits)
    return (bitstring >> pos) & 1

def parity_sign(bitstring: int, mode: int, n_qubits: int) -> int:
    """
    Fermionic JW sign for acting on 'mode':
    (-1)^(number of occupied modes with label < mode).
    IMPORTANT: this is NOT the same as counting lower integer bit positions
    once mode 0 is mapped to the MSB.
    """
    occ_before = 0
    for k in range(mode):
        occ_before += mode_is_occupied(bitstring, k, n_qubits)
    return -1 if (occ_before % 2 == 1) else 1

def apply_annihilate(bitstring: int, mode: int, n_qubits: int):
    pos = mode_to_bitpos(mode, n_qubits)
    if ((bitstring >> pos) & 1) == 0:
        return None, 0
    sign = parity_sign(bitstring, mode, n_qubits)
    return bitstring & ~(1 << pos), sign

def apply_create(bitstring: int, mode: int, n_qubits: int):
    pos = mode_to_bitpos(mode, n_qubits)
    if ((bitstring >> pos) & 1) == 1:
        return None, 0
    sign = parity_sign(bitstring, mode, n_qubits)
    return bitstring | (1 << pos), sign

def build_lih_geometry(li_h_bond_angstrom: float):
    """
    Return a linear LiH geometry centered at the origin.
    """
    r = li_h_bond_angstrom / 2.0
    return [
        ("Li", (0.0, 0.0, -r)),
        ("H",  (0.0, 0.0, +r)),
    ]

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


def restrict_operator_to_subspace(op_mat, basis_bitstrings):
    idx = np.asarray(basis_bitstrings, dtype=int)
    return op_mat[idx, :][:, idx].tocsc()


def print_fci_state(
    v_sub,
    basis_bitstrings,
    n_spatial,
    topk=None,
    amp_tol=1e-10,
    sort_by_weight=True,
):

    coeffs = np.asarray(v_sub, dtype=np.complex128)
    coeffs = coeffs / np.linalg.norm(coeffs)

    rows = []
    for idx, (b, c) in enumerate(zip(basis_bitstrings, coeffs)):
        amp = abs(c)
        if amp < amp_tol:
            continue

        wt = amp * amp
        occ_a, occ_b = occ_lists_alpha_beta(int(b), n_spatial)
        omega_mask = omega_mask_from_bitstring(int(b), n_spatial)
        omega_pat = format_omega_mask(omega_mask, n_spatial)
        sen = popcount(omega_mask)
        bitstr = format(int(b), f"0{2*n_spatial}b")

        rows.append({
            "idx": idx,
            "bitstring": bitstr,
            "occ_a": occ_a,
            "occ_b": occ_b,
            "omega": omega_pat,
            "sen": sen,
            "coeff": c,
            "weight": wt,
        })

    if sort_by_weight:
        rows.sort(key=lambda r: r["weight"], reverse=True)

    if topk is not None:
        rows = rows[:topk]

    print("\n=== FCI state in N-electron determinant basis ===")
    print(f"Components shown: {len(rows)}")
    print("rank   idx        bitstring            occ_a        occ_b      Ω-mask  sen        coeff                         weight      cumulative")

    cumulative = 0.0
    for rank, r in enumerate(rows, start=1):
        cumulative += r["weight"]
        c = r["coeff"]
        coeff_str = f"{c.real:+.10f}{c.imag:+.10f}j"
        print(
            f"{rank:>4d} {r['idx']:>6d}   {r['bitstring']}   "
            f"{str(r['occ_a']):>10s}  {str(r['occ_b']):>10s}   "
            f"{r['omega']:>5s}   {r['sen']:>3d}   "
            f"{coeff_str:>28s}   {r['weight']:>10.8f}   {cumulative:>10.8f}"
        )

    shown_weight = float(sum(r["weight"] for r in rows))
    total_weight = float(np.sum(np.abs(coeffs) ** 2))
    print(f"\nShown weight = {shown_weight:.10f}")
    print(f"Total weight = {total_weight:.10f}")

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

def build_U_from_thetas(n, thetas, pairs): #U size = n_spatial x n_spatial
    U = np.eye(n)
    for th, (p, q) in zip(thetas, pairs):
        U = U @ givens(n, p, q, th)
    return U

@dataclass
class OptLog:
    V: list
    nOmega: list
    x: list

def variance_restricted(gamma_a, gamma_b, Gamma_ab, x_params, pairs):
    n = gamma_a.shape[0]
    m = len(pairs)

    # Unpack orbital rotations and operator angles
    thetas = x_params[:m]
    phi1, phi2 = x_params[m], x_params[m+1]

    # Spherical parameterization for sqrt(a^2 + b^2 + c^2) = 1
    a = np.sin(phi1) * np.cos(phi2)
    b = np.sin(phi1) * np.sin(phi2)
    c = np.cos(phi1)

    U = build_U_from_thetas(n, thetas, pairs)
    Ua = U.T @ gamma_a @ U
    Ub = U.T @ gamma_b @ U

    exp_vals = np.zeros(n, dtype=float)
    V_total = 0.0

    for i in range(n):
        u = U[:, i]
        G_i = np.einsum("p,q,r,s,pqrs->", u, u, u, u, Gamma_ab, optimize=True).real
        N_a = Ua[i, i].real
        N_b = Ub[i, i].real

        # < \tilde{\Omega}_i >
        exp_omega = a * N_a + b * N_b + c * G_i
        exp_vals[i] = exp_omega

        # < \tilde{\Omega}_i^2 >
        exp_omega_sq = a**2 * N_a + b**2 * N_b + (2*a*b + 2*a*c + 2*b*c + c**2) * G_i

        # Exact Variance: <O^2> - <O>^2
        V_total += float(exp_omega_sq - exp_omega**2)

    return V_total, exp_vals, U, a, b, c

def optimize_variance_restricted(gamma_a, gamma_b, Gamma_ab):
    np.random.seed(RANDOM_SEED)
    n = gamma_a.shape[0]
    pairs = pair_list_for_n(n)
    m = len(pairs)
    num_params = m + 2 # +2 for phi1, phi2

    def obj(x):
        V, _, _, _, _, _ = variance_restricted(gamma_a, gamma_b, Gamma_ab, x, pairs)
        return V

    best = None
    for r in range(N_RESTARTS):
        x0 = np.zeros(num_params)
        if r == 0:
            # Initialize close to standard seniority: a=1, b=1, c=-2 (Normalized by sqrt(6))
            x0[m] = np.arccos(-2.0 / np.sqrt(6.0)) # phi1 for c
            x0[m+1] = np.pi / 4.0                  # phi2 for a, b
        else:
            x0[:m] = ANGLE_INIT_SCALE * np.random.randn(m)
            x0[m] = np.random.uniform(0, np.pi)
            x0[m+1] = np.random.uniform(0, 2*np.pi)

        log = OptLog(V=[], nOmega=[], x=[])

        def callback(xk):
            V, nO, _, _, _, _ = variance_restricted(gamma_a, gamma_b, Gamma_ab, xk, pairs)
            log.V.append(V); log.nOmega.append(nO); log.x.append(np.array(xk, copy=True))

        if OPT_METHOD.upper() == "POWELL":
            res = minimize(obj, x0=x0, method="Powell", options={"maxiter": MAXITER, "disp": False})
            callback(res.x)
        else:
            res = minimize(obj, x0=x0, method=OPT_METHOD, options={"maxiter": MAXITER, "disp": False}, callback=callback)

        V_fin = obj(res.x)
        if best is None or V_fin < best["V"]:
            best = {"res": res, "log": log, "V": V_fin, "pairs": pairs, "a": np.sin(res.x[m])*np.cos(res.x[m+1]), "b": np.sin(res.x[m])*np.sin(res.x[m+1]), "c": np.cos(res.x[m])}

    return best

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

def closed_shell_hf_bitstring(n_electrons, n_spatial):
    if n_electrons % 2 != 0:
        raise ValueError("This helper assumes closed-shell (even electron count).")

    n_qubits = 2 * n_spatial
    occ = n_electrons // 2
    b = 0

    for i in range(occ):
        a_mode = 2 * i
        b_mode = 2 * i + 1
        b |= (1 << mode_to_bitpos(a_mode, n_qubits))
        b |= (1 << mode_to_bitpos(b_mode, n_qubits))

    return b

def omega_mask_from_bitstring(bitstring: int, n_spatial: int) -> int:
    """
    Returns an n_spatial-bit mask in ORBITAL LABEL ORDER.
    Bit i of the returned mask corresponds to spatial orbital i.
    """
    n_qubits = 2 * n_spatial
    mask = 0
    for i in range(n_spatial):
        oa = mode_is_occupied(bitstring, 2 * i,     n_qubits)
        ob = mode_is_occupied(bitstring, 2 * i + 1, n_qubits)
        if oa ^ ob:
            mask |= (1 << i)
    return mask

def occ_lists_alpha_beta(bitstring: int, n_spatial: int):
    n_qubits = 2 * n_spatial
    occ_a = [i for i in range(n_spatial) if mode_is_occupied(bitstring, 2 * i,     n_qubits)]
    occ_b = [i for i in range(n_spatial) if mode_is_occupied(bitstring, 2 * i + 1, n_qubits)]
    return occ_a, occ_b

def format_omega_mask(mask: int, n_spatial: int) -> str: #ai code to make it easier to look at
    """Human-friendly Ω-pattern string, i=0 on the left."""
    return "".join(str((mask >> i) & 1) for i in range(n_spatial))

def build_seniority_sectors(basis_bitstrings, n_spatial: int): #partition in2 sectors
    sectors = {}
    for k, b in enumerate(basis_bitstrings):
        m = omega_mask_from_bitstring(int(b), n_spatial)
        sectors.setdefault(m, []).append(k)
    return sectors

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

    print("\n=== Check: R D R† vs direct rotated seniority operator ===")
    print(f"||R†R - I||_F                      = {np.linalg.norm(R.conj().T @ R - np.eye(dim_sub)):.6e}")
    print(f"||R D_total R† - S_direct_total||_F = {fro_total:.6e}")
    print(f"relative total mismatch            = {rel_total:.6e}")

    # Equivalent check in the rotated basis: R† S_direct R should be diagonal and equal to D_total
    backrot_total = R.conj().T @ S_direct_total @ R
    diag_mismatch = np.linalg.norm(backrot_total - D_total)
    offdiag_only = backrot_total - np.diag(np.diag(backrot_total))
    print(f"||R† S_direct_total R - D_total||_F = {diag_mismatch:.6e}")
    print(f"offdiag norm of R†S_directR         = {np.linalg.norm(offdiag_only):.6e}")

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
            f"||R D_i R† - Ω_i(U)||_F = {fro_i:.6e}   "
            f"rel = {rel_i:.6e}   "
            f"||R†Ω_iR - D_i||_F = {diag_err_i:.6e}   "
            f"offdiag = {np.linalg.norm(offdiag_i):.6e}"
        )

    # --- optional state expectation check in the old basis ---
    if psi_old is not None:
        psi = np.asarray(psi_old, dtype=np.complex128)
        psi = psi / np.linalg.norm(psi)

        exp_via_R = np.vdot(psi, S_via_R_total @ psi)
        exp_direct = np.vdot(psi, S_direct_total @ psi)

        print("\nExpectation on supplied state (old basis):")
        print(f"<psi|R D_total R†|psi>             = {float(np.real_if_close(exp_via_R)):.12f}")
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

def build_hf_occ_modes(bitstring: int, n_qubits: int):
    return [m for m in range(n_qubits) if mode_is_occupied(bitstring, m, n_qubits)]

def build_hf_virt_modes(bitstring: int, n_qubits: int):
    return [m for m in range(n_qubits) if not mode_is_occupied(bitstring, m, n_qubits)]

def excite(bitstring: int, annihilators, creators, n_qubits: int):
    """
    Apply a_p^† ... a_q^† a_j ... a_i to a determinant bitstring, returning (new_bitstring, phase)
    using your JW sign convention via apply_annihilate/apply_create.
    """
    b = int(bitstring)
    phase = 1

    # annihilate in a chosen order (common: descending mode index can reduce sign bugs)
    for m in annihilators:
        b2, s = apply_annihilate(b, m, n_qubits)
        if b2 is None:
            return None, 0
        b, phase = b2, phase * s

    # create
    for m in creators:
        b2, s = apply_create(b, m, n_qubits)
        if b2 is None:
            return None, 0
        b, phase = b2, phase * s

    return b, phase

def build_cisd_basis_bitstrings(hf_b: int, n_qubits: int):
    occ = build_hf_occ_modes(hf_b, n_qubits)
    virt = build_hf_virt_modes(hf_b, n_qubits)

    cisd_set = {int(hf_b)}

    # Singles: i -> a
    for i in occ:
        for a in virt:
            b_new, _ = excite(hf_b, [i], [a], n_qubits)
            if b_new is not None:
                cisd_set.add(int(b_new))

    # Doubles: i,j -> a,b
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            i, j = occ[ii], occ[jj]
            for aa in range(len(virt)):
                for bb in range(aa + 1, len(virt)):
                    a, b = virt[aa], virt[bb]
                    b_new, _ = excite(hf_b, [j, i], [a, b], n_qubits)  # note ordering choice
                    if b_new is not None:
                        cisd_set.add(int(b_new))

    return sorted(cisd_set)

def solve_cisd_state(H_sub, basis_bitstrings, hf_b, n_qubits):
    # Map determinant bitstring -> index in your fixed-N basis ordering
    det_to_subidx = {int(b): k for k, b in enumerate(basis_bitstrings)}

    cisd_dets = build_cisd_basis_bitstrings(hf_b, n_qubits)
    cisd_subidx = [det_to_subidx[b] for b in cisd_dets if b in det_to_subidx]

    H_cisd = H_sub[cisd_subidx, :][:, cisd_subidx].tocsc()
    e, v = spla.eigsh(H_cisd, k=1, which="SA")
    v_cisd = v[:, 0]

    return float(np.real(e[0])), v_cisd, cisd_subidx

def omega_eigenvalue_on_orbital(oa: int, ob: int, a: float, b: float, c: float) -> float:
    """Eigenvalue of Omega_i = a n_ia + b n_ib + c n_ia n_ib on one orbital's occupation."""
    return float(a * oa + b * ob + c * (oa * ob))


def omega_eigenvalues_from_bitstring(
    bitstring: int,
    n_spatial: int,
    n_qubits: int,
    a: float,
    b: float,
    c: float,
    tol_decimals: int = 8,
) -> tuple[float, ...]:
    """Joint eigenvalue vector (omega_0, ..., omega_{n_spatial-1}) of all local Omega_i."""
    values = []
    for i in range(n_spatial):
        oa = mode_is_occupied(bitstring, 2 * i, n_qubits)
        ob = mode_is_occupied(bitstring, 2 * i + 1, n_qubits)
        values.append(
            round(omega_eigenvalue_on_orbital(oa, ob, a, b, c), tol_decimals)
        )
    return tuple(values)


def generalized_eigenvalue_from_bitstring(bitstring: int, n_spatial: int, n_qubits: int, a: float, b: float, c: float) -> float:
    """Sum eigenvalue of S_tot = sum_i Omega_i on a determinant (diagnostic only)."""
    eigval = 0.0
    for i in range(n_spatial):
        oa = mode_is_occupied(bitstring, 2 * i, n_qubits)
        ob = mode_is_occupied(bitstring, 2 * i + 1, n_qubits)
        eigval += omega_eigenvalue_on_orbital(oa, ob, a, b, c)
    return eigval

def build_generalized_sectors(basis_bitstrings, n_spatial, n_qubits, a, b, c, tol_decimals=8):
    """
    Partition fixed-N determinants into joint eigenspaces of local operators
        Omega_i = a n_{i,alpha} + b n_{i,beta} + c n_{i,alpha} n_{i,beta}.

    Each sector key is the tuple (omega_0, ..., omega_{n_spatial-1}), not sum_i omega_i.
    """
    sectors: dict[tuple[float, ...], list[int]] = {}
    for k, bit_str in enumerate(basis_bitstrings):
        sector_key = omega_eigenvalues_from_bitstring(
            int(bit_str), n_spatial, n_qubits, a, b, c, tol_decimals=tol_decimals
        )
        sectors.setdefault(sector_key, []).append(k)

    return sectors

def shannon_entropy_from_weights(weights, eps=1e-15):
    w = np.asarray(weights, dtype=float)
    w = w[w > eps]
    return float(-np.sum(w * np.log(w))) if w.size else 0.0

def shannon_block_decomposition(H_dense, psi_vec, sectors_dict):
    weights_fine = []
    sector_weights = {}

    for msk, idxs in sectors_dict.items():
        psi_s = psi_vec[idxs]
        ws = float(np.vdot(psi_s, psi_s).real)
        sector_weights[msk] = ws

        d = len(idxs)
        if d == 0:
            continue

        H_blk = H_dense[np.ix_(idxs, idxs)]
        H_blk = 0.5 * (H_blk + H_blk.conj().T)

        evals_blk, evecs_blk = np.linalg.eigh(H_blk)
        c_eig = evecs_blk.conj().T @ psi_s
        weights_fine.extend((np.abs(c_eig) ** 2).tolist())

    I_S = shannon_entropy_from_weights(weights_fine)
    I_SS = shannon_entropy_from_weights(list(sector_weights.values()))

    p_sum = float(np.sum(weights_fine))
    if abs(p_sum - 1.0) > 1e-6:
        print(f"  [warn] Σ_{'{'}s,i{'}'} w_s,i = {p_sum:.8f} (expected ~1).")

    return I_S, I_SS, sector_weights

@dataclass
class EnergySectorDiagnostics:
    E_dec_min: float
    best_sector: object
    best_sector_dim: int
    E_coupled: float
    K_coupled: int
    coupled_converged: bool
    E_BO: float
    n_sectors: int


def skipped_energy_sector_diagnostics() -> EnergySectorDiagnostics:
    """Placeholder when dense sector / rotation diagnostics are skipped."""
    nan = float("nan")
    return EnergySectorDiagnostics(
        E_dec_min=nan,
        best_sector=None,
        best_sector_dim=0,
        E_coupled=nan,
        K_coupled=0,
        coupled_converged=False,
        E_BO=nan,
        n_sectors=0,
    )


def diagonalize_sector_blocks(H_dense, sectors_dict):
    """
    Diagonalize each symmetry block H(s) independently.

    Returns
    -------
    sector_data : dict
        key -> {
            "idxs": list of determinant indices in this sector,
            "evals": ndarray of block eigenvalues,
            "evecs_full": list of full-space vectors (same basis ordering as H_dense)
        }
    """
    dim = H_dense.shape[0]
    sector_data = {}

    for key, idxs in sectors_dict.items():
        blk = H_dense[np.ix_(idxs, idxs)]
        blk = 0.5 * (blk + blk.conj().T)

        evals, evecs = np.linalg.eigh(blk)

        evecs_full = []
        for j in range(evecs.shape[1]):
            v = np.zeros(dim, dtype=np.complex128)
            v[np.asarray(idxs, dtype=int)] = evecs[:, j]
            evecs_full.append(v)

        sector_data[key] = {
            "idxs": idxs,
            "evals": evals,
            "evecs_full": evecs_full,
        }

    return sector_data

def decoupled_energy_test(H_dense, sectors_dict):
    """
    Decoupled-energy test:
        E_dec_min = min_s lambda_min(H(s))
    """
    best_E = None
    best_key = None
    best_dim = 0

    for key, idxs in sectors_dict.items():
        blk = H_dense[np.ix_(idxs, idxs)]
        blk = 0.5 * (blk + blk.conj().T)
        e0 = float(np.linalg.eigvalsh(blk)[0])

        if best_E is None or e0 < best_E:
            best_E = e0
            best_key = key
            best_dim = len(idxs)

    return best_E, best_key, best_dim

def coupled_energy_test(H_dense, sector_data, E_exact=None, tol=1e-8, max_total_vectors=None):
    """
    Coupled-energy test:
      1) collect low-energy eigenvectors from all sectors,
      2) sort by their block eigen-energies,
      3) project H into span of first K vectors,
      4) increase K until projected ground-state energy reaches E_exact within tol.

    If E_exact is None, uses all available vectors.
    """
    candidates = []
    for key, data in sector_data.items():
        for e, v in zip(data["evals"], data["evecs_full"]):
            candidates.append((float(e), key, v))

    candidates.sort(key=lambda t: t[0])

    if max_total_vectors is None:
        max_total_vectors = len(candidates)

    chosen_vecs = []
    chosen_keys = []
    E_proj = None
    K_final = 0
    converged = False

    for K in range(1, min(max_total_vectors, len(candidates)) + 1):
        e, key, v = candidates[K - 1]
        chosen_vecs.append(v)
        chosen_keys.append(key)

        V = np.column_stack(chosen_vecs)
        H_proj = V.conj().T @ H_dense @ V
        H_proj = 0.5 * (H_proj + H_proj.conj().T)

        evals_proj = np.linalg.eigvalsh(H_proj)
        E_proj = float(evals_proj[0])
        K_final = K

        if E_exact is not None and abs(E_proj - E_exact) <= tol:
            converged = True
            break

    return E_proj, K_final, converged, chosen_keys[:K_final]

def bo_like_coupled_energy_test(H_dense, sector_data):
    """
    BO-like coupled test:
      take one lowest-energy eigenvector from each sector,
      build projected Hamiltonian in their span,
      return its ground-state energy.
    """
    ordered = sorted(sector_data.items(), key=lambda kv: float(kv[1]["evals"][0]))

    vecs = []
    chosen_keys = []
    for key, data in ordered:
        vecs.append(data["evecs_full"][0])
        chosen_keys.append(key)

    V = np.column_stack(vecs)
    H_proj = V.conj().T @ H_dense @ V
    H_proj = 0.5 * (H_proj + H_proj.conj().T)

    evals_proj = np.linalg.eigvalsh(H_proj)
    return float(evals_proj[0]), len(vecs), chosen_keys

def shared_abc_energy_indicators(
    H_dense,
    basis_bitstrings,
    n_spatial,
    n_qubits,
    a,
    b,
    c,
    U_spatial=None,
    E_exact=None,
    tol=1e-8,
    label="",
):
    """
    Energy-based indicators for the shared-(a,b,c) workflow.

    If U_spatial is None:
        use H_dense directly and build sectors in the current determinant basis.

    If U_spatial is not None:
        rotate H into the orbital-rotated determinant basis using R,
        then build sectors there, matching your entropy workflow.
    """
    if E_exact is None:
        E_exact = float(np.linalg.eigvalsh(0.5 * (H_dense + H_dense.conj().T))[0])

    # Work in the rotated determinant basis if U is supplied
    if U_spatial is not None:
        R = orbital_rotation_representation_R(U_spatial, basis_bitstrings, n_spatial)
        H_work = R.conj().T @ H_dense @ R
        H_work = 0.5 * (H_work + H_work.conj().T)
    else:
        H_work = 0.5 * (H_dense + H_dense.conj().T)

    sectors = build_generalized_sectors(
        basis_bitstrings, n_spatial, n_qubits, a, b, c
    )

    sector_data = diagonalize_sector_blocks(H_work, sectors)

    E_dec_min, best_sector, best_sector_dim = decoupled_energy_test(H_work, sectors)
    E_coupled, K_coupled, converged, _ = coupled_energy_test(
        H_work, sector_data, E_exact=E_exact, tol=tol
    )
    E_BO, _, _ = bo_like_coupled_energy_test(H_work, sector_data)

    print(f"\n=== Energy indicators: {label} ===")
    print(f"E_exact            = {E_exact:+.12f}")
    print(f"E_dec_min          = {E_dec_min:+.12f}")
    print(f"best sector        = {best_sector}")
    print(f"best sector dim    = {best_sector_dim}")
    print(f"E_coupled          = {E_coupled:+.12f}")
    print(f"K_coupled          = {K_coupled}")
    print(f"coupled converged  = {converged}")
    print(f"E_BO               = {E_BO:+.12f}")
    print(f"n_sectors          = {len(sectors)}")

    payload = {
        "E_dec_min": E_dec_min,
        "best_sector": best_sector,
        "best_sector_dim": best_sector_dim,
        "E_coupled": E_coupled,
        "K_coupled": K_coupled,
        "coupled_converged": converged,
        "E_BO": E_BO,
        "n_sectors": len(sectors),
    }
    try:
        return EnergySectorDiagnostics(**payload)
    except TypeError:
        # Compatibility fallback for environments where the class exists
        # but was defined without a dataclass-generated __init__.
        diagnostics = EnergySectorDiagnostics()
        for key, value in payload.items():
            setattr(diagnostics, key, value)
        return diagnostics

def build_lih_geometry(li_h_bond_angstrom: float):
    r = li_h_bond_angstrom / 2.0
    return [
        ("Li", (0.0, 0.0, -r)),
        ("H",  (0.0, 0.0, +r)),
    ]

def build_h2o_geometry(oh_bond_angstrom: float, hoh_angle_deg: float = 104.5):
    angle_rad = np.radians(hoh_angle_deg / 2.0)
    x = oh_bond_angstrom * np.sin(angle_rad)
    y = oh_bond_angstrom * np.cos(angle_rad)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (x, y, 0.0)),
        ("H", (-x, y, 0.0)),
    ]

def build_h4_linear_geometry(h_h_bond_angstrom: float):
    """
    Linear H4 chain centered at origin:
      H - H - H - H
    nearest-neighbor spacing = h_h_bond_angstrom
    """
    d = h_h_bond_angstrom
    coords = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    return [("H", (0.0, 0.0, z)) for z in coords]

def build_h4_square_geometry(side_angstrom: float):
    """
    H4 square centered at origin in the xy-plane.

    side_angstrom:
        side length of the square
    """
    s = side_angstrom / 2.0
    return [
        ("H", (-s, -s, 0.0)),
        ("H", (+s, -s, 0.0)),
        ("H", (+s, +s, 0.0)),
        ("H", (-s, +s, 0.0)),
    ]

def build_n2_geometry(bond_angstrom: float):
    """Linear N2 centered at the origin; x is the N–N bond length in Å."""
    r = bond_angstrom / 2.0
    return [
        ("N", (0.0, 0.0, -r)),
        ("N", (0.0, 0.0, +r)),
    ]


def build_h4_rectangle_geometry(long_side_angstrom: float, aspect_ratio: float = 1.5):
    """
    H4 rectangle centered at origin in the xy-plane.

    long_side_angstrom:
        length of the longer side

    aspect_ratio:
        long_side / short_side
        must be > 0
    """
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive.")

    a = long_side_angstrom / 2.0
    b = (long_side_angstrom / aspect_ratio) / 2.0

    return [
        ("H", (-a, -b, 0.0)),
        ("H", (+a, -b, 0.0)),
        ("H", (+a, +b, 0.0)),
        ("H", (-a, +b, 0.0)),
    ]

def get_geometry_and_description(molecule: str, x: float, **kwargs):
    from hamiltonian_geometry import get_geometry_and_description as _get_geometry_and_description

    return _get_geometry_and_description(molecule, x, **kwargs)

def build_total_operator(U_spatial, n_spatial, a, b, c, tol=1e-12):
    S_ferm = FermionOperator()
    for i in range(n_spatial):
        S_ferm += rotated_seniority_orbital_fermion(
            U_spatial, i, n_spatial, a, b, c, tol=tol
        )
    return normal_ordered(S_ferm)

def comm_state_norm_sq(H_mat, S_mat, psi, check_eigenstate=False, atol=1e-10):
    """
    Returns:
        norm2 = || [H,S] psi ||^2
        norm  = || [H,S] psi ||

    For Hermitian H,S:
        ||[H,S]psi||^2 = - <psi| [H,S]^2 |psi>.
    """
    Apsi = H_mat.dot(S_mat.dot(psi)) - S_mat.dot(H_mat.dot(psi))
    norm2 = np.real(np.vdot(Apsi, Apsi))

    # consistency check
    A2psi = H_mat.dot(S_mat.dot(Apsi)) - S_mat.dot(H_mat.dot(Apsi))
    exp = np.vdot(psi, A2psi)
    if not np.allclose(exp, -norm2, atol=atol):
        raise AssertionError("Expected <psi|[H,S]^2|psi> = -||[H,S]psi||^2")

    if check_eigenstate:
        E0 = np.vdot(psi, H_mat.dot(psi))
        resid = np.linalg.norm(H_mat.dot(psi) - E0 * psi)
        if resid < atol:
            Spsi = S_mat.dot(psi)
            HSpsi = H_mat.dot(Spsi)
            norm3 = (
                np.vdot(HSpsi, HSpsi)
                - 2 * E0 * np.vdot(Spsi, HSpsi)
                + E0**2 * np.vdot(Spsi, Spsi)
            )
            if not np.allclose(norm2, np.real(norm3), atol=atol):
                raise AssertionError("Eigenstate expansion check failed.")

    return float(norm2), float(np.sqrt(max(norm2, 0.0)))

def analyze_individual_symmetry_operators(H_mat, psi, U_spatial, n_spatial, n_qubits, a, b, c,
                                          label="", tol=1e-12, check_eigenstate=True):
    """
    For each local operator S_i, print:
      - <S_i>
      - ||[H,S_i] psi||^2
      - ||[H,S_i] psi||

    Returns:
      dict with per-orbital values and summed commutator-squared
    """
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)

    exp_vals = []
    comm_sq_vals = []
    comm_vals = []

    print(f"\n=== Individual operator analysis: {label} ===")
    print(" i        <S_i>                  ||[H,S_i]psi||^2         ||[H,S_i]psi||")

    for i in range(n_spatial):
        Si_ferm = normal_ordered(
            rotated_seniority_orbital_fermion(
                U_spatial, i, n_spatial, a, b, c, tol=tol
            )
        )
        Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)

        exp_i = expectation_value(Si_mat, psi)
        comm_sq_i, comm_i = comm_state_norm_sq(
            H_mat, Si_mat, psi, check_eigenstate=check_eigenstate
        )

        exp_vals.append(exp_i)
        comm_sq_vals.append(comm_sq_i)
        comm_vals.append(comm_i)

        print(
            f"{i:2d}   "
            f"{exp_i.real:+.12f}{exp_i.imag:+.2e}j   "
            f"{comm_sq_i:+.12e}   "
            f"{comm_i:+.12e}"
        )

    total_comm_sq = float(np.sum(comm_sq_vals))
    total_exp = np.sum(exp_vals)

    print(f"sum_i <S_i>                 = {total_exp.real:+.12f}{total_exp.imag:+.2e}j")
    print(f"sum_i ||[H,S_i]psi||^2      = {total_comm_sq:+.12e}")

    return {
        "exp_vals": exp_vals,
        "comm_sq_vals": comm_sq_vals,
        "comm_vals": comm_vals,
        "sum_exp": total_exp,
        "sum_comm_sq": total_comm_sq,
    }

def expectation_value(op_mat, psi):
    return np.vdot(psi, op_mat.dot(psi))

def analyze_single_operator_leakage(H_mat, S_mat, psi, label="", atol=1e-12):
    """
    For one operator S, decompose
        S|psi> = s|psi> + |delta>
    where
        s = <psi|S|psi>
        <psi|delta> = 0

    Prints:
      - <psi|S|psi>
      - ||delta||
      - <delta|H|delta>
      - <H>_delta = <delta|H|delta> / <delta|delta>
      - <H>_delta - E0
    """
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)

    E0 = np.real(np.vdot(psi, H_mat.dot(psi)))

    Spsi = S_mat.dot(psi)
    s = np.vdot(psi, Spsi)
    delta = Spsi - s * psi

    delta_norm2 = np.real(np.vdot(delta, delta))
    delta_norm = float(np.sqrt(max(delta_norm2, 0.0)))
    delta_H_delta = np.vdot(delta, H_mat.dot(delta))

    print(f"\n[{label}] leakage check")
    print(f"  <psi|S|psi>                = {s.real:+.12f}{s.imag:+.3e}j")
    print(f"  ||delta||                  = {delta_norm:.12e}")
    print(f"  <delta|H|delta>            = {delta_H_delta.real:+.12e}{delta_H_delta.imag:+.3e}j")

    if delta_norm2 > atol:
        E_delta = delta_H_delta / delta_norm2
        print(f"  <H>_delta                  = {E_delta.real:+.12f}{E_delta.imag:+.3e}j")
        print(f"  <H>_delta - E0             = {E_delta.real - E0:+.12e}")
    else:
        E_delta = np.nan
        print("  <H>_delta                  = undefined (delta ~ 0)")

    return {
        "s": s,
        "delta": delta,
        "delta_norm": delta_norm,
        "delta_norm2": delta_norm2,
        "delta_H_delta": delta_H_delta,
        "E_delta": E_delta,
        "E0": E0,
    }

def analyze_individual_symmetry_operators_with_leakage(
    H_mat,
    psi,
    U_spatial,
    n_spatial,
    n_qubits,
    a,
    b,
    c,
    label="",
    tol=1e-12,
    check_eigenstate=True,
):
    """
    For each local operator S_i, print:
      - <S_i>
      - ||delta_i||
      - <delta_i|H|delta_i>
      - <H>_{delta_i} - E0
      - ||[H,S_i]psi||^2
      - ||[H,S_i]psi||

    Returns a dict of per-orbital diagnostics plus summed commutator squared.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)

    exp_vals = []
    delta_norm_vals = []
    delta_energy_vals = []
    delta_Eavg_vals = []
    comm_sq_vals = []
    comm_vals = []

    print(f"\n=== Individual operator + leakage analysis: {label} ===")
    print(" i        <S_i>                  ||delta_i||           <delta_i|H|delta_i>       <H>_delta_i-E0         ||[H,S_i]psi||^2")

    for i in range(n_spatial):
        Si_ferm = normal_ordered(
            rotated_seniority_orbital_fermion(
                U_spatial, i, n_spatial, a, b, c, tol=tol
            )
        )
        Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)

        leak = analyze_single_operator_leakage(
            H_mat, Si_mat, psi, label=f"{label} / S_{i}", atol=tol
        )

        comm_sq_i, comm_i = comm_state_norm_sq(
            H_mat, Si_mat, psi, check_eigenstate=check_eigenstate
        )

        E_shift = np.nan
        if leak["delta_norm2"] > tol and np.isfinite(np.real(leak["E_delta"])):
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
            f"{comm_sq_i:+.12e}"
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


def analyze_individual_symmetry_operators_with_leakage_subspace(
    h_sub,
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    U_spatial,
    n_spatial: int,
    n_qubits: int,
    a: float,
    b: float,
    c: float,
    label: str = "",
    tol: float = 1e-12,
    check_eigenstate: bool = True,
):
    """Leakage / commutator analysis on the fixed-N subspace (memory-safe)."""
    psi = np.asarray(v_sub, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)

    exp_vals = []
    delta_norm_vals = []
    delta_energy_vals = []
    delta_Eavg_vals = []
    comm_sq_vals = []
    comm_vals = []

    print(f"\n=== Individual operator + leakage analysis (subspace): {label} ===")
    print(
        " i        <S_i>                  ||delta_i||           "
        "<delta_i|H|delta_i>       <H>_delta_i-E0         ||[H,S_i]psi||^2"
    )

    for i in range(n_spatial):
        Si_ferm = normal_ordered(
            rotated_seniority_orbital_fermion(U_spatial, i, n_spatial, a, b, c, tol=tol)
        )
        Si_full = fermion_to_sparse_qubit(Si_ferm, n_qubits)
        Si_sub = restrict_operator_to_subspace(Si_full, basis_bitstrings)

        leak = analyze_single_operator_leakage(
            h_sub, Si_sub, psi, label=f"{label} / S_{i}", atol=tol
        )

        comm_sq_i, comm_i = comm_state_norm_sq(
            h_sub, Si_sub, psi, check_eigenstate=check_eigenstate
        )

        E_shift = np.nan
        if leak["delta_norm2"] > tol and np.isfinite(np.real(leak["E_delta"])):
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
            f"{comm_sq_i:+.12e}"
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
