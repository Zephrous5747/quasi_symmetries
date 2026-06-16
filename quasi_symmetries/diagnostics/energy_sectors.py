"""Energy-sector diagnostics and leakage analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from quasi_symmetries.config import OP_COEF_TOL
from quasi_symmetries.fermion.bitstring import (
    build_cisd_basis_bitstrings,
    mode_is_occupied,
    occ_lists_alpha_beta,
    omega_mask_from_bitstring,
)
from quasi_symmetries.fermion.subspace import restrict_operator_to_subspace
from quasi_symmetries.fermion.operators import (
    fermion_to_sparse_qubit,
    rotated_seniority_orbital_fermion,
)
from quasi_symmetries.optimization.rotations import orbital_rotation_representation_R
from openfermion import normal_ordered

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


def _all_sector_eigenpair_candidates(
    sector_data,
) -> list[tuple[float, object, np.ndarray, int]]:
    """All block eigenpairs (energy, sector key, full-space vector, block index)."""
    candidates: list[tuple[float, object, np.ndarray, int]] = []
    for key, data in sector_data.items():
        for block_index, (energy, vector) in enumerate(
            zip(data["evals"], data["evecs_full"])
        ):
            candidates.append((float(energy), key, vector, int(block_index)))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _max_coupling_to_span(
    h_dense: np.ndarray,
    candidate: np.ndarray,
    chosen_vecs: list[np.ndarray],
) -> float:
    if not chosen_vecs:
        return float("inf")
    h_cand = h_dense @ candidate
    return max(float(abs(np.vdot(chosen, h_cand))) for chosen in chosen_vecs)


def _projected_ground_energy(h_dense: np.ndarray, vecs: list[np.ndarray]) -> float:
    v = np.column_stack(vecs)
    h_proj = v.conj().T @ h_dense @ v
    h_proj = 0.5 * (h_proj + h_proj.conj().T)
    return float(np.linalg.eigvalsh(h_proj)[0])


def coupled_energy_test(
    H_dense,
    sector_data,
    E_exact=None,
    tol=1e-8,
    max_total_vectors=None,
    coupling_tol: float = 1e-12,
    energy_change_tol: float = 1e-12,
):
    """
    Coupled-energy test over sector-block eigenvectors.

    Candidates are all block eigenvectors sorted by block eigenenergy. Greedily
    add a candidate when it (1) has nonzero Hamiltonian coupling to the current
    span and (2) improves the projected ground energy toward E_exact (or lowers
    E_proj when E_exact is unavailable). Multiple passes over the sorted list
    allow later additions once newly coupled sectors enter the span.

    K is the minimum number of sector eigenvectors whose projected Hamiltonian
    ground energy matches E_exact within tol.
    """
    candidates = _all_sector_eigenpair_candidates(sector_data)
    if not candidates:
        return None, 0, False, []

    if max_total_vectors is None:
        max_total_vectors = len(candidates)

    chosen_vecs: list[np.ndarray] = []
    chosen_keys: list[tuple[object, int]] = []
    chosen_indices: set[int] = set()
    e_proj: float | None = None
    converged = False

    while True:
        added_this_pass = False
        for index, (_energy, key, vec, block_index) in enumerate(candidates):
            if index in chosen_indices:
                continue
            if len(chosen_vecs) >= max_total_vectors:
                break

            if chosen_vecs:
                if _max_coupling_to_span(H_dense, vec, chosen_vecs) <= coupling_tol:
                    continue
                e_new = _projected_ground_energy(H_dense, [*chosen_vecs, vec])
                if e_proj is not None:
                    if E_exact is not None:
                        if abs(e_new - E_exact) >= abs(e_proj - E_exact) - energy_change_tol:
                            continue
                    elif abs(e_new - e_proj) <= energy_change_tol:
                        continue
            else:
                e_new = _projected_ground_energy(H_dense, [vec])

            chosen_indices.add(index)
            chosen_vecs.append(vec)
            chosen_keys.append((key, block_index))
            e_proj = e_new
            added_this_pass = True

            if E_exact is not None and abs(e_proj - E_exact) <= tol:
                converged = True
                break

        if converged:
            break
        if not added_this_pass or len(chosen_vecs) >= max_total_vectors:
            break

    if E_exact is not None and e_proj is not None and abs(e_proj - E_exact) <= tol:
        converged = True

    return e_proj, len(chosen_vecs), converged, chosen_keys
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


def build_sectors_with_exact_symmetries(
    basis_bitstrings,
    n_spatial: int,
    n_qubits: int,
    a: float,
    b: float,
    c: float,
    exact_parity_diagonals: dict[str, np.ndarray],
    *,
    restrict_to_target_sector: bool = True,
    target: int = 1,
    tol_decimals: int = 8,
) -> tuple[dict[tuple, list[int]], list[int]]:
    """
    Joint sectors of exact parity operators and local Omega_i eigenvalues.

    Returns (sectors, allowed_indices). When restrict_to_target_sector is True,
    only determinants with all exact parities equal to target are included.
    """
    from quasi_symmetries.symmetry.exact import ground_state_sector_indices

    n_det = len(basis_bitstrings)
    if restrict_to_target_sector and exact_parity_diagonals:
        allowed = ground_state_sector_indices(exact_parity_diagonals, target=target)
    else:
        allowed = list(range(n_det))

    parity_names = tuple(sorted(exact_parity_diagonals))
    sectors: dict[tuple, list[int]] = {}
    allowed_set = set(allowed)
    for index in allowed:
        exact_key = tuple(
            int(np.sign(exact_parity_diagonals[name][index])) for name in parity_names
        )
        omega_key = omega_eigenvalues_from_bitstring(
            int(basis_bitstrings[index]),
            n_spatial,
            n_qubits,
            a,
            b,
            c,
            tol_decimals=tol_decimals,
        )
        key = exact_key + omega_key
        sectors.setdefault(key, []).append(index)
    return sectors, allowed


def restrict_state_and_hamiltonian(
    h_dense: np.ndarray,
    psi: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict dense H and state vector to a determinant subset."""
    idx = np.asarray(indices, dtype=int)
    h_block = h_dense[np.ix_(idx, idx)]
    h_block = 0.5 * (h_block + h_block.conj().T)
    psi_block = psi[idx]
    norm = np.linalg.norm(psi_block)
    if norm == 0:
        raise ValueError("State vector has zero norm on the restricted sector.")
    return h_block, psi_block / norm


def energy_sector_diagnostics_symmetry_restricted(
    h_dense: np.ndarray,
    basis_bitstrings,
    n_spatial: int,
    n_qubits: int,
    a: float,
    b: float,
    c: float,
    exact_parity_diagonals: dict[str, np.ndarray],
    *,
    u_spatial=None,
    energy_fci: float | None = None,
    tol: float = 1e-8,
    label: str = "",
) -> tuple[EnergySectorDiagnostics, dict[str, int | float]]:
    """
  Energy indicators after restricting to the exact-symmetry +1 sector.

    Returns diagnostics and summary metadata (allowed dim, full dim, sector count).
    """
    sectors_full, allowed = build_sectors_with_exact_symmetries(
        basis_bitstrings,
        n_spatial,
        n_qubits,
        a,
        b,
        c,
        exact_parity_diagonals,
        restrict_to_target_sector=True,
    )

    if u_spatial is not None:
        r = orbital_rotation_representation_R(u_spatial, basis_bitstrings, n_spatial)
        h_work = r.conj().T @ h_dense @ r
        h_work = 0.5 * (h_work + h_work.conj().T)
    else:
        h_work = 0.5 * (h_dense + h_dense.conj().T)

    if energy_fci is None:
        energy_fci = float(np.linalg.eigvalsh(h_work)[0])

    sector_data = diagonalize_sector_blocks(h_work, sectors_full)
    e_dec_min, best_sector, best_sector_dim = decoupled_energy_test(h_work, sectors_full)
    e_coupled, k_coupled, converged, _ = coupled_energy_test(
        h_work, sector_data, E_exact=energy_fci, tol=tol
    )
    e_bo, _, _ = bo_like_coupled_energy_test(h_work, sector_data)

    print(f"\n=== Symmetry-restricted energy indicators: {label} ===")
    print(f"allowed dim        = {len(allowed)} / {len(basis_bitstrings)}")
    print(f"E_exact            = {energy_fci:+.12f}")
    print(f"E_dec_min          = {e_dec_min:+.12f}")
    print(f"best sector        = {best_sector}")
    print(f"best sector dim    = {best_sector_dim}")
    print(f"E_coupled          = {e_coupled:+.12f}")
    print(f"K_coupled          = {k_coupled}")
    print(f"coupled converged  = {converged}")
    print(f"E_BO               = {e_bo:+.12f}")
    print(f"n_sectors          = {len(sectors_full)}")

    diagnostics = EnergySectorDiagnostics(
        E_dec_min=e_dec_min,
        best_sector=best_sector,
        best_sector_dim=best_sector_dim,
        E_coupled=e_coupled,
        K_coupled=k_coupled,
        coupled_converged=converged,
        E_BO=e_bo,
        n_sectors=len(sectors_full),
    )
    meta = {
        "allowed_dim": len(allowed),
        "full_dim": len(basis_bitstrings),
        "n_sectors": len(sectors_full),
    }
    return diagnostics, meta


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
