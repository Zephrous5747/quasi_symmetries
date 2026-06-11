"""Parity-parent Hamiltonian algebra from incidence vectors (parity_H_P.pdf)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp
from openfermion import FermionOperator

from optimization_abc_utils import mode_is_occupied


def spatial_pair_groups(n_spatial: int) -> list[frozenset[int]]:
    """DOCI pair groups g_p = {p_alpha, p_beta} as spin-orbital indices."""
    return [frozenset({2 * p, 2 * p + 1}) for p in range(n_spatial)]


def singleton_groups(spin_orbitals: Iterable[int]) -> list[frozenset[int]]:
    """CASSCF singleton groups g_p = {p} for selected spin orbitals."""
    return [frozenset({int(p)}) for p in spin_orbitals]


def build_incidence_matrix(groups: list[frozenset[int]], n_spin_orbitals: int) -> np.ndarray:
    """Return Gamma[a, p] = 1 if spin orbital p lies in group a."""
    k = len(groups)
    gamma = np.zeros((k, n_spin_orbitals), dtype=np.int8)
    for a, group in enumerate(groups):
        for p in group:
            if not (0 <= p < n_spin_orbitals):
                raise ValueError(f"Spin orbital {p} out of range for n_spin_orbitals={n_spin_orbitals}.")
            gamma[a, p] = 1
    return gamma


def charge_vector_for_modes(modes: Iterable[int], groups: list[frozenset[int]], n_spin_orbitals: int) -> np.ndarray:
    """Binary parity charge q(M) in F_2^k for a fermionic monomial's mode list."""
    gamma = build_incidence_matrix(groups, n_spin_orbitals)
    q = np.zeros(len(groups), dtype=np.int8)
    for mode in modes:
        q ^= gamma[:, int(mode)]
    return q


def onebody_allowed(p: int, q: int, groups: list[frozenset[int]], n_spin_orbitals: int) -> bool:
    """Eq. (9): a†_p a_q allowed iff gamma_p = gamma_q."""
    gp = charge_vector_for_modes([p], groups, n_spin_orbitals)
    gq = charge_vector_for_modes([q], groups, n_spin_orbitals)
    return np.array_equal(gp, gq)


def twobody_allowed(
    p: int,
    q: int,
    r: int,
    s: int,
    groups: list[frozenset[int]],
    n_spin_orbitals: int,
) -> bool:
    """Eq. (10): a†_p a†_q a_s a_r allowed iff gamma_p xor gamma_q = gamma_r xor gamma_s."""
    left = charge_vector_for_modes([p, q], groups, n_spin_orbitals)
    right = charge_vector_for_modes([r, s], groups, n_spin_orbitals)
    return np.array_equal(left, right)


def term_allowed(term: tuple[tuple[int, int], ...], groups: list[frozenset[int]], n_spin_orbitals: int) -> bool:
    """Selection rule for a normal-ordered fermion term."""
    if term == ():
        return True
    modes = [mode for mode, _ in term]
    if len(modes) == 2:
        return onebody_allowed(modes[0], modes[1], groups, n_spin_orbitals)
    if len(modes) == 4:
        return twobody_allowed(modes[0], modes[1], modes[2], modes[3], groups, n_spin_orbitals)
    return False


def project_fermion_to_parent(
    operator: FermionOperator,
    groups: list[frozenset[int]],
    n_spin_orbitals: int,
    *,
    coef_tol: float = 1e-15,
) -> FermionOperator:
    """Keep only terms consistent with the generator-resolved parent (Eq. 11)."""
    projected = FermionOperator()
    for term, coef in operator.terms.items():
        if abs(coef) <= coef_tol:
            continue
        if term_allowed(term, groups, n_spin_orbitals):
            projected += FermionOperator(term, coef)
    return projected


def reflection_eigenvalue(bitstring: int, group: Iterable[int], n_qubits: int) -> int:
    """Eigenvalue of R_g = (-1)^(sum_{p in g} n_p) on one determinant."""
    occ = sum(mode_is_occupied(int(bitstring), int(p), n_qubits) for p in group)
    return -1 if occ % 2 else 1


def reflection_diagonal(
    group: Iterable[int],
    basis_bitstrings: Iterable[int],
    n_qubits: int,
) -> np.ndarray:
    """Diagonal matrix elements of a single parity reflection on a fixed-N basis."""
    return np.array(
        [float(reflection_eigenvalue(int(b), group, n_qubits)) for b in basis_bitstrings],
        dtype=np.float64,
    )


def max_reflection_commutator_frobenius(
    h_sub: sp.spmatrix,
    reflection_diag: np.ndarray,
) -> float:
    """
    ||[H, R]||_F for diagonal involutory reflection R.

    [H, R]_ij = H_ij (R_j - R_i), so parent Hamiltonians give zero when R_i != R_j implies H_ij = 0.
    """
    h = h_sub.toarray() if sp.issparse(h_sub) else np.asarray(h_sub)
    r = np.asarray(reflection_diag, dtype=np.float64)
    diff = r[np.newaxis, :] - r[:, np.newaxis]
    commutator = h * diff
    return float(np.linalg.norm(commutator))


def pair_parity_diagonal(orbital: int, basis_bitstrings: Iterable[int], n_spatial: int) -> np.ndarray:
    """Diagonal of R^pair_p = (-1)^(n_palpha + n_pbeta)."""
    group = spatial_pair_groups(n_spatial)[orbital]
    n_qubits = 2 * n_spatial
    return reflection_diagonal(group, basis_bitstrings, n_qubits)


def project_h_sub_to_reflections(
    h_sub: sp.spmatrix,
    reflection_diagonals: list[np.ndarray],
) -> np.ndarray:
    """
    Project a fixed-N Hamiltonian onto operators commuting with all listed reflections.

    For involutory diagonal reflections, [H, R] = 0 requires H_ij = 0 whenever R_i != R_j.
    """
    h = h_sub.toarray() if sp.issparse(h_sub) else np.asarray(h_sub)
    mask = np.ones(h.shape, dtype=bool)
    for diag in reflection_diagonals:
        same_sector = diag[:, np.newaxis] == diag[np.newaxis, :]
        mask &= same_sector
    return h * mask


def project_h_sub_to_pair_parent(
    h_sub: sp.spmatrix,
    basis_bitstrings: Iterable[int],
    n_spatial: int,
) -> np.ndarray:
    """Matrix projection onto the DOCI pair-parity parent (commutes with every R^pair_p)."""
    diags = [
        pair_parity_diagonal(orbital, basis_bitstrings, n_spatial)
        for orbital in range(n_spatial)
    ]
    return project_h_sub_to_reflections(h_sub, diags)


def project_h_sub_to_singleton_parent(
    h_sub: sp.spmatrix,
    basis_bitstrings: Iterable[int],
    spin_orbitals: Iterable[int],
    n_qubits: int,
) -> np.ndarray:
    """Matrix projection onto the CASSCF singleton-parity parent on selected spin orbitals."""
    diags = [
        reflection_diagonal({int(p)}, basis_bitstrings, n_qubits)
        for p in spin_orbitals
    ]
    return project_h_sub_to_reflections(h_sub, diags)


def casscf_spin_orbital_partition(
    n_electrons: int,
    n_spatial: int,
    *,
    n_active_spatial: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Closed-shell CASSCF partition into inactive (I), active (A), virtual (V) spin orbitals.

    Inactive spatial orbitals are the lowest n_electrons/2 spatial orbitals (doubly occupied).
    Active spatial orbitals fill the middle; virtual orbitals are the remainder.
    """
    if n_electrons % 2 != 0:
        raise ValueError("casscf_spin_orbital_partition assumes closed-shell even electron count.")
    n_inactive_spatial = n_electrons // 2
    if n_active_spatial is None:
        n_active_spatial = n_spatial - n_inactive_spatial
    if n_inactive_spatial + n_active_spatial > n_spatial:
        raise ValueError("Inactive plus active spatial orbitals exceed n_spatial.")

    inactive_spin = [2 * i for i in range(n_inactive_spatial)] + [2 * i + 1 for i in range(n_inactive_spatial)]
    virtual_start = n_inactive_spatial + n_active_spatial
    virtual_spin = [2 * i for i in range(virtual_start, n_spatial)] + [
        2 * i + 1 for i in range(virtual_start, n_spatial)
    ]
    inactive_set = set(inactive_spin)
    virtual_set = set(virtual_spin)
    active_spin = [p for p in range(2 * n_spatial) if p not in inactive_set and p not in virtual_set]
    return inactive_spin, active_spin, virtual_spin


def external_singleton_spin_orbitals(
    n_electrons: int,
    n_spatial: int,
    *,
    n_active_spatial: int | None = None,
) -> list[int]:
    """Spin orbitals carrying CASSCF singleton parities: inactive union virtual (PDF Sec. 4)."""
    inactive, _, virtual = casscf_spin_orbital_partition(
        n_electrons,
        n_spatial,
        n_active_spatial=n_active_spatial,
    )
    return inactive + virtual


def parity_variance_sum(
    state: np.ndarray,
    reflection_diagonals: list[np.ndarray],
) -> float:
    """Sum of Var(R_a) = 1 - <R_a>^2 for involutory diagonal reflections."""
    weights = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    total = 0.0
    for diag in reflection_diagonals:
        expectation = float(np.dot(weights, np.asarray(diag, dtype=np.float64)))
        total += max(0.0, 1.0 - expectation**2)
    return total


def pair_parity_variance_sum(
    state: np.ndarray,
    basis_bitstrings: Iterable[int],
    n_spatial: int,
) -> float:
    """DOCI cost: sum_p Var(R^pair_p) on a fixed-N state vector."""
    diags = [
        pair_parity_diagonal(orbital, basis_bitstrings, n_spatial)
        for orbital in range(n_spatial)
    ]
    return parity_variance_sum(state, diags)


def singleton_parity_variance_sum(
    state: np.ndarray,
    basis_bitstrings: Iterable[int],
    external_spin_orbitals: Iterable[int],
    n_qubits: int,
) -> float:
    """CASSCF external cost: sum_{p in I cup V} Var((-1)^n_p)."""
    diags = [
        reflection_diagonal({int(p)}, basis_bitstrings, n_qubits)
        for p in external_spin_orbitals
    ]
    return parity_variance_sum(state, diags)


def rotate_h_sub_dense(
    u_spatial: np.ndarray,
    h_sub: sp.spmatrix | np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
) -> np.ndarray:
    """Rotate a fixed-N Hamiltonian by an orbital unitary: H(U) = R^dagger H R."""
    from quartet_optimization_utils import orbital_rotation_representation_R_fast

    h = h_sub.toarray() if sp.issparse(h_sub) else np.asarray(h_sub, dtype=np.complex128)
    r = orbital_rotation_representation_R_fast(u_spatial, basis_bitstrings, n_spatial)
    rotated = r.conj().T @ h @ r
    return 0.5 * (rotated + rotated.conj().T)


def projection_error_frobenius(
    h_sub: sp.spmatrix | np.ndarray,
    projected: np.ndarray,
) -> float:
    """||H - P_parent(H)||_F for Eq. (32) parent-matching diagnostics."""
    h = h_sub.toarray() if sp.issparse(h_sub) else np.asarray(h_sub, dtype=np.complex128)
    diff = h - projected
    return float(np.linalg.norm(diff))


def relative_projection_error(
    h_sub: sp.spmatrix | np.ndarray,
    projected: np.ndarray,
) -> float:
    h = h_sub.toarray() if sp.issparse(h_sub) else np.asarray(h_sub, dtype=np.complex128)
    num = projection_error_frobenius(h, projected)
    denom = max(float(np.linalg.norm(h)), 1e-15)
    return num / denom


def parent_ground_state(h_parent: np.ndarray) -> tuple[float, np.ndarray]:
    """Lowest eigenpair of a dense parent Hamiltonian."""
    h = 0.5 * (h_parent + h_parent.conj().T)
    evals, evecs = np.linalg.eigh(h)
    psi = evecs[:, 0].astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return float(np.real(evals[0])), psi


def casscf_sector_indices(
    basis_bitstrings: Iterable[int],
    inactive_spin: Iterable[int],
    virtual_spin: Iterable[int],
    n_qubits: int,
) -> list[int]:
    """
    Determinant indices in the CASSCF sector R_i=-1 on inactive, R_a=+1 on virtual (PDF Eq. 16).

    Occupied inactive and empty virtual spin orbitals.
    """
    inactive_set = set(int(p) for p in inactive_spin)
    virtual_set = set(int(p) for p in virtual_spin)
    indices: list[int] = []
    for index, bitstring in enumerate(basis_bitstrings):
        bitstring = int(bitstring)
        if any(mode_is_occupied(bitstring, p, n_qubits) == 0 for p in inactive_set):
            continue
        if any(mode_is_occupied(bitstring, p, n_qubits) == 1 for p in virtual_set):
            continue
        indices.append(index)
    return indices


def casscf_sector_ground_energy(h_parent: np.ndarray, sector_indices: list[int]) -> float:
    """Exact active-space ground energy within a fixed CAS sector."""
    if not sector_indices:
        raise ValueError("Empty CASSCF sector.")
    idx = np.asarray(sector_indices, dtype=int)
    block = h_parent[np.ix_(idx, idx)]
    block = 0.5 * (block + block.conj().T)
    return float(np.linalg.eigvalsh(block)[0])


def modes_of_term(term: tuple[tuple[int, int], ...]) -> list[int]:
    return [int(mode) for mode, _ in term]


def spin_sets_from_partition(
    n_electrons: int,
    n_spatial: int,
) -> tuple[set[int], set[int], set[int]]:
    inactive, active, virtual = casscf_spin_orbital_partition(n_electrons, n_spatial)
    return set(inactive), set(active), set(virtual)


def incidence_rows_dependent(groups: list[frozenset[int]], n_spin_orbitals: int) -> bool:
    """True when incidence rows contain linear redundancy (PDF remark after Eq.~11)."""
    gamma = build_incidence_matrix(groups, n_spin_orbitals)
    rank = int(np.linalg.matrix_rank(gamma.astype(np.int8)))
    return rank < len(groups)


def reflection_product_row(
    groups: list[frozenset[int]],
    generator_indices: Iterable[int],
    n_spin_orbitals: int,
) -> np.ndarray:
    """Binary row for a reflection product hat R_B = prod_{a in B} hat R_a."""
    gamma = build_incidence_matrix(groups, n_spin_orbitals)
    row = np.zeros(gamma.shape[0], dtype=np.int8)
    for index in generator_indices:
        row[int(index)] = 1
    return row


def term_allowed_parity_polynomial(
    term: tuple[tuple[int, int], ...],
    groups: list[frozenset[int]],
    product_rows: list[np.ndarray],
    n_spin_orbitals: int,
) -> bool:
    """Eq.~(13): allowed when B·q(M)=0 (mod 2) for every reflection product row in support."""
    if term == ():
        return True
    q = charge_vector_for_modes(modes_of_term(term), groups, n_spin_orbitals)
    for row in product_rows:
        if int(np.dot(row, q) % 2) != 0:
            return False
    return True


def term_allowed_dyall(
    term: tuple[tuple[int, int], ...],
    groups: list[frozenset[int]],
    inactive_spin: set[int],
    active_spin: set[int],
    virtual_spin: set[int],
    n_spin_orbitals: int,
) -> bool:
    """
    Structured Dyall subset of the CAS singleton parent (PDF Sec.~4.2).

    Retains CAS-allowed one-body terms and active-active two-body terms only.
    """
    if not term_allowed(term, groups, n_spin_orbitals):
        return False
    if len(term) <= 2:
        return True
    if len(term) == 4:
        modes = set(modes_of_term(term))
        return modes <= active_spin
    return False


def term_allowed_extended_cas(
    term: tuple[tuple[int, int], ...],
    groups: list[frozenset[int]],
    inactive_spin: set[int],
    active_spin: set[int],
    virtual_spin: set[int],
    n_spin_orbitals: int,
) -> bool:
    """
    Extended CAS parent: adds external density and inactive/virtual-active cross terms (Eq.~20).
    """
    if not term_allowed(term, groups, n_spin_orbitals):
        return False
    if len(term) <= 2:
        return True
    if len(term) == 4:
        modes = set(modes_of_term(term))
        external = inactive_spin | virtual_spin
        if modes <= active_spin or modes <= external:
            return True
        return bool(modes & external and modes & active_spin)
    return False


def project_fermion_with_filter(
    operator: FermionOperator,
    predicate,
    groups: list[frozenset[int]],
    inactive_spin: set[int],
    active_spin: set[int],
    virtual_spin: set[int],
    n_spin_orbitals: int,
    *,
    coef_tol: float = 1e-15,
) -> FermionOperator:
    projected = FermionOperator()
    for term, coef in operator.terms.items():
        if abs(coef) <= coef_tol:
            continue
        if predicate(term, groups, inactive_spin, active_spin, virtual_spin, n_spin_orbitals):
            projected += FermionOperator(term, coef)
    return projected


def project_fermion_to_cas_parent(
    operator: FermionOperator,
    external_spin: Iterable[int],
    n_spin_orbitals: int,
    **kwargs,
) -> FermionOperator:
    groups = singleton_groups(external_spin)
    return project_fermion_to_parent(operator, groups, n_spin_orbitals, **kwargs)


def project_fermion_to_dyall(
    operator: FermionOperator,
    n_electrons: int,
    n_spatial: int,
    **kwargs,
) -> FermionOperator:
    inactive, active, virtual = casscf_spin_orbital_partition(n_electrons, n_spatial)
    inactive_set, active_set, virtual_set = set(inactive), set(active), set(virtual)
    groups = singleton_groups(inactive + virtual)
    return project_fermion_with_filter(
        operator,
        term_allowed_dyall,
        groups,
        inactive_set,
        active_set,
        virtual_set,
        2 * n_spatial,
        **kwargs,
    )


def project_fermion_to_extended_cas(
    operator: FermionOperator,
    n_electrons: int,
    n_spatial: int,
    **kwargs,
) -> FermionOperator:
    inactive, active, virtual = casscf_spin_orbital_partition(n_electrons, n_spatial)
    inactive_set, active_set, virtual_set = set(inactive), set(active), set(virtual)
    groups = singleton_groups(inactive + virtual)
    return project_fermion_with_filter(
        operator,
        term_allowed_extended_cas,
        groups,
        inactive_set,
        active_set,
        virtual_set,
        2 * n_spatial,
        **kwargs,
    )


def build_h_sub_from_fermion(
    h_fermion: FermionOperator,
    basis_bitstrings: list[int],
    n_qubits: int,
) -> np.ndarray:
    try:
        from hamiltonian_generation import build_fixed_n_hamiltonian_direct
    except ImportError as exc:
        raise ImportError(f"PySCF/hamiltonian_generation required: {exc}") from exc
    return build_fixed_n_hamiltonian_direct(h_fermion, basis_bitstrings, n_qubits).toarray()


def quartet_product_rows(n_spatial: int, edges: Iterable[tuple[int, int]]) -> list[np.ndarray]:
    """Parity-polynomial rows for products hat R_p hat R_q on spatial pair groups."""
    groups = spatial_pair_groups(n_spatial)
    n_spin = 2 * n_spatial
    rows: list[np.ndarray] = []
    for p, q in edges:
        rows.append(reflection_product_row(groups, (p, q), n_spin))
    return rows


def project_h_sub_to_polynomial_parent(
    h_sub: sp.spmatrix | np.ndarray,
    basis_bitstrings: Iterable[int],
    n_spatial: int,
    edges: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Matrix projection commuting with all listed pair-product reflections."""
    groups = spatial_pair_groups(n_spatial)
    n_qubits = 2 * n_spatial
    diags: list[np.ndarray] = []
    for p, q in edges:
        d_p = pair_parity_diagonal(p, basis_bitstrings, n_spatial)
        d_q = pair_parity_diagonal(q, basis_bitstrings, n_spatial)
        diags.append(d_p * d_q)
    return project_h_sub_to_reflections(h_sub, diags)


def determinant_excitation_spin_orbitals(
    bit_i: int,
    bit_j: int,
    n_qubits: int,
) -> tuple[list[int], list[int]]:
    """Created and annihilated spin orbitals when coupling determinants i -> j."""
    created: list[int] = []
    annihilated: list[int] = []
    for mode in range(n_qubits):
        occ_i = mode_is_occupied(bit_i, mode, n_qubits)
        occ_j = mode_is_occupied(bit_j, mode, n_qubits)
        if occ_i < occ_j:
            created.append(mode)
        elif occ_i > occ_j:
            annihilated.append(mode)
    return created, annihilated


def matrix_element_allowed_dyall(
    bit_i: int,
    bit_j: int,
    active_spin: set[int],
    n_qubits: int,
) -> bool:
    """Dyall two-body support on fixed-N matrix elements (PDF Sec.~4.2)."""
    created, annihilated = determinant_excitation_spin_orbitals(bit_i, bit_j, n_qubits)
    n_exc = len(created) + len(annihilated)
    if n_exc <= 2:
        return True
    if n_exc == 4:
        return set(created + annihilated) <= active_spin
    return False


def matrix_element_allowed_extended_cas(
    bit_i: int,
    bit_j: int,
    inactive_spin: set[int],
    active_spin: set[int],
    virtual_spin: set[int],
    n_qubits: int,
) -> bool:
    """Extended-CAS two-body support on fixed-N matrix elements (Eq.~20)."""
    created, annihilated = determinant_excitation_spin_orbitals(bit_i, bit_j, n_qubits)
    n_exc = len(created) + len(annihilated)
    if n_exc <= 2:
        return True
    if n_exc == 4:
        modes = set(created + annihilated)
        external = inactive_spin | virtual_spin
        if modes <= active_spin or modes <= external:
            return True
        return bool(modes & external and modes & active_spin)
    return False


def _mask_h_sub_by_excitation_predicate(
    h_cas: np.ndarray,
    basis_bitstrings: list[int],
    predicate,
) -> np.ndarray:
    masked = h_cas.copy()
    for row, bit_i in enumerate(basis_bitstrings):
        for col, bit_j in enumerate(basis_bitstrings):
            if abs(masked[row, col]) <= 0.0:
                continue
            if not predicate(int(bit_i), int(bit_j)):
                masked[row, col] = 0.0
    return masked


def project_h_sub_to_dyall_parent(
    h_sub: sp.spmatrix | np.ndarray,
    basis_bitstrings: Iterable[int],
    n_electrons: int,
    n_spatial: int,
    *,
    n_active_spatial: int | None = None,
) -> np.ndarray:
    """CAS singleton parent followed by Dyall excitation masking."""
    basis_list = [int(b) for b in basis_bitstrings]
    n_qubits = 2 * n_spatial
    inactive, active, virtual = casscf_spin_orbital_partition(
        n_electrons,
        n_spatial,
        n_active_spatial=n_active_spatial,
    )
    external = external_singleton_spin_orbitals(
        n_electrons,
        n_spatial,
        n_active_spatial=n_active_spatial,
    )
    h_cas = project_h_sub_to_singleton_parent(h_sub, basis_list, external, n_qubits)
    active_set = set(active)

    def predicate(bit_i: int, bit_j: int) -> bool:
        return matrix_element_allowed_dyall(bit_i, bit_j, active_set, n_qubits)

    return _mask_h_sub_by_excitation_predicate(np.asarray(h_cas), basis_list, predicate)


def project_h_sub_to_extended_cas_parent(
    h_sub: sp.spmatrix | np.ndarray,
    basis_bitstrings: Iterable[int],
    n_electrons: int,
    n_spatial: int,
    *,
    n_active_spatial: int | None = None,
) -> np.ndarray:
    """CAS singleton parent followed by extended-CAS excitation masking."""
    basis_list = [int(b) for b in basis_bitstrings]
    n_qubits = 2 * n_spatial
    inactive, active, virtual = casscf_spin_orbital_partition(
        n_electrons,
        n_spatial,
        n_active_spatial=n_active_spatial,
    )
    external = external_singleton_spin_orbitals(
        n_electrons,
        n_spatial,
        n_active_spatial=n_active_spatial,
    )
    h_cas = project_h_sub_to_singleton_parent(h_sub, basis_list, external, n_qubits)
    inactive_set, active_set, virtual_set = set(inactive), set(active), set(virtual)

    def predicate(bit_i: int, bit_j: int) -> bool:
        return matrix_element_allowed_extended_cas(
            bit_i,
            bit_j,
            inactive_set,
            active_set,
            virtual_set,
            n_qubits,
        )

    return _mask_h_sub_by_excitation_predicate(np.asarray(h_cas), basis_list, predicate)
