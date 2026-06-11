"""Exact point-group symmetries as products of orbital pair parities."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp

from quasi_symmetries.optimization.quartet import (
    quartet_parity_diagonal,
    single_orbital_parity_value,
)
from quasi_symmetries.symmetry.labels import (
    MoleculeSymmetryLabels,
    h2o_mirror_orbital_sets,
    load_symmetry_labels,
    n2_ungerade_orbital_indices,
)
from quasi_symmetries.theory.parity_parent import max_reflection_commutator_frobenius


def product_parity_diagonal(
    basis_bitstrings: Iterable[int],
    orbitals: tuple[int, ...],
    n_spatial: int,
) -> np.ndarray:
    """Diagonal of prod_i (-1)^(n_ialpha + n_ibeta) over listed spatial orbitals."""
    basis_list = list(basis_bitstrings)
    if not orbitals:
        return np.ones(len(basis_list), dtype=np.float64)
    if len(orbitals) == 2:
        p, q = int(orbitals[0]), int(orbitals[1])
        edge = (p, q) if p < q else (q, p)
        return quartet_parity_diagonal(basis_list, edge, n_spatial)

    out = np.empty(len(basis_list), dtype=np.float64)
    for index, bitstring in enumerate(basis_list):
        parity = 1
        for orbital in orbitals:
            parity *= single_orbital_parity_value(int(bitstring), int(orbital), n_spatial)
        out[index] = float(parity)
    return out


def _mirror_parity_from_set(
    basis_bitstrings: Iterable[int],
    orbitals: tuple[int, ...],
    n_spatial: int,
) -> np.ndarray:
  return product_parity_diagonal(basis_bitstrings, orbitals, n_spatial)


def h2o_mirror_parities(
    basis_bitstrings: Iterable[int],
    labels: MoleculeSymmetryLabels,
) -> dict[str, np.ndarray]:
    """Return exact C2v mirror parities for H2O."""
    if labels.molecule != "h2o":
        raise ValueError("h2o_mirror_parities requires molecule='h2o'.")
    n_spatial = len(labels.irrep_labels)
    mirrors = h2o_mirror_orbital_sets(labels.irrep_labels)
    result: dict[str, np.ndarray] = {}
    for name, orbitals in mirrors.items():
        if not orbitals:
            continue
        result[name] = _mirror_parity_from_set(basis_bitstrings, orbitals, n_spatial)
    return result


def n2_inversion_parity(
    basis_bitstrings: Iterable[int],
    labels: MoleculeSymmetryLabels,
) -> np.ndarray:
    """Return spatial inversion parity prod_{u} (-1)^(n_iα + n_iβ) for N2."""
    if labels.molecule != "n2":
        raise ValueError("n2_inversion_parity requires molecule='n2'.")
    n_spatial = len(labels.irrep_labels)
    ungerade = tuple(n2_ungerade_orbital_indices(labels.irrep_labels))
    return product_parity_diagonal(basis_bitstrings, ungerade, n_spatial)


def exact_symmetry_parities_for_molecule(
    basis_bitstrings: Iterable[int],
    labels: MoleculeSymmetryLabels,
) -> dict[str, np.ndarray]:
    """Molecule-specific exact involutory parity diagonals."""
    if labels.molecule == "h2o":
        return h2o_mirror_parities(basis_bitstrings, labels)
    if labels.molecule == "n2":
        return {"P_inv": n2_inversion_parity(basis_bitstrings, labels)}
    raise ValueError(f"No exact symmetry map for molecule '{labels.molecule}'.")


def build_exact_symmetry_sectors(
    basis_bitstrings: Iterable[int],
    parity_diagonals: dict[str, np.ndarray],
) -> dict[tuple[int, ...], list[int]]:
    """Partition determinants by joint eigenvalues of named parity operators."""
    basis_list = list(basis_bitstrings)
    names = tuple(sorted(parity_diagonals))
    sectors: dict[tuple[int, ...], list[int]] = {}
    for index, _bitstring in enumerate(basis_list):
        key = tuple(int(np.sign(parity_diagonals[name][index])) for name in names)
        sectors.setdefault(key, []).append(index)
    return sectors


def ground_state_sector_indices(
    parity_diagonals: dict[str, np.ndarray],
    *,
    target: int = 1,
) -> list[int]:
    """Indices where every listed parity equals target (+1 for bosonic ground states)."""
    if not parity_diagonals:
        return []
    length = len(next(iter(parity_diagonals.values())))
    indices: list[int] = []
    for index in range(length):
        if all(int(np.sign(diag[index])) == int(target) for diag in parity_diagonals.values()):
            indices.append(index)
    return indices


def restrict_indices_to_sector(
    parity_diagonals: dict[str, np.ndarray],
    *,
    target: int = 1,
) -> list[int]:
    return ground_state_sector_indices(parity_diagonals, target=target)


def restrict_matrix_to_indices(matrix: np.ndarray | sp.spmatrix, indices: list[int]) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if sp.issparse(matrix):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix, dtype=np.complex128)
    block = dense[np.ix_(idx, idx)]
    return 0.5 * (block + block.conj().T)


def verify_exact_symmetries(
    h_sub: sp.spmatrix | np.ndarray,
    parity_diagonals: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return Frobenius commutator norms ||[H, Q]||_F for each parity operator."""
    errors: dict[str, float] = {}
    for name, diagonal in parity_diagonals.items():
        errors[name] = max_reflection_commutator_frobenius(h_sub, diagonal)
    return errors


def exact_symmetry_data_for_reference(
    ref: dict[str, Any],
    *,
    molecule: str | None = None,
) -> tuple[MoleculeSymmetryLabels, dict[str, np.ndarray]] | None:
    """Load labels and parity diagonals for a cached reference state."""
    labels = load_symmetry_labels(ref, molecule=molecule)
    if labels is None:
        return None
    parities = exact_symmetry_parities_for_molecule(ref["basis_bitstrings"], labels)
    return labels, parities
