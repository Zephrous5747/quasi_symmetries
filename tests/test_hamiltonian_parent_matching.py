"""Eq. (32): Hamiltonian parent-matching norm vs parity variance."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.optimization import build_U_from_thetas, pair_list_for_n
from quasi_symmetries.theory.parity_parent import (
    external_singleton_spin_orbitals,
    pair_parity_variance_sum,
    projection_error_frobenius,
    project_h_sub_to_pair_parent,
    project_h_sub_to_singleton_parent,
    relative_projection_error,
    rotate_h_sub_dense,
)
from quasi_symmetries.optimization.quartet import rotate_state_to_orbital_frame
from tests.helpers import hf_state_vector, load_h4_reference, random_unitary


class TestHamiltonianParentMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.psi_hf = hf_state_vector(cls.ref)
        cls.n_spatial = cls.ref["n_spatial"]
        cls.h_dense = cls.ref["h_sub"].toarray()
        cls.h_pair_parent = project_h_sub_to_pair_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            cls.n_spatial,
        )
        external = external_singleton_spin_orbitals(
            cls.ref["n_electrons"],
            cls.n_spatial,
        )
        cls.h_cas_parent = project_h_sub_to_singleton_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            external,
            cls.ref["n_qubits"],
        )

    def test_projection_idempotent_pair_parent(self) -> None:
        twice = project_h_sub_to_pair_parent(
            self.h_pair_parent,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        self.assertLess(projection_error_frobenius(self.h_pair_parent, twice), 1e-10)

    def test_pair_parent_has_zero_projection_error(self) -> None:
        self.assertLess(
            projection_error_frobenius(self.h_pair_parent, self.h_pair_parent),
            1e-10,
        )

    def test_physical_hamiltonian_has_positive_pair_projection_error(self) -> None:
        projected = project_h_sub_to_pair_parent(
            self.h_dense,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        self.assertGreater(relative_projection_error(self.h_dense, projected), 1e-6)

    def test_cas_parent_has_zero_singleton_projection_error(self) -> None:
        external = external_singleton_spin_orbitals(
            self.ref["n_electrons"],
            self.n_spatial,
        )
        projected = project_h_sub_to_singleton_parent(
            self.h_cas_parent,
            self.ref["basis_bitstrings"],
            external,
            self.ref["n_qubits"],
        )
        self.assertLess(projection_error_frobenius(self.h_cas_parent, projected), 1e-10)

    def test_rotated_parent_breaks_fixed_basis_reflection_commutativity(self) -> None:
        """
        H(U) = R† H R does not commute with canonical-basis R_p unless R is trivial.

        Parity reflections in the PDF are orbital-frame objects; our fixed determinant
        basis keeps R_p diagonal in the original labels while only H is rotated.
        """
        u = random_unitary(self.n_spatial, seed=43)
        h_rot = rotate_h_sub_dense(
            u,
            self.h_pair_parent,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        from quasi_symmetries.theory.parity_parent import max_reflection_commutator_frobenius, pair_parity_diagonal

        max_comm = 0.0
        for orbital in range(self.n_spatial):
            diag = pair_parity_diagonal(orbital, self.ref["basis_bitstrings"], self.n_spatial)
            max_comm = max(max_comm, max_reflection_commutator_frobenius(h_rot, diag))
        self.assertGreater(max_comm, 1e-6)

    def test_rotated_physical_hamiltonian_projection_error_changes(self) -> None:
        pairs = pair_list_for_n(self.n_spatial)
        thetas = 0.25 * np.random.default_rng(43).standard_normal(len(pairs))
        u = build_U_from_thetas(self.n_spatial, thetas, pairs)
        h_rot = rotate_h_sub_dense(u, self.h_dense, self.ref["basis_bitstrings"], self.n_spatial)
        err_identity = relative_projection_error(
            self.h_dense,
            project_h_sub_to_pair_parent(
                self.h_dense,
                self.ref["basis_bitstrings"],
                self.n_spatial,
            ),
        )
        err_rotated = relative_projection_error(
            h_rot,
            project_h_sub_to_pair_parent(
                h_rot,
                self.ref["basis_bitstrings"],
                self.n_spatial,
            ),
        )
        self.assertGreater(abs(err_identity - err_rotated), 1e-8)

    def test_eq32_commutator_and_projection_both_detect_non_parent_physical_h(self) -> None:
        """
        Eq. (32) uses ||H - P_parent[H]||; Eq. (8) uses [H, R_a] = 0.

        In a fixed determinant basis both vanish on the parent class and are positive
        for the physical Hamiltonian, but they are not numerically identical measures.
        """
        from quasi_symmetries.theory.parity_parent import max_reflection_commutator_frobenius, pair_parity_diagonal

        comm_sum = 0.0
        for orbital in range(self.n_spatial):
            diag = pair_parity_diagonal(orbital, self.ref["basis_bitstrings"], self.n_spatial)
            comm_sum += max_reflection_commutator_frobenius(self.h_dense, diag) ** 2

        proj_err = projection_error_frobenius(
            self.h_dense,
            project_h_sub_to_pair_parent(
                self.h_dense,
                self.ref["basis_bitstrings"],
                self.n_spatial,
            ),
        )
        parent_comm = 0.0
        for orbital in range(self.n_spatial):
            diag = pair_parity_diagonal(orbital, self.ref["basis_bitstrings"], self.n_spatial)
            parent_comm = max(parent_comm, max_reflection_commutator_frobenius(self.h_pair_parent, diag))

        self.assertGreater(comm_sum, 1e-12)
        self.assertGreater(proj_err, 1e-8)
        self.assertLess(parent_comm, 1e-8)
        self.assertLess(projection_error_frobenius(self.h_pair_parent, self.h_pair_parent), 1e-10)

    def test_hf_zero_pair_variance_implies_zero_projection_for_parent_not_physical(self) -> None:
        rotated = rotate_state_to_orbital_frame(
            self.psi_hf,
            self.ref["basis_bitstrings"],
            np.eye(self.n_spatial),
            self.n_spatial,
        )
        var_sum = pair_parity_variance_sum(
            rotated,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        self.assertLess(var_sum, 1e-12)
        self.assertGreater(
            relative_projection_error(
                self.h_dense,
                project_h_sub_to_pair_parent(
                    self.h_dense,
                    self.ref["basis_bitstrings"],
                    self.n_spatial,
                ),
            ),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
