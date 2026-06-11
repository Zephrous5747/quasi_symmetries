"""Matrix-only Dyall hierarchy when PySCF is unavailable."""

from __future__ import annotations

import unittest

import numpy as np

from parity_parent_hamiltonians import (
    casscf_spin_orbital_partition,
    external_singleton_spin_orbitals,
    max_reflection_commutator_frobenius,
    projection_error_frobenius,
    project_h_sub_to_dyall_parent,
    project_h_sub_to_extended_cas_parent,
    project_h_sub_to_pair_parent,
    project_h_sub_to_singleton_parent,
    reflection_diagonal,
    relative_projection_error,
)
from tests.helpers import load_h4_reference, load_lih_reference

LIH_ACTIVE_SPATIAL = 2


class TestDyallExtendedMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.external = external_singleton_spin_orbitals(
            cls.ref["n_electrons"],
            cls.ref["n_spatial"],
        )
        cls.h_dense = cls.ref["h_sub"].toarray()
        cls.h_cas = project_h_sub_to_singleton_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            cls.external,
            cls.ref["n_qubits"],
        )
        cls.h_pair = project_h_sub_to_pair_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            cls.ref["n_spatial"],
        )

    def test_cas_projection_error_le_pair_on_physical_h(self) -> None:
        err_cas = relative_projection_error(self.h_dense, self.h_cas)
        err_pair = relative_projection_error(self.h_dense, self.h_pair)
        self.assertGreaterEqual(err_cas, err_pair - 1e-12)

    def test_pair_and_cas_parents_are_idempotent(self) -> None:
        cas_twice = project_h_sub_to_singleton_parent(
            self.h_cas,
            self.ref["basis_bitstrings"],
            self.external,
            self.ref["n_qubits"],
        )
        pair_twice = project_h_sub_to_pair_parent(
            self.h_pair,
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
        )
        self.assertLess(projection_error_frobenius(self.h_cas, cas_twice), 1e-10)
        self.assertLess(projection_error_frobenius(self.h_pair, pair_twice), 1e-10)

    def test_matrix_dyall_and_extended_projection_errors_ordered(self) -> None:
        h_dyall = project_h_sub_to_dyall_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_electrons"],
            self.ref["n_spatial"],
        )
        h_extended = project_h_sub_to_extended_cas_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_electrons"],
            self.ref["n_spatial"],
        )
        err_dyall = relative_projection_error(self.h_dense, h_dyall)
        err_extended = relative_projection_error(self.h_dense, h_extended)
        err_cas = relative_projection_error(self.h_dense, self.h_cas)
        self.assertGreaterEqual(err_dyall, err_extended - 1e-12)
        self.assertGreaterEqual(err_cas, err_extended - 1e-12)

    def test_matrix_dyall_commutes_with_external_singletons(self) -> None:
        h_dyall = project_h_sub_to_dyall_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_electrons"],
            self.ref["n_spatial"],
        )
        for spin_orb in self.external[:2]:
            diag = reflection_diagonal({spin_orb}, self.ref["basis_bitstrings"], self.ref["n_qubits"])
            comm = max_reflection_commutator_frobenius(h_dyall, diag)
            self.assertLess(comm, 1e-8)


class TestDyallExtendedMatrixLiH(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_lih_reference()
        cls.external = external_singleton_spin_orbitals(
            cls.ref["n_electrons"],
            cls.ref["n_spatial"],
            n_active_spatial=LIH_ACTIVE_SPATIAL,
        )
        cls.h_dense = cls.ref["h_sub"].toarray()
        cls.h_cas = project_h_sub_to_singleton_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            cls.external,
            cls.ref["n_qubits"],
        )
        cls.h_dyall = project_h_sub_to_dyall_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            cls.ref["n_electrons"],
            cls.ref["n_spatial"],
            n_active_spatial=LIH_ACTIVE_SPATIAL,
        )
        cls.h_extended = project_h_sub_to_extended_cas_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            cls.ref["n_electrons"],
            cls.ref["n_spatial"],
            n_active_spatial=LIH_ACTIVE_SPATIAL,
        )

    def test_virtual_partition_adds_external_singleton_generators(self) -> None:
        full_external = external_singleton_spin_orbitals(
            self.ref["n_electrons"],
            self.ref["n_spatial"],
        )
        h_full_cas = project_h_sub_to_singleton_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            full_external,
            self.ref["n_qubits"],
        )
        inactive, _, virtual = casscf_spin_orbital_partition(
            self.ref["n_electrons"],
            self.ref["n_spatial"],
            n_active_spatial=LIH_ACTIVE_SPATIAL,
        )
        self.assertGreater(len(virtual), 0)
        self.assertGreater(len(self.external), len(full_external))
        # More singleton parities => stricter parent => no larger matrix support.
        self.assertLessEqual(
            float(np.linalg.norm(self.h_cas)),
            float(np.linalg.norm(h_full_cas)) + 1e-8,
        )
        self.assertGreater(
            relative_projection_error(self.h_dense, self.h_cas),
            relative_projection_error(self.h_dense, h_full_cas),
        )

    def test_matrix_dyall_hierarchy_with_virtual_orbitals(self) -> None:
        err_dyall = relative_projection_error(self.h_dense, self.h_dyall)
        err_extended = relative_projection_error(self.h_dense, self.h_extended)
        err_cas = relative_projection_error(self.h_dense, self.h_cas)
        self.assertGreaterEqual(err_dyall, err_extended - 1e-12)
        self.assertGreaterEqual(err_cas, err_extended - 1e-12)

    def test_dyall_parent_idempotent_on_lih(self) -> None:
        twice = project_h_sub_to_dyall_parent(
            self.h_dyall,
            self.ref["basis_bitstrings"],
            self.ref["n_electrons"],
            self.ref["n_spatial"],
            n_active_spatial=LIH_ACTIVE_SPATIAL,
        )
        self.assertLess(projection_error_frobenius(self.h_dyall, twice), 1e-9)


if __name__ == "__main__":
    unittest.main()
