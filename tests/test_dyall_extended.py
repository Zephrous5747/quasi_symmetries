"""Dyall and extended-Dyall parent hierarchy on H4 (PDF Sec.~4.2)."""

from __future__ import annotations

import unittest

from parity_parent_hamiltonians import (
    external_singleton_spin_orbitals,
    max_reflection_commutator_frobenius,
    projection_error_frobenius,
    project_fermion_to_dyall,
    project_fermion_to_extended_cas,
    project_h_sub_to_singleton_parent,
    reflection_diagonal,
    relative_projection_error,
    spin_sets_from_partition,
    term_allowed_dyall,
    term_allowed_extended_cas,
)
from tests.helpers import H4_GEOMETRY, H4_MOLECULE, load_h4_reference, molecular_fermion_hamiltonian


class TestDyallExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.external = external_singleton_spin_orbitals(
            cls.ref["n_electrons"],
            cls.ref["n_spatial"],
        )
        cls.inactive, cls.active, cls.virtual = spin_sets_from_partition(
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

    def _require_fermion(self):
        try:
            return molecular_fermion_hamiltonian(H4_MOLECULE, H4_GEOMETRY)
        except unittest.SkipTest:
            self.skipTest("PySCF unavailable.")

    def test_extended_allows_more_two_body_than_dyall(self) -> None:
        h_fermion, _, n_spin = self._require_fermion()
        from parity_parent_hamiltonians import singleton_groups, term_allowed

        groups = singleton_groups(self.external)
        dyall_count = 0
        extended_count = 0
        cas_count = 0
        for term in h_fermion.terms:
            if len(term) != 4:
                continue
            if term_allowed(term, groups, n_spin):
                cas_count += 1
            if term_allowed_dyall(term, groups, self.inactive, self.active, self.virtual, n_spin):
                dyall_count += 1
            if term_allowed_extended_cas(
                term, groups, self.inactive, self.active, self.virtual, n_spin
            ):
                extended_count += 1
        self.assertGreaterEqual(extended_count, dyall_count)
        self.assertGreaterEqual(cas_count, extended_count)

    def test_fermion_dyall_projection_error_le_extended(self) -> None:
        h_fermion, _, _ = self._require_fermion()
        from parity_parent_hamiltonians import (
            build_h_sub_from_fermion,
            project_fermion_to_parent,
            singleton_groups,
        )

        groups = singleton_groups(self.external)
        n_spin = self.ref["n_qubits"]
        h_sub = self.h_dense
        h_dyall = build_h_sub_from_fermion(
            project_fermion_to_dyall(h_fermion, self.ref["n_electrons"], self.ref["n_spatial"]),
            self.ref["basis_bitstrings"],
            self.ref["n_qubits"],
        )
        h_ext = build_h_sub_from_fermion(
            project_fermion_to_extended_cas(h_fermion, self.ref["n_electrons"], self.ref["n_spatial"]),
            self.ref["basis_bitstrings"],
            self.ref["n_qubits"],
        )
        h_cas = build_h_sub_from_fermion(
            project_fermion_to_parent(h_fermion, groups, n_spin),
            self.ref["basis_bitstrings"],
            self.ref["n_qubits"],
        )
        err_dyall = relative_projection_error(h_sub, h_dyall)
        err_ext = relative_projection_error(h_sub, h_ext)
        err_cas = relative_projection_error(h_sub, h_cas)
        self.assertGreaterEqual(err_dyall, err_ext - 1e-12)
        self.assertGreaterEqual(err_cas, err_ext - 1e-12)

    def test_dyall_and_extended_commute_with_external_singletons(self) -> None:
        h_fermion, _, _ = self._require_fermion()
        from parity_parent_hamiltonians import build_h_sub_from_fermion

        for projector, label in (
            (project_fermion_to_dyall, "dyall"),
            (project_fermion_to_extended_cas, "extended"),
        ):
            h_parent = build_h_sub_from_fermion(
                projector(h_fermion, self.ref["n_electrons"], self.ref["n_spatial"]),
                self.ref["basis_bitstrings"],
                self.ref["n_qubits"],
            )
            for spin_orb in list(self.inactive)[:2]:
                diag = reflection_diagonal({spin_orb}, self.ref["basis_bitstrings"], self.ref["n_qubits"])
                comm = max_reflection_commutator_frobenius(h_parent, diag)
                self.assertLess(comm, 1e-8, msg=f"{label} spin={spin_orb}")

    def test_matrix_cas_parent_is_idempotent(self) -> None:
        twice = project_h_sub_to_singleton_parent(
            self.h_cas,
            self.ref["basis_bitstrings"],
            self.external,
            self.ref["n_qubits"],
        )
        self.assertLess(projection_error_frobenius(self.h_cas, twice), 1e-10)


if __name__ == "__main__":
    unittest.main()
