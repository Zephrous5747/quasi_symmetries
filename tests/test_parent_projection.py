"""Verify molecular Hamiltonians project to zero commutators with parity reflections."""

from __future__ import annotations

import unittest

from parity_parent_hamiltonians import (
    casscf_spin_orbital_partition,
    max_reflection_commutator_frobenius,
    pair_parity_diagonal,
    project_h_sub_to_pair_parent,
    project_h_sub_to_singleton_parent,
    reflection_diagonal,
)
from tests.helpers import load_h4_reference


class TestParentProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()

    def test_pair_parent_commutes_with_all_pair_reflections(self) -> None:
        h_parent = project_h_sub_to_pair_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
        )
        for orbital in range(self.ref["n_spatial"]):
            diag = pair_parity_diagonal(orbital, self.ref["basis_bitstrings"], self.ref["n_spatial"])
            comm = max_reflection_commutator_frobenius(h_parent, diag)
            self.assertLess(comm, 1e-8, msg=f"orbital={orbital}")

    def test_physical_hamiltonian_has_nonzero_pair_commutator(self) -> None:
        h_phys = self.ref["h_sub"]
        max_comm = 0.0
        for orbital in range(self.ref["n_spatial"]):
            diag = pair_parity_diagonal(orbital, self.ref["basis_bitstrings"], self.ref["n_spatial"])
            max_comm = max(max_comm, max_reflection_commutator_frobenius(h_phys, diag))
        self.assertGreater(max_comm, 1e-6)

    def test_singleton_projection_commutes_on_inactive_virtual(self) -> None:
        inactive, _, virtual = casscf_spin_orbital_partition(
            self.ref["n_electrons"],
            self.ref["n_spatial"],
        )
        h_parent = project_h_sub_to_singleton_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            inactive + virtual,
            self.ref["n_qubits"],
        )
        for spin_orb in inactive[:2] + virtual[:2]:
            if spin_orb in virtual:
                continue  # H4 has no virtual spin orbitals at this geometry
            diag = reflection_diagonal({spin_orb}, self.ref["basis_bitstrings"], self.ref["n_qubits"])
            comm = max_reflection_commutator_frobenius(h_parent, diag)
            self.assertLess(comm, 1e-8, msg=f"spin_orbital={spin_orb}")


if __name__ == "__main__":
    unittest.main()
