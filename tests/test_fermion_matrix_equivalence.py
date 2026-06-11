"""Fermion-projected vs matrix-projected parents must agree on H4."""

from __future__ import annotations

import unittest

import numpy as np

from parity_parent_hamiltonians import (
    build_h_sub_from_fermion,
    external_singleton_spin_orbitals,
    project_fermion_to_parent,
    project_h_sub_to_pair_parent,
    project_h_sub_to_singleton_parent,
    singleton_groups,
    spatial_pair_groups,
)
from tests.helpers import H4_GEOMETRY, H4_MOLECULE, load_h4_reference, molecular_fermion_hamiltonian


class TestFermionMatrixEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        try:
            cls.h_fermion, _, _ = molecular_fermion_hamiltonian(H4_MOLECULE, H4_GEOMETRY)
            cls.pyscf_available = True
        except unittest.SkipTest:
            cls.h_fermion = None
            cls.pyscf_available = False

    def setUp(self) -> None:
        if not self.pyscf_available:
            self.skipTest("PySCF unavailable.")

    def test_pair_parent_fermion_matches_matrix(self) -> None:
        groups = spatial_pair_groups(self.ref["n_spatial"])
        h_fermion_parent = project_fermion_to_parent(
            self.h_fermion,
            groups,
            self.ref["n_qubits"],
        )
        h_fermion_sub = build_h_sub_from_fermion(
            h_fermion_parent,
            self.ref["basis_bitstrings"],
            self.ref["n_qubits"],
        )
        h_matrix = project_h_sub_to_pair_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
        )
        diff = h_fermion_sub - h_matrix
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(h_matrix), 1e-15))
        self.assertLess(rel, 1e-8)

    def test_cas_parent_fermion_matches_matrix(self) -> None:
        external = external_singleton_spin_orbitals(
            self.ref["n_electrons"],
            self.ref["n_spatial"],
        )
        groups = singleton_groups(external)
        h_fermion_parent = project_fermion_to_parent(
            self.h_fermion,
            groups,
            self.ref["n_qubits"],
        )
        h_fermion_sub = build_h_sub_from_fermion(
            h_fermion_parent,
            self.ref["basis_bitstrings"],
            self.ref["n_qubits"],
        )
        h_matrix = project_h_sub_to_singleton_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            external,
            self.ref["n_qubits"],
        )
        diff = h_fermion_sub - h_matrix
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(h_matrix), 1e-15))
        self.assertLess(rel, 1e-8)


if __name__ == "__main__":
    unittest.main()
