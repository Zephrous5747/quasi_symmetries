"""Unit tests for parity incidence rules and parent projection."""

from __future__ import annotations

import unittest

import numpy as np
from openfermion import FermionOperator

from parity_parent_hamiltonians import (
    build_incidence_matrix,
    charge_vector_for_modes,
    onebody_allowed,
    project_fermion_to_parent,
    singleton_groups,
    spatial_pair_groups,
    term_allowed,
    twobody_allowed,
)


class TestParityAlgebra(unittest.TestCase):
    def test_pair_group_incidence(self) -> None:
        groups = spatial_pair_groups(3)
        gamma = build_incidence_matrix(groups, 6)
        self.assertEqual(gamma.shape, (3, 6))
        self.assertEqual(int(gamma[0, 0]), 1)
        self.assertEqual(int(gamma[0, 1]), 1)
        self.assertEqual(int(gamma[0, 2]), 0)

    def test_singleton_onebody_rule(self) -> None:
        groups = singleton_groups([0, 1, 4, 5])
        n = 6
        # Active spin orbitals 2,3 are outside all singleton groups -> gamma_p = gamma_q = 0.
        self.assertTrue(onebody_allowed(2, 3, groups, n))
        self.assertFalse(onebody_allowed(0, 2, groups, n))
        self.assertFalse(onebody_allowed(0, 1, groups, n))

    def test_pair_twobody_rule(self) -> None:
        groups = spatial_pair_groups(2)
        n = 4
        self.assertTrue(twobody_allowed(0, 1, 2, 3, groups, n))
        self.assertTrue(twobody_allowed(0, 2, 1, 3, groups, n))

    def test_charge_vector_xor(self) -> None:
        groups = spatial_pair_groups(2)
        q = charge_vector_for_modes([0, 1], groups, 4)
        self.assertTrue(np.array_equal(q, np.zeros(2, dtype=np.int8)))
        q_cross = charge_vector_for_modes([0, 2], groups, 4)
        self.assertTrue(np.array_equal(q_cross, np.array([1, 1], dtype=np.int8)))

    def test_project_fermion_doci_parent(self) -> None:
        groups = spatial_pair_groups(2)
        n = 4
        op = FermionOperator("0^ 2") + FermionOperator("0^ 1^ 2 3", 0.5)
        projected = project_fermion_to_parent(op, groups, n)
        self.assertFalse(term_allowed(((0, 1), (2, 0)), groups, n))
        self.assertEqual(len(projected.terms), 1)
        self.assertAlmostEqual(float(np.real(projected.terms[((0, 1), (1, 1), (2, 0), (3, 0))])), 0.5)

    def test_constant_term_always_allowed(self) -> None:
        groups = singleton_groups([0])
        self.assertTrue(term_allowed((), groups, 2))


if __name__ == "__main__":
    unittest.main()
