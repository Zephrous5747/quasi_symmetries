"""Parity polynomials (Eq.~13), overlapping groups, redundant generators on H4."""

from __future__ import annotations

import unittest

import numpy as np
from openfermion import FermionOperator

from parity_parent_hamiltonians import (
    build_incidence_matrix,
    charge_vector_for_modes,
    incidence_rows_dependent,
    max_reflection_commutator_frobenius,
    pair_parity_diagonal,
    project_fermion_to_parent,
    project_h_sub_to_polynomial_parent,
    project_h_sub_to_pair_parent,
    project_h_sub_to_reflections,
    quartet_product_rows,
    reflection_diagonal,
    reflection_product_row,
    singleton_groups,
    spatial_pair_groups,
    term_allowed_parity_polynomial,
)
from quartet_optimization_utils import (
    matching_edges,
    optimize_fixed_edge_quartets,
    quartet_cost_for_u,
)
from tests.helpers import (
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    load_h4_reference,
    random_unitary,
)


class TestParityPolynomialOverlapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()

    def test_redundant_generators_same_projection(self) -> None:
        groups = spatial_pair_groups(self.ref["n_spatial"])
        redundant = [*groups, groups[0]]
        self.assertTrue(incidence_rows_dependent(redundant, self.ref["n_qubits"]))
        op = FermionOperator("0^ 1^ 2 3", 0.25) + FermionOperator("0^ 2")
        p_minimal = project_fermion_to_parent(op, groups, self.ref["n_qubits"])
        p_redundant = project_fermion_to_parent(op, redundant, self.ref["n_qubits"])
        self.assertEqual(len(p_minimal.terms), len(p_redundant.terms))

    def test_overlapping_groups_projection(self) -> None:
        # Overlapping spin-orbital groups on four modes (PDF Sec.~6).
        groups = [frozenset({0, 1, 2}), frozenset({2, 3})]
        gamma = build_incidence_matrix(groups, 4)
        self.assertEqual(int(gamma[0, 2]), 1)
        self.assertEqual(int(gamma[1, 2]), 1)
        op = FermionOperator("0^ 3") + FermionOperator("0^ 1", 0.5)
        projected = project_fermion_to_parent(op, groups, 4)
        self.assertEqual(len(projected.terms), 1)
        self.assertAlmostEqual(float(np.real(projected.terms[((0, 1), (1, 0))])), 0.5)

    def test_quartet_polynomial_rows_match_product_reflections(self) -> None:
        edges = [(0, 1), (2, 3)]
        rows = quartet_product_rows(self.ref["n_spatial"], edges)
        groups = spatial_pair_groups(self.ref["n_spatial"])
        for (p, q), row in zip(edges, rows):
            expected = reflection_product_row(groups, (p, q), self.ref["n_qubits"])
            self.assertTrue(np.array_equal(row, expected))

    def test_polynomial_parent_commutes_with_quartet_products(self) -> None:
        from parity_parent_hamiltonians import max_reflection_commutator_frobenius

        edges = [(0, 1), (2, 3)]
        h_parent = project_h_sub_to_polynomial_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
            edges,
        )
        from parity_parent_hamiltonians import pair_parity_diagonal

        for p, q in edges:
            d_p = pair_parity_diagonal(p, self.ref["basis_bitstrings"], self.ref["n_spatial"])
            d_q = pair_parity_diagonal(q, self.ref["basis_bitstrings"], self.ref["n_spatial"])
            comm = max_reflection_commutator_frobenius(h_parent, d_p * d_q)
            self.assertLess(comm, 1e-8)

    def test_polynomial_filter_matches_matrix_projection_support(self) -> None:
        edges = [(0, 1), (2, 3)]
        groups = spatial_pair_groups(self.ref["n_spatial"])
        rows = quartet_product_rows(self.ref["n_spatial"], edges)
        try:
            from tests.helpers import H4_GEOMETRY, H4_MOLECULE, molecular_fermion_hamiltonian

            h_fermion, _, _ = molecular_fermion_hamiltonian(H4_MOLECULE, H4_GEOMETRY)
        except unittest.SkipTest:
            self.skipTest("PySCF unavailable.")
        allowed = 0
        for term in h_fermion.terms:
            if term_allowed_parity_polynomial(term, groups, rows, self.ref["n_qubits"]):
                allowed += 1
        self.assertGreater(allowed, 0)

    def test_polynomial_parent_subset_of_pair_parent(self) -> None:
        edges = [(0, 1), (2, 3)]
        h_pair = project_h_sub_to_pair_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
        )
        h_poly = project_h_sub_to_polynomial_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
            edges,
        )
        # Fewer generators => typically retains more matrix elements.
        self.assertGreaterEqual(
            float(np.linalg.norm(h_poly)),
            float(np.linalg.norm(h_pair)) - 1e-8,
        )

    def test_molecular_overlapping_groups_commute_with_projected_h(self) -> None:
        """Overlapping spin-orbital groups on H4 (PDF Sec.~6), not the 4-mode toy."""
        groups = [frozenset({0, 1, 2}), frozenset({2, 3, 4})]
        diags = [
            reflection_diagonal(group, self.ref["basis_bitstrings"], self.ref["n_qubits"])
            for group in groups
        ]
        h_parent = project_h_sub_to_reflections(self.ref["h_sub"], diags)
        for diag in diags:
            comm = max_reflection_commutator_frobenius(h_parent, diag)
            self.assertLess(comm, 1e-8)

    def test_fci_polynomial_quartet_optimization_decreases_variance(self) -> None:
        psi = np.asarray(self.ref["v_sub"], dtype=np.complex128)
        psi /= np.linalg.norm(psi)
        edges = matching_edges(self.ref["n_spatial"])
        u_start = random_unitary(self.ref["n_spatial"], seed=81)
        initial = quartet_cost_for_u(
            psi,
            self.ref["basis_bitstrings"],
            u_start,
            self.ref["n_spatial"],
            edges,
        )
        self.assertGreater(initial, 1e-4)

        best = optimize_fixed_edge_quartets(
            psi,
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
            edges,
            n_restarts=N_RESTARTS,
            random_seed=81,
            include_zero_start=False,
            parallel=False,
            maxfev=OPT_MAXFEV,
            maxiter=OPT_MAXITER,
        )
        self.assertLess(best["cost"], 0.5 * initial)

    def test_polynomial_rows_match_pair_products_on_h4(self) -> None:
        edges = matching_edges(self.ref["n_spatial"])
        rows = quartet_product_rows(self.ref["n_spatial"], edges)
        groups = spatial_pair_groups(self.ref["n_spatial"])
        for (p, q), row in zip(edges, rows):
            expected = reflection_product_row(groups, (p, q), self.ref["n_qubits"])
            self.assertTrue(np.array_equal(row, expected))
            d_product = (
                pair_parity_diagonal(p, self.ref["basis_bitstrings"], self.ref["n_spatial"])
                * pair_parity_diagonal(q, self.ref["basis_bitstrings"], self.ref["n_spatial"])
            )
            h_poly = project_h_sub_to_polynomial_parent(
                self.ref["h_sub"],
                self.ref["basis_bitstrings"],
                self.ref["n_spatial"],
                [(p, q)],
            )
            comm = max_reflection_commutator_frobenius(h_poly, d_product)
            self.assertLess(comm, 1e-8)


if __name__ == "__main__":
    unittest.main()
