"""Cross-check variance formulas and rotation representations."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.optimization import build_U_from_thetas, pair_list_for_n
from quasi_symmetries.optimization.quartet import (
    parity_stats_from_diagonal,
    rotate_state_to_orbital_frame,
    single_parity_diagonal,
)
from tests.helpers import hf_state_vector, load_h4_reference


class TestInternalConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.psi = hf_state_vector(cls.ref)

    def test_diagonal_vs_explicit_pair_parity_variance(self) -> None:
        n_spatial = self.ref["n_spatial"]
        u = np.eye(n_spatial, dtype=np.complex128)
        rotated = rotate_state_to_orbital_frame(
            self.psi,
            self.ref["basis_bitstrings"],
            u,
            n_spatial,
        )
        weights = np.abs(rotated) ** 2
        for orbital in range(n_spatial):
            diag = single_parity_diagonal(self.ref["basis_bitstrings"], orbital, n_spatial)
            stats = parity_stats_from_diagonal(rotated, diag)
            expectation = float(np.dot(weights, diag))
            var_explicit = max(0.0, 1.0 - expectation**2)
            self.assertLess(abs(stats.expectation - expectation), 1e-12)
            self.assertLess(abs(stats.variance - var_explicit), 1e-12)

    def test_pair_parity_zero_variance_at_identity_frame(self) -> None:
        n_spatial = self.ref["n_spatial"]
        u = np.eye(n_spatial, dtype=np.complex128)
        rotated = rotate_state_to_orbital_frame(
            self.psi,
            self.ref["basis_bitstrings"],
            u,
            n_spatial,
        )
        total_var = 0.0
        for orbital in range(n_spatial):
            diag = single_parity_diagonal(self.ref["basis_bitstrings"], orbital, n_spatial)
            total_var += parity_stats_from_diagonal(rotated, diag).variance
        self.assertLess(total_var, 1e-12)

    def test_rotated_frame_changes_variance(self) -> None:
        n_spatial = self.ref["n_spatial"]
        pairs = pair_list_for_n(n_spatial)
        rng = np.random.default_rng(7)
        thetas = 0.35 * rng.standard_normal(len(pairs))
        u = build_U_from_thetas(n_spatial, thetas, pairs)
        rotated = rotate_state_to_orbital_frame(
            self.psi,
            self.ref["basis_bitstrings"],
            u,
            n_spatial,
        )
        total_var = 0.0
        for orbital in range(n_spatial):
            diag = single_parity_diagonal(self.ref["basis_bitstrings"], orbital, n_spatial)
            total_var += parity_stats_from_diagonal(rotated, diag).variance
        self.assertGreater(total_var, 1e-6)


if __name__ == "__main__":
    unittest.main()
