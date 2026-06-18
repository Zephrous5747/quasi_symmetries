"""Synthetic exact tests for archived ABC variance optimization."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.optimization import (
    compute_spin_rdms_from_subspace_state,
    pair_list_for_n,
)
from quasi_symmetries_abc.optimization.variance import (
    optimize_variance_restricted,
    variance_restricted,
)
from tests.helpers import (
    VAR_TOL,
    hf_state_vector,
    load_h4_reference,
)


class TestAbcVarianceExactRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.psi = hf_state_vector(cls.ref)
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)

    def test_fixed_abc_zero_variance_at_identity_frame(self) -> None:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_subspace_state(
            self.psi,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        m = len(self.pairs)
        x0 = np.zeros(m + 2)
        x0[m] = np.arccos(-2.0 / np.sqrt(6.0))
        x0[m + 1] = np.pi / 4.0
        v, _, _, _, _, _ = variance_restricted(gamma_a, gamma_b, gamma_ab, x0, self.pairs)
        self.assertLess(v, VAR_TOL)

    def test_optimize_fixed_abc_seniority_via_production_optimizer(self) -> None:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_subspace_state(
            self.psi,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        best = optimize_variance_restricted(gamma_a, gamma_b, gamma_ab)
        self.assertLess(best["V"], VAR_TOL)


if __name__ == "__main__":
    unittest.main()
