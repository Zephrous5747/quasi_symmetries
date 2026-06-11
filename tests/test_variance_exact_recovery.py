"""Synthetic exact tests: HF eigenstates should recover zero parity variance after optimization."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize

from quasi_symmetries.optimization import (
    ANGLE_INIT_SCALE,
    build_U_from_thetas,
    compute_spin_rdms_from_subspace_state,
    optimize_variance_restricted,
    pair_list_for_n,
    variance_restricted,
)
from quasi_symmetries.optimization.quartet import (
    matching_edges,
    optimize_fixed_edge_quartets,
    quartet_cost_for_u,
    single_parity_diagonal,
    parity_stats_from_diagonal,
    rotate_state_to_orbital_frame,
)
from tests.helpers import (
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    VAR_TOL,
    hf_state_vector,
    load_h4_reference,
    random_unitary,
)


class TestVarianceExactRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.psi = hf_state_vector(cls.ref)
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)

    def _pair_parity_cost(self, u: np.ndarray) -> float:
        rotated = rotate_state_to_orbital_frame(
            self.psi,
            self.ref["basis_bitstrings"],
            u,
            self.n_spatial,
        )
        total = 0.0
        for orbital in range(self.n_spatial):
            diag = single_parity_diagonal(self.ref["basis_bitstrings"], orbital, self.n_spatial)
            total += parity_stats_from_diagonal(rotated, diag).variance
        return float(total)

    def test_hf_zero_pair_variance_identity_frame(self) -> None:
        u = np.eye(self.n_spatial, dtype=np.complex128)
        self.assertLess(self._pair_parity_cost(u), 1e-12)

    def test_optimize_pair_parity_from_random_unitary(self) -> None:
        u_start = random_unitary(self.n_spatial, seed=11)
        self.assertGreater(self._pair_parity_cost(u_start), 1e-4)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return self._pair_parity_cost(u)

        rng = np.random.default_rng(11)
        best_cost = float("inf")
        for restart in range(N_RESTARTS):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={"maxiter": OPT_MAXITER, "maxfev": OPT_MAXFEV, "disp": False},
            )
            best_cost = min(best_cost, float(objective(res.x)))

        self.assertLess(best_cost, VAR_TOL)

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

    def test_optimize_quartet_matching_from_random_unitary(self) -> None:
        u_start = random_unitary(self.n_spatial, seed=17)
        edges = matching_edges(self.n_spatial)
        initial_cost = quartet_cost_for_u(
            self.psi,
            self.ref["basis_bitstrings"],
            u_start,
            self.n_spatial,
            edges,
        )
        self.assertGreater(initial_cost, 1e-4)

        best = optimize_fixed_edge_quartets(
            self.psi,
            self.ref["basis_bitstrings"],
            self.n_spatial,
            edges,
            n_restarts=N_RESTARTS,
            random_seed=17,
            include_zero_start=False,
            parallel=False,
            maxfev=OPT_MAXFEV,
            maxiter=OPT_MAXITER,
        )
        self.assertLess(best["cost"], VAR_TOL)


if __name__ == "__main__":
    unittest.main()
