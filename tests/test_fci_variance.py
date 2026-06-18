"""FCI-state variance behavior on H4 (not an exact parity eigenstate)."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize

from quasi_symmetries.optimization import (
    build_U_from_thetas,
    pair_list_for_n,
)
from quasi_symmetries.theory.parity_parent import pair_parity_variance_sum
from quasi_symmetries.optimization.quartet import (
    matching_edges,
    optimize_fixed_edge_quartets,
    rotate_state_to_orbital_frame,
)
from quasi_symmetries.optimization import ANGLE_INIT_SCALE
from tests.helpers import (
    LIH_N_RESTARTS,
    LIH_OPT_MAXFEV,
    LIH_OPT_MAXITER,
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    load_h4_reference,
    load_lih_reference,
    random_unitary,
)


class _FciVarianceTests:
    molecule: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if cls.molecule == "h4":
            cls.ref = load_h4_reference()
            cls.n_restarts = N_RESTARTS
            cls.maxfev = OPT_MAXFEV
            cls.maxiter = OPT_MAXITER
            cls.random_seed = 53
            cls.quartet_seed = 59
            cls.label = "h4_fci"
        elif cls.molecule == "lih":
            cls.ref = load_lih_reference()
            cls.n_restarts = LIH_N_RESTARTS
            cls.maxfev = LIH_OPT_MAXFEV
            cls.maxiter = LIH_OPT_MAXITER
            cls.random_seed = 55
            cls.quartet_seed = 57
            cls.label = "lih_fci"
        else:
            raise ValueError(f"Unknown molecule '{cls.molecule}'.")

        cls.psi = np.asarray(cls.ref["v_sub"], dtype=np.complex128)
        cls.psi /= np.linalg.norm(cls.psi)
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)

    def _pair_cost(self, u: np.ndarray) -> float:
        rotated = rotate_state_to_orbital_frame(
            self.psi,
            self.ref["basis_bitstrings"],
            u,
            self.n_spatial,
        )
        return pair_parity_variance_sum(
            rotated,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )

    def _optimize_pair_from(self, u_start: np.ndarray, seed: int) -> float:
        initial = self._pair_cost(u_start)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return self._pair_cost(u)

        rng = np.random.default_rng(seed)
        best = initial
        for _ in range(self.n_restarts):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={"maxiter": self.maxiter, "maxfev": self.maxfev, "disp": False},
            )
            best = min(best, float(objective(res.x)))
        return best

    def test_fci_has_positive_pair_variance_at_identity(self) -> None:
        u = np.eye(self.n_spatial, dtype=np.complex128)
        self.assertGreater(self._pair_cost(u), 1e-8)

    def test_fci_pair_optimization_does_not_increase_best_cost(self) -> None:
        u_start = random_unitary(self.n_spatial, seed=self.random_seed)
        initial = self._pair_cost(u_start)
        best = self._optimize_pair_from(u_start, self.random_seed)
        self.assertLessEqual(best, initial + 1e-10)

    def test_fci_pair_optimization_decreases_cost_measurably(self) -> None:
        u_start = random_unitary(self.n_spatial, seed=self.random_seed + 1)
        initial = self._pair_cost(u_start)
        best = self._optimize_pair_from(u_start, self.random_seed + 1)
        self.assertLess(best, 0.5 * initial)

    def test_fci_quartet_optimization_does_not_increase_cost(self) -> None:
        edges = matching_edges(self.n_spatial)
        u_start = random_unitary(self.n_spatial, seed=self.quartet_seed)
        from quasi_symmetries.optimization.quartet import quartet_cost_for_u

        initial = quartet_cost_for_u(
            self.psi,
            self.ref["basis_bitstrings"],
            u_start,
            self.n_spatial,
            edges,
        )
        best = optimize_fixed_edge_quartets(
            self.psi,
            self.ref["basis_bitstrings"],
            self.n_spatial,
            edges,
            n_restarts=self.n_restarts,
            random_seed=self.quartet_seed,
            include_zero_start=False,
            parallel=False,
            maxfev=self.maxfev,
            maxiter=self.maxiter,
        )
        self.assertLessEqual(best["cost"], initial + 1e-10)


class TestFciVarianceH4(_FciVarianceTests, unittest.TestCase):
    molecule = "h4"


class TestFciVarianceLiH(_FciVarianceTests, unittest.TestCase):
    molecule = "lih"


if __name__ == "__main__":
    unittest.main()
