"""Eq.~(32): projection norm tracked alongside parity-variance optimization."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize

from optimization_abc_utils import ANGLE_INIT_SCALE, build_U_from_thetas, pair_list_for_n
from parity_parent_hamiltonians import (
    pair_parity_variance_sum,
    projection_error_frobenius,
    project_h_sub_to_pair_parent,
    relative_projection_error,
    rotate_h_sub_dense,
)
from quartet_optimization_utils import rotate_state_to_orbital_frame
from tests.helpers import (
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    LIH_N_RESTARTS,
    LIH_OPT_MAXFEV,
    LIH_OPT_MAXITER,
    load_h4_reference,
    load_lih_reference,
    random_unitary,
)


class _Eq32OptimizerTests:
    molecule: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if cls.molecule == "h4":
            cls.ref = load_h4_reference()
            cls.n_restarts = N_RESTARTS
            cls.maxfev = OPT_MAXFEV
            cls.maxiter = OPT_MAXITER
            cls.seed = 91
        elif cls.molecule == "lih":
            cls.ref = load_lih_reference()
            cls.n_restarts = LIH_N_RESTARTS
            cls.maxfev = LIH_OPT_MAXFEV
            cls.maxiter = LIH_OPT_MAXITER
            cls.seed = 93
        else:
            raise ValueError(f"Unknown molecule '{cls.molecule}'.")

        cls.psi = np.asarray(cls.ref["v_sub"], dtype=np.complex128)
        cls.psi /= np.linalg.norm(cls.psi)
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)
        cls.h_dense = cls.ref["h_sub"].toarray()

    def _pair_variance(self, u: np.ndarray) -> float:
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

    def _projection_error(self, u: np.ndarray) -> float:
        h_rot = rotate_h_sub_dense(u, self.h_dense, self.ref["basis_bitstrings"], self.n_spatial)
        projected = project_h_sub_to_pair_parent(
            h_rot,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        return relative_projection_error(h_rot, projected)

    def test_variance_and_projection_both_improve_from_random_start(self) -> None:
        u_start = random_unitary(self.n_spatial, self.seed)
        var_start = self._pair_variance(u_start)
        proj_start = self._projection_error(u_start)
        self.assertGreater(var_start, 1e-4)
        self.assertGreater(proj_start, 1e-6)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return self._pair_variance(u)

        rng = np.random.default_rng(self.seed)
        best_thetas = None
        best_var = var_start
        for _ in range(self.n_restarts):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={"maxiter": self.maxiter, "maxfev": self.maxfev, "disp": False},
            )
            trial = float(objective(res.x))
            if trial < best_var:
                best_var = trial
                best_thetas = res.x

        assert best_thetas is not None
        u_best = build_U_from_thetas(self.n_spatial, best_thetas, self.pairs)
        proj_best = self._projection_error(u_best)

        self.assertLess(best_var, 0.5 * var_start)
        self.assertLess(proj_best, proj_start)

    def test_variance_minimum_does_not_imply_zero_projection_error(self) -> None:
        u_start = random_unitary(self.n_spatial, self.seed + 2)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return self._pair_variance(u)

        rng = np.random.default_rng(self.seed + 2)
        best_thetas = np.zeros(len(self.pairs))
        best_var = self._pair_variance(u_start)
        for _ in range(self.n_restarts):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={"maxiter": self.maxiter, "maxfev": self.maxfev, "disp": False},
            )
            trial = float(objective(res.x))
            if trial < best_var:
                best_var = trial
                best_thetas = res.x

        u_best = build_U_from_thetas(self.n_spatial, best_thetas, self.pairs)
        h_rot = rotate_h_sub_dense(u_best, self.h_dense, self.ref["basis_bitstrings"], self.n_spatial)
        projected = project_h_sub_to_pair_parent(
            h_rot,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )

        self.assertLess(best_var, 0.5)
        self.assertGreater(self._projection_error(u_best), 1e-6)
        self.assertGreater(projection_error_frobenius(h_rot, projected), 1e-8)


class TestEq32OptimizerCouplingH4(_Eq32OptimizerTests, unittest.TestCase):
    molecule = "h4"


class TestEq32OptimizerCouplingLiH(_Eq32OptimizerTests, unittest.TestCase):
    molecule = "lih"


if __name__ == "__main__":
    unittest.main()
