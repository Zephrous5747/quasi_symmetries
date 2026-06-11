"""Synthetic parent-Hamiltonian ground states must recover zero parity variance."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize

from optimization_abc_utils import ANGLE_INIT_SCALE, build_U_from_thetas, pair_list_for_n
from parity_parent_hamiltonians import (
    external_singleton_spin_orbitals,
    pair_parity_variance_sum,
    parent_ground_state,
    project_h_sub_to_pair_parent,
    project_h_sub_to_singleton_parent,
    singleton_parity_variance_sum,
)
from quartet_optimization_utils import (
    matching_edges,
    optimize_fixed_edge_quartets,
    quartet_cost_for_u,
    rotate_state_to_orbital_frame,
)
from tests.helpers import (
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    VAR_TOL,
    load_h4_reference,
    random_unitary,
)


class TestSyntheticParentEigenstate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)

        h_pair = project_h_sub_to_pair_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            cls.n_spatial,
        )
        _, cls.psi_pair = parent_ground_state(h_pair)

        external = external_singleton_spin_orbitals(
            cls.ref["n_electrons"],
            cls.n_spatial,
        )
        h_cas = project_h_sub_to_singleton_parent(
            cls.ref["h_sub"],
            cls.ref["basis_bitstrings"],
            external,
            cls.ref["n_qubits"],
        )
        _, cls.psi_cas = parent_ground_state(h_cas)

    def test_pair_parent_ground_state_not_hf(self) -> None:
        from tests.helpers import hf_state_vector

        hf = hf_state_vector(self.ref)
        overlap = abs(np.vdot(hf, self.psi_pair))
        self.assertLess(overlap, 1.0 - 1e-6)

    def test_pair_parent_ground_state_zero_pair_variance_at_identity(self) -> None:
        rotated = rotate_state_to_orbital_frame(
            self.psi_pair,
            self.ref["basis_bitstrings"],
            np.eye(self.n_spatial),
            self.n_spatial,
        )
        total = pair_parity_variance_sum(
            rotated,
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        self.assertLess(total, 1e-10)

    def test_optimize_pair_variance_from_random_unitary_pair_parent(self) -> None:
        u_start = random_unitary(self.n_spatial, seed=31)

        def pair_cost(u: np.ndarray) -> float:
            rotated = rotate_state_to_orbital_frame(
                self.psi_pair,
                self.ref["basis_bitstrings"],
                u,
                self.n_spatial,
            )
            return pair_parity_variance_sum(
                rotated,
                self.ref["basis_bitstrings"],
                self.n_spatial,
            )

        self.assertGreater(pair_cost(u_start), 1e-4)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return pair_cost(u)

        rng = np.random.default_rng(31)
        best = float("inf")
        for _ in range(N_RESTARTS):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={"maxiter": OPT_MAXITER, "maxfev": OPT_MAXFEV, "disp": False},
            )
            best = min(best, float(objective(res.x)))

        self.assertLess(best, VAR_TOL)

    def test_optimize_quartet_variance_pair_parent_ground_state(self) -> None:
        edges = matching_edges(self.n_spatial)
        u_start = random_unitary(self.n_spatial, seed=37)
        initial = quartet_cost_for_u(
            self.psi_pair,
            self.ref["basis_bitstrings"],
            u_start,
            self.n_spatial,
            edges,
        )
        self.assertGreater(initial, 1e-4)

        best = optimize_fixed_edge_quartets(
            self.psi_pair,
            self.ref["basis_bitstrings"],
            self.n_spatial,
            edges,
            n_restarts=N_RESTARTS,
            random_seed=37,
            include_zero_start=False,
            parallel=False,
            maxfev=OPT_MAXFEV,
            maxiter=OPT_MAXITER,
        )
        self.assertLess(best["cost"], VAR_TOL)

    def test_singleton_parent_ground_state_zero_external_variance_at_identity(self) -> None:
        external = external_singleton_spin_orbitals(
            self.ref["n_electrons"],
            self.n_spatial,
        )
        rotated = rotate_state_to_orbital_frame(
            self.psi_cas,
            self.ref["basis_bitstrings"],
            np.eye(self.n_spatial),
            self.n_spatial,
        )
        total = singleton_parity_variance_sum(
            rotated,
            self.ref["basis_bitstrings"],
            external,
            self.ref["n_qubits"],
        )
        self.assertLess(total, 1e-10)

    def test_optimize_singleton_variance_singleton_parent_ground_state(self) -> None:
        external = external_singleton_spin_orbitals(
            self.ref["n_electrons"],
            self.n_spatial,
        )

        def singleton_cost(u: np.ndarray) -> float:
            rotated = rotate_state_to_orbital_frame(
                self.psi_cas,
                self.ref["basis_bitstrings"],
                u,
                self.n_spatial,
            )
            return singleton_parity_variance_sum(
                rotated,
                self.ref["basis_bitstrings"],
                external,
                self.ref["n_qubits"],
            )

        u_start = random_unitary(self.n_spatial, seed=41)
        self.assertGreater(singleton_cost(u_start), 1e-4)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return singleton_cost(u)

        rng = np.random.default_rng(41)
        best = float("inf")
        for _ in range(N_RESTARTS):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={"maxiter": OPT_MAXITER, "maxfev": OPT_MAXFEV, "disp": False},
            )
            best = min(best, float(objective(res.x)))

        self.assertLess(best, VAR_TOL)


if __name__ == "__main__":
    unittest.main()
