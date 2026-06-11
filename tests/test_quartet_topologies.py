"""Quartet edge topologies beyond matching on H4."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.optimization.quartet import (
    balanced_tree_plus_edges,
    hub_edges,
    matching_edges,
    optimize_fixed_edge_quartets,
    ring_edges,
    run_matching_greedy_baseline,
)
from tests.helpers import (
    LIH_N_RESTARTS,
    LIH_OPT_MAXFEV,
    LIH_OPT_MAXITER,
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    VAR_TOL,
    hf_state_vector,
    load_h4_reference,
    load_lih_reference,
)


class _QuartetTopologyTests:
    molecule: str = ""
    state: str = "hf"

    @classmethod
    def setUpClass(cls) -> None:
        if cls.molecule == "h4":
            cls.ref = load_h4_reference()
            cls.n_restarts = N_RESTARTS
            cls.maxfev = OPT_MAXFEV
            cls.maxiter = OPT_MAXITER
        elif cls.molecule == "lih":
            cls.ref = load_lih_reference()
            cls.n_restarts = LIH_N_RESTARTS
            cls.maxfev = LIH_OPT_MAXFEV
            cls.maxiter = LIH_OPT_MAXITER
        else:
            raise ValueError(f"Unknown molecule '{cls.molecule}'.")

        if cls.state == "hf":
            cls.psi = hf_state_vector(cls.ref)
        elif cls.state == "fci":
            cls.psi = np.asarray(cls.ref["v_sub"], dtype=np.complex128)
            cls.psi /= np.linalg.norm(cls.psi)
        else:
            raise ValueError(f"Unknown state '{cls.state}'.")
        cls.n_spatial = cls.ref["n_spatial"]

    def _optimize_topology(self, edges: list[tuple[int, int]], seed: int) -> float:
        best = optimize_fixed_edge_quartets(
            self.psi,
            self.ref["basis_bitstrings"],
            self.n_spatial,
            edges,
            n_restarts=self.n_restarts,
            random_seed=seed,
            include_zero_start=True,
            parallel=False,
            maxfev=self.maxfev,
            maxiter=self.maxiter,
        )
        return float(best["cost"])


class TestQuartetTopologiesH4(_QuartetTopologyTests, unittest.TestCase):
    molecule = "h4"
    state = "hf"

    def test_ring_topology_hf_exact_recovery(self) -> None:
        cost = self._optimize_topology(ring_edges(self.n_spatial), seed=61)
        self.assertLess(cost, VAR_TOL)

    def test_balanced_tree_topology_hf_exact_recovery(self) -> None:
        cost = self._optimize_topology(balanced_tree_plus_edges(self.n_spatial), seed=67)
        self.assertLess(cost, VAR_TOL)

    def test_hub_topology_hf_exact_recovery(self) -> None:
        cost = self._optimize_topology(hub_edges(self.n_spatial), seed=69)
        self.assertLess(cost, VAR_TOL)

    def test_matching_greedy_topology_hf_exact_recovery(self) -> None:
        result = run_matching_greedy_baseline(
            self.psi,
            self.ref["basis_bitstrings"],
            self.n_spatial,
            n_restarts=self.n_restarts,
            include_zero_start=True,
            parallel_restarts=False,
            maxfev=self.maxfev,
            maxiter=self.maxiter,
        )
        self.assertLess(float(result["final"]["cost"]), VAR_TOL)

    def test_ring_and_matching_both_zero_at_identity_for_hf(self) -> None:
        from quasi_symmetries.optimization.quartet import quartet_cost_for_u

        u = np.eye(self.n_spatial, dtype=np.complex128)
        for edges in (
            ring_edges(self.n_spatial),
            balanced_tree_plus_edges(self.n_spatial),
            hub_edges(self.n_spatial),
            matching_edges(self.n_spatial),
        ):
            cost = quartet_cost_for_u(
                self.psi,
                self.ref["basis_bitstrings"],
                u,
                self.n_spatial,
                edges,
            )
            self.assertLess(cost, 1e-12)


class TestQuartetTopologiesLiHFci(_QuartetTopologyTests, unittest.TestCase):
    molecule = "lih"
    state = "fci"

    def test_hub_topology_does_not_increase_cost(self) -> None:
        from quasi_symmetries.optimization.quartet import quartet_cost_for_u

        edges = hub_edges(self.n_spatial)
        u_start = np.eye(self.n_spatial, dtype=np.complex128)
        initial = quartet_cost_for_u(
            self.psi,
            self.ref["basis_bitstrings"],
            u_start,
            self.n_spatial,
            edges,
        )
        cost = self._optimize_topology(edges, seed=75)
        self.assertLessEqual(cost, initial + 1e-10)
        self.assertLess(cost, 0.05)


if __name__ == "__main__":
    unittest.main()
