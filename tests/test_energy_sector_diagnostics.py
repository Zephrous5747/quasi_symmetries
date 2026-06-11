"""Post-optimization E_dec and coarse sector-entropy checks on H4 and LiH."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.diagnostics.n2_action import (
    OrbitalRotationAction,
    RotatedHamiltonian,
    _coarse_entropy,
    _quartet_sectors,
)
from quasi_symmetries.optimization import (
    SparseSubspaceHamiltonian,
    build_U_from_thetas,
    decoupled_energy_lazy,
    pair_list_for_n,
)
from quasi_symmetries.optimization.quartet import (
    matching_edges,
    optimize_fixed_edge_quartets,
    quartet_parity_diagonal,
)
from tests.helpers import (
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    LIH_N_RESTARTS,
    LIH_OPT_MAXFEV,
    LIH_OPT_MAXITER,
    load_h4_reference,
    load_lih_reference,
)


def _sector_entropy_from_state(
    psi_rot: np.ndarray,
    basis_bitstrings: list[int],
    edges: list[tuple[int, int]],
    n_spatial: int,
) -> float:
    weights = np.abs(psi_rot) ** 2
    diagonals = [
        quartet_parity_diagonal(basis_bitstrings, edge, n_spatial) for edge in edges
    ]
    sector_weights: dict[tuple[float, ...], float] = {}
    for index, weight in enumerate(weights):
        key = tuple(float(diagonal[index]) for diagonal in diagonals)
        sector_weights[key] = sector_weights.get(key, 0.0) + float(weight)
    return _coarse_entropy(sector_weights)


class _EnergySectorTests:
    molecule: str = ""

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

        cls.psi = np.asarray(cls.ref["v_sub"], dtype=np.complex128)
        cls.psi /= np.linalg.norm(cls.psi)
        cls.n_spatial = cls.ref["n_spatial"]
        cls.edges = matching_edges(cls.n_spatial)
        cls.sectors = _quartet_sectors(cls.ref["basis_bitstrings"], cls.edges, cls.n_spatial)

    def _optimize_quartets(self, seed: int) -> dict:
        return optimize_fixed_edge_quartets(
            self.psi,
            self.ref["basis_bitstrings"],
            self.n_spatial,
            self.edges,
            n_restarts=self.n_restarts,
            random_seed=seed,
            include_zero_start=False,
            parallel=False,
            maxfev=self.maxfev,
            maxiter=self.maxiter,
        )

    def test_fci_quartet_optimization_lowers_variance_and_entropy(self) -> None:
        best = self._optimize_quartets(seed=71 if self.molecule == "h4" else 73)
        self.assertLess(best["cost"], 0.05)

        u = best["u_spatial"]
        action = OrbitalRotationAction(u, self.ref["basis_bitstrings"], self.n_spatial)
        psi_rot = action.apply_dagger(self.psi)
        entropy_opt = _sector_entropy_from_state(
            psi_rot,
            self.ref["basis_bitstrings"],
            self.edges,
            self.n_spatial,
        )
        entropy_id = _sector_entropy_from_state(
            self.psi,
            self.ref["basis_bitstrings"],
            self.edges,
            self.n_spatial,
        )
        self.assertLess(entropy_opt, entropy_id)

    def test_edec_defined_and_changes_under_orbital_rotation(self) -> None:
        h_identity = SparseSubspaceHamiltonian(self.ref["h_sub"])
        edec_identity, _, _ = decoupled_energy_lazy(h_identity, self.sectors)

        pairs = pair_list_for_n(self.n_spatial)
        thetas = 0.1 * np.random.default_rng(79).standard_normal(len(pairs))
        u = build_U_from_thetas(self.n_spatial, thetas, pairs)
        action = OrbitalRotationAction(u, self.ref["basis_bitstrings"], self.n_spatial)
        h_rot = RotatedHamiltonian(self.ref["h_sub"], action)
        edec_rotated, _, _ = decoupled_energy_lazy(h_rot, self.sectors)

        self.assertTrue(np.isfinite(edec_identity))
        self.assertTrue(np.isfinite(edec_rotated))
        self.assertGreater(abs(edec_identity - edec_rotated), 1e-6)

    def test_optimized_frame_edec_remains_close_to_fci(self) -> None:
        best = self._optimize_quartets(seed=83 if self.molecule == "h4" else 85)
        action = OrbitalRotationAction(
            best["u_spatial"],
            self.ref["basis_bitstrings"],
            self.n_spatial,
        )
        h_rot = RotatedHamiltonian(self.ref["h_sub"], action)
        edec_opt, _, _ = decoupled_energy_lazy(h_rot, self.sectors)
        self.assertLess(abs(edec_opt - float(self.ref["energy_fci"])), 0.05)


class TestEnergySectorDiagnosticsH4(_EnergySectorTests, unittest.TestCase):
    molecule = "h4"


class TestEnergySectorDiagnosticsLiH(_EnergySectorTests, unittest.TestCase):
    molecule = "lih"


if __name__ == "__main__":
    unittest.main()
