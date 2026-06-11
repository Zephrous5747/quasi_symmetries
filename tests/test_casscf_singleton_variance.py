"""Phase 1C: CASSCF singleton-parity variance and CAS sector checks on H4."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize

from optimization_abc_utils import ANGLE_INIT_SCALE, build_U_from_thetas, pair_list_for_n
from parity_parent_hamiltonians import (
    casscf_sector_ground_energy,
    casscf_sector_indices,
    casscf_spin_orbital_partition,
    external_singleton_spin_orbitals,
    parent_ground_state,
    project_h_sub_to_singleton_parent,
    singleton_parity_variance_sum,
)
from quartet_optimization_utils import rotate_state_to_orbital_frame
from tests.helpers import (
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    VAR_TOL,
    hf_state_vector,
    load_h4_reference,
    random_unitary,
)


class TestCasscfSingletonVariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.psi_hf = hf_state_vector(cls.ref)
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)
        cls.external = external_singleton_spin_orbitals(
            cls.ref["n_electrons"],
            cls.n_spatial,
        )

    def _singleton_cost(self, u: np.ndarray) -> float:
        rotated = rotate_state_to_orbital_frame(
            self.psi_hf,
            self.ref["basis_bitstrings"],
            u,
            self.n_spatial,
        )
        return singleton_parity_variance_sum(
            rotated,
            self.ref["basis_bitstrings"],
            self.external,
            self.ref["n_qubits"],
        )

    def test_hf_zero_external_singleton_variance_at_identity(self) -> None:
        u = np.eye(self.n_spatial, dtype=np.complex128)
        self.assertLess(self._singleton_cost(u), 1e-12)

    def test_optimize_singleton_variance_from_random_unitary(self) -> None:
        u_start = random_unitary(self.n_spatial, seed=23)
        self.assertGreater(self._singleton_cost(u_start), 1e-4)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return self._singleton_cost(u)

        rng = np.random.default_rng(23)
        best_cost = float("inf")
        for _ in range(N_RESTARTS):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={"maxiter": OPT_MAXITER, "maxfev": OPT_MAXFEV, "disp": False},
            )
            best_cost = min(best_cost, float(objective(res.x)))

        self.assertLess(best_cost, VAR_TOL)

    def test_cas_sector_ground_energy_matches_parent_diagonalization(self) -> None:
        inactive, _, virtual = casscf_spin_orbital_partition(
            self.ref["n_electrons"],
            self.n_spatial,
        )
        h_parent = project_h_sub_to_singleton_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            inactive + virtual,
            self.ref["n_qubits"],
        )
        sector = casscf_sector_indices(
            self.ref["basis_bitstrings"],
            inactive,
            virtual,
            self.ref["n_qubits"],
        )
        e_sector = casscf_sector_ground_energy(h_parent, sector)
        e_full, _ = parent_ground_state(h_parent)
        self.assertLess(abs(e_sector - e_full), 1e-8)


if __name__ == "__main__":
    unittest.main()
