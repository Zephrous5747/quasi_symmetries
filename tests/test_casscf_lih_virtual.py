"""CASSCF singleton parity with virtual orbitals on LiH (reduced active space)."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize

from quasi_symmetries.optimization import ANGLE_INIT_SCALE, build_U_from_thetas, pair_list_for_n
from quasi_symmetries.theory.parity_parent import (
    casscf_sector_ground_energy,
    casscf_sector_indices,
    casscf_spin_orbital_partition,
    external_singleton_spin_orbitals,
    max_reflection_commutator_frobenius,
    parent_ground_state,
    project_h_sub_to_singleton_parent,
    reflection_diagonal,
    singleton_parity_variance_sum,
)
from quasi_symmetries.optimization.quartet import rotate_state_to_orbital_frame
from tests.helpers import (
    LIH_N_RESTARTS,
    LIH_OPT_MAXFEV,
    LIH_OPT_MAXITER,
    VAR_TOL,
    hf_state_vector,
    load_lih_reference,
    random_unitary,
)

LIH_ACTIVE_SPATIAL = 2


class TestCasscfLihVirtual(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_lih_reference()
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)
        cls.psi_hf = hf_state_vector(cls.ref)
        cls.inactive, cls.active, cls.virtual = casscf_spin_orbital_partition(
            cls.ref["n_electrons"],
            cls.n_spatial,
            n_active_spatial=LIH_ACTIVE_SPATIAL,
        )
        cls.external = external_singleton_spin_orbitals(
            cls.ref["n_electrons"],
            cls.n_spatial,
            n_active_spatial=LIH_ACTIVE_SPATIAL,
        )

    def test_partition_includes_virtual_spin_orbitals(self) -> None:
        self.assertGreater(len(self.virtual), 0)
        self.assertTrue(set(self.virtual).issubset(set(self.external)))

    def test_hf_zero_external_singleton_variance_at_identity(self) -> None:
        rotated = rotate_state_to_orbital_frame(
            self.psi_hf,
            self.ref["basis_bitstrings"],
            np.eye(self.n_spatial, dtype=np.complex128),
            self.n_spatial,
        )
        cost = singleton_parity_variance_sum(
            rotated,
            self.ref["basis_bitstrings"],
            self.external,
            self.ref["n_qubits"],
        )
        self.assertLess(cost, 1e-12)

    def test_optimize_singleton_variance_from_random_unitary(self) -> None:
        u_start = random_unitary(self.n_spatial, seed=97)

        def cost(u: np.ndarray) -> float:
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

        self.assertGreater(cost(u_start), 1e-4)

        def objective(thetas: np.ndarray) -> float:
            u = build_U_from_thetas(self.n_spatial, thetas, self.pairs)
            return cost(u)

        rng = np.random.default_rng(97)
        best_cost = float("inf")
        for _ in range(LIH_N_RESTARTS):
            x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(self.pairs))
            res = minimize(
                objective,
                x0=x0,
                method="Powell",
                options={
                    "maxiter": LIH_OPT_MAXITER,
                    "maxfev": LIH_OPT_MAXFEV,
                    "disp": False,
                },
            )
            best_cost = min(best_cost, float(objective(res.x)))

        self.assertLess(best_cost, VAR_TOL)

    def test_cas_sector_nonempty_and_matches_parent_ground_energy(self) -> None:
        h_parent = project_h_sub_to_singleton_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.external,
            self.ref["n_qubits"],
        )
        sector = casscf_sector_indices(
            self.ref["basis_bitstrings"],
            self.inactive,
            self.virtual,
            self.ref["n_qubits"],
        )
        self.assertGreater(len(sector), 0)
        e_sector = casscf_sector_ground_energy(h_parent, sector)
        e_full, _ = parent_ground_state(h_parent)
        self.assertLess(abs(e_sector - e_full), 1e-6)

    def test_singleton_parent_commutes_with_virtual_reflections(self) -> None:
        h_parent = project_h_sub_to_singleton_parent(
            self.ref["h_sub"],
            self.ref["basis_bitstrings"],
            self.external,
            self.ref["n_qubits"],
        )
        for spin_orb in self.virtual[:2]:
            diag = reflection_diagonal({spin_orb}, self.ref["basis_bitstrings"], self.ref["n_qubits"])
            comm = max_reflection_commutator_frobenius(h_parent, diag)
            self.assertLess(comm, 1e-8)


if __name__ == "__main__":
    unittest.main()
