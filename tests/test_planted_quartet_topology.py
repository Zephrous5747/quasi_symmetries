"""Eq. (9)-(10) parent Hamiltonians: confusion + multi-protocol optimization survey."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.theory.parity_parent import (
    max_reflection_commutator_frobenius,
    pair_parity_diagonal,
    project_h_sub_to_pair_parent,
    project_h_sub_to_polynomial_parent,
)
from tests.helpers import (
    K_COUPLED_TOL,
    N_RESTARTS,
    OPT_MAXFEV,
    OPT_MAXITER,
    PARENT_PROTOCOL_NAMES,
    PLANTED_SEED,
    build_parent_hamiltonian,
    load_h4_reference,
    load_lih_reference,
    quartet_topology_builders,
    run_parent_hamiltonian_survey,
)


class TestParentHamiltonianConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.n_spatial = cls.ref["n_spatial"]
        cls.builders = quartet_topology_builders()

    def test_quartet_parents_commute_with_their_product_reflections(self) -> None:
        for parent_protocol in ("matching", "ring", "hub", "balanced_tree"):
            with self.subTest(parent_protocol=parent_protocol):
                parent = build_parent_hamiltonian(self.ref, parent_protocol)
                edges = self.builders[parent_protocol](self.n_spatial)
                for p, q in edges:
                    d_p = pair_parity_diagonal(p, self.ref["basis_bitstrings"], self.n_spatial)
                    d_q = pair_parity_diagonal(q, self.ref["basis_bitstrings"], self.n_spatial)
                    comm = max_reflection_commutator_frobenius(parent["h_dense"], d_p * d_q)
                    self.assertLess(comm, 1e-8)

    def test_seniority_parent_commutes_with_pair_reflections(self) -> None:
        parent = build_parent_hamiltonian(self.ref, "seniority")
        for orbital in range(self.n_spatial):
            diag = pair_parity_diagonal(orbital, self.ref["basis_bitstrings"], self.n_spatial)
            comm = max_reflection_commutator_frobenius(parent["h_dense"], diag)
            self.assertLess(comm, 1e-8)

    def test_h4_quartet_parents_share_fci_ground_energy(self) -> None:
        energies = [
            build_parent_hamiltonian(self.ref, name)["energy"]
            for name in ("matching", "ring", "hub", "balanced_tree")
        ]
        self.assertTrue(max(energies) - min(energies) < 1e-8)

    def test_h4_seniority_parent_has_distinct_energy(self) -> None:
        quartet_energy = build_parent_hamiltonian(self.ref, "matching")["energy"]
        seniority_energy = build_parent_hamiltonian(self.ref, "seniority")["energy"]
        self.assertNotAlmostEqual(quartet_energy, seniority_energy, places=4)

    def test_h4_parent_psi_is_hp_ground_state_not_fci(self) -> None:
        psi_fci = np.asarray(self.ref["v_sub"], dtype=np.complex128).ravel()
        psi_fci /= np.linalg.norm(psi_fci)
        e_fci = float(self.ref["energy_fci"])

        for parent_protocol in ("matching", "seniority"):
            with self.subTest(parent_protocol=parent_protocol):
                parent = build_parent_hamiltonian(self.ref, parent_protocol)
                h = parent["h_dense"]
                psi = parent["psi"]
                energy = parent["energy"]
                residual = float(np.linalg.norm(h @ psi - energy * psi))
                self.assertLess(residual, 1e-10)
                self.assertAlmostEqual(float(np.real(np.vdot(psi, h @ psi))), energy, places=10)

                overlap = abs(np.vdot(psi_fci, psi))
                if parent_protocol == "seniority":
                    self.assertLess(overlap, 0.999)
                    self.assertGreater(energy, e_fci + 0.01)
                else:
                    # Quartet parent on H4 is near-degenerate with FCI but psi is recomputed.
                    self.assertGreater(1.0 - overlap, 1e-5)


class TestParentHamiltonianSurvey(unittest.TestCase):
    def test_h4_survey_shape(self) -> None:
        ref = load_h4_reference()
        rows = run_parent_hamiltonian_survey(
            ref,
            apply_confusion=True,
            confusion_seed=PLANTED_SEED,
            n_restarts=N_RESTARTS,
            random_seed=PLANTED_SEED,
            maxfev=OPT_MAXFEV,
            maxiter=OPT_MAXITER,
        )
        parent_names = {row["ParentProtocol"] for row in rows}
        self.assertEqual(parent_names, set(PARENT_PROTOCOL_NAMES))
        self.assertEqual(len(rows), len(PARENT_PROTOCOL_NAMES) ** 2)

    def test_lih_parent_construction(self) -> None:
        ref = load_lih_reference()
        parent = build_parent_hamiltonian(ref, "seniority")
        self.assertEqual(parent["construction"], "pair_parity_parent")
        self.assertEqual(len(ref["basis_bitstrings"]), 495)


if __name__ == "__main__":
    unittest.main()
