"""Cross-check sparse OrbitalRotationAction against dense R_fast on H4 and LiH."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.diagnostics.n2_action import OrbitalRotationAction
from quasi_symmetries.optimization import build_U_from_thetas, pair_list_for_n
from quasi_symmetries.theory.parity_parent import rotate_h_sub_dense
from quasi_symmetries.optimization.quartet import orbital_rotation_representation_R_fast
from tests.helpers import load_h4_reference, load_lih_reference, random_unitary


class _RotationEquivalenceTests:
    molecule: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if cls.molecule == "h4":
            cls.ref = load_h4_reference()
        elif cls.molecule == "lih":
            cls.ref = load_lih_reference()
        else:
            raise ValueError(f"Unknown molecule '{cls.molecule}'.")
        cls.n_spatial = cls.ref["n_spatial"]
        cls.pairs = pair_list_for_n(cls.n_spatial)
        cls.h_dense = cls.ref["h_sub"].toarray()

    def _unitaries(self) -> list[np.ndarray]:
        return [
            np.eye(self.n_spatial, dtype=np.complex128),
            random_unitary(self.n_spatial, seed=101),
            build_U_from_thetas(
                self.n_spatial,
                0.12 * np.random.default_rng(101).standard_normal(len(self.pairs)),
                self.pairs,
            ),
        ]

    def test_apply_matches_dense_r(self) -> None:
        rng = np.random.default_rng(202)
        for u in self._unitaries():
            action = OrbitalRotationAction(u, self.ref["basis_bitstrings"], self.n_spatial)
            r_fast = orbital_rotation_representation_R_fast(
                u,
                self.ref["basis_bitstrings"],
                self.n_spatial,
            )
            psi = rng.standard_normal(len(self.ref["basis_bitstrings"])) + 1j * rng.standard_normal(
                len(self.ref["basis_bitstrings"])
            )
            diff = np.linalg.norm(action.apply(psi) - r_fast @ psi)
            self.assertLess(diff, 1e-10)

    def test_apply_dagger_matches_dense_r_conjugate_transpose(self) -> None:
        rng = np.random.default_rng(203)
        for u in self._unitaries():
            action = OrbitalRotationAction(u, self.ref["basis_bitstrings"], self.n_spatial)
            r_fast = orbital_rotation_representation_R_fast(
                u,
                self.ref["basis_bitstrings"],
                self.n_spatial,
            )
            psi = rng.standard_normal(len(self.ref["basis_bitstrings"])) + 1j * rng.standard_normal(
                len(self.ref["basis_bitstrings"])
            )
            diff = np.linalg.norm(action.apply_dagger(psi) - r_fast.conj().T @ psi)
            self.assertLess(diff, 1e-10)

    def test_rotated_hamiltonian_matches_dense_formula(self) -> None:
        for u in self._unitaries():
            action = OrbitalRotationAction(u, self.ref["basis_bitstrings"], self.n_spatial)
            r_fast = orbital_rotation_representation_R_fast(
                u,
                self.ref["basis_bitstrings"],
                self.n_spatial,
            )
            psi = np.random.default_rng(204).standard_normal(self.h_dense.shape[0]) + 1j * np.random.default_rng(
                205
            ).standard_normal(self.h_dense.shape[0])

            h_rot_dense = rotate_h_sub_dense(u, self.h_dense, self.ref["basis_bitstrings"], self.n_spatial)
            lhs = action.apply_dagger(self.h_dense @ action.apply(psi))
            rhs = h_rot_dense @ psi
            rel = float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-15))
            self.assertLess(rel, 1e-9)

            fro_diff = float(np.linalg.norm(r_fast.conj().T @ self.h_dense @ r_fast - h_rot_dense))
            self.assertLess(fro_diff, 1e-9)


class TestOrbitalRotationEquivalenceH4(_RotationEquivalenceTests, unittest.TestCase):
    molecule = "h4"


class TestOrbitalRotationEquivalenceLiH(_RotationEquivalenceTests, unittest.TestCase):
    molecule = "lih"


if __name__ == "__main__":
    unittest.main()
