"""Tests for coupled-energy (K_coupled) diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.optimization import (
    coupled_energy_test,
    decoupled_energy_test,
    diagonalize_sector_blocks,
)


def _coupled_energy_test_legacy(H_dense, sector_data, E_exact=None, tol=1e-8, max_total_vectors=None):
    """Previous all-eigenvector greedy expansion (for regression comparison)."""
    candidates = []
    for key, data in sector_data.items():
        for e, v in zip(data["evals"], data["evecs_full"]):
            candidates.append((float(e), key, v))
    candidates.sort(key=lambda item: item[0])

    if max_total_vectors is None:
        max_total_vectors = len(candidates)

    chosen_vecs = []
    chosen_keys = []
    e_proj = None
    k_final = 0
    converged = False

    for k in range(1, min(max_total_vectors, len(candidates)) + 1):
        _e, key, vec = candidates[k - 1]
        chosen_vecs.append(vec)
        chosen_keys.append(key)
        v = np.column_stack(chosen_vecs)
        h_proj = v.conj().T @ H_dense @ v
        h_proj = 0.5 * (h_proj + h_proj.conj().T)
        e_proj = float(np.linalg.eigvalsh(h_proj)[0])
        k_final = k
        if E_exact is not None and abs(e_proj - E_exact) <= tol:
            converged = True
            break

    return e_proj, k_final, converged, chosen_keys[:k_final]


def _toy_hamiltonian_and_sectors() -> tuple[np.ndarray, dict, float]:
    """
    Four sectors: A(2D), B(2D), C(1D), D(1D).

    A has ground 0.0 and excited 0.05; B ground 0.1 with coupling to A.
    C and D are decoupled at higher energy.
    """
    dim = 6
    h = np.zeros((dim, dim), dtype=np.complex128)
    h[0, 0] = 0.0
    h[1, 1] = 0.05
    h[2, 2] = 0.1
    h[3, 3] = 0.11
    h[4, 4] = 0.2
    h[5, 5] = 0.3
    coupling = 0.02
    h[0, 2] = h[2, 0] = coupling

    sectors = {
        "A": [0, 1],
        "B": [2, 3],
        "C": [4],
        "D": [5],
    }
    sector_data = diagonalize_sector_blocks(h, sectors)
    e_exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T))[0])
    return h, sector_data, e_exact


def _toy_with_excited_cross_coupling() -> tuple[np.ndarray, dict, float]:
    """Sector A excited state couples to sector B and lowers the projected energy."""
    dim = 3
    h = np.zeros((dim, dim), dtype=np.complex128)
    h[0, 0] = 0.0
    h[1, 1] = 0.08
    h[2, 2] = 0.12
    h[0, 2] = h[2, 0] = 0.02
    h[1, 2] = h[2, 1] = 0.05

    sectors = {"A": [0, 1], "B": [2]}
    sector_data = diagonalize_sector_blocks(h, sectors)
    e_exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T))[0])
    return h, sector_data, e_exact


class CoupledEnergyTest(unittest.TestCase):
    def test_new_metric_uses_fewer_vectors_than_legacy(self) -> None:
        h, sector_data, e_exact = _toy_hamiltonian_and_sectors()

        e_new, k_new, converged_new, keys_new = coupled_energy_test(
            h, sector_data, E_exact=e_exact, tol=1e-10
        )
        e_old, k_old, converged_old, _ = _coupled_energy_test_legacy(
            h, sector_data, E_exact=e_exact, tol=1e-10
        )

        self.assertTrue(converged_new)
        self.assertTrue(converged_old)
        self.assertLess(k_new, k_old)
        self.assertEqual(k_new, 2)
        self.assertEqual(len(keys_new), 2)
        self.assertAlmostEqual(e_new, e_exact, places=10)
        self.assertAlmostEqual(e_old, e_exact, places=10)

    def test_skips_zero_coupling_sector_grounds(self) -> None:
        h, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        _e, _k, _converged, keys = coupled_energy_test(
            h, sector_data, E_exact=e_exact, tol=1e-10
        )
        sector_keys = {key for key, _index in keys}
        self.assertNotIn("C", sector_keys)
        self.assertNotIn("D", sector_keys)

    def test_skips_decoupled_same_sector_excited_states(self) -> None:
        h, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        _e, k, _converged, keys = coupled_energy_test(
            h, sector_data, E_exact=e_exact, tol=1e-10
        )
        self.assertEqual(k, 2)
        a_indices = [index for key, index in keys if key == "A"]
        self.assertEqual(a_indices, [0])

    def test_edec_within_tol_gives_k_one(self) -> None:
        h, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        e_dec, _, _ = decoupled_energy_test(h, {"A": [0, 1], "B": [2, 3], "C": [4], "D": [5]})
        # Use loose tol so decoupled minimum is within tolerance.
        tol = abs(e_dec - e_exact) + 1e-6
        _e, k, converged, _ = coupled_energy_test(
            h, sector_data, E_exact=e_exact, tol=tol
        )
        self.assertTrue(converged)
        self.assertEqual(k, 1)

    def test_allows_excited_state_when_it_couples_and_lowers_energy(self) -> None:
        h, sector_data, e_exact = _toy_with_excited_cross_coupling()

        e_new, k_new, converged_new, keys_new = coupled_energy_test(
            h, sector_data, E_exact=e_exact, tol=1e-10
        )
        e_two_ground, _, _, keys_two = coupled_energy_test(
            h,
            sector_data,
            E_exact=None,
            max_total_vectors=2,
        )

        self.assertTrue(converged_new)
        self.assertGreaterEqual(k_new, 3)
        self.assertIn(("A", 1), keys_new)
        self.assertLess(e_new, e_two_ground)
        self.assertLess(abs(e_new - e_exact), abs(e_two_ground - e_exact))


if __name__ == "__main__":
    unittest.main()
