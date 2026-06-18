"""Tests for sparse mixed-pool energy-sector diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from quasi_symmetries.diagnostics.mixed_pool import mixed_pool_sectors
from quasi_symmetries.diagnostics.sparse_energy import (
    SparseSubspaceHamiltonian,
    coupled_energy_lazy,
    decoupled_energy_lazy,
    energy_sector_diagnostics_sparse,
)
from quasi_symmetries.optimization.quartet import MixedOperatorPool


def _toy_sparse_hamiltonian() -> tuple[sp.csc_matrix, dict[tuple[int, ...], list[int]], float]:
    """Four basis states, two sectors from single parity on orbital 0."""
    dim = 4
    h_dense = np.diag([0.0, 0.05, 0.1, 0.15]).astype(np.complex128)
    h_sub = sp.csc_matrix(h_dense)
    basis_bitstrings = [0b0011, 0b0101, 0b1001, 0b1100]
    pool = MixedOperatorPool(singles=(0,), quartets=())
    sectors = mixed_pool_sectors(basis_bitstrings, pool, n_spatial=2)
    e_exact = 0.0
    return h_sub, sectors, e_exact


class MixedPoolSparseEnergyTests(unittest.TestCase):
    def test_sector_count_matches_mixed_pool_sectors(self) -> None:
        h_sub, sectors, _ = _toy_sparse_hamiltonian()
        self.assertEqual(len(sectors), 2)
        self.assertEqual(sum(len(idxs) for idxs in sectors.values()), h_sub.shape[0])

    def test_edec_within_tol_gives_k_one(self) -> None:
        h_sub, sectors, e_exact = _toy_sparse_hamiltonian()
        h_op = SparseSubspaceHamiltonian(h_sub)
        e_dec, _, _ = decoupled_energy_lazy(h_op, sectors)
        self.assertAlmostEqual(e_dec, e_exact, places=10)

        result = energy_sector_diagnostics_sparse(
            h_op,
            sectors,
            e_exact,
            tol=1e-3,
            states_per_sector=2,
        )
        self.assertEqual(result["Kcoupled"], 1)
        self.assertTrue(result["Coupled_Converged"])

    def test_coupled_energy_lazy_matches_edec_when_k_one(self) -> None:
        h_sub, sectors, e_exact = _toy_sparse_hamiltonian()
        h_op = SparseSubspaceHamiltonian(h_sub)
        e_coupled, k_coupled, converged, _ = coupled_energy_lazy(
            h_op,
            sectors,
            E_exact=e_exact,
            tol=1e-3,
        )
        e_dec, _, _ = decoupled_energy_lazy(h_op, sectors)
        self.assertEqual(k_coupled, 1)
        self.assertTrue(converged)
        self.assertAlmostEqual(e_coupled, e_dec, places=10)

    def test_energy_sector_sparse_early_k_one_skips_coupled(self) -> None:
        h_sub, sectors, e_exact = _toy_sparse_hamiltonian()
        h_op = SparseSubspaceHamiltonian(h_sub)
        result = energy_sector_diagnostics_sparse(
            h_op,
            sectors,
            e_exact,
            tol=1e-3,
            profile=True,
        )
        profile = result.pop("_profile")
        self.assertEqual(result["Kcoupled"], 1)
        self.assertEqual(profile.get("coupled_seconds"), 0.0)
        h_sub, sectors, e_exact = _toy_sparse_hamiltonian()
        h_op = SparseSubspaceHamiltonian(h_sub)
        e_coupled, k_coupled, converged, _ = coupled_energy_lazy(
            h_op,
            sectors,
            E_exact=e_exact,
            tol=1e-3,
        )
        e_dec, _, _ = decoupled_energy_lazy(h_op, sectors)
        self.assertEqual(k_coupled, 1)
        self.assertTrue(converged)
        self.assertAlmostEqual(e_coupled, e_dec, places=10)


if __name__ == "__main__":
    unittest.main()
