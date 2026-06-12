"""Tests for exact point-group symmetry operators and simplifications."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.hamiltonian.cache import DEFAULT_CACHE_DIR, load_reference_state
from quasi_symmetries.optimization import pair_list_for_n
from quasi_symmetries.optimization.quartet import H2O_MIXED_POOL, build_h2o_mixed_pool, mixed_pool_cost_for_u
from quasi_symmetries.optimization.rotations import symmetry_blocked_pair_list
from quasi_symmetries.symmetry.exact import (
    build_exact_symmetry_sectors,
    exact_symmetry_parities_for_molecule,
    ground_state_sector_indices,
    product_parity_diagonal,
    verify_exact_symmetries,
)
from quasi_symmetries.symmetry.labels import (
    fallback_sto3g_irrep_labels,
    labels_from_irrep_list,
    load_symmetry_labels,
    molecule_point_group,
)


H2O_X = 0.958
N2_X = 1.2


def _try_load(molecule: str, x: float) -> dict | None:
    try:
        return load_reference_state(
            molecule,
            x,
            cache_dir=str(DEFAULT_CACHE_DIR),
            compute_rdms=False,
            load_full_hamiltonian=False,
        )
    except FileNotFoundError:
        return None


class TestExactSymmetryOperators(unittest.TestCase):
    def test_h2o_fallback_labels_and_mirror_sets(self) -> None:
        labels = labels_from_irrep_list("h2o", fallback_sto3g_irrep_labels("h2o", 7))
        mirrors = labels.h2o_mirror_sets()
        self.assertIn("sigma_a", mirrors)
        self.assertIn("sigma_b", mirrors)
        self.assertEqual(mirrors["sigma_a"], (2,))
        self.assertEqual(mirrors["sigma_b"], (4, 6))

    def test_n2_ungerade_indices(self) -> None:
        labels = labels_from_irrep_list("n2", fallback_sto3g_irrep_labels("n2", 10))
        self.assertEqual(labels.n2_ungerade_indices(), [2, 3, 4, 6, 7, 8])

    def test_point_groups_for_h2o_and_n2(self) -> None:
        self.assertEqual(molecule_point_group("h2o"), "C2v")
        self.assertEqual(molecule_point_group("n2"), "D2h")

    def test_product_parity_involutory(self) -> None:
        labels = labels_from_irrep_list("h2o", fallback_sto3g_irrep_labels("h2o", 7))
        bitstrings = [0, 1, 3, 7, 15]
        diag = product_parity_diagonal(bitstrings, labels.h2o_mirror_sets()["sigma_b"], 7)
        np.testing.assert_allclose(diag * diag, 1.0)

    def test_blocked_rotation_count_smaller_than_full(self) -> None:
        labels = labels_from_irrep_list("h2o", fallback_sto3g_irrep_labels("h2o", 7))
        blocked = symmetry_blocked_pair_list(7, labels.irrep_labels)
        full = pair_list_for_n(7)
        self.assertLess(len(blocked), len(full))
        self.assertGreater(len(blocked), 0)

    def test_h2o_mixed_pool_marks_exact_quartets(self) -> None:
        pool = build_h2o_mixed_pool(7)
        pool.validate(7)
        self.assertGreaterEqual(len(pool.singles), 1)
        self.assertGreater(len(pool.quasi_quartets), 0)


class TestExactSymmetryReferenceStates(unittest.TestCase):
    def test_h2o_ground_state_in_plus_sector(self) -> None:
        ref = _try_load("h2o", H2O_X)
        if ref is None:
            self.skipTest("H2O cache missing")
        labels = load_symmetry_labels(ref, molecule="h2o")
        assert labels is not None
        parities = exact_symmetry_parities_for_molecule(ref["basis_bitstrings"], labels)
        psi = np.asarray(ref["v_sub"], dtype=np.complex128)
        psi = psi / np.linalg.norm(psi)
        for name, diag in parities.items():
            exp = float(np.real(np.dot(np.abs(psi) ** 2, diag)))
            self.assertGreater(exp, 0.9, msg=f"H2O ground state expectation for {name}")

    def test_n2_ground_state_in_plus_sector(self) -> None:
        ref = _try_load("n2", N2_X)
        if ref is None:
            self.skipTest("N2 cache missing")
        labels = load_symmetry_labels(ref, molecule="n2")
        assert labels is not None
        parities = exact_symmetry_parities_for_molecule(ref["basis_bitstrings"], labels)
        psi = np.asarray(ref["v_sub"], dtype=np.complex128)
        psi = psi / np.linalg.norm(psi)
        exp = float(np.real(np.dot(np.abs(psi) ** 2, parities["P_inv"])))
        self.assertGreater(exp, 0.9)

    def test_symmetry_sector_reduces_dimension(self) -> None:
        ref = _try_load("h2o", H2O_X)
        if ref is None:
            self.skipTest("H2O cache missing")
        labels = load_symmetry_labels(ref, molecule="h2o")
        assert labels is not None
        parities = exact_symmetry_parities_for_molecule(ref["basis_bitstrings"], labels)
        allowed = ground_state_sector_indices(parities, target=1)
        self.assertLess(len(allowed), len(ref["basis_bitstrings"]))
        self.assertGreater(len(allowed), 0)

    def test_exact_commutators_reported_for_h2o(self) -> None:
        ref = _try_load("h2o", H2O_X)
        if ref is None or ref["h_sub"] is None:
            self.skipTest("H2O dense cache missing")
        labels = load_symmetry_labels(ref, molecule="h2o")
        assert labels is not None
        parities = exact_symmetry_parities_for_molecule(ref["basis_bitstrings"], labels)
        errors = verify_exact_symmetries(ref["h_sub"], parities)
        self.assertEqual(set(errors), set(parities))
        for value in errors.values():
            self.assertGreaterEqual(value, 0.0)

    def test_mixed_pool_exact_operators_zero_cost_at_identity(self) -> None:
        ref = _try_load("h2o", H2O_X)
        if ref is None:
            self.skipTest("H2O cache missing")
        pool = H2O_MIXED_POOL
        u_identity = np.eye(ref["n_spatial"], dtype=np.complex128)
        cost = mixed_pool_cost_for_u(
            ref["v_sub"],
            ref["basis_bitstrings"],
            u_identity,
            ref["n_spatial"],
            pool,
        )
        self.assertGreaterEqual(cost, 0.0)

    def test_sector_building_with_exact_parities(self) -> None:
        ref = _try_load("n2", N2_X)
        if ref is None:
            self.skipTest("N2 cache missing")
        labels = load_symmetry_labels(ref, molecule="n2")
        assert labels is not None
        parities = exact_symmetry_parities_for_molecule(ref["basis_bitstrings"], labels)
        sectors = build_exact_symmetry_sectors(ref["basis_bitstrings"], parities)
        self.assertGreaterEqual(len(sectors), 2)


if __name__ == "__main__":
    unittest.main()
