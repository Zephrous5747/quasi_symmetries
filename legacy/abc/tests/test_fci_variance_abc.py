"""FCI-state ABC commutator behavior on H4 (archived fixed-abc workflow)."""

from __future__ import annotations

import unittest

import numpy as np

from quasi_symmetries.optimization import (
    OP_COEF_TOL,
    analyze_individual_symmetry_operators_with_leakage_subspace,
)
from tests.helpers import STANDARD_ABC, load_h4_reference


class TestFciAbcVariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()
        cls.psi = np.asarray(cls.ref["v_sub"], dtype=np.complex128)
        cls.psi /= np.linalg.norm(cls.psi)
        cls.n_spatial = cls.ref["n_spatial"]

    def test_fci_fixed_abc_commutators_positive_at_identity(self) -> None:
        a, b, c = STANDARD_ABC
        result = analyze_individual_symmetry_operators_with_leakage_subspace(
            self.ref["h_sub"],
            self.psi,
            self.ref["basis_bitstrings"],
            np.eye(self.n_spatial),
            self.n_spatial,
            self.ref["n_qubits"],
            a,
            b,
            c,
            label="h4_fci",
            tol=OP_COEF_TOL,
            check_eigenstate=False,
        )
        self.assertGreater(float(result["sum_comm_sq"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
