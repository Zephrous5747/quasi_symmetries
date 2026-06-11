"""check_R_vs_direct_seniority on H4."""

from __future__ import annotations

import unittest

import numpy as np

from optimization_abc_utils import (
    ANGLE_INIT_SCALE,
    build_U_from_thetas,
    check_R_vs_direct_seniority,
    pair_list_for_n,
)
from tests.helpers import STANDARD_ABC, load_h4_reference


class TestSeniorityRotationConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = load_h4_reference()

    def test_identity_frame_seniority_rotation(self) -> None:
        a, b, c = STANDARD_ABC
        u = np.eye(self.ref["n_spatial"], dtype=np.complex128)
        out = check_R_vs_direct_seniority(
            u,
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
            self.ref["n_qubits"],
            a,
            b,
            c,
        )
        self.assertLess(out["rel_total"], 1e-8)

    def test_givens_chain_frame_seniority_rotation(self) -> None:
        """Use the same Givens-chain parameterization as the variance optimizer."""
        a, b, c = STANDARD_ABC
        pairs = pair_list_for_n(self.ref["n_spatial"])
        thetas = ANGLE_INIT_SCALE * np.random.default_rng(71).standard_normal(len(pairs))
        u = build_U_from_thetas(self.ref["n_spatial"], thetas, pairs)
        out = check_R_vs_direct_seniority(
            u,
            self.ref["basis_bitstrings"],
            self.ref["n_spatial"],
            self.ref["n_qubits"],
            a,
            b,
            c,
        )
        self.assertLess(out["rel_total"], 1e-6)


if __name__ == "__main__":
    unittest.main()
