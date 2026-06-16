"""Compute sparse E_dec / K diagnostics for N2 parity-seniority optimizations."""

from __future__ import annotations

import argparse
from pathlib import Path

from quasi_symmetries.config import TABLES_DIR
from quasi_symmetries.diagnostics.n2_action import run_parity_seniority


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=TABLES_DIR / "n2_parity_seniority_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=TABLES_DIR / "n2_parity_seniority_action_diagnostics.csv",
    )
    args = parser.parse_args()
    run_parity_seniority(input_csv=args.input_csv, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
