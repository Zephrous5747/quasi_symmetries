"""Compute sparse E_dec / K diagnostics for N2 mixed-pool optimizations."""

from __future__ import annotations

import argparse
from pathlib import Path

from quasi_symmetries.config import TABLES_DIR
from quasi_symmetries.diagnostics.n2_action import benchmark_first_mixed_pool, run_mixed_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=TABLES_DIR / "n2_mixed_pool_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=TABLES_DIR / "n2_mixed_pool_action_diagnostics.csv",
    )
    parser.add_argument(
        "--benchmark-first",
        action="store_true",
        help="Benchmark the first row only (timing + sector stats).",
    )
    args = parser.parse_args()

    if args.benchmark_first:
        benchmark_first_mixed_pool()
        return

    run_mixed_pool(input_csv=args.input_csv, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
